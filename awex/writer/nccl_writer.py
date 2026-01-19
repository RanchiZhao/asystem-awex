# Licensed to the Awex developers under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import gc
import os
import time

import torch
import torch.distributed as dist

from awex import logging
from awex.transfer.nccl_comm import batch_send_recv, nccl_build_send_ops
from awex.transfer.transfer_plan import TransferPlanBuilder
from awex.util.common import compute_statistics, get_ip_address
from awex.util.gpu import print_current_gpu_status
from awex.util.process_group import init_weights_update_group, setup_batch_isend_irecv
from awex.util.system_util import count_open_fds
from awex.util.tensor_util import (
    cuda_ipc_serialize,
    group_tensors_by_shape_and_dtype,
    ipc_serialize,
    release_tensors,
)
from awex.writer.weights_writer import WeightsExchangeShardingWriter

logger = logging.getLogger(__name__)


class NCCLWeightsWriter(WeightsExchangeShardingWriter):
    def _initialize(self):
        super()._initialize()
        logger.info(
            f"Start to initialize NCCL weights writer for rank {self.transfer_rank}"
        )
        if self.enable_colocate_mode:
            self._init_writer_in_colocate_mode()
            return
        logger.info(f"Start to build transfer plan for rank {self.transfer_rank}")
        self.transfer_plan = TransferPlanBuilder(
            self.infer_world_size,
            self.training_world_size,
            self.num_infer_engines,
            self.enable_debug_mode,
            hf_config=self.hf_config,
        ).build_local_transfer_plan(
            self.infer_params_meta,
            self.parameters_meta,
            self.transfer_rank,
        )
        self.recv_ranks = list(self.transfer_plan.operations.keys())
        logger.info(
            f"Writer rank {self.transfer_rank}: Built transfer plan to send to ranks: {self.recv_ranks}"
        )
        logger.info(
            f"Writer rank {self.transfer_rank}: Operations per rank: {[(rank, len(ops)) for rank, ops in self.transfer_plan.operations.items()]}"
        )
        self.recv_ranks_sample = (
            self.recv_ranks[:8] + ["..."] + self.recv_ranks[-8:]
            if len(self.recv_ranks) > 16
            else self.recv_ranks
        )
        self.num_to_sends = sum(
            len(operations) for operations in self.transfer_plan.operations.values()
        )
        logger.info(f"Finished building transfer plan for rank {self.transfer_rank}")
        logger.info(
            f"Start to get master info from meta server for rank {self.transfer_rank}"
        )
        master_info = self.meta_server_client.get_object(
            "master_info", timeout=self.timeout
        )
        master_address, master_port = master_info
        logger.info(
            f"Get master info from meta server for rank {self.transfer_rank}: {master_info}"
        )
        self._set_device()
        self.weights_update_group = init_weights_update_group(
            master_address,
            master_port,
            self.transfer_rank,
            self.transfer_world_size,
            "weights_exchange",
            role="train",
        )
        logger.info(f"Initialized NCCL weights writer for rank {self.transfer_rank}")
        # Add a barrier to ensure all processes are ready
        dist.barrier(
            group=self.weights_update_group, device_ids=[torch.cuda.current_device()]
        )
        logger.info(f"Barrier passed for weights writer with rank {self.transfer_rank}")
        if self.transfer_rank == self.transfer_world_size - 1:
            logger.info(
                f"Start to test NCCL ready for rank {self.transfer_rank}, world size {self.transfer_world_size}"
            )
            dist.send(
                torch.tensor(1).cuda(),
                dst=0,
                group=self.weights_update_group,
            )
            logger.info(
                f"NCCL ready: send tensor to rank {self.transfer_world_size - 1} from rank {self.transfer_rank}"
            )
        setup_batch_isend_irecv(
            self.weights_update_group, self.transfer_rank, self.transfer_world_size
        )
        logger.info(
            f"Finished initializing NCCL weights writer for rank {self.transfer_rank}"
        )

    def _set_device(self):
        device = torch.cuda.current_device()
        gpu_id = int(os.environ.get("DEVICE", device)) % torch.cuda.device_count()
        logger.info(
            f"[NCCLWeightsWriter] Set device to {gpu_id} for rank {self.transfer_rank}, device env is {os.environ.get('DEVICE')}, "
            f"previous device is {device}, device_count is {torch.cuda.device_count()}, "
            f"CUDA_VISIBLE_DEVICES env is {os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )
        torch.cuda.set_device(gpu_id)

    def _init_writer_in_colocate_mode(self):
        self.ipc_backend = self.config.get(
            "weights_exchange_ipc_backend", "cuda"
        )
        # Don't get IPC tensors here since every step, the memory address for weights will change
        # because we use offloading for moving GPU tensors to CPU and back later
        ip_address = get_ip_address()
        self._set_device()
        device_id = torch.cuda.current_device()

        # CRITICAL: Wait for inference side to finish cleanup
        # The inference side uses two-phase cleanup with epoch:
        # - inference_cleanup_starting: signals cleanup is starting
        # - inference_cleanup_done: signals cleanup is complete
        # We wait for inference_cleanup_done to ensure cleanup is complete before registering
        cleanup_done_key = "inference_cleanup_done"
        cleanup_barrier_key = "inference_cleanup_barrier"  # Legacy key for backwards compatibility
        import time
        max_retries = 120  # Wait up to 120 seconds for inference to start and complete cleanup
        cleanup_found = False
        for i in range(max_retries):
            try:
                # Try new key first
                self.meta_server_client.get_object(cleanup_done_key, timeout=3)
                logger.info(
                    f"[NCCLWeightsWriter] Rank {self.transfer_rank}: inference cleanup_done found"
                )
                cleanup_found = True
                break
            except Exception:
                try:
                    # Fallback to legacy key for backwards compatibility
                    self.meta_server_client.get_object(cleanup_barrier_key, timeout=2)
                    logger.info(
                        f"[NCCLWeightsWriter] Rank {self.transfer_rank}: inference cleanup_barrier (legacy) found"
                    )
                    cleanup_found = True
                    break
                except Exception:
                    if i < max_retries - 1:
                        if i % 10 == 0:
                            logger.info(
                                f"[NCCLWeightsWriter] Rank {self.transfer_rank}: waiting for inference cleanup... ({i+1}/{max_retries})"
                            )
                        time.sleep(1)
                    else:
                        logger.warning(
                            f"[NCCLWeightsWriter] Rank {self.transfer_rank}: timeout waiting for inference cleanup, proceeding anyway"
                        )

        self.meta_server_client.add_object_to_set(
            "training_device_rank_entries", (ip_address, device_id, self.transfer_rank)
        )
        # [TRAIN_REGISTER] Log registration details for debugging rank mismatch
        megatron_rank = self.transfer_rank - self.infer_world_size
        pp_rank = getattr(self.rank_info, 'pp_rank', 'N/A')
        tp_rank = getattr(self.rank_info, 'tp_rank', 'N/A')
        logger.info(
            f"[TRAIN_REGISTER] ip={ip_address} device={device_id} transfer_rank={self.transfer_rank} "
            f"megatron_rank={megatron_rank} pp_rank={pp_rank} tp_rank={tp_rank}"
        )

        # Clean up stale IPC keys from previous runs
        for step_id in [1]:
            key_suffix = f"_{ip_address}_{device_id}_{step_id}"
            serialized_weights_key = f"training_serialized_weights{key_suffix}"
            update_finished_key = f"weights_update_finished{key_suffix}"
            self.meta_server_client.delete_if_exists(serialized_weights_key)
            self.meta_server_client.delete_if_exists(update_finished_key)
        logger.info(
            f"Initialized NCCL weights writer for rank {self.transfer_rank} in colocate mode "
            f"(cleaned stale keys for {ip_address}:{device_id})"
        )

    @torch.no_grad()
    def _write_weights(self, step_id, **kwargs):
        """
        Asynchronously send weights to inference ranks using torch.distributed.isend.

        This method implements a pipelined approach where:
        1. For each sender rank, we maintain a queue of operations to send
        2. We start isend operations in parallel to all sender ranks
        3. When a send completes, we immediately start the next send to that rank
        4. We continue until all operations to all sender ranks are completed

        Args:
            step_id: The training step ID used as communication tag
            **kwargs: Additional keyword arguments (unused)
        """
        rank_coordinate = self.transfer_rank
        logger.info(
            f"Start to send weights using NCCL to {len(self.transfer_plan.operations)} "
            f"ranks({self.recv_ranks_sample}) from rank {rank_coordinate} "
            f"with {self.num_to_sends} sends"
        )
        start_time = time.time()
        parameters = self.convert_parameters()
        logger.info("Writer: Converting parameters completed, building send ops")
        p2p_op_list, _ = nccl_build_send_ops(
            parameters, self.transfer_plan, self.weights_update_group, -1
        )
        logger.info(
            f"Writer: Built {len(p2p_op_list)} send operations to "
            f"{len(self.transfer_plan.operations)} ranks"
        )

        # Execute all sends via batch_send_recv to get consistent interleaving
        # and per-peer stream assignment without relying directly on
        # batch_isend_irecv.
        logger.info(
            f"Writer: Executing {len(p2p_op_list)} send ops via batch_send_recv"
        )
        batch_send_recv(
            send_ops=p2p_op_list, recv_ops=[], blocking=True, use_group=True
        )
        torch.cuda.synchronize(device=torch.cuda.current_device())
        duration = time.time() - start_time
        logger.info(
            f"Finished sending weights for step {step_id} using NCCL to {len(self.transfer_plan.operations)} ranks({self.recv_ranks_sample}) "
            f"from rank {rank_coordinate} with {self.num_to_sends} sends, took {duration:.4f} seconds"
        )
        compute_statistics(
            self._history_write_weights_time,
            step_id,
            duration,
            "Send weights using NCCL",
        )
        dist.barrier(
            group=self.weights_update_group, device_ids=[torch.cuda.current_device()]
        )
        logger.info(
            f"Barrier passed for writer step {step_id} with rank {self.transfer_rank}"
        )

    @torch.no_grad()
    def _prepare_params_for_colocate(self):
        logger.info(
            f"Start to write weights in colocate mode for rank {self.transfer_rank}"
        )
        # CRITICAL: Resume memory before accessing model parameters
        # In Slime's offload_train mode, model weights are on CPU after sleep()
        # We need to bring them back to GPU before convert_parameters() can work
        self.train_engine.resume_memory_occupation("weights")
        self.train_engine.release_grad_memory()
        converted = self.convert_parameters()
        tensors, names = [], []
        for name, tensor in converted.items():
            assert not tensor.requires_grad
            tensors.append(tensor)
            names.append(name)
        return tensors, names

    @torch.no_grad()
    def _write_weights_in_colocate_mode(self, step_id, **kwargs):
        total_start = time.time()

        # [PROFILE] Step 1: Prepare params (resume + convert)
        t0 = time.time()
        tensors, names = self._prepare_params_for_colocate()
        num_tensors = len(tensors)
        t1 = time.time()
        prepare_time = t1 - t0
        logger.info(f"[PROFILE] train_rank={self.transfer_rank} step={step_id} prepare_params: {prepare_time:.3f}s ({num_tensors} tensors)")

        # [PROFILE] Step 2: Group tensors
        if self.ipc_backend == "cpu":
            tensors = [t.cpu() for t in tensors]
        logger.info(
            f"Start to group tensors by shape and dtype for rank {self.transfer_rank}"
        )
        # this will copy tensor by concatenate
        group_tensors, metadata = group_tensors_by_shape_and_dtype(tensors)
        torch.cuda.synchronize(device=torch.cuda.current_device())
        t2 = time.time()
        group_time = t2 - t1
        logger.info(f"[PROFILE] train_rank={self.transfer_rank} step={step_id} group_tensors: {group_time:.3f}s ({len(group_tensors)} groups)")

        # [VERIFY] Log group_tensors summary for IPC correctness verification
        for i, gt in enumerate(group_tensors[:3]):
            logger.info(f"[IPC_SEND] rank={self.transfer_rank} group_{i}: shape={gt.shape} mean={gt.float().mean():.6f} sum={gt.float().sum():.6f}")
        print_current_gpu_status(
            f"after group_tensors_by_shape_and_dtype for rank {self.transfer_rank}"
        )
        logger.info(f"Open fds before serialize: {count_open_fds()}")

        # NOTE: Do NOT call release_tensors(tensors) here!
        # tensors are references to model parameters (via detach()), not copies.
        # Calling release_tensors() would destroy the model parameter storage,
        # causing "CUDA error: invalid argument" when Slime tries to restore("ref").
        # The data has already been cloned into group_tensors by group_tensors_by_shape_and_dtype().
        del tensors

        # [PROFILE] Step 3: Offload weights
        t3 = time.time()
        self.train_engine.release_memory_occupation("weights")
        self.meta_server_client.add_object_to_set(
            "all_training_offloaded_weights", self.transfer_rank
        )
        t4 = time.time()
        offload_time = t4 - t3
        logger.info(f"[PROFILE] train_rank={self.transfer_rank} step={step_id} offload_weights: {offload_time:.3f}s")
        print_current_gpu_status(
            f"after offloaded weights for rank {self.transfer_rank}"
        )

        # [PROFILE] Step 4: Serialize IPC
        if self.ipc_backend == "cpu":
            group_shared = [tensor.cpu().share_memory_() for tensor in group_tensors]
            serialized_weights = ipc_serialize((group_shared, metadata, names))
        else:
            group_shared = [tensor.cuda().share_memory_() for tensor in group_tensors]
            serialized_weights = cuda_ipc_serialize((group_shared, metadata, names))
        torch.cuda.synchronize(device=torch.cuda.current_device())
        t5 = time.time()
        serialize_time = t5 - t4
        logger.info(f"[PROFILE] train_rank={self.transfer_rank} step={step_id} ipc_serialize: {serialize_time:.3f}s")
        logger.info(f"Open fds after serialize: {count_open_fds()}")

        # [PROFILE] Step 5: Put to metaserver and wait
        ip_address = get_ip_address()
        device_id = torch.cuda.current_device()
        key_suffix = f"_{ip_address}_{device_id}_{step_id}"
        serialized_weights_key = f"training_serialized_weights{key_suffix}"
        update_finished_key = f"weights_update_finished{key_suffix}"
        # CRITICAL: Delete old keys before putting new data to prevent stale IPC handles
        # If old data from a previous run exists, inference may read stale CUDA IPC handles
        # which point to memory from a dead process, causing "invalid resource handle" errors
        self.meta_server_client.delete_if_exists(serialized_weights_key)
        self.meta_server_client.delete_if_exists(update_finished_key)
        self.meta_server_client.put_object(
            serialized_weights_key,
            (self.transfer_rank, self.rank_info, serialized_weights),
        )
        t6 = time.time()
        put_time = t6 - t5
        logger.info(f"[PROFILE] train_rank={self.transfer_rank} step={step_id} metaserver_put: {put_time:.3f}s")

        # Wait for inference engines to finish processing
        logger.info(
            f"[Writer Rank {self.transfer_rank}] Waiting for inference engines to finish "
            f"(key={update_finished_key}, timeout={self.timeout}s, step_id={step_id})"
        )
        self.meta_server_client.get_object(update_finished_key, timeout=self.timeout)
        logger.info(
            f"[Writer Rank {self.transfer_rank}] Received completion signal from inference engines (step_id={step_id})"
        )
        t7 = time.time()
        wait_time = t7 - t6
        logger.info(f"[PROFILE] train_rank={self.transfer_rank} step={step_id} wait_inference: {wait_time:.3f}s")

        # Cleanup - simplified (Reader no longer waits for write_finished)
        self.meta_server_client.delete_if_exists(update_finished_key)
        self.meta_server_client.delete_if_exists(serialized_weights_key)
        release_tensors(group_tensors)
        release_tensors(group_shared)
        del group_tensors
        del group_shared
        gc.collect()  # Single gc.collect() is sufficient
        torch.cuda.synchronize(device=torch.cuda.current_device())
        torch.cuda.empty_cache()
        t8 = time.time()
        cleanup_time = t8 - t7
        total_time = t8 - total_start

        logger.info(
            f"[PROFILE] train_rank={self.transfer_rank} step={step_id} SUMMARY: "
            f"prepare={prepare_time:.2f}s group={group_time:.2f}s offload={offload_time:.2f}s "
            f"serialize={serialize_time:.2f}s put={put_time:.2f}s wait={wait_time:.2f}s "
            f"cleanup={cleanup_time:.2f}s total={total_time:.2f}s"
        )
