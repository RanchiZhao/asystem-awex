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
from awex.reader.weights_reader import WorkerWeightsReader
from awex.transfer.nccl_comm import batch_send_recv, nccl_build_recv_ops
from awex.transfer.transfer_plan import TransferPlanBuilder
from awex.util.common import (
    compute_statistics,
    get_free_port,
    get_ip_address,
)
from awex.util.gpu import get_gpu_status, print_current_gpu_status
from awex.util.system_util import count_open_fds
from awex.util.tensor_util import (
    cuda_ipc_deserialize,
    ipc_deserialize,
    reconstruct_tensors_from_groups,
)

logger = logging.getLogger(__name__)


class NCCLWorkerWeightsReader(WorkerWeightsReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transfer_plan = None
        self.weights_update_group = None
        self.send_ranks = None
        self.send_ranks_sample = None
        self.num_to_recvs = None
        self.rank_coordinate = None

    def initialize(self):
        super().initialize()

        # Build transfer plan (both colocate and non-colocate modes need this)
        plan_builder = TransferPlanBuilder(
            self.infer_world_size,
            self.training_world_size,
            self.num_engines,
            self.enable_debug_mode,
            hf_config=self.hf_config,
            enable_colocate_mode=self.enable_colocate_mode,
        )
        self.transfer_plan = plan_builder.build_local_transfer_plan(
            self.parameters_meta,
            self.training_params_meta,
            self.transfer_rank,
        )

        # Plan A: Try to use pre-initialized AWEX colocate group
        # The group is created during SGLang startup with name: awex_colocate_global
        # (ONE global group for ALL inference workers across ALL engines)
        preinitialized_group = self._try_get_preinitialized_group()

        if preinitialized_group is not None:
            logger.info(
                f"[NCCLWeightsReader] Using pre-initialized AWEX colocate group for rank {self.transfer_rank}"
            )
            self.weights_update_group = preinitialized_group
            self._use_preinitialized_group = True
        else:
            # Fall back to creating a new group (original behavior)
            logger.info(
                f"[NCCLWeightsReader] No pre-initialized group found, creating new group for rank {self.transfer_rank}"
            )
            self._use_preinitialized_group = False

            if self.transfer_rank == 0:
                master_address = get_ip_address()
                master_port = get_free_port()
                master_info = (master_address, master_port)
                self.meta_server_client.put_object("master_info", master_info)
                logger.info(
                    f"Put master info to meta server for rank {self.transfer_rank}: {master_info}"
                )
            else:
                master_info = self.meta_server_client.get_object(
                    "master_info", timeout=self.timeout
                )
                master_address, master_port = master_info
                logger.info(
                    f"Get master info from meta server for rank {self.transfer_rank}: {master_info}"
                )
            logger.info(
                f"Start to initialize NCCL weights writer for rank {self.transfer_rank}"
            )

            from awex.util.process_group import (
                init_weights_update_group,
                setup_batch_isend_irecv,
            )

            gpu_id = self.scheduler.gpu_id
            logger.info(
                f"[NCCLWeightsReader] Set device to {gpu_id} for rank {self.transfer_rank}, "
                f"device env is {os.environ.get('DEVICE')}, "
                f"previous device is {torch.cuda.current_device()}, "
                f"device_count is {torch.cuda.device_count()}, "
                f"CUDA_VISIBLE_DEVICES env is {os.environ.get('CUDA_VISIBLE_DEVICES')}"
            )
            torch.cuda.set_device(gpu_id)
            world_size = (
                self.infer_world_size
                if self.enable_colocate_mode
                else self.transfer_world_size
            )
            self.weights_update_group = init_weights_update_group(
                master_address=master_address,
                master_port=master_port,
                rank=self.transfer_rank,
                world_size=world_size,
                group_name="weights_exchange",
                role="inference",
            )
            logger.info(
                f"Initialized NCCL weights reader for rank {self.transfer_rank}, engine rank {self.engine_rank}"
            )
            # Add a barrier to ensure all processes are ready
            dist.barrier(
                group=self.weights_update_group, device_ids=[torch.cuda.current_device()]
            )
            logger.info(f"Barrier passed for weights reader with rank {self.transfer_rank}")
            if self.transfer_rank == 0:
                logger.info(
                    f"Start to test NCCL ready for rank {self.transfer_rank}, world size {self.transfer_world_size}"
                )
                dist.recv(
                    torch.tensor(1).cuda(),
                    src=world_size - 1,
                    group=self.weights_update_group,
                )
                logger.info(
                    f"NCCL ready: recv tensor from rank 0 for rank {self.transfer_rank}"
                )
            if (
                self.enable_colocate_mode
                and self.transfer_rank == self.infer_world_size - 1
            ):
                dist.send(
                    torch.tensor(1).cuda(),
                    dst=0,
                    group=self.weights_update_group,
                )
            setup_batch_isend_irecv(
                self.weights_update_group, self.transfer_rank, world_size
            )

        self.send_ranks = list(self.transfer_plan.operations.keys())
        self.send_ranks_sample = (
            self.send_ranks[:8] + ["..."] + self.send_ranks[-8:]
            if len(self.send_ranks) > 16
            else self.send_ranks
        )
        self.num_to_recvs = sum(
            len(operations) for operations in self.transfer_plan.operations.values()
        )
        self.rank_coordinate = (
            f"{self.engine_rank}-{self.rank_info.global_rank}-{self.transfer_rank}"
        )
        # In colocate mode, initialize additional colocate-specific state
        if self.enable_colocate_mode:
            self._init_reader_in_colocate_mode()
        self.deserialized_weights = {}
        logger.info(
            f"Created NCCL weights reader for rank {self.rank_info.global_rank}, engine rank {self.engine_rank}"
        )

    def _try_get_preinitialized_group(self):
        """
        Try to get a pre-initialized AWEX colocate group from model_runner.

        The group is created during SGLang startup by init_awex_colocate_group()
        with name: awex_colocate_global (ONE global group for ALL inference workers)

        Returns:
            The pre-initialized group if found, None otherwise.
        """
        if not self.enable_colocate_mode:
            return None

        try:
            # Access model_runner through scheduler.tp_worker
            model_runner = self.scheduler.tp_worker.model_runner
            model_update_groups = getattr(model_runner, '_model_update_group', {})

            # CRITICAL FIX: Use the global group name, not per-engine group name
            # All 128 inference workers share ONE NCCL group for P2P with training workers
            group_name = "awex_colocate_global"

            if group_name in model_update_groups:
                logger.info(
                    f"[NCCLWeightsReader] Found pre-initialized group '{group_name}' in model_runner"
                )
                return model_update_groups[group_name]
            else:
                logger.info(
                    f"[NCCLWeightsReader] Pre-initialized group '{group_name}' not found. "
                    f"Available groups: {list(model_update_groups.keys())}"
                )
                return None
        except Exception as e:
            logger.warning(
                f"[NCCLWeightsReader] Failed to get pre-initialized group: {e}"
            )
            return None

    def _init_reader_in_colocate_mode(self):
        """Initialize reader for colocate mode with MetaServer registration.

        This method checks for pre-initialized state from TpModelWorker first.
        If available, it uses the pre-initialized inference_device_mapping
        and skips the cleanup/registration steps (which require ALL workers
        to be synchronized, but execute_task_in_model_worker only runs on ~2 workers).

        The training side registration and transfer plan building still happens here
        because the training side might not be ready during TpModelWorker init.
        """
        ip_address = get_ip_address()
        device_id = torch.cuda.current_device()

        # Check if we have pre-initialized state from TpModelWorker
        # This state was collected when ALL workers were synchronized during init
        pre_init_state = None
        try:
            if hasattr(self.scheduler, 'tp_worker') and self.scheduler.tp_worker is not None:
                pre_init_state = self.scheduler.tp_worker.get_awex_colocate_init_state()
        except Exception as e:
            logger.warning(f"[NCCLWeightsReader] Failed to get pre-init state: {e}")

        if pre_init_state is not None and pre_init_state.get("initialized", False):
            # Use pre-initialized state - this was collected when ALL workers were synchronized
            logger.info(
                f"[NCCLWeightsReader] Using pre-initialized colocate state for {ip_address}:{device_id} "
                f"(transfer_rank={pre_init_state['transfer_rank']}, "
                f"infer_world_size={pre_init_state['infer_world_size']})"
            )
            self.inference_device_mapping = pre_init_state["inference_device_mapping"]
            logger.info(
                f"[NCCLWeightsReader] Pre-init inference_device_mapping has {len(self.inference_device_mapping)} entries"
            )
        else:
            # Fallback: Do the original initialization (requires ALL workers to be synchronized)
            # This path is taken when pre-init is not available (e.g., older SGLang version)
            logger.warning(
                f"[NCCLWeightsReader] No pre-init state available, falling back to original init "
                f"(this requires ALL workers to call this method simultaneously!)"
            )

            # CRITICAL: Clean up stale keys from previous runs at initialization time
            # This prevents reading old IPC handles from a previous run
            for step_id in [1]:
                key_suffix = f"_{ip_address}_{device_id}_{step_id}"
                serialized_weights_key = f"training_serialized_weights{key_suffix}"
                update_finished_key = f"weights_update_finished{key_suffix}"
                self.meta_server_client.delete_if_exists(serialized_weights_key)
                self.meta_server_client.delete_if_exists(update_finished_key)

            # Clean up stale device rank entry sets from previous runs
            # Only the TRUE global rank 0 (engine_rank=0 AND local_rank=0) does cleanup
            # to avoid race conditions when multiple engines each have their own "rank 0"
            cleanup_barrier_key = "inference_cleanup_barrier"
            is_global_rank_zero = (self.engine_rank == 0 and self.rank_info.global_rank == 0)

            if is_global_rank_zero:
                # True global rank 0: first delete the barrier to ensure other ranks will wait
                self.meta_server_client.delete_if_exists(cleanup_barrier_key)
                # Clean up all stale sets
                self.meta_server_client.delete_if_exists("inference_device_rank_entries")
                self.meta_server_client.delete_if_exists("training_device_rank_entries")
                self.meta_server_client.delete_if_exists("all_training_offloaded_weights")
                # Signal cleanup is done
                self.meta_server_client.put_object(cleanup_barrier_key, True)
                logger.info(
                    f"[NCCLWeightsReader] TRUE global rank 0 (engine={self.engine_rank}, "
                    f"local={self.rank_info.global_rank}) cleaned stale device rank entry sets"
                )
            else:
                # Wait for global rank 0 cleanup - with retry in case we're faster than rank 0's delete
                import time
                max_retries = 30
                for i in range(max_retries):
                    try:
                        self.meta_server_client.get_object(cleanup_barrier_key, timeout=10)
                        break
                    except Exception:
                        if i < max_retries - 1:
                            time.sleep(1)
                        else:
                            raise

            logger.info(
                f"[NCCLWeightsReader] Cleaned stale keys for {ip_address}:{device_id} "
                f"before initialization"
            )

            self.meta_server_client.add_object_to_set(
                "inference_device_rank_entries",
                (get_ip_address(), torch.cuda.current_device(), self.transfer_rank),
            )
            self.meta_server_client.wait_set_until_size(
                "inference_device_rank_entries", self.infer_world_size, timeout=self.timeout
            )
            self.inference_device_mapping = self.meta_server_client.get_set(
                "inference_device_rank_entries"
            )
            self.inference_device_mapping = {
                (ip_address, device_id): transfer_rank
                for ip_address, device_id, transfer_rank in self.inference_device_mapping
            }

        # Wait for training side to register (training side might not be ready during TpModelWorker init)
        logger.info(
            f"[NCCLWeightsReader] Waiting for {self.training_world_size} training workers to register..."
        )
        self.meta_server_client.wait_set_until_size(
            "training_device_rank_entries",
            self.training_world_size,
            timeout=self.timeout,
        )
        device_rank_entries = self.meta_server_client.get_set(
            "training_device_rank_entries"
        )
        self.training_device_mapping = {
            (ip_address, device_id): transfer_rank
            for ip_address, device_id, transfer_rank in device_rank_entries
        }

        # Debug: Log both mappings to diagnose IP/device mismatches
        inference_ips = set(ip for ip, dev in self.inference_device_mapping.keys())
        training_ips = set(ip for ip, dev in self.training_device_mapping.keys())
        logger.info(
            f"[NCCLWeightsReader] Inference IPs: {sorted(inference_ips)}"
        )
        logger.info(
            f"[NCCLWeightsReader] Training IPs: {sorted(training_ips)}"
        )
        logger.info(
            f"[NCCLWeightsReader] Inference mapping sample: {list(self.inference_device_mapping.items())[:5]}"
        )
        logger.info(
            f"[NCCLWeightsReader] Training mapping sample: {list(self.training_device_mapping.items())[:5]}"
        )

        self.train_to_infer_device_mapping = {}
        self.infer_to_train_device_mapping = {}
        for ip_address, device_id, transfer_rank in device_rank_entries:
            key = (ip_address, device_id)
            if key not in self.inference_device_mapping:
                # Log detailed error info for debugging
                logger.error(
                    f"[NCCLWeightsReader] Training device {key} not found in inference_device_mapping! "
                    f"This usually means training and inference are not on the same machines (not colocate). "
                    f"Available inference devices: {list(self.inference_device_mapping.keys())[:10]}..."
                )
                raise KeyError(
                    f"Training device {key} not found in inference_device_mapping. "
                    f"Training IPs: {sorted(training_ips)}, Inference IPs: {sorted(inference_ips)}. "
                    f"Ensure training and inference workers are on the same machines for colocate mode."
                )
            infer_rank = self.inference_device_mapping[key]
            self.train_to_infer_device_mapping[transfer_rank] = infer_rank
            self.infer_to_train_device_mapping[infer_rank] = transfer_rank
        plan_builder = TransferPlanBuilder(
            self.infer_world_size,
            self.training_world_size,
            self.num_engines,
            self.enable_debug_mode,
            hf_config=self.hf_config,
            enable_colocate_mode=self.enable_colocate_mode,
        )
        self.send_transfer_plan = plan_builder.build_local_transfer_plan(
            self.parameters_meta,
            self.training_params_meta,
            self.infer_to_train_device_mapping[self.transfer_rank],
        )
        from awex.transfer.nccl_stream_batch import NcclColocateStreamBatchTransport

        self.colocate_transport = NcclColocateStreamBatchTransport(
            self.transfer_rank, self.infer_world_size
        )
        logger.info(
            f"Initialized NCCL weights reader for rank {self.transfer_rank} in colocate mode"
        )

    def pre_update_weights(self, step_id, **kwargs):
        pass

    def collect_training_weights(self, step_id, **kwargs):
        if not self.enable_colocate_mode:
            return
        # Can't serialize IPC tensors at initialization since every step, the memory address for weights will change
        # because we use offloading for moving GPU tensors to CPU and back later
        # We'll get serialized weights from meta server each step instead
        # Get serialized weights from meta server
        ip_address = get_ip_address()
        device_id = torch.cuda.current_device()
        key = f"training_serialized_weights_{ip_address}_{device_id}_{step_id}"
        logger.info(
            f"Start to get serialized ipc weights {key} for rank {self.rank_coordinate}"
        )
        self.send_rank, self.send_rank_info, serialized_weights = (
            self.meta_server_client.get_object(key, timeout=self.timeout)
        )
        # [CRITICAL_VERIFY] Check if IPC send_rank matches mapping expectation
        expected_train_rank = self.infer_to_train_device_mapping.get(self.transfer_rank)
        if self.send_rank != expected_train_rank:
            logger.error(
                f"[RANK_MISMATCH] rank={self.rank_coordinate} transfer_rank={self.transfer_rank} "
                f"IPC send_rank={self.send_rank} != expected_train_rank={expected_train_rank} "
                f"(from infer_to_train_device_mapping). This causes COLOCATE_MISMATCH!"
            )
        else:
            logger.info(
                f"[RANK_VERIFY_OK] rank={self.rank_coordinate} transfer_rank={self.transfer_rank} "
                f"IPC send_rank={self.send_rank} matches expected_train_rank={expected_train_rank}"
            )
        logger.info(
            f"Finished getting serialized ipc weights {key} for rank {self.rank_coordinate}"
        )
        logger.info(
            f"GPU status before deserialization:\n{get_gpu_status()} for rank {self.rank_coordinate}"
        )
        logger.info(f"Open fds before deserialization: {count_open_fds()}")
        # Deserialize weights into tensors
        if self.ipc_backend == "cpu":
            group_shared, metadata, names = ipc_deserialize(serialized_weights)
            group_shared = [t.to(device_id) for t in group_shared]
        else:
            group_shared, metadata, names = cuda_ipc_deserialize(serialized_weights)
        torch.cuda.synchronize(device=torch.cuda.current_device())
        # [VERIFY] Log group_shared summary for IPC correctness verification
        for i, gs in enumerate(group_shared[:3]):
            logger.info(f"[IPC_RECV] rank={self.rank_coordinate} group_{i}: shape={gs.shape} mean={gs.float().mean():.6f} sum={gs.float().sum():.6f}")
        tensors = reconstruct_tensors_from_groups(group_shared, metadata)
        torch.cuda.synchronize(device=torch.cuda.current_device())
        self.deserialized_weights = dict(zip(names, tensors))
        logger.info(
            f"Deserialized {len(self.deserialized_weights)} parameters and {len(group_shared)} groups"
        )
        logger.info(
            f"GPU status after deserialization for rank {self.rank_coordinate}:\n{get_gpu_status()}"
        )
        logger.info(f"Open fds after deserialization: {count_open_fds()}")

    def _update_weights(self, step_id, **kwargs):
        """
        Asynchronously receive weights from training ranks using torch.distributed.irecv.

        This method implements a pipelined approach where:
        1. For each sender rank, we maintain a queue of operations to receive
        2. We start irecv operations in parallel from all sender ranks
        3. When a receive completes, we immediately start the next receive from that rank
        4. We continue until all operations from all sender ranks are completed

        Args:
            step_id: The training step ID
            **kwargs: Additional keyword arguments (unused)
        """
        logger.info(
            f"Start to update weights using NCCL for step {step_id} from "
            f"{len(self.transfer_plan.operations)} ranks({self.send_ranks_sample}) "
            f"for rank {self.rank_coordinate}."
        )
        start_time = time.time()

        # Build receive ops once for logging, then execute them via
        # batch_send_recv to keep scheduling consistent with the writer.
        p2p_op_list = nccl_build_recv_ops(
            self.parameters, self.transfer_plan, self.weights_update_group
        )
        logger.info(
            f"Reader: Built {len(p2p_op_list)} recv operations from "
            f"{len(self.transfer_plan.operations)} training ranks"
        )

        logger.info(
            f"Reader: Executing {len(p2p_op_list)} recv ops via batch_send_recv"
        )
        batch_send_recv(
            send_ops=[], recv_ops=p2p_op_list, blocking=True, use_group=True
        )
        torch.cuda.synchronize(device=torch.cuda.current_device())
        duration = time.time() - start_time
        logger.info(
            f"Finished receiving weights for step {step_id} using NCCL "
            f"from {len(self.transfer_plan.operations)} ranks({self.send_ranks_sample}) "
            f"to rank {self.rank_coordinate} with {self.num_to_recvs} receives, took {duration:.4f} seconds"
        )
        compute_statistics(
            self._history_update_weights_time,
            step_id,
            duration,
            "Receive weights using NCCL",
        )
        dist.barrier(
            group=self.weights_update_group, device_ids=[torch.cuda.current_device()]
        )
        logger.info(
            f"Barrier passed for reader step {step_id} with rank {self.transfer_rank}"
        )

    def _update_weights_in_colocate_mode(self, step_id, **kwargs):
        assert self.enable_colocate_mode, "Colocate mode is not enabled"
        total_start = time.time()

        # [PROFILE] Step 1: Collect training weights (IPC deserialization)
        t0 = time.time()
        self.collect_training_weights(step_id, **kwargs)
        t1 = time.time()
        ipc_time = t1 - t0
        logger.info(f"[PROFILE] rank={self.transfer_rank} step={step_id} ipc_deserialize: {ipc_time:.3f}s")

        # [PROFILE] Step 2: Refresh parameter views
        # CRITICAL: Refresh self.parameters to ensure views point to current storage
        logger.info(f"[COLOCATE] Refreshing parameter views before P2P transfer for step {step_id}")
        old_param_count = len(self.parameters)
        self.parameters = {
            hf_name: hf_param
            for name, param in self.model.named_parameters()
            for hf_name, hf_param in self.weight_converter.convert_param(name, param)
        }
        t2 = time.time()
        view_refresh_time = t2 - t1
        logger.info(f"[PROFILE] rank={self.transfer_rank} step={step_id} view_refresh: {view_refresh_time:.3f}s ({len(self.parameters)} params)")

        # Verify expert views share storage with original tensor (only on first sync)
        if step_id == 0:
            self._verify_expert_view_storage()

        # [PROFILE] Step 3: P2P transfer
        logger.info(
            f"Start to update weights using NCCL for step {step_id} from {len(self.transfer_plan.operations)} "
            f"ranks({self.send_ranks_sample}) for rank {self.rank_coordinate}."
        )

        # [DEBUG] Check for missing keys in deserialized_weights before P2P
        if self.deserialized_weights is not None and self.send_transfer_plan is not None:
            required_send_params = set()
            for peer_rank, ops in self.send_transfer_plan.operations.items():
                for op in ops:
                    required_send_params.add(op.send_shard_meta.name)
            available_params = set(self.deserialized_weights.keys())
            missing_params = required_send_params - available_params
            if missing_params:
                logger.error(
                    f"[DEBUG] rank={self.transfer_rank} MISSING PARAMS in deserialized_weights: "
                    f"{list(missing_params)[:10]}... (total {len(missing_params)} missing)"
                )
                logger.error(
                    f"[DEBUG] rank={self.transfer_rank} available params sample: "
                    f"{list(available_params)[:10]}... (total {len(available_params)})"
                )
                logger.error(
                    f"[DEBUG] rank={self.transfer_rank} required params sample: "
                    f"{list(required_send_params)[:10]}... (total {len(required_send_params)})"
                )

        t3 = time.time()
        self.colocate_transport.update_weights_in_colocate_mode(
            self.train_to_infer_device_mapping,
            self.infer_to_train_device_mapping,
            self.transfer_rank,
            self.rank_coordinate,
            self.infer_world_size,
            self.send_transfer_plan,
            self.transfer_plan,
            self.weights_update_group,
            self.deserialized_weights,
            self.parameters,
            step_id=step_id,
        )
        torch.cuda.synchronize()
        t4 = time.time()
        p2p_time = t4 - t3
        logger.info(f"[PROFILE] rank={self.transfer_rank} step={step_id} p2p_transfer: {p2p_time:.3f}s")

        print_current_gpu_status(
            f"after weights update using NCCL for rank {self.rank_coordinate}"
        )
        self.deserialized_weights = None
        duration = time.time() - total_start
        compute_statistics(
            self._history_update_weights_time,
            step_id,
            duration,
            "Receive weights using NCCL",
        )

        # [PROFILE] Step 4: Signal and barrier
        t5 = time.time()
        ip_address = get_ip_address()
        device_id = torch.cuda.current_device()
        key_suffix = f"_{ip_address}_{device_id}_{step_id}"
        # Signal completion to training process
        update_finished_key = f"weights_update_finished{key_suffix}"
        self.meta_server_client.put_object(update_finished_key, True)
        dist.barrier(
            group=self.weights_update_group, device_ids=[torch.cuda.current_device()]
        )
        t6 = time.time()
        barrier_time = t6 - t5
        logger.info(f"[PROFILE] rank={self.transfer_rank} step={step_id} barrier: {barrier_time:.3f}s")

        # [PROFILE] Step 5: Cleanup
        # NOTE: Removed wait for write_finished - Reader doesn't need to wait for Writer cleanup
        # Data has already been copied via P2P, so Writer can cleanup asynchronously
        t7 = time.time()
        cleanup_time = t7 - t6
        total_time = t7 - total_start

        logger.info(
            f"[PROFILE] rank={self.transfer_rank} step={step_id} SUMMARY: "
            f"ipc={ipc_time:.2f}s view={view_refresh_time:.2f}s p2p={p2p_time:.2f}s "
            f"barrier={barrier_time:.2f}s cleanup={cleanup_time:.2f}s total={total_time:.2f}s"
        )
