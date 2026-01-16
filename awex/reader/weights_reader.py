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

import os
import pickle
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch

from awex import logging
from awex.meta.infer_meta_resolver import InferParamMetaResolver
from awex.meta.meta_resolver import (
    ParameterMeta,
)
from awex.meta.meta_server import MetaServerClient
from awex.models.registry import get_infer_weights_converter
from awex.sharding.param_sharding import (
    get_rank_info_extractor,
)
from awex.util.common import (
    check_train_infer_params_meta,
    compute_statistics,
    simple_hf_config,
    stripped_env_vars,
)
from awex.util.tensor_util import (
    check_and_log_nan_values,
    compare_and_log_tensor_differences,
)

logger = logging.getLogger(__name__)


def _log_gpu_memory(label: str, rank: int = None):
    """Log GPU memory usage for debugging memory issues."""
    try:
        if rank is None:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        if not torch.cuda.is_available():
            return

        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)

        # Get max memory if available
        try:
            max_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            free = max_memory - reserved
        except:
            max_memory = 0
            free = 0

        logger.info(
            f"[GPU_MEM] {label} | rank={rank} device={device} | "
            f"allocated={allocated:.2f}GB reserved={reserved:.2f}GB free={free:.2f}GB total={max_memory:.2f}GB"
        )
    except Exception as e:
        logger.warning(f"[GPU_MEM] Failed to log memory for {label}: {e}")


class WeightExchangeReader(ABC):
    def __init__(self, inference_engine):
        self.inference_engine = inference_engine
        self.enable_colocate_mode = inference_engine.config.enable_colocate_mode
        self.infer_config = inference_engine.config
        self.weights_comm_nccl_group_size = (
            self.infer_config.weights_comm_nccl_group_size
        )

    @abstractmethod
    def initialize(self, **kwargs):
        """Initialize the weight exchange reader."""
        pass

    @abstractmethod
    def update_weights(self, step_id, **kwargs):
        pass


class FileWeightExchangeReader(WeightExchangeReader):
    def __init__(self, inference_engine):
        super().__init__(inference_engine)

    def initialize(self, **kwargs):
        """Initialize file-based weight exchange reader (no-op)."""
        pass

    def update_weights(self, step_id, **kwargs):
        if self.enable_colocate_mode:
            self.inference_engine.resume_memory_occupation()
        self.inference_engine.update_weights_from_disk(
            kwargs["path"], kwargs.get("load_format")
        )


class WeightsReader(WeightExchangeReader):
    parameters_meta: List[ParameterMeta]

    def __init__(self, inference_engine, meta_resolver: InferParamMetaResolver = None):
        import time
        super().__init__(inference_engine)
        self.infer_engine_config = self.inference_engine.config
        if meta_resolver is None:
            logger.info(f"[PROFILE] WeightsReader.__init__: Creating InferParamMetaResolver...")
            t0 = time.time()
            meta_resolver = InferParamMetaResolver(
                inference_engine,
                num_engines=inference_engine.num_engines,
                engine_rank=inference_engine.engine_rank,
                convert_params=True,
            )
            t1 = time.time()
            logger.info(f"[PROFILE] WeightsReader.__init__: InferParamMetaResolver created in {t1-t0:.2f}s")
        self.meta_resolver = meta_resolver
        self.parameters_meta = []
        self.hf_config = inference_engine.hf_config
        config = inference_engine.config
        self.model_arch_name = meta_resolver.get_model_arch_name()
        self.meta_server_addr = config.meta_server_addr
        logger.info(f"Meta server address: {self.meta_server_addr}")
        self.meta_server_client = MetaServerClient(*self.meta_server_addr.split(":"))
        self.num_engines = config.num_engines
        logger.info("Put number of inference engines to meta server")
        self.engine_rank = self.inference_engine.engine_rank
        self.tp_size = config.tp_size
        self.pp_size = config.pp_size
        self.infer_world_size = self.num_engines * self.tp_size * self.pp_size
        self.validated_steps = 0
        self.start_step = -1
        self.weights_validation_steps = config.weights_validation_steps
        self.validate_weights_every_n_steps = config.validate_weights_every_n_steps
        self.dump_weights_list_for_validation = config.dump_weights_list_for_validation
        self.dump_weights_dir_for_validation = (
            config.dump_weights_dir_for_validation or os.getcwd()
        )
        self.ipc_backend = config.weights_exchange_ipc_backend
        self.timeout = 10000
        self.lock = threading.Lock()
        self.initialized = False
        logger.info(
            f"DP rank id: {self.engine_rank}, num_engines: {self.num_engines}, "
            f"engine_rank: {self.engine_rank}, infer_world_size: {self.infer_world_size}"
        )

    def initialize(self, **kwargs):
        _log_gpu_memory("WeightsReader.initialize START")
        # In colocate mode with TP spanning multiple nodes:
        # - Only node_rank=0 engines call initialize()
        # - So actual initialized count = num_engines / nnodes
        # - nnodes = tp_size / gpus_per_node (e.g., 64 / 8 = 8)
        #
        # We need to report the ACTUAL number of engines that will initialize,
        # not the total num_engines, so training side doesn't wait forever.
        if self.enable_colocate_mode:
            # Calculate nnodes: tp_size / local_world_size (GPUs per node)
            # In DeepSeek-V3: tp_size=64 across 8 nodes, each node has 8 GPUs
            gpus_per_node = int(os.environ.get("NPROC_PER_NODE", 8))
            nnodes = max(1, self.tp_size // gpus_per_node)
            actual_num_engines = max(1, self.num_engines // nnodes) 
            # [COLOCATE] Calculated actual_num_engines=2 (num_engines=16, nnodes=8, gpus_per_node=8)
            logger.info(
                f"[COLOCATE] Calculated actual_num_engines={actual_num_engines} "
                f"(num_engines={self.num_engines}, nnodes={nnodes}, gpus_per_node={gpus_per_node})"
            )
        else:
            actual_num_engines = self.num_engines
        logger.info(
            f"[WeightsReader] Engine rank {self.engine_rank} putting num_infer_engines={actual_num_engines} to MetaServer"
        )
        self.meta_server_client.put_object("num_infer_engines", actual_num_engines)
        logger.info(
            f"[WeightsReader] Engine rank {self.engine_rank} successfully put num_infer_engines={actual_num_engines} to MetaServer"
        )
        # NOTE: In colocate mode with TP spanning multiple nodes, we CANNOT call
        # release_memory_occupation here because:
        # 1. Only node_rank=0 workers call WeightsReader.initialize()
        # 2. SGLang's release_memory_occupation has torch.distributed.barrier(tp_cpu_group)
        # 3. tp_cpu_group includes ALL TP workers across all nodes (e.g., 64 workers)
        # 4. Only 8 workers (node 0) would call the barrier -> deadlock
        #
        # Memory release will be handled by Slime's offload_rollout which broadcasts
        # to all workers through Ray, ensuring all TP workers participate.
        #
        # Keeping the code here commented for reference:
        # if self.enable_colocate_mode:
        #     _log_gpu_memory("WeightsReader.initialize BEFORE release_memory_occupation")
        #     logger.info("Start to release memory after inference engine initialized")
        #     self.inference_engine.release_memory_occupation()
        #     logger.info("Finished releasing memory after inference engine initialized")
        #     _log_gpu_memory("WeightsReader.initialize AFTER release_memory_occupation")
        logger.info(
            f"[WeightsReader.initialize] Skipping release_memory_occupation in colocate mode "
            f"(will be handled by Slime's offload_rollout)"
        )
        self.meta_server_client.add_object_to_set(
            "num_inited_inference_engines", self.engine_rank
        )
        # NOTE: We don't call _initialize() here because it requires training_params_meta
        # which is not available until training side has initialized.
        # The _initialize() will be called lazily in update_weights().
        # But we need to mark basic initialization as done.
        _log_gpu_memory("WeightsReader.initialize END")

    def _initialize(self):
        logger.info(
            f"Initializing weights exchange reader for engine rank {self.engine_rank}"
        )
        self.parameters_meta = self.meta_resolver.get_parameters_meta()
        logger.info(
            "Finished querying and building parameters meta from all tp workers"
        )
        self.infer_conf = {
            "infer_atten_tp_size": self.meta_resolver.rank0_info.attn_tp_size,
            "router_dtype": getattr(self.hf_config, "router_dtype", "bf16"),
            "infer_engine_config": self.infer_engine_config,
            "hf_config": simple_hf_config(self.hf_config),
            "infer_world_size": self.infer_world_size,
        }
        self.meta_server_client.put_object("infer_conf", self.infer_conf)
        logger.info(f"Put inference config {self.infer_conf} to meta server")
        if self.engine_rank == 0:
            self.meta_server_client.put_object(
                "infer_params_meta", self.parameters_meta
            )
            logger.info("Put inference parameters meta to meta server")
        logger.info(
            f"Start to get training parameters meta from meta server for engine rank {self.engine_rank}"
        )
        self.training_params_meta = self.meta_server_client.get_object(
            "training_params_meta", timeout=self.timeout
        )
        logger.info("Finished getting training parameters meta from meta server")
        self.training_world_size = self.training_params_meta[0].shards[0].world_size
        config = self.inference_engine.config
        # In colocate mode, training and inference may have different parallelism (e.g., EP=8 vs EP=1)
        # so shape mismatches are expected. Only raise exceptions in non-colocate, non-debug mode.
        should_raise = not config.enable_debug_mode and not config.enable_colocate_mode
        check_train_infer_params_meta(
            self.training_params_meta,
            self.parameters_meta,
            raise_exception=should_raise,
            hf_config=self.hf_config,
        )
        logger.info("Start to send parameters meta to tp workers")
        infer_parameters_meta_bytes = pickle.dumps(self.parameters_meta)
        train_parameters_meta_bytes = pickle.dumps(self.training_params_meta)
        infer_conf_bytes = pickle.dumps(self.infer_conf)
        self.inference_engine.execute_task_in_model_worker(
            self._init_in_tp_worker,
            infer_conf_bytes=infer_conf_bytes,
            parameters_meta_bytes=infer_parameters_meta_bytes,
            training_params_meta_bytes=train_parameters_meta_bytes,
            engine_rank=self.engine_rank,
            num_engines=self.num_engines,
            meta_server_addr=self.meta_server_addr,
            weights_comm_backend=config.comm_backend,
            enable_debug_mode=config.enable_debug_mode,
            debug_mode_config=config.debug_mode_config,
            disable_pipeline=config.disable_weights_exchange_pipeline,
            enable_colocate_mode=self.enable_colocate_mode,
            ipc_backend=self.ipc_backend,
            weights_comm_nccl_group_size=self.weights_comm_nccl_group_size,
        )
        logger.info(
            f"Finished full initialization of weights reader for engine rank {self.engine_rank}"
        )

    @staticmethod
    def _init_in_tp_worker(
        infer_conf_bytes: bytes,
        parameters_meta_bytes: bytes,
        training_params_meta_bytes: bytes,
        engine_rank: int,
        num_engines: int,
        meta_server_addr: str,
        weights_comm_backend: str,
        debug_mode_config: Dict[str, Any],
        disable_pipeline: bool,
        enable_colocate_mode: bool,
        ipc_backend: str,
        **kwargs,
    ):
        """Cache meta to avoid send to to worker everytime when update weights"""
        # TODO(mubai) EPLB: needs to rebuild weights meta if robust experts rebalance.
        model = kwargs["model"]
        model_context = kwargs["model_context"]
        scheduler = model_context["scheduler"]
        infer_conf = pickle.loads(infer_conf_bytes)
        parameters_meta = pickle.loads(parameters_meta_bytes)
        training_params_meta = pickle.loads(training_params_meta_bytes)
        if weights_comm_backend == "nccl":
            from awex.reader.nccl_reader import NCCLWorkerWeightsReader

            cls = NCCLWorkerWeightsReader
        elif weights_comm_backend == "astate":
            from awex.reader.astate_reader import AStateWorkerWeightsReader

            cls = AStateWorkerWeightsReader
        scheduler.awes_weights_reader = cls(
            "sglang",  # engine_name
            model,
            model_context,
            infer_conf,
            engine_rank,
            num_engines,
            meta_server_addr,
            parameters_meta,
            training_params_meta,
            enable_debug_mode=kwargs.get("enable_debug_mode", False),
            debug_mode_config=debug_mode_config,
            disable_pipeline=disable_pipeline,
            enable_colocate_mode=enable_colocate_mode,
            ipc_backend=ipc_backend,
            weights_comm_nccl_group_size=kwargs.get("weights_comm_nccl_group_size"),
        )
        scheduler.awes_weights_reader.initialize()

    def update_weights(self, step_id, **kwargs):
        with self.lock:
            _log_gpu_memory(f"WeightsReader.update_weights START step={step_id}")
            if not self.initialized:
                logger.info(
                    f"Start to initialize weights exchange reader for engine rank {self.engine_rank}"
                )
                self._initialize()
                self.initialized = True
                logger.info(
                    f"Finished initializing weights exchange reader for engine rank {self.engine_rank}"
                )
                _log_gpu_memory(f"WeightsReader.update_weights AFTER _initialize step={step_id}")
            self._pre_validate_weights(step_id, **kwargs)
            start_time = time.time()
            logger.info(
                f"Start to update weights for step {step_id} for engine rank {self.engine_rank}"
            )
            if self.enable_colocate_mode:
                _log_gpu_memory(f"WeightsReader.update_weights BEFORE release step={step_id}")
                # NOTE: This release_memory_occupation call should be a NO-OP in normal flow because:
                # 1. Slime's offload_rollout() runs FIRST and broadcasts release to ALL workers
                # 2. SGLang's release_memory_occupation has idempotent check (offload_tags)
                # 3. Since memory is already released, this returns early without hitting barrier
                # If this ever deadlocks, it means Slime's offload timing changed.
                self.inference_engine.release_memory_occupation()
                _log_gpu_memory(f"WeightsReader.update_weights AFTER release step={step_id}")
                self._pre_update_weights(step_id=step_id)
                _log_gpu_memory(f"WeightsReader.update_weights AFTER _pre_update_weights (resume weights) step={step_id}")
            # TODO(mubai) EPLB: needs to rebuild weights meta if experts rebalance.
            _log_gpu_memory(f"WeightsReader.update_weights BEFORE execute_task step={step_id}")
            self.inference_engine.execute_task_in_model_worker(
                self._update_parameters_in_tp_worker, step_id=step_id
            )
            _log_gpu_memory(f"WeightsReader.update_weights AFTER execute_task step={step_id}")
            duration = time.time() - start_time
            logger.info(
                f"Finished updating weights for step {step_id} for engine rank {self.engine_rank}, took {duration} seconds"
            )
            self._validate_weights(
                step_id,
                dump_weights_list_for_validation=self.dump_weights_list_for_validation,
                dump_weights_dir_for_validation=self.dump_weights_dir_for_validation,
                **kwargs,
            )
            if self.enable_colocate_mode:
                _log_gpu_memory(f"WeightsReader.update_weights BEFORE resume_kvcache step={step_id}")
                self._resume_kvcache_memory_occupation()
                _log_gpu_memory(f"WeightsReader.update_weights AFTER resume_kvcache step={step_id}")

    def _resume_weights_memory_occupation(self):
        """Resume weights memory for validation flow.

        IMPORTANT: resume_memory_occupation has torch.distributed.barrier(tp_cpu_group)
        which requires ALL TP workers to participate. Since only node_rank=0 workers
        call WeightsReader methods, we MUST broadcast via execute_task_in_model_worker.
        """
        assert self.enable_colocate_mode
        logger.info(
            "Start to resume weights memory occupation, waiting for all train ranks to offload optimizer"
        )
        self.meta_server_client.get_object(
            "all_training_offloaded_optimizers", timeout=self.timeout
        )
        logger.info(
            "All train ranks have offloaded optimizer states, broadcasting resume to all TP workers"
        )
        self.inference_engine.execute_task_in_model_worker(
            self._resume_weights_in_tp_worker
        )
        logger.info("Finished resuming weights memory occupation")

    @staticmethod
    def _resume_weights_in_tp_worker(**kwargs):
        """Resume weights memory on all TP workers.

        This is called via execute_task_in_model_worker so ALL 64 workers participate,
        allowing the barrier inside resume_memory_occupation to succeed.
        """
        model_context = kwargs["model_context"]
        scheduler = model_context["scheduler"]

        from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
        if GPU_MEMORY_TYPE_WEIGHTS in scheduler.offload_tags:
            scheduler.offload_tags.remove(GPU_MEMORY_TYPE_WEIGHTS)
            scheduler.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
            import torch
            torch.distributed.barrier(scheduler.tp_cpu_group)
            # Import static state back
            if hasattr(scheduler, 'stashed_model_static_state'):
                from sglang.srt.managers.scheduler_update_weights_mixin import _import_static_state
                _import_static_state(
                    scheduler.tp_worker.model_runner.model,
                    scheduler.stashed_model_static_state,
                )
                del scheduler.stashed_model_static_state
            logger.info(f"[_resume_weights_in_tp_worker] Resumed weights memory")
        else:
            logger.info(f"[_resume_weights_in_tp_worker] weights not in offload_tags, skipping")

    def _resume_kvcache_memory_occupation(self):
        assert self.enable_colocate_mode
        self.meta_server_client.add_object_to_set(
            "finished_weights_update_engines", self.engine_rank
        )
        # IMPORTANT: resume_memory_occupation has torch.distributed.barrier(tp_cpu_group)
        # which requires ALL TP workers to participate. Since only node_rank=0 workers
        # call WeightsReader methods, we MUST broadcast via execute_task_in_model_worker.
        logger.info(
            f"[_resume_kvcache_memory_occupation] Broadcasting kv_cache resume to all TP workers"
        )
        self.inference_engine.execute_task_in_model_worker(
            self._resume_kvcache_in_tp_worker
        )
        logger.info(
            f"Finished resuming kvcache memory occupation for engine rank {self.engine_rank}"
        )

    @staticmethod
    def _resume_kvcache_in_tp_worker(**kwargs):
        """Resume KV cache memory on all TP workers.

        This is called via execute_task_in_model_worker so ALL 64 workers participate,
        allowing the barrier inside resume_memory_occupation to succeed.
        """
        model_context = kwargs["model_context"]
        scheduler = model_context["scheduler"]

        from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
        if GPU_MEMORY_TYPE_KV_CACHE in scheduler.offload_tags:
            scheduler.offload_tags.remove(GPU_MEMORY_TYPE_KV_CACHE)
            scheduler.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)
            logger.info(f"[_resume_kvcache_in_tp_worker] Resumed kv_cache memory")
        else:
            logger.info(f"[_resume_kvcache_in_tp_worker] kv_cache not in offload_tags, skipping")

    def _pre_validate_weights(self, step_id, **kwargs):
        if self.validated_steps == 0:
            self.start_step = step_id
        if self.validated_steps >= self.weights_validation_steps:
            return
        if (step_id - self.start_step) % self.validate_weights_every_n_steps != 0:
            return
        logger.info(f"Start to pre-validate weights for step {step_id}")
        model_path = kwargs["path"]
        start_time = time.time()
        last_log_time = time.time()
        load_key = "weights_ready_for_load"
        while not self.meta_server_client.has_key(load_key):
            current_time = time.time()
            if current_time - last_log_time >= 10:
                logger.info(
                    f"Reader is waiting {current_time - start_time} seconds for {load_key} to be ready "
                    f"for validation for step {step_id}, model_path: {model_path}"
                )
                last_log_time = current_time
            time.sleep(0.5)
        logger.info(
            f"Weights for step {step_id} are ready for reader, model_path: {model_path}"
        )
        if self.enable_colocate_mode:
            self._resume_weights_memory_occupation()
        self.inference_engine.update_weights_from_disk(
            model_path, kwargs.get("load_format")
        )
        logger.info(
            f"Finished updating weights from disk for step {step_id}, model_path: {model_path}"
        )
        self.weights_meta = self.inference_engine.execute_task_in_model_worker(
            self._pre_validate_weights_on_tp_worker,
            step_id=step_id,
        )
        send_key = "weights_ready_for_send"
        self.meta_server_client.add_object_to_set(send_key, self.engine_rank)
        ready_engines = self.meta_server_client.get_object(send_key)
        logger.info(
            f"Inference engine instances has read weights for step {step_id} from {model_path}: {ready_engines}"
        )
        start_time = time.time()
        last_log_time = time.time()
        while len(ready_engines) != self.num_engines:
            current_time = time.time()
            if current_time - last_log_time >= 10:
                logger.info(
                    f"Waiting {current_time - start_time} seconds for all inference engine instances to "
                    f"read weights for step {step_id}, model_path: {model_path}"
                )
                last_log_time = current_time
            new_ready_engines = self.meta_server_client.get_object(send_key)
            if new_ready_engines is None or len(new_ready_engines) == 0:
                # deleted by writer
                break
            if new_ready_engines != ready_engines:
                logger.info(
                    f"Inference engine instances has read weights for step {step_id} from {model_path}: {new_ready_engines}"
                )
                ready_engines = new_ready_engines
            time.sleep(0.5)
        logger.info(
            f"All inference engine instances has read weights for step {step_id} from {model_path}"
        )
        if self.enable_colocate_mode:
            # IMPORTANT: release_memory_occupation has torch.distributed.barrier(tp_cpu_group)
            # which requires ALL TP workers to participate. Must broadcast via execute_task_in_model_worker.
            logger.info(
                "[_pre_validate_weights] Broadcasting release_memory_occupation to all TP workers"
            )
            self.inference_engine.execute_task_in_model_worker(
                self._release_memory_in_tp_worker
            )

    @staticmethod
    def _release_memory_in_tp_worker(**kwargs):
        """Release memory on all TP workers.

        This is called via execute_task_in_model_worker so ALL 64 workers participate,
        allowing the barrier inside release_memory_occupation to succeed.
        """
        model_context = kwargs["model_context"]
        scheduler = model_context["scheduler"]

        from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS
        tags_to_release = []
        for tag in [GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS]:
            if tag not in scheduler.offload_tags:
                tags_to_release.append(tag)

        if not tags_to_release:
            logger.info(f"[_release_memory_in_tp_worker] All tags already offloaded, skipping")
            return

        for tag in tags_to_release:
            scheduler.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags_to_release:
            scheduler.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
            scheduler.flush_cache()

        if GPU_MEMORY_TYPE_WEIGHTS in tags_to_release:
            # Export static state before releasing
            from sglang.srt.managers.scheduler_update_weights_mixin import _export_static_state
            scheduler.stashed_model_static_state = _export_static_state(
                scheduler.tp_worker.model_runner.model
            )
            import torch
            torch.distributed.barrier(scheduler.tp_cpu_group)
            scheduler.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)

        logger.info(f"[_release_memory_in_tp_worker] Released memory for tags: {tags_to_release}")

    @classmethod
    def _pre_validate_weights_on_tp_worker(cls, step_id, **kwargs):
        model = kwargs["model"]
        scheduler = kwargs["model_context"]["scheduler"]
        logger.info(
            f"Start to copy parameters for step {step_id} for consistency check"
        )
        torch.cuda.synchronize()
        scheduler._asystem_copied_parameters = {}
        for name, param in model.named_parameters():
            scheduler._asystem_copied_parameters[name] = (
                param.detach().cpu().contiguous()
            )
        logger.info(
            f"Finished copying parameters for step {step_id} for consistency check"
        )
        for name, param in model.named_parameters():
            param.data.fill_(0)
            logger.info(f"Set parameter {name} to 0")
        torch.cuda.synchronize()

    def _validate_weights(
        self,
        step_id,
        dump_weights_list_for_validation: Optional[List[str]] = None,
        dump_weights_dir_for_validation: str = ".",
        **kwargs,
    ):
        if self.validated_steps == 0:
            self.start_step = step_id
        if self.validated_steps >= self.weights_validation_steps:
            return
        if (step_id - self.start_step) % self.validate_weights_every_n_steps != 0:
            return
        self.validated_steps += 1
        logger.info(f"Start to validate weights for step {step_id}")
        verify_results = self.inference_engine.execute_task_in_model_worker(
            self._verify_weights_on_tp_worker,
            step_id=step_id,
            dump_weights_list_for_validation=dump_weights_list_for_validation,
            dump_weights_dir_for_validation=dump_weights_dir_for_validation,
        )
        for tp_rank, tp_results in enumerate(verify_results):
            if not all(tp_results.values()):
                not_consistent_weights = [
                    name for name, result in tp_results.items() if not result
                ]
                logger.error(
                    f"Weights for step {step_id} is not consistent for tp rank {tp_rank}: {not_consistent_weights}, "
                    f"total {len(tp_results)} weights"
                )
            else:
                logger.info(
                    f"Weights for step {step_id} is consistent for tp rank {tp_rank}, "
                    f"total {len(tp_results)} weights"
                )

    @classmethod
    def _verify_weights_on_tp_worker(
        cls,
        step_id,
        dump_weights_list_for_validation=None,
        dump_weights_dir_for_validation=".",
        **kwargs,
    ):
        model = kwargs["model"]
        scheduler = kwargs["model_context"]["scheduler"]
        logger.info("Start to verify parameters")
        torch.cuda.synchronize()
        results = {}
        dump_weights_list_for_validation = set(dump_weights_list_for_validation or [])
        for name, tensor_from_hg in scheduler._asystem_copied_parameters.items():
            if name in dump_weights_list_for_validation:
                abs_path = os.path.abspath(
                    os.path.join(
                        dump_weights_dir_for_validation,
                        f"reader_{os.getpid()}_from_hg_{step_id}.{name}.pt",
                    )
                )
                torch.save(tensor_from_hg, abs_path)
                logger.info(
                    f"[Reader] Saved parameter {name} loaded from hg to {abs_path}"
                )
        for name, tensor_from_commu in model.named_parameters():
            if name in dump_weights_list_for_validation:
                abs_path = os.path.abspath(
                    os.path.join(
                        dump_weights_dir_for_validation,
                        f"reader_{os.getpid()}_from_commu_{step_id}.{name}.pt",
                    )
                )
                torch.save(tensor_from_commu.detach().cpu().contiguous(), abs_path)
                logger.info(
                    f"[Reader] Saved parameter {name} loaded from communication to {abs_path}"
                )
        for name, param in model.named_parameters():
            copied_param = scheduler._asystem_copied_parameters.pop(name, None)
            if copied_param is None:
                logger.error(f"Parameter {name} not found in copied parameters")
                results[name] = False
                continue

            hg_param = copied_param
            # Check for NaN values in both parameters
            param = param.detach().cpu()
            param_has_nan = check_and_log_nan_values(param, name)
            copied_param_has_nan = check_and_log_nan_values(
                hg_param, f"copied param {name}"
            )

            if param_has_nan or copied_param_has_nan:
                results[name] = False
                continue

            if not compare_and_log_tensor_differences(
                param, hg_param, name, exact_match=True
            ):
                results[name] = False
            else:
                logger.info(f"Weights for {name} is consistent")
                results[name] = True

        # Clean up the copied parameters dictionary after verification
        scheduler._asystem_copied_parameters.clear()

        if all(results.values()):
            logger.info(f"Weights for step {step_id} is consistent")
        else:
            not_consistent_weights = [
                name for name, result in results.items() if not result
            ]
            logger.error(
                f"Weights for step {step_id} is not consistent: {not_consistent_weights}, total {len(results)} weights"
            )
        logger.info("Finished verifying parameters")
        return results

    def _pre_update_weights(self, step_id, **kwargs):
        if not self.enable_colocate_mode:
            return
        _log_gpu_memory(f"WeightsReader._pre_update_weights START step={step_id}")
        # Wait for training to offload weights BEFORE broadcasting to workers
        # This ensures all training ranks have released weights before we try to resume
        self.meta_server_client.wait_set_until_size(
            "all_training_offloaded_weights",
            self.training_world_size,
            timeout=self.timeout,
        )
        logger.info(
            f"[_pre_update_weights] Training has offloaded weights, now broadcasting resume to all TP workers"
        )
        # IMPORTANT: resume_memory_occupation has torch.distributed.barrier(tp_cpu_group)
        # which requires ALL TP workers to participate. Since only node_rank=0 workers
        # call WeightsReader methods, we MUST broadcast via execute_task_in_model_worker
        # to ensure all 64 workers call resume_memory_occupation together.
        _log_gpu_memory(f"WeightsReader._pre_update_weights BEFORE resume_weights (via broadcast) step={step_id}")
        self.inference_engine.execute_task_in_model_worker(
            self._pre_update_weights_in_tp_worker,
            step_id=step_id,
            resume_weights=True,  # Signal to resume weights inside the worker
        )
        _log_gpu_memory(f"WeightsReader._pre_update_weights AFTER resume_weights (via broadcast) step={step_id}")
        logger.info(
            f"Finished pre-updating weights for step {step_id} in colocate mode on engine rank {self.engine_rank}"
        )

    @classmethod
    def _pre_update_weights_in_tp_worker(cls, **kwargs):
        model_context = kwargs["model_context"]
        scheduler = model_context["scheduler"]
        resume_weights = kwargs.pop("resume_weights", False)

        # Resume weights memory occupation - this is called on ALL TP workers
        # so the barrier inside resume_memory_occupation will succeed
        if resume_weights:
            logger.info(
                f"[_pre_update_weights_in_tp_worker] Calling resume_memory_occupation('weights') "
                f"on rank {scheduler.tp_rank if hasattr(scheduler, 'tp_rank') else 'unknown'}"
            )
            # Use memory_saver_adapter.resume directly to avoid idempotent issues
            # Since Slime's offload_rollout called release_memory_occupation on all workers,
            # we need to resume on all workers too
            from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
            if GPU_MEMORY_TYPE_WEIGHTS in scheduler.offload_tags:
                scheduler.offload_tags.remove(GPU_MEMORY_TYPE_WEIGHTS)
                scheduler.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
                import torch
                torch.distributed.barrier(scheduler.tp_cpu_group)
                # Import static state back
                if hasattr(scheduler, 'stashed_model_static_state'):
                    from sglang.srt.managers.scheduler_update_weights_mixin import _import_static_state
                    _import_static_state(
                        scheduler.tp_worker.model_runner.model,
                        scheduler.stashed_model_static_state,
                    )
                    del scheduler.stashed_model_static_state
                logger.info(f"[_pre_update_weights_in_tp_worker] Resumed weights memory")
            else:
                logger.info(f"[_pre_update_weights_in_tp_worker] weights not in offload_tags, skipping resume")

        weights_reader = scheduler.awes_weights_reader
        weights_reader.pre_update_weights(**kwargs)

    @classmethod
    def _update_parameters_in_tp_worker(cls, **kwargs):
        model_context = kwargs["model_context"]
        scheduler = model_context["scheduler"]
        weights_reader = scheduler.awes_weights_reader
        weights_reader.update_weights(**kwargs)


class WorkerWeightsReader:
    def __init__(
        self,
        engine_name,
        model,
        model_context,
        infer_conf,
        engine_rank,
        num_engines,
        meta_server_addr: str,
        parameters_meta: List[ParameterMeta],
        training_params_meta: List[ParameterMeta],
        enable_debug_mode: bool = False,
        debug_mode_config: Dict[str, Any] = None,
        disable_pipeline: bool = False,
        enable_colocate_mode: bool = False,
        ipc_backend: str = "cuda",
        weights_comm_nccl_group_size: int = None,
    ):
        self.engine_name = engine_name
        self.model = model
        self.model_context = model_context
        self.infer_conf = infer_conf
        self.hf_config = infer_conf["hf_config"]
        self.model_arch_name = self.hf_config.architectures[0]
        self.scheduler = model_context["scheduler"]
        self.infer_engine_config = model_context["infer_engine_config"]
        self.engine_rank = engine_rank
        self.num_engines = num_engines
        self.enable_debug_mode = enable_debug_mode
        self.enable_nccl_debug_mode = (debug_mode_config or {}).get(
            "enable_nccl_debug_mode", False
        )
        self.disable_pipeline = disable_pipeline
        self.enable_colocate_mode = enable_colocate_mode
        self.ipc_backend = ipc_backend
        self.weights_comm_nccl_group_size = weights_comm_nccl_group_size

        self.train_to_infer_device_mapping = None
        self.infer_to_train_device_mapping = None
        logger.info(
            f"Disable pipeline for weights reader: {self.disable_pipeline} enable_colocate_mode {enable_colocate_mode}"
        )
        self.parameters_meta = parameters_meta
        self.training_params_meta = training_params_meta
        self.training_world_size = training_params_meta[0].shards[0].world_size
        self.infer_instance_world_size = parameters_meta[0].shards[0].world_size
        self.infer_world_size = num_engines * self.infer_instance_world_size
        self.transfer_world_size = self.training_world_size + self.infer_world_size
        self.rank_info = get_rank_info_extractor(engine_name)(
            model_context, engine_rank
        )
        logger.info(f"Reader rank info: {self.rank_info}")
        self.transfer_rank = (
            +self.engine_rank * self.infer_instance_world_size
            + self.rank_info.global_rank
        )
        self.meta_server_addr = meta_server_addr
        self.meta_server_client = MetaServerClient(*self.meta_server_addr.split(":"))
        self.weight_converter = get_infer_weights_converter(
            self.engine_name,
            self.model_arch_name,
            self.hf_config,
            self.rank_info,
            self.infer_engine_config,
        )
        self.current_worker_parameters_meta = [
            p.to_local_parameter_meta(self.rank_info.global_rank)
            for p in self.parameters_meta
        ]
        self.total_local_num_elements = sum(
            shard.numel
            for p in self.current_worker_parameters_meta
            for shard in p.shards
        )
        self.total_local_param_size = sum(
            shard.numel * shard.dtype.itemsize
            for p in self.current_worker_parameters_meta
            for shard in p.shards
        )
        logger.info(
            f"[Reader {self.transfer_rank}] Total local number of elements: {self.total_local_num_elements}, "
            f"total local parameter size: {self.total_local_param_size}"
        )
        self.timeout = 10000
        self._history_update_weights_time = {}
        logger.info(f"Env varabbles for weights reader: {stripped_env_vars()}")
        logger.info(
            f"Created weights reader for rank {self.rank_info.global_rank}, engine rank {self.engine_rank}"
        )

    def initialize(self):
        self.parameters = {
            hf_name: hf_param
            for name, param in self.model.named_parameters()
            for hf_name, hf_param in self.weight_converter.convert_param(name, param)
        }
        # [DEBUG] Verify expert views share storage with original tensor
        self._verify_expert_view_storage()

    def _verify_expert_view_storage(self):
        """Verify that expert parameter views share storage with the original w13_weight tensor."""
        import torch
        # Find original w13_weight tensors and their expert views
        original_tensors = {}
        for name, param in self.model.named_parameters():
            if "w13_weight" in name or "w2_weight" in name:
                original_tensors[name] = param
                logger.info(f"[VIEW_CHECK] Found original tensor: {name} shape={param.shape}")

        if not original_tensors:
            logger.info("[VIEW_CHECK] No expert tensors found (not MoE model)")
            return

        # Log some converter output parameter names for debugging
        sample_expert_params = [n for n in self.parameters.keys() if "experts." in n and "gate_up_proj" in n][:5]
        logger.info(f"[VIEW_CHECK] Sample expert param names: {sample_expert_params}")
        logger.info(f"[VIEW_CHECK] Original tensor names: {list(original_tensors.keys())[:5]}")

        # Check if views share storage with original
        view_check_results = {}
        for hf_name, hf_param in self.parameters.items():
            if "experts." in hf_name and ("gate_up_proj" in hf_name or "down_proj" in hf_name):
                # Find the corresponding original tensor
                # e.g., model.layers.0.mlp.experts.80.gate_up_proj.weight -> model.layers.0.mlp.experts.w13_weight
                if "gate_up_proj" in hf_name:
                    # Parse: model.layers.0.mlp.experts.80.gate_up_proj.weight
                    parts = hf_name.split(".")
                    # Find layer index
                    layer_idx = None
                    for i, p in enumerate(parts):
                        if p == "layers" and i + 1 < len(parts):
                            layer_idx = parts[i + 1]
                            break
                    if layer_idx:
                        # Construct original name: model.layers.{layer_idx}.mlp.experts.w13_weight
                        original_name = f"model.layers.{layer_idx}.mlp.experts.w13_weight"
                    else:
                        original_name = None
                elif "down_proj" in hf_name:
                    parts = hf_name.split(".")
                    layer_idx = None
                    for i, p in enumerate(parts):
                        if p == "layers" and i + 1 < len(parts):
                            layer_idx = parts[i + 1]
                            break
                    if layer_idx:
                        original_name = f"model.layers.{layer_idx}.mlp.experts.w2_weight"
                    else:
                        original_name = None
                else:
                    continue

                if original_name and original_name in original_tensors:
                    original_tensor = original_tensors[original_name]
                    # Check if hf_param shares storage with original
                    shares_storage = hf_param.storage().data_ptr() == original_tensor.storage().data_ptr()
                    # Extract expert_id
                    expert_id = None
                    for i, p in enumerate(parts):
                        if parts[i-1] == "experts" and p.isdigit():
                            expert_id = int(p)
                            break
                    view_check_results[hf_name] = {
                        "original": original_name,
                        "expert_id": expert_id,
                        "shares_storage": shares_storage,
                        "view_shape": tuple(hf_param.shape),
                        "original_shape": tuple(original_tensor.shape),
                    }

        # Log summary
        num_share = sum(1 for v in view_check_results.values() if v["shares_storage"])
        num_total = len(view_check_results)
        logger.info(f"[VIEW_CHECK] {num_share}/{num_total} expert views share storage with original tensor")

        # Log first few for debugging
        for i, (name, result) in enumerate(view_check_results.items()):
            if i < 5 or not result["shares_storage"]:
                logger.info(
                    f"[VIEW_CHECK] {name}: shares_storage={result['shares_storage']}, "
                    f"original={result['original']}, expert_id={result['expert_id']}, "
                    f"view_shape={result['view_shape']}, original_shape={result['original_shape']}"
                )
                if i >= 5 and result["shares_storage"]:
                    break

    def pre_update_weights(self, step_id, **kwargs):
        pass

    def update_weights(self, step_id, **kwargs):
        start_time = time.time()
        torch.cuda.synchronize()
        if self.enable_colocate_mode:
            self._update_weights_in_colocate_mode(step_id, **kwargs)
        else:
            self._update_weights(step_id, **kwargs)
        logger.info(
            f"Start to flush cache for step {step_id} for rank {self.transfer_rank}"
        )
        flash_cache_success = self.scheduler.flush_cache()
        assert flash_cache_success, "Cache flush failed after updating weights"
        logger.info(
            f"Finished flushing cache for step {step_id} for rank {self.transfer_rank}"
        )
        torch.cuda.synchronize()
        duration = time.time() - start_time
        compute_statistics(
            self._history_update_weights_time, step_id, duration, "Update weights"
        )

    def _update_weights(self, step_id, **kwargs):
        logger.info(
            f"Start to update weights for step {step_id} for rank {self.engine_rank}-{self.rank_info.global_rank}"
        )
        tensor_pairs = []
        for parameter_meta in self.current_worker_parameters_meta:
            name = parameter_meta.name
            parameter = self.parameters[name]
            if len(parameter_meta.shards) != 1:
                raise ValueError(f"Current shard is None for parameter: {name}")
            tensor_pairs.append(
                (
                    name,
                    parameter,
                    parameter_meta.shards[0],
                    parameter_meta,
                )
            )
        self.read_tensors(step_id, tensor_pairs, **kwargs)
        self.finish_step(step_id)
        logger.info(
            f"Finished updating weights for step {step_id} for rank {self.engine_rank}-{self.rank_info.global_rank}"
        )

    def _update_weights_in_colocate_mode(self, step_id, **kwargs):
        self._update_weights(step_id, **kwargs)

    def finish_step(self, step_id):
        pass

    def read_tensors(
        self,
        step_id: int,
        tensor_pairs: List,
        **kwargs,
    ):
        pass


def get_weights_exchange_reader(inference_engine) -> WeightExchangeReader:
    if inference_engine.config.comm_backend == "file":
        return FileWeightExchangeReader(inference_engine)
    return WeightsReader(inference_engine)
