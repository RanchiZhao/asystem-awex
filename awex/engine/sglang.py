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

import time
from typing import Any, Dict, List, Optional, Union

from awex import logging
from awex.config import InferenceConfig
from awex.engine.core import InferenceEngine
from awex.reader.weights_reader import get_weights_exchange_reader
from awex.util.gpu import get_gpu_status

logger = logging.getLogger(__name__)


class SGLangEngine(InferenceEngine):
    def __init__(self, config: Union[Dict[str, Any], InferenceConfig], sgl_engine):
        super().__init__(sgl_engine.tokenizer_manager.model_config)
        if isinstance(config, dict):
            config = InferenceConfig.from_dict(config)
        self._config = config
        self._sgl_engine = sgl_engine
        self.node_rank = config.node_rank
        self.released_tags = set()
        self.weights_exchange_reader = None
        self.rank_coordinate = f"{config.engine_rank}-{self.node_rank}"
        self._initialized = False

        # In cross-node colocate mode, each engine independently handles weight updates
        # because execute_task_in_model_worker cannot broadcast across nodes.
        # When awex_per_node_mode is True, all engines (not just node_rank=0) initialize
        # WeightsReader and handle weight updates for their local 8 GPUs.
        self._awex_per_node_mode = getattr(config, 'awex_per_node_mode', False)
        if self._awex_per_node_mode:
            logger.info(
                f"[SGLangEngine] {self.rank_coordinate}: Per-node mode enabled - "
                f"this engine will independently handle weight updates"
            )

    @property
    def engine_name(self):
        return "sglang"

    @property
    def config(self):
        return self._config

    def initialize(self) -> None:
        import time
        # Determine if this engine should initialize weights exchange reader.
        #
        # There are three modes:
        # 1. awex_per_node_mode=True: All engines initialize independently
        #    (each handles its own 8 GPUs)
        # 2. Multi-node engine (tp_size > 8 or nodes_per_engine > 1):
        #    All nodes in each engine must initialize so that all workers can participate
        #    in init_weights_update_group together. This applies regardless of num_engines.
        #    Example: num_engines=2, tp_size=64 means each engine uses 8 nodes,
        #    so all 8 nodes per engine must initialize.
        # 3. Regular mode: Only node_rank=0 initializes
        #
        # The key insight for mode 2: init_weights_update_group requires ALL
        # workers (64 in our case) to call it simultaneously. If only node_rank=0
        # initializes (8 workers), the other 56 workers won't participate,
        # causing deadlock.

        # Detect multi-node engine mode
        nnodes = getattr(self.config, 'nnodes', None)
        tp_size = getattr(self.config, 'tp_size', None)
        num_gpus_per_node = 8  # Standard assumption

        # Check if each engine spans multiple nodes
        # This is true when tp_size > num_gpus_per_node, regardless of num_engines
        # For example: 2 engines with tp_size=64 each means each engine uses 8 nodes
        is_multi_node_engine = False
        if tp_size is not None and tp_size > num_gpus_per_node:
            # tp_size > 8 implies this engine spans multiple nodes
            is_multi_node_engine = True
        elif nnodes is not None:
            # Calculate nodes per engine
            nodes_per_engine = nnodes // max(self.config.num_engines, 1)
            if nodes_per_engine > 1:
                is_multi_node_engine = True

        should_initialize = (
            self._awex_per_node_mode or
            is_multi_node_engine or
            self.config.node_rank == 0
        )

        logger.info(
            f"[SGLangEngine] {self.rank_coordinate}: Initialize decision - "
            f"per_node_mode={self._awex_per_node_mode}, "
            f"multi_node_engine={is_multi_node_engine} (nnodes={nnodes}, tp_size={tp_size}, num_engines={self.config.num_engines}), "
            f"node_rank={self.config.node_rank}, should_initialize={should_initialize}"
        )

        if should_initialize:
            logger.info(
                f"Start to initialize weights exchange reader for {self.rank_coordinate} "
                f"(per_node_mode={self._awex_per_node_mode})"
            )
            self._initialized = True
            t0 = time.time()
            logger.info(f"[PROFILE] {self.rank_coordinate} Calling get_weights_exchange_reader...")
            self.weights_exchange_reader = get_weights_exchange_reader(self)
            t1 = time.time()
            logger.info(f"[PROFILE] {self.rank_coordinate} get_weights_exchange_reader took {t1-t0:.2f}s")
            logger.info(f"[PROFILE] {self.rank_coordinate} Calling weights_exchange_reader.initialize()...")
            self.weights_exchange_reader.initialize()
            t2 = time.time()
            logger.info(f"[PROFILE] {self.rank_coordinate} initialize() took {t2-t1:.2f}s")
            logger.info(
                f"Finished initializing weights exchange reader for {self.rank_coordinate}"
            )
        else:
            logger.info(
                f"Skip initializing weights exchange reader for {self.rank_coordinate}"
            )

    def update_weights_from_disk(
        self, model_path: str, load_format: Optional[str] = None
    ):
        """Update model weights for inference."""
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call setup_model() first.")
        logger.info(
            f"Start to update weights from disk for step {self.global_step} for "
            f"{self.rank_coordinate}, path: {model_path}, load_format: {load_format}"
        )
        if self.node_rank != 0:
            logger.info("Non-zero rank node, skipping update weights from disk")
            return
        self._sgl_engine.update_weights_from_disk(
            model_path=model_path, load_format=load_format
        )
        logger.info(
            f"Finished updating weights from disk for step {self.global_step} for "
            f"{self.rank_coordinate}, path: {model_path}, load_format: {load_format}"
        )

    def update_weights(self, **kwargs):
        # Use step_id from kwargs if provided, otherwise fallback to global_step
        step_id = kwargs.pop('step_id', self.global_step)

        # In multi-node engine mode, all nodes must participate in update_weights
        # because each node's inference engine receives weights for its GPUs.
        # In per-node mode, all engines also execute independently.
        # In standard single-node mode, only node_rank=0 workers have initialized weights_exchange_reader
        tp_size = getattr(self._config, 'tp_size', None)
        num_gpus_per_node = 8
        is_multi_node_engine = tp_size is not None and tp_size > num_gpus_per_node

        should_update = self._awex_per_node_mode or is_multi_node_engine or self.node_rank == 0

        if not should_update:
            logger.info(
                f"Non-zero rank node {self.rank_coordinate}, skipping update_weights"
            )
            return

        logger.info(
            f"Start to update weights for step {step_id} for {self.rank_coordinate} "
            f"(per_node_mode={self._awex_per_node_mode})"
        )
        start_time = time.time()
        self.weights_exchange_reader.update_weights(step_id=step_id, **kwargs)
        duration = time.time() - start_time
        logger.info(
            f"Finished updating weights for step {step_id} for {self.rank_coordinate}, "
            f"took {duration:.3f} seconds"
        )

    def release_memory_occupation(self, tags: Optional[List[str]] = None) -> None:
        tags = tags or ["kv_cache", "weights"]
        if isinstance(tags, str):
            tags = [tags]
        # In multi-node engine mode, all nodes can release memory
        tp_size = getattr(self._config, 'tp_size', None)
        num_gpus_per_node = 8
        is_multi_node_engine = tp_size is not None and tp_size > num_gpus_per_node

        should_release = self._awex_per_node_mode or is_multi_node_engine or self.node_rank == 0
        if self._initialized and should_release:
            logger.info(
                f"Release memory occupation {tags}, released_tags {self.released_tags}"
            )
            if set(tags) - self.released_tags != set(tags):
                tags = list(set(tags) - self.released_tags)
            self.released_tags.update(tags)
            if not tags:
                logger.info("No memory occupation to release")
                return
            logger.info(f"Start to release memory occupation {tags}")
            logger.info(f"GPU status before release:\n{get_gpu_status()}")
            self._sgl_engine.release_memory_occupation(tags=tags)
            logger.info("Finished releasing memory occupation")
            logger.info(f"GPU status after release:\n{get_gpu_status()}")

    def resume_memory_occupation(self, tags: Optional[List[str]] = None) -> None:
        """Resume memory occupation for the engine.
        tags: kv_cache, weights, default is both
        """
        tags = tags or ["kv_cache", "weights"]
        if isinstance(tags, str):
            tags = [tags]
        # In multi-node engine mode, all nodes can resume memory
        tp_size = getattr(self._config, 'tp_size', None)
        num_gpus_per_node = 8
        is_multi_node_engine = tp_size is not None and tp_size > num_gpus_per_node

        should_resume = self._awex_per_node_mode or is_multi_node_engine or self.node_rank == 0
        if self._initialized and should_resume:
            logger.info(
                f"Resume memory occupation {tags}, released_tags {self.released_tags}"
            )
            tags = list(self.released_tags & set(tags))
            self.released_tags.difference_update(tags)
            if not tags:
                logger.info("No memory occupation to resume")
                return
            logger.info(f"Start to resume memory occupation {tags}")
            logger.info(f"GPU status before resume:\n{get_gpu_status()}")
            self._sgl_engine.resume_memory_occupation(tags=tags)
            logger.info("Finished resuming memory occupation")
            logger.info(f"GPU status after resume:\n{get_gpu_status()}")

    def execute_task_in_model_worker(self, fn, **kwargs):
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call `initialize` first.")
        # In multi-node engine mode, all nodes need to execute tasks (not just node_rank=0)
        # This is required for colocate mode where each node's inference engine
        # must participate in weight synchronization
        tp_size = getattr(self._config, 'tp_size', None)
        num_gpus_per_node = 8
        is_multi_node_engine = tp_size is not None and tp_size > num_gpus_per_node

        should_execute = self._awex_per_node_mode or is_multi_node_engine or self.node_rank == 0
        if not should_execute:
            raise RuntimeError(
                f"Non-zero rank node {self.rank_coordinate} is not allowed to "
                f"execute task in model workers"
            )
        return self._sgl_engine.execute_task_in_model_worker(fn, **kwargs)

    @property
    def num_engines(self):
        return self._config.num_engines

    @property
    def engine_rank(self):
        return self._config.engine_rank


def extract_sgl_config(config: Dict[str, Any]) -> Dict[str, Any]:
    from sglang.srt.server_args import ServerArgs

    engine_kwargs = {
        k: v for k, v in config.items() if k in ServerArgs.__dataclass_fields__
    }
    return engine_kwargs
