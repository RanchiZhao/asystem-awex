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

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

from awex.meta.meta_resolver import ParamMetaResolver, logger
from awex.meta.weight_meta import (
    ParameterMeta,
    ParameterReplicaMeta,
    ParameterShardMeta,
    compute_total_model_size,
    dump_parameters_meta,
)
from awex.models.registry import get_infer_weights_converter
from awex.sharding import get_rank_info_extractor, get_sharding_strategy_builder
from awex.sharding.param_sharding import ShardingType
from awex.sharding.rank_info import RankInfo
from awex.util.common import to_dict


class InferParamMetaResolver(ParamMetaResolver):
    def __init__(
        self,
        inference_engine,
        convert_params=False,
        num_engines=1,
        engine_rank=0,
    ):
        """
        Args:
            inference_engine: The inference engine object that can execute tasks in model workers.
            convert_params: Whether to convert the parameters to the Hugging Face format.
        """
        import time
        super().__init__(inference_engine.hf_config)
        self._inference_engine = inference_engine
        self.infer_engine_config = inference_engine.config
        self.engine_name = inference_engine.engine_name
        self.convert_params = convert_params
        self.num_engines = num_engines
        self.engine_rank = engine_rank

        logger.info(f"[PROFILE] InferParamMetaResolver.__init__: engine_rank={engine_rank}, convert_params={convert_params}")
        # Log vocab_size from HF config to help diagnose padding issues
        vocab_size = getattr(inference_engine.hf_config, "vocab_size", None)
        logger.info(f"[VOCAB_DEBUG] HF config vocab_size={vocab_size}")

        suffix = f"{engine_rank}_{os.getpid()}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.json"
        if self._inference_engine.config.enable_debug_mode:
            non_converted_params_raw_meta = (
                inference_engine.execute_task_in_model_worker(
                    self._get_model_param_info,
                    engine_name=self.engine_name,
                    infer_engine_config=self.infer_engine_config,
                    engine_rank=engine_rank,
                    convert_params=False,
                )
            )
            filename = f"infer_params_non_converted_raw_meta_{suffix}"
            abs_filename = os.path.abspath(filename)
            with open(abs_filename, "w") as f:
                json.dump(to_dict(non_converted_params_raw_meta), f, indent=4)
            logger.info(
                f"Inference rank {engine_rank}, non_converted_params_raw_meta: {abs_filename}"
            )
        logger.info(f"[PROFILE] InferParamMetaResolver: Starting execute_task_in_model_worker for _get_model_param_info...")
        t0 = time.time()
        self._params_raw_meta = inference_engine.execute_task_in_model_worker(
            self._get_model_param_info,
            engine_name=self.engine_name,
            infer_engine_config=self.infer_engine_config,
            engine_rank=engine_rank,
            convert_params=self.convert_params,
        )
        t1 = time.time()
        logger.info(f"[PROFILE] InferParamMetaResolver: execute_task_in_model_worker completed in {t1-t0:.2f}s")
        if self._inference_engine.config.enable_debug_mode:
            filename = f"infer_params_raw_meta_{suffix}"
            abs_filename = os.path.abspath(filename)
            with open(abs_filename, "w") as f:
                json.dump(to_dict(self._params_raw_meta), f, indent=4)
            logger.info(
                f"Inference rank {engine_rank}, params_raw_meta: {abs_filename}"
            )

        # In colocate mode, we only have LOCAL metadata (one rank).
        # In non-colocate mode, we have metadata from ALL ranks.
        # We use the available metadata to initialize, prioritizing rank 0 if available.
        enable_colocate = getattr(self._inference_engine.config, "enable_colocate_mode", False)
        if enable_colocate:
            # Colocate mode: use local metadata (only one entry)
            if len(self._params_raw_meta) != 1:
                logger.warning(f"Colocate mode: expected 1 local meta, got {len(self._params_raw_meta)}")
            local_meta = self._params_raw_meta[0]
            self._rank0_meta = local_meta  # Use local as reference
            logger.info(
                f"[COLOCATE] Using local metadata from rank {local_meta['rank_info'].global_rank} "
                f"as reference (world_size={local_meta['rank_info'].world_size})"
            )
        else:
            # Non-colocate mode: find rank 0's metadata
            rank0_params = [
                info for info in self._params_raw_meta if info["rank_info"].global_rank == 0
            ]
            if len(rank0_params) != 1:
                logger.error(f"Expected 1 rank0 meta, got {rank0_params}")
                raise ValueError(f"Expected 1 rank0 meta, got {len(rank0_params)}")
            [self._rank0_meta] = rank0_params

        self.rank0_info = self._rank0_meta["rank_info"]
        self._world_size = self.rank0_info.world_size
        self._model_arch_name = self._rank0_meta["model_arch_name"]
        self._sharding_strategy = get_sharding_strategy_builder(self.engine_name)(
            self._model_arch_name,
            self.infer_engine_config,
            self.rank0_info,
        )
        # Log which sharding strategy class is being used
        logger.info(
            f"[SHARDING_CLASS] Using {type(self._sharding_strategy).__name__} for {self._model_arch_name}, "
            f"enable_dp_attention={self._sharding_strategy.enable_dp_attention}, "
            f"enable_dp_lm_head={self._sharding_strategy.enable_dp_lm_head}"
        )
        self._params_meta = self._build_params_meta()
        if self._inference_engine.config.enable_debug_mode:
            filename = f"infer_params_meta_{suffix}"
            abs_filename = os.path.abspath(filename)
            with open(abs_filename, "w") as f:
                json.dump(dump_parameters_meta(self._params_meta), f, indent=4)
            logger.info(f"Inference rank {engine_rank}, params_meta: {abs_filename}")
        self.total_numel = sum(param.global_numel for param in self._params_meta)
        self.total_size = compute_total_model_size(self._params_meta)
        logger.info(
            f"Total number of elements in the model: {self.total_numel}, total size: {self.total_size} bytes"
        )

    def get_model_arch_name(self) -> str:
        return self._model_arch_name

    def get_parameters_meta(self) -> List[ParameterMeta]:
        """
        Returns the list of ParameterMeta objects for all parameters in the model.
        """
        return self._params_meta

    def _get_params_raw_meta(self) -> List[Dict[str, Any]]:
        return self._params_raw_meta

    def _get_sharding_info(
        self, name: str, rank_info: RankInfo, param_meta: Dict[str, Any]
    ) -> Tuple[ShardingType, int, int]:
        return self._sharding_strategy.get_sharding_strategy(
            name, rank_info=rank_info, param_meta=param_meta
        )

    def _build_params_meta(self) -> List[ParameterMeta]:
        """
        Override parent method to handle colocate mode.

        In colocate mode, we only have metadata from a single rank (local metadata).
        We use sharding strategy to infer the complete shard structure for all ranks.
        """
        all_params_raw_meta = self._get_params_raw_meta()
        enable_colocate = getattr(self._inference_engine.config, "enable_colocate_mode", False)

        logger.info(
            f"[COLOCATE_CHECK] enable_colocate={enable_colocate}, "
            f"num_raw_meta={len(all_params_raw_meta)}, "
            f"config_type={type(self._inference_engine.config).__name__}"
        )

        # In colocate mode with only local metadata, use inference-based approach
        if enable_colocate and len(all_params_raw_meta) == 1:
            logger.info(
                f"[COLOCATE] Building params meta from local metadata only "
                f"(inferring {self.rank0_info.attn_tp_size} shards from sharding strategy)"
            )
            return self._build_params_meta_from_local(all_params_raw_meta[0])

        # Otherwise, use parent's implementation (has all ranks' metadata)
        logger.info(
            f"[COLOCATE_CHECK] Using parent _build_params_meta (non-colocate or has all ranks)"
        )
        return super()._build_params_meta()

    def _build_params_meta_from_local(self, local_meta: Dict[str, Any]) -> List[ParameterMeta]:
        """
        Build complete ParameterMeta list from a single rank's metadata.

        In colocate mode, we only have local metadata from one rank. We use the
        sharding strategy to infer the complete shard structure for all ranks.

        For each parameter:
        1. Get sharding info (sharding_type, sharding_dim, num_shards) from strategy
        2. Infer global shape from local shape and num_shards
        3. Generate virtual shards for all ranks with computed offsets

        Args:
            local_meta: Metadata dict from a single rank containing 'rank_info' and 'params_meta'

        Returns:
            List of ParameterMeta with complete shard information
        """
        rank_info: RankInfo = local_meta["rank_info"]
        local_params_meta = local_meta["params_meta"]

        logger.info(
            f"[COLOCATE] Inferring params meta from local rank {rank_info.global_rank}: "
            f"attn_tp_size={rank_info.attn_tp_size}, ep_size={rank_info.ep_size}, "
            f"tp_size={rank_info.tp_size}, world_size={rank_info.world_size}"
        )

        params_meta_list = []

        for param_meta in local_params_meta:
            name = param_meta["name"]
            local_shape = param_meta["shape"]
            local_numel = param_meta["numel"]
            dtype = param_meta["dtype"]

            # Get sharding strategy for this parameter
            sharding_type, sharding_dim, num_shards = self._get_sharding_info(
                name, rank_info, param_meta
            )

            # Debug log for key parameters (attention, shared_experts, embedding, lm_head)
            should_log = any(k in name for k in [
                "kv_b_proj", "q_b_proj", "o_proj",  # attention
                "shared_experts",  # shared experts (key issue!)
                "embed_tokens", "lm_head",  # embedding and output
            ])
            if should_log:
                logger.info(
                    f"[COLOCATE_DEBUG] {name}: "
                    f"sharding_type={sharding_type}, num_shards={num_shards}, "
                    f"sharding_dim={sharding_dim}, local_shape={local_shape}, "
                    f"enable_dp_attention={self._sharding_strategy.enable_dp_attention}, "
                    f"strategy_class={type(self._sharding_strategy).__name__}"
                )

            # Compute global shape
            # IMPORTANT: EP_SHARDING is different from TP_SHARDING!
            # - TP_SHARDING: A single tensor is split across ranks → global_numel = local_numel * num_shards
            # - EP_SHARDING: Different experts are assigned to different ranks → global_numel = local_numel
            #   Each expert parameter (e.g., experts.20.weight) is COMPLETE on its rank, not a shard.
            #   The "sharding" is at the expert level, not the tensor level.
            num_dims = len(local_shape)
            if sharding_type == ShardingType.NO_SHARDING or num_shards == 1:
                global_shape = tuple(local_shape)
                global_numel = local_numel
            elif sharding_type == ShardingType.EP_SHARDING:
                # EP sharding: experts are distributed across ranks, not tensor-sharded
                # Each expert's parameters are complete, so global_numel = local_numel
                global_shape = tuple(local_shape)
                global_numel = local_numel
                # For EP, num_shards represents how many ranks have this parameter name
                # (i.e., each rank has different experts but may have same local expert indices)
            else:
                global_shape = tuple(
                    local_shape[i] * num_shards if i == sharding_dim else local_shape[i]
                    for i in range(num_dims)
                )
                global_numel = local_numel * num_shards

            # Generate shards for all ranks
            # IMPORTANT: For EP_SHARDING, each expert is complete (not a shard of a larger tensor).
            # We only create 1 shard representing this expert on its assigned rank.
            # For TP/DP_TP sharding, we create num_shards virtual shards.
            shards = []
            if sharding_type == ShardingType.EP_SHARDING:
                # EP sharding: single shard representing this complete expert
                shard = ParameterShardMeta(
                    name=name,
                    tp_rank=0,
                    attn_tp_rank=0,
                    pp_rank=rank_info.pp_rank,
                    ep_rank=rank_info.ep_rank,  # Use actual ep_rank from local metadata
                    ep_tp_rank=0,
                    global_rank=rank_info.global_rank,
                    engine_rank=rank_info.engine_rank,
                    world_size=rank_info.world_size,
                    shape=local_shape,
                    numel=local_numel,
                    dtype=dtype,
                    global_offset=tuple(0 for _ in range(num_dims)),
                    sharding_type=sharding_type,
                    num_shards=1,  # Each expert is complete, not sharded
                    sharding_dim=sharding_dim,
                )
                shards.append(shard)
            else:
                # TP/DP_TP/EP_TP sharding: create virtual shards for all ranks
                for shard_idx in range(num_shards):
                    # Compute global offset for this shard
                    global_offset = tuple(
                        shard_idx * local_shape[i] if i == sharding_dim else 0
                        for i in range(num_dims)
                    )

                    # Determine rank values based on sharding type
                    if sharding_type == ShardingType.TP_SHARDING:
                        tp_rank = shard_idx
                        attn_tp_rank = shard_idx % rank_info.attn_tp_size
                        ep_rank = 0
                        ep_tp_rank = 0
                    elif sharding_type == ShardingType.DP_TP_SHARDING:
                        tp_rank = shard_idx
                        attn_tp_rank = shard_idx
                        ep_rank = 0
                        ep_tp_rank = 0
                    elif sharding_type == ShardingType.EP_TP_SHARDING:
                        tp_rank = shard_idx
                        attn_tp_rank = shard_idx % rank_info.attn_tp_size
                        ep_rank = shard_idx // rank_info.ep_tp_size if rank_info.ep_tp_size > 0 else 0
                        ep_tp_rank = shard_idx % rank_info.ep_tp_size if rank_info.ep_tp_size > 0 else 0
                    else:
                        # NO_SHARDING or unknown
                        tp_rank = 0
                        attn_tp_rank = 0
                        ep_rank = 0
                        ep_tp_rank = 0

                    shard = ParameterShardMeta(
                        name=name,
                        tp_rank=tp_rank,
                        attn_tp_rank=attn_tp_rank,
                        pp_rank=rank_info.pp_rank,
                        ep_rank=ep_rank,
                        ep_tp_rank=ep_tp_rank,
                        global_rank=shard_idx,  # Virtual global rank
                        engine_rank=rank_info.engine_rank,
                        world_size=rank_info.world_size,
                        shape=local_shape,
                        numel=local_numel,
                        dtype=dtype,
                        global_offset=global_offset,
                        sharding_type=sharding_type,
                        num_shards=num_shards,
                        sharding_dim=sharding_dim,
                    )
                    shards.append(shard)

            # Create ParameterMeta with single replica containing all shards
            param = ParameterMeta(
                name=name,
                global_numel=global_numel,
                global_shape=global_shape,
                dtype=dtype,
                shards=shards,
                replicas=[ParameterReplicaMeta(shards=shards)],
            )
            params_meta_list.append(param)

            # Log for key parameters
            if "embed_tokens" in name or "lm_head" in name or "qkv_proj" in name or ("experts" in name and "layers.0." in name):
                logger.info(
                    f"[COLOCATE_INFER] {name}: sharding_type={sharding_type.name}, "
                    f"num_shards={num_shards}, local_shape={local_shape}, "
                    f"global_shape={global_shape}"
                )

        total_shards = sum(len(p.replicas[0].shards) for p in params_meta_list)
        logger.info(
            f"[COLOCATE] Built {len(params_meta_list)} parameters with {total_shards} total shards "
            f"(inferred from single rank)"
        )

        return params_meta_list

    @staticmethod
    def _get_model_param_info(
        engine_name, infer_engine_config, convert_params=False, engine_rank=0, **kwargs
    ):
        """
        Static method to extract parameter meta information from a model and its context.
        Args:
            kwargs: Should contain 'model' and 'model_context'.
        Returns:
            dict: Metadata for the current rank, including rank_info, params_meta, and model_arch_name.
        """
        model = kwargs["model"]
        model_context = kwargs["model_context"]
        params_meta = []
        rank_info = get_rank_info_extractor(engine_name)(model_context, engine_rank)
        model_arch_name = type(model).__name__
        meta = {
            "rank_info": rank_info,
            "params_meta": params_meta,
            "model_arch_name": model_arch_name,
        }
        sglang_to_hf_weight_converter = get_infer_weights_converter(
            engine_name,
            model_arch_name,
            hf_config=model.config,
            infer_engine_config=infer_engine_config,
            rank_info=rank_info,
        )
        params = []
        logger.info(f"[DEBUG] Start iterating model.named_parameters(), rank={rank_info.global_rank}, convert_params={convert_params}")
        import time
        t_start = time.time()
        param_count = 0
        convert_time = 0.0
        for name, param in model.named_parameters():
            if convert_params:
                t_convert_start = time.time()
                try:
                    converted = sglang_to_hf_weight_converter.convert_param(name, param)
                    for hf_name, hf_param in converted:
                        params.append((hf_name, hf_param))
                except Exception as e:
                    logger.error(f"[ERROR] convert_param failed for {name}: {e}")
                    raise
                convert_time += time.time() - t_convert_start
            else:
                params.append((name, param))
            param_count += 1
            if param_count % 200 == 0:
                elapsed = time.time() - t_start
                logger.info(f"[DEBUG] Processed {param_count} parameters in {elapsed:.2f}s (convert_time={convert_time:.2f}s), rank={rank_info.global_rank}")
        t_end = time.time()
        logger.info(f"[DEBUG] Finished iterating {param_count} parameters in {t_end-t_start:.2f}s (convert_time={convert_time:.2f}s), rank={rank_info.global_rank}")
        for name, param in params:
            if not param.is_contiguous():
                logger.info(
                    f"Parameter {name} is not contiguous, shape: {param.shape}, "
                    f"rank: {rank_info.global_rank}"
                )
            params_meta.append(
                {
                    "name": name,
                    "numel": param.numel(),
                    "shape": tuple(param.shape),
                    "dtype": param.dtype,
                }
            )
        return meta
