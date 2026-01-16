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

from awex.sharding.rank_info import RankInfo


def get_sglang_sharding_strategy(
    model_name: str, infer_engine_config, rank_info: RankInfo, **kwargs
):
    """
    Get the sharding strategy class for a given model architecture name.
    """
    from awex import logging
    from awex.models import get_sharding_strategy

    logger = logging.getLogger(__name__)

    cls = get_sharding_strategy(model_name)

    # Debug: log the config object type and attributes
    config_type = type(infer_engine_config).__name__
    raw_dp_attention = getattr(infer_engine_config, "enable_dp_attention", "ATTR_NOT_FOUND")
    raw_dp_lm_head = getattr(infer_engine_config, "enable_dp_lm_head", "ATTR_NOT_FOUND")
    logger.info(
        f"[SHARDING_DEBUG] Config type: {config_type}, "
        f"raw enable_dp_attention={raw_dp_attention}, raw enable_dp_lm_head={raw_dp_lm_head}"
    )

    # Safely get config values with defaults (some configs may have None values)
    enable_dp_attention = getattr(infer_engine_config, "enable_dp_attention", False) or False
    enable_dp_lm_head = getattr(infer_engine_config, "enable_dp_lm_head", False) or False
    moe_dense_tp_size = getattr(infer_engine_config, "moe_dense_tp_size", 1) or 1
    ep_size = getattr(infer_engine_config, "ep_size", 1) or 1
    # MoE A2A backend determines shared_experts TP behavior in SGLang
    # When using deepep/mooncake, shared_experts are NOT TP-sharded
    moe_a2a_backend = getattr(infer_engine_config, "moe_a2a_backend", "none") or "none"

    logger.info(
        f"[SHARDING] Creating sharding strategy for {model_name}: "
        f"enable_dp_attention={enable_dp_attention}, enable_dp_lm_head={enable_dp_lm_head}, "
        f"tp_size={rank_info.tp_size}, attn_tp_size={rank_info.attn_tp_size}, ep_size={ep_size}, "
        f"moe_a2a_backend={moe_a2a_backend}"
    )

    return cls(
        engine_name="sglang",
        enable_dp_attention=enable_dp_attention,
        enable_dp_lm_head=enable_dp_lm_head,
        moe_dense_tp_size=moe_dense_tp_size,
        tp_size=rank_info.tp_size,
        ep_size=ep_size,
        ep_tp_size=rank_info.ep_tp_size,
        rank_info=rank_info,
        moe_a2a_backend=moe_a2a_backend,
        **kwargs,
    )


def get_sglang_rank_info(model_context, engine_rank) -> RankInfo:
    scheduler = model_context["scheduler"]
    infer_engine_config = scheduler.server_args
    if infer_engine_config.dp_size != 1 and not infer_engine_config.enable_dp_attention:
        raise ValueError(
            f"DP size is not 1, but {infer_engine_config.dp_size}. This is not supported yet."
        )
    tp_size = model_context["tp_size"]
    tp_rank = model_context["tp_rank"]
    ep_size = infer_engine_config.ep_size
    if ep_size > 1:
        ep_tp_size = tp_size // ep_size
        ep_tp_rank = tp_rank % ep_tp_size
        ep_rank = tp_rank // ep_tp_size
    else:
        assert ep_size == 1, "ep_size must be 1"
        ep_rank = 0
        ep_tp_size = 1
        ep_tp_rank = 0
    return RankInfo(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=model_context["pp_rank"],
        pp_size=model_context["pp_size"],
        dp_size=model_context["dp_size"],
        dp_rank=0,
        ep_rank=ep_rank,
        ep_size=ep_size,
        ep_tp_rank=ep_tp_rank,
        ep_tp_size=ep_tp_size,
        attn_tp_rank=model_context["attn_tp_rank"],
        attn_tp_size=model_context["attn_tp_size"],
        attn_dp_rank=model_context["attn_dp_rank"],
        world_size=model_context["world_size"],
        global_rank=model_context["global_rank"],
        engine_rank=engine_rank,
        local_rank=model_context["local_rank"],
        is_infer=True,
    )
