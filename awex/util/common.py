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
import math
import os
import pickle
import socket
import struct
from enum import Enum
from typing import List

import torch

from awex import logging

logger = logging.getLogger(__name__)


def configure_logging(level=logging.INFO, force=True):
    logging.basicConfig(
        level=level,
        format="%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s -- %(process)d -- %(message)s",
        force=force,
    )


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, (
        f"{numerator} is not divisible by {denominator}"
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def get_ip_address():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return socket.gethostbyname(socket.gethostname())


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def to_binary(data):
    # Serialize messages using pickle
    pickled_data = pickle.dumps(data)
    # Get the length of the pickled data
    data_len = len(pickled_data)
    # Create the binary response: length of data (4 bytes) + pickled data
    return struct.pack("!I", data_len) + pickled_data


def from_binary(binary):
    # Extract the length of the pickled data
    data_len = struct.unpack("!I", binary[:4])[0]
    # Extract and unpickle the data
    pickled_data = binary[4 : 4 + data_len]
    data = pickle.loads(pickled_data)
    return data


def to_dict(param_meta, ignore_keys=None) -> dict:
    """Convert the parameter meta to a dict."""
    ignore_keys = ignore_keys or set()

    def convert_value(v):
        if isinstance(v, Enum):
            return v.value  # Handle enums
        if isinstance(v, (tuple, list)):
            return [convert_value(x) for x in v]
        if isinstance(v, torch.dtype):
            return str(v)  # Handle torch.dtype
        if isinstance(v, slice):
            return str(v)  # Handle slice
        if isinstance(v, dict):
            return {k: convert_value(v) for k, v in v.items() if k not in ignore_keys}
        if hasattr(v, "__dict__"):
            return {
                k: convert_value(v)
                for k, v in v.__dict__.items()
                if not k.startswith("_") and k not in ignore_keys
            }
        if hasattr(v, "__slots__"):
            return {
                k: convert_value(getattr(v, k))
                for k in v.__slots__
                if not k.startswith("_") and k not in ignore_keys
            }
        return v

    param_dict = convert_value(param_meta)
    return param_dict


def to_json(param_meta, ignore_keys=None) -> str:
    """Convert the parameter meta to a json string."""
    return json.dumps(to_dict(param_meta, ignore_keys), indent=2)


def compute_statistics(stage_history: dict, step_id: int, duration: float, stage: str):
    if stage not in stage_history:
        stage_history[stage] = []
    history = stage_history[stage]
    history.append(duration)
    if len(history) > 10000:
        history.pop(0)
    if step_id == 2:
        # first step contains init time
        history.pop(history.index(max(history)))
    num_updates = len(history)
    stage_history[stage] = history = sorted(history)
    avg_time = sum(history) / num_updates
    median_time = history[num_updates // 2]
    max_time = history[-1]
    min_time = history[0]
    logger.info(
        f"{stage} time statistics for step {step_id}: average time: {avg_time:.4f} seconds, median time: {median_time:.4f} seconds, "
        f"min time: {min_time:.4f} seconds,  max time: {max_time:.4f} seconds"
    )


def _is_acceptable_dtype_mismatch(
    param_name: str,
    train_dtype,
    infer_dtype,
) -> bool:
    """
    Check if a dtype mismatch is acceptable and can be handled at transfer time.

    Acceptable mismatches:
    - Router bias (e_score_correction_bias): float32 <-> bfloat16
      Megatron may store in float32, SGLang expects bfloat16 (or vice versa)
    """
    # Define acceptable conversions (bidirectional)
    ACCEPTABLE_FLOAT_CONVERSIONS = {
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float16),
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float16),
        (torch.float16, torch.bfloat16),
    }

    # Router bias - dtype conversion is handled at transfer time
    if "e_score_correction_bias" in param_name:
        dtype_pair = (train_dtype, infer_dtype)
        if dtype_pair in ACCEPTABLE_FLOAT_CONVERSIONS:
            logger.info(
                f"[DTYPE_CHECK] {param_name}: Allowing dtype mismatch "
                f"({train_dtype} -> {infer_dtype}), will convert at transfer time"
            )
            return True

    return False


def _is_vocab_padding_mismatch(
    param_name: str,
    infer_param_meta,
    train_param_meta,
    hf_config=None,
) -> bool:
    """
    Check if this is an embedding/lm_head parameter with vocab padding mismatch.

    Megatron pads vocab_size to be divisible by (make_vocab_size_divisible_by * tp_size).
    For example: original=129280, padded=130048 (diff=768).

    This mismatch is expected and will be handled at transfer time by slicing.
    """
    # Only check for embedding and lm_head
    if not any(key in param_name for key in ["embed_tokens", "lm_head", "word_embeddings"]):
        return False

    if hf_config is None:
        return False

    vocab_size = getattr(hf_config, "vocab_size", None)
    if vocab_size is None:
        return False

    # Get the vocab dimension (dim 0 for embedding/lm_head)
    train_vocab_dim = train_param_meta.global_shape[0]
    infer_vocab_dim = infer_param_meta.global_shape[0]

    # Check if training has more elements (padded)
    if train_vocab_dim > infer_vocab_dim and train_vocab_dim > vocab_size:
        # Training is padded, inference uses HF vocab_size
        # Verify inference matches HF config
        if infer_vocab_dim == vocab_size:
            logger.info(
                f"[VOCAB_PADDING] {param_name}: Allowing vocab padding mismatch "
                f"(train={train_vocab_dim}, infer={infer_vocab_dim}, hf_config={vocab_size}), "
                f"will slice at transfer time"
            )
            return True

    # Also check the reverse (inference padded more than training - less common)
    if infer_vocab_dim > train_vocab_dim:
        # This shouldn't happen normally, but log it
        logger.warning(
            f"[VOCAB_PADDING] {param_name}: Unexpected - inference vocab ({infer_vocab_dim}) "
            f"> training vocab ({train_vocab_dim})"
        )

    return False


def _is_qkv_param_with_kv_replication(
    param_name: str,
    infer_param_meta,
    train_param_meta,
    hf_config=None,
) -> bool:
    """
    Check if this is a qkv_proj parameter that has KV head replication.

    SGLang replicates KV heads when infer_tp > num_kv_heads, causing:
    - Training global shape: (num_q_heads + num_kv_heads*2) * head_dim
    - Inference global shape: (num_q_heads + infer_tp*2) * head_dim  (with replicated KV)

    This is expected and should not be treated as an error.
    """
    if "qkv_proj" not in param_name:
        return False

    if hf_config is None:
        return False

    num_kv_heads = getattr(hf_config, "num_key_value_heads", None)
    if num_kv_heads is None:
        return False

    infer_tp_size = len(infer_param_meta.replicas[0].shards)

    # KV replication happens when infer_tp > num_kv_heads
    if infer_tp_size <= num_kv_heads:
        return False

    # Verify the shape difference matches expected KV replication
    # Inference has extra (infer_tp - num_kv_heads) * 2 * head_dim elements per shard
    train_global = train_param_meta.global_shape[0]
    infer_global = infer_param_meta.global_shape[0]

    head_dim = getattr(hf_config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(hf_config, "hidden_size", None)
        num_q_heads = getattr(hf_config, "num_attention_heads", None)
        if hidden_size and num_q_heads:
            head_dim = hidden_size // num_q_heads
        else:
            return False

    # Expected difference: (infer_tp - num_kv_heads) * 2 * head_dim
    expected_diff = (infer_tp_size - num_kv_heads) * 2 * head_dim
    actual_diff = infer_global - train_global

    if actual_diff == expected_diff:
        logger.info(
            f"[QKV_VALIDATION] {param_name}: Allowing shape difference due to KV replication. "
            f"train_global={train_global}, infer_global={infer_global}, "
            f"expected_diff={expected_diff}, infer_tp={infer_tp_size}, num_kv_heads={num_kv_heads}"
        )
        return True

    return False


def check_train_infer_params_meta(
    training_params_meta: List,
    infer_parameters_meta: List,
    raise_exception: bool = False,
    hf_config=None,
    skip_numel_check: bool = False,
):
    """
    Check consistency between training and inference parameter metadata.

    Args:
        training_params_meta: Training side parameter metadata
        infer_parameters_meta: Inference side parameter metadata
        raise_exception: Whether to raise exception on error
        hf_config: HuggingFace config for special case handling
        skip_numel_check: If True, skip numel/shape checks (for colocate mode where
                          training and inference sharding are different)

    Returns:
        int: Number of errors found (0 means all checks passed)
    """
    error_count = 0
    error_messages = []

    infer_meta = {param_meta.name: param_meta for param_meta in infer_parameters_meta}
    train_meta = {param_meta.name: param_meta for param_meta in training_params_meta}
    common_params = set(infer_meta.keys()) & set(train_meta.keys())

    logger.info(
        f"[METADATA_CHECK] Checking {len(common_params)} common parameters "
        f"(train: {len(train_meta)}, infer: {len(infer_meta)})"
    )

    # Check for parameter count mismatch
    if len(common_params) != len(infer_meta) or len(common_params) != len(train_meta):
        train_only = set(train_meta.keys()) - common_params
        infer_only = set(infer_meta.keys()) - common_params

        # Separate params into categories
        def categorize_param(name):
            """Categorize param as expert, fused_mla_scale, or other."""
            if ".experts." in name:
                return "expert"
            # FP8 scale params for fused MLA (fused vs split naming mismatch is expected)
            if "weight_scale" in name and any(x in name for x in ["q_a_proj", "kv_a_proj", "fused_qkv_a_proj"]):
                return "mla_scale"
            return "other"

        train_only_experts = {p for p in train_only if categorize_param(p) == "expert"}
        train_only_mla_scales = {p for p in train_only if categorize_param(p) == "mla_scale"}
        train_only_other = train_only - train_only_experts - train_only_mla_scales

        infer_only_experts = {p for p in infer_only if categorize_param(p) == "expert"}
        infer_only_mla_scales = {p for p in infer_only if categorize_param(p) == "mla_scale"}
        infer_only_other = infer_only - infer_only_experts - infer_only_mla_scales

        # Expert params mismatch is expected in colocate mode (EP sharding)
        if train_only_experts or infer_only_experts:
            logger.warning(
                f"[METADATA_CHECK] Expert params mismatch (expected in colocate mode): "
                f"train_only={len(train_only_experts)} experts, infer_only={len(infer_only_experts)} experts"
            )

        # MLA scale params mismatch is expected (fused vs split naming difference)
        if train_only_mla_scales or infer_only_mla_scales:
            logger.warning(
                f"[METADATA_CHECK] MLA scale params mismatch (expected due to fused vs split): "
                f"train_only={len(train_only_mla_scales)}, infer_only={len(infer_only_mla_scales)}"
            )

        # Other params mismatch is a real error
        if train_only_other or infer_only_other:
            error_msg = (
                f"Unexpected params mismatch: "
                f"train_only={train_only_other}, infer_only={infer_only_other}"
            )
            error_count += 1
            error_messages.append(error_msg)
            logger.error(f"[METADATA_CHECK] {error_msg}")
            if raise_exception:
                raise ValueError(
                    f"Inconsistent non-expert/non-scale parameters: "
                    f"train_only={train_only_other}, infer_only={infer_only_other}"
                )

    for param_name in common_params:
        infer_param_meta = infer_meta[param_name]
        train_param_meta = train_meta[param_name]

        # Global skip_numel_check takes precedence (for colocate mode with different sharding)
        if skip_numel_check:
            skip_shape_check = True
            if infer_param_meta.global_numel != train_param_meta.global_numel:
                # Log as warning instead of error when skip_numel_check is True
                logger.warning(
                    f"[METADATA_CHECK] SKIPPED numel check for {param_name}: "
                    f"infer={infer_param_meta.global_numel} vs train={train_param_meta.global_numel} "
                    f"(skip_numel_check=True, colocate mode with different sharding)"
                )
        else:
            # Skip shape/numel validation for qkv_proj with KV head replication
            skip_shape_check = _is_qkv_param_with_kv_replication(
                param_name, infer_param_meta, train_param_meta, hf_config
            )

            # Skip shape/numel validation for embedding/lm_head with vocab padding mismatch
            if not skip_shape_check:
                skip_shape_check = _is_vocab_padding_mismatch(
                    param_name, infer_param_meta, train_param_meta, hf_config
                )

        if not skip_shape_check:
            if infer_param_meta.global_numel != train_param_meta.global_numel:
                error_msg = (
                    f"Inconsistent number of elements for parameter {param_name}: "
                    f"{infer_param_meta.global_numel} != {train_param_meta.global_numel}"
                )
                error_count += 1
                error_messages.append(error_msg)
                if raise_exception:
                    raise ValueError(error_msg)
                else:
                    logger.error(f"[METADATA_CHECK] {error_msg}")
            if infer_param_meta.global_shape != train_param_meta.global_shape:
                error_msg = (
                    f"Inconsistent shape for parameter {param_name}: "
                    f"{infer_param_meta.global_shape} != {train_param_meta.global_shape}"
                )
                error_count += 1
                error_messages.append(error_msg)
                if raise_exception:
                    raise ValueError(error_msg)
                else:
                    logger.error(f"[METADATA_CHECK] {error_msg}")
        if infer_param_meta.dtype != train_param_meta.dtype:
            # Check if this dtype mismatch is acceptable (e.g., router bias float32 <-> bfloat16)
            if _is_acceptable_dtype_mismatch(
                param_name, train_param_meta.dtype, infer_param_meta.dtype
            ):
                # This is a warning, not an error - will convert at transfer time
                pass
            else:
                error_msg = (
                    f"Inconsistent dtype for parameter {param_name}: "
                    f"{infer_param_meta.dtype} != {train_param_meta.dtype}"
                )
                error_count += 1
                error_messages.append(error_msg)
                if raise_exception:
                    raise ValueError(error_msg)
                else:
                    logger.error(f"[METADATA_CHECK] {error_msg}")
        infer_tp_size = len(infer_param_meta.replicas[0].shards)
        train_tp_size = len(train_param_meta.replicas[0].shards)
        # Valid cases:
        # 1. Scatter: infer_tp_size >= train_tp_size and infer_tp_size % train_tp_size == 0
        #    Example: train TP=8, infer TP=64 -> each train shard goes to 8 infer shards
        # 2. Gather: train_tp_size >= infer_tp_size and train_tp_size % infer_tp_size == 0
        #    Example: train TP=8, infer TP=1 -> 8 train shards gathered into 1 infer shard
        #    (Used for shared_experts with moe_a2a_backend=deepep where NO_SHARDING on infer)
        is_scatter_valid = infer_tp_size % train_tp_size == 0
        is_gather_valid = train_tp_size % infer_tp_size == 0
        if not (is_scatter_valid or is_gather_valid):
            error_msg = (
                f"Inference for parameter {param_name} has incompatible tp_size: "
                f"infer {infer_tp_size} train {train_tp_size} (neither evenly divides the other)"
            )
            error_count += 1
            error_messages.append(error_msg)
            if raise_exception:
                raise ValueError(error_msg)
            else:
                logger.error(f"[METADATA_CHECK] {error_msg}")

    # Print summary
    if error_count > 0:
        logger.error(
            f"[METADATA_CHECK] ========== SUMMARY: {error_count} ERRORS FOUND =========="
        )
        # Print first few errors in summary
        for i, msg in enumerate(error_messages[:5]):
            logger.error(f"[METADATA_CHECK]   [{i+1}] {msg}")
        if len(error_messages) > 5:
            logger.error(f"[METADATA_CHECK]   ... and {len(error_messages) - 5} more errors")
        logger.error(
            f"[METADATA_CHECK] =========================================================="
        )
    else:
        logger.info(f"[METADATA_CHECK] All {len(common_params)} parameters validated successfully")

    return error_count


def pretty_bytes(size_bytes):
    if size_bytes == 0:
        return "0B"
    units = ["B", "KB", "MB", "GB", "TB"]
    index = int(math.floor(math.log(size_bytes, 1024)))
    power = math.pow(1024, index)
    converted_size = round(size_bytes / power, 2)
    return f"{converted_size} {units[index]}"


def stripped_env_vars():
    vars = {}
    for k, v in os.environ.items():
        if "secret" not in k.lower():
            vars[k] = v
    vars.pop("LS_COLORS", None)
    return vars


class AttrDict(dict):
    """Dictionary that allows attribute-style access."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure that attributes refer to dictionary keys
        self.__dict__ = self


def simple_hf_config(hg_config):
    config = hg_config.to_dict()
    final_config = {}
    for k, v in config.items():
        if "sglang" in str(v):
            logger.warning(f"Skipping sglang config {k}: {v}")
            continue
        final_config[k] = v
    return AttrDict(**config)
