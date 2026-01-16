"""
Unit tests for DeepSeek-V3 model plugin.

These tests verify:
1. Parameter name conversion from Megatron to HuggingFace format
2. MLA (Multi-head Latent Attention) parameter handling
3. Sharding strategy for MLA parameters
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from awex.sharding.rank_info import RankInfo
from awex.sharding.param_sharding import ShardingType


# ----------------------
# HELPER FUNCTIONS
# ----------------------


def make_rank_info(
    tp_rank=0,
    tp_size=1,
    pp_rank=0,
    pp_size=1,
    dp_size=1,
    dp_rank=0,
    ep_rank=0,
    ep_size=1,
    ep_tp_rank=0,
    ep_tp_size=1,
    attn_tp_rank=0,
    attn_tp_size=1,
    attn_dp_rank=0,
    world_size=1,
    global_rank=0,
    local_rank=0,
    engine_rank=0,
    is_infer=False,
):
    return RankInfo(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=pp_rank,
        pp_size=pp_size,
        dp_size=dp_size,
        dp_rank=dp_rank,
        ep_rank=ep_rank,
        ep_size=ep_size,
        ep_tp_rank=ep_tp_rank,
        ep_tp_size=ep_tp_size,
        attn_tp_rank=attn_tp_rank,
        attn_tp_size=attn_tp_size,
        attn_dp_rank=attn_dp_rank,
        world_size=world_size,
        global_rank=global_rank,
        local_rank=local_rank,
        engine_rank=engine_rank,
        is_infer=is_infer,
    )


def create_mock_hf_config():
    """Create a mock HuggingFace config for DeepSeek-V3."""
    config = MagicMock()
    config.architectures = ["DeepseekV3ForCausalLM"]
    config.num_hidden_layers = 61
    config.hidden_size = 7168
    config.num_attention_heads = 128
    config.num_key_value_heads = 128  # num_query_groups
    config.n_routed_experts = 256  # DeepSeek-V3 uses n_routed_experts, not num_experts
    config.q_lora_rank = 1536
    config.kv_lora_rank = 512
    config.kv_channels = 128
    config.quantization_config = None  # Explicitly set to None for no FP8
    return config


# ----------------------
# TEST: Model Registration
# ----------------------


def test_deepseek_v3_model_registered():
    """Verify DeepSeek-V3 model is registered in AWEX."""
    from awex.models.registry import ModelRegistry

    # Check all variants are registered
    for model_name in [
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "DeepseekV3ForCausalLMNextN",
    ]:
        config = ModelRegistry.get_model_config(model_name)
        assert config is not None, f"Model {model_name} should be registered"
        # Registry returns either ModelConfig or dict
        if hasattr(config, 'sharding_strategy'):
            assert config.sharding_strategy is not None
        else:
            # It's a dict from CONFIG list
            assert "sharding_strategy" in config or hasattr(config, "__getitem__")


# ----------------------
# TEST: Sharding Strategy
# ----------------------


def test_mla_sharding_strategy_attention_params():
    """Test sharding strategy for MLA attention parameters."""
    from awex.models.deepseek_v3 import DeepSeekV3ShardingStrategy

    rank_info = make_rank_info(tp_rank=0, tp_size=8)
    strategy = DeepSeekV3ShardingStrategy(
        engine_name="mcore",
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=8,
        tp_size=8,
        ep_size=32,
        ep_tp_size=1,
        rank_info=rank_info,
    )

    # MLA Q projections should be TP sharded
    sharding_type, dim, num_shards = strategy.get_attention_sharding_strategy(
        "model.layers.0.self_attn.q_a_proj.weight"
    )
    assert sharding_type == ShardingType.TP_SHARDING
    assert num_shards == 8

    # MLA KV projections should be TP sharded
    sharding_type, dim, num_shards = strategy.get_attention_sharding_strategy(
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"
    )
    assert sharding_type == ShardingType.TP_SHARDING
    assert num_shards == 8


def test_mla_sharding_strategy_layernorm_not_sharded():
    """Test that LayerNorm parameters are not sharded."""
    from awex.models.deepseek_v3 import DeepSeekV3ShardingStrategy

    rank_info = make_rank_info(tp_rank=0, tp_size=8)
    strategy = DeepSeekV3ShardingStrategy(
        engine_name="mcore",
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=8,
        tp_size=8,
        ep_size=32,
        ep_tp_size=1,
        rank_info=rank_info,
    )

    # LayerNorm should NOT be sharded
    sharding_type, dim, num_shards = strategy.get_attention_sharding_strategy(
        "model.layers.0.self_attn.q_a_layernorm.weight"
    )
    assert sharding_type == ShardingType.NO_SHARDING
    assert num_shards == 1


def test_expert_bias_not_sharded():
    """Test that expert bias (e_score_correction_bias) is not sharded."""
    from awex.models.deepseek_v3 import DeepSeekV3ShardingStrategy

    rank_info = make_rank_info(tp_rank=0, tp_size=8)
    strategy = DeepSeekV3ShardingStrategy(
        engine_name="mcore",
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=8,
        tp_size=8,
        ep_size=32,
        ep_tp_size=1,
        rank_info=rank_info,
    )

    sharding_type, dim, num_shards = strategy.get_sharding_strategy(
        "model.layers.3.mlp.gate.e_score_correction_bias"
    )
    assert sharding_type == ShardingType.NO_SHARDING


# ----------------------
# TEST: Converter - MLA Parameters
# ----------------------


def test_mcore_converter_mla_q_down_proj():
    """Test Megatron to HF conversion for Q down projection."""
    from awex.models.deepseek_v3 import McoreToHFWeightConverterDeepSeekV3

    hf_config = create_mock_hf_config()
    rank_info = make_rank_info(tp_rank=0, tp_size=8, pp_rank=0, pp_size=4)
    infer_conf = {"infer_atten_tp_size": 8, "router_dtype": "bf16"}

    converter = McoreToHFWeightConverterDeepSeekV3(hf_config, rank_info, infer_conf)

    # Create dummy parameter
    param = torch.randn(1536, 7168)  # q_lora_rank x hidden_size

    # Test conversion
    result = converter._convert_mla_attention_param(
        "self_attention.linear_q_down_proj.weight", param, "0"
    )

    assert len(result) == 1
    name, tensor = result[0]
    assert name == "self_attn.q_a_proj.weight"
    assert tensor.shape == param.shape


def test_mcore_converter_mla_kv_down_proj():
    """Test Megatron to HF conversion for KV down projection."""
    from awex.models.deepseek_v3 import McoreToHFWeightConverterDeepSeekV3

    hf_config = create_mock_hf_config()
    rank_info = make_rank_info(tp_rank=0, tp_size=8, pp_rank=0, pp_size=4)
    infer_conf = {"infer_atten_tp_size": 8, "router_dtype": "bf16"}

    converter = McoreToHFWeightConverterDeepSeekV3(hf_config, rank_info, infer_conf)

    # Create dummy parameter
    param = torch.randn(512, 7168)  # kv_lora_rank x hidden_size

    result = converter._convert_mla_attention_param(
        "self_attention.linear_kv_down_proj.weight", param, "0"
    )

    assert len(result) == 1
    name, tensor = result[0]
    assert name == "self_attn.kv_a_proj_with_mqa.weight"
    assert tensor.shape == param.shape


def test_mcore_converter_mla_layernorm():
    """Test Megatron to HF conversion for MLA LayerNorm."""
    from awex.models.deepseek_v3 import McoreToHFWeightConverterDeepSeekV3

    hf_config = create_mock_hf_config()
    rank_info = make_rank_info(tp_rank=0, tp_size=8, pp_rank=0, pp_size=4)
    infer_conf = {"infer_atten_tp_size": 8, "router_dtype": "bf16"}

    converter = McoreToHFWeightConverterDeepSeekV3(hf_config, rank_info, infer_conf)

    # Q LayerNorm
    param = torch.randn(1536)
    result = converter._convert_mla_attention_param(
        "self_attention.linear_q_up_proj.layer_norm_weight", param, "0"
    )
    assert len(result) == 1
    assert result[0][0] == "self_attn.q_a_layernorm.weight"

    # KV LayerNorm
    param = torch.randn(512)
    result = converter._convert_mla_attention_param(
        "self_attention.linear_kv_up_proj.layer_norm_weight", param, "0"
    )
    assert len(result) == 1
    assert result[0][0] == "self_attn.kv_a_layernorm.weight"


def test_mcore_converter_output_proj():
    """Test Megatron to HF conversion for output projection."""
    from awex.models.deepseek_v3 import McoreToHFWeightConverterDeepSeekV3

    hf_config = create_mock_hf_config()
    rank_info = make_rank_info(tp_rank=0, tp_size=8, pp_rank=0, pp_size=4)
    infer_conf = {"infer_atten_tp_size": 8, "router_dtype": "bf16"}

    converter = McoreToHFWeightConverterDeepSeekV3(hf_config, rank_info, infer_conf)

    param = torch.randn(7168, 7168)
    result = converter._convert_mla_attention_param(
        "self_attention.linear_proj.weight", param, "0"
    )

    assert len(result) == 1
    assert result[0][0] == "self_attn.o_proj.weight"


# ----------------------
# TEST: Converter - MoE Parameters
# ----------------------


def test_mcore_converter_expert_bias():
    """Test Megatron to HF conversion for expert bias."""
    from awex.models.deepseek_v3 import McoreToHFWeightConverterDeepSeekV3

    hf_config = create_mock_hf_config()
    rank_info = make_rank_info(tp_rank=0, tp_size=8, pp_rank=0, pp_size=4, ep_rank=0, ep_size=32)
    infer_conf = {"infer_atten_tp_size": 8, "router_dtype": "bf16"}

    converter = McoreToHFWeightConverterDeepSeekV3(hf_config, rank_info, infer_conf)

    param = torch.randn(256)  # num_experts
    name, tensor = converter._convert_expert_bias_param(
        "mlp.router.expert_bias", param, "3"
    )

    assert name == "mlp.gate.e_score_correction_bias"
    assert tensor.dtype == torch.bfloat16


# ----------------------
# TEST: SGLang Converter
# ----------------------


def test_sglang_converter_no_fusion():
    """Test SGLang converter doesn't fuse QKV or gate_up."""
    from awex.models.deepseek_v3 import SGlangToHFWeightConverterDeepSeekV3

    hf_config = create_mock_hf_config()
    rank_info = make_rank_info(tp_rank=0, tp_size=8)
    infer_config = MagicMock()
    infer_config.tp_size = 8
    infer_config.ep_size = 64

    converter = SGlangToHFWeightConverterDeepSeekV3(hf_config, infer_config, rank_info)

    assert converter._fuse_qkv("any_param") is False
    assert converter._fuse_gate_up_proj("any_param") is False


def test_sglang_converter_mla_fusion_split():
    """Test SGLang converter splits fused_qkv_a_proj_with_mqa into views.

    This is critical for P2P transfer: SGLang uses fused_qkv_a_proj_with_mqa internally,
    but TransferPlan matches by HF names (q_a_proj, kv_a_proj_with_mqa).
    The converter must split the fused tensor into views so writes update the original.
    """
    from awex.models.deepseek_v3 import SGlangToHFWeightConverterDeepSeekV3

    hf_config = create_mock_hf_config()
    hf_config.q_lora_rank = 1536
    hf_config.kv_lora_rank = 512
    hf_config.qk_rope_head_dim = 64

    rank_info = make_rank_info(tp_rank=0, tp_size=8)
    infer_config = MagicMock()
    infer_config.tp_size = 8
    infer_config.ep_size = 64

    converter = SGlangToHFWeightConverterDeepSeekV3(hf_config, infer_config, rank_info)

    # Create a fused tensor: shape (q_lora_rank + kv_lora_rank + qk_rope_head_dim, hidden_size)
    # = (1536 + 512 + 64, 7168) = (2112, 7168)
    hidden_size = 7168
    fused_size = 1536 + 512 + 64  # 2112
    fused_param = torch.randn(fused_size, hidden_size)

    # Convert the fused parameter
    result = converter._convert_attention_param(
        "self_attn.fused_qkv_a_proj_with_mqa.weight", fused_param, "0"
    )

    # Should split into 2 parameters
    assert len(result) == 2

    # Check names and shapes
    names = {r[0] for r in result}
    assert "self_attn.q_a_proj.weight" in names
    assert "self_attn.kv_a_proj_with_mqa.weight" in names

    for name, tensor in result:
        if "q_a_proj" in name:
            assert tensor.shape == (1536, hidden_size), f"q_a_proj shape mismatch: {tensor.shape}"
        elif "kv_a_proj_with_mqa" in name:
            assert tensor.shape == (576, hidden_size), f"kv_a_proj shape mismatch: {tensor.shape}"

    # CRITICAL: Verify views share storage with original tensor
    q_a_tensor = next(t for n, t in result if "q_a_proj" in n)
    kv_a_tensor = next(t for n, t in result if "kv_a_proj" in n)

    assert q_a_tensor.data_ptr() == fused_param.data_ptr(), "q_a_proj should share storage"
    # kv_a should point to offset in fused tensor
    expected_kv_offset = fused_param.data_ptr() + 1536 * hidden_size * fused_param.element_size()
    assert kv_a_tensor.data_ptr() == expected_kv_offset, "kv_a_proj should share storage at offset"


def test_sglang_converter_fp8_scale_split():
    """Test SGLang converter splits FP8 scale parameters for fused_qkv_a_proj_with_mqa.

    FP8 quantization adds auxiliary parameters like weight_scale, weight_scale_inv.
    For fused_qkv_a_proj_with_mqa, these SHOULD be split to match training-side parameters
    which have separate q_a_proj and kv_a_proj_with_mqa scale parameters.

    FP8 scale shape: (ceil(2112/128), X) = (17, 56) for fused tensor
    Split into:
    - q_a_proj scale: (ceil(1536/128), 56) = (12, 56)
    - kv_a_proj_with_mqa scale: (ceil(576/128), 56) = (5, 56)
    """
    from awex.models.deepseek_v3 import SGlangToHFWeightConverterDeepSeekV3

    hf_config = create_mock_hf_config()
    hf_config.q_lora_rank = 1536
    hf_config.kv_lora_rank = 512
    hf_config.qk_rope_head_dim = 64

    rank_info = make_rank_info(tp_rank=0, tp_size=8)
    infer_config = MagicMock()
    infer_config.tp_size = 8
    infer_config.ep_size = 64

    converter = SGlangToHFWeightConverterDeepSeekV3(hf_config, infer_config, rank_info)

    # Simulate FP8 block quant scale: shape (17, X)
    # where 17 = ceil(1536/128) + ceil(576/128) = 12 + 5 = 17
    scale_param = torch.randn(17, 56)

    # Should split into q_a_proj and kv_a_proj_with_mqa scales
    result = converter._convert_attention_param(
        "self_attn.fused_qkv_a_proj_with_mqa.weight_scale", scale_param, "0"
    )

    # Should return 2 split parameters
    assert len(result) == 2, f"Expected 2 split params, got {len(result)}"

    names = {r[0] for r in result}
    assert "self_attn.q_a_proj.weight_scale" in names
    assert "self_attn.kv_a_proj_with_mqa.weight_scale" in names

    for name, tensor in result:
        if "q_a_proj" in name:
            assert tensor.shape == (12, 56), f"q_a_proj scale shape mismatch: {tensor.shape}"
        elif "kv_a_proj_with_mqa" in name:
            assert tensor.shape == (5, 56), f"kv_a_proj scale shape mismatch: {tensor.shape}"


# ----------------------
# TEST: MLA Sharding Dimension
# ----------------------


def test_get_mla_sharding_dim():
    """Test MLA-specific sharding dimension lookup."""
    from awex.models.deepseek_v3 import get_mla_sharding_dim

    # Q projections - dim 0
    assert get_mla_sharding_dim("model.layers.0.self_attn.q_a_proj.weight") == 0
    assert get_mla_sharding_dim("model.layers.0.self_attn.q_b_proj.weight") == 0

    # KV projections - dim 0
    assert get_mla_sharding_dim("model.layers.0.self_attn.kv_a_proj_with_mqa.weight") == 0
    assert get_mla_sharding_dim("model.layers.0.self_attn.kv_b_proj.weight") == 0

    # Output projection - dim 1
    assert get_mla_sharding_dim("model.layers.0.self_attn.o_proj.weight") == 1

    # LayerNorm - dim 0 (but not sharded anyway)
    assert get_mla_sharding_dim("model.layers.0.self_attn.q_a_layernorm.weight") == 0


# ----------------------
# TEST: FP8 Quantization
# ----------------------


def create_mock_hf_config_with_fp8():
    """Create a mock HuggingFace config with FP8 quantization enabled."""
    config = MagicMock()
    config.architectures = ["DeepseekV3ForCausalLM"]
    config.num_hidden_layers = 61
    config.hidden_size = 7168
    config.num_attention_heads = 128
    config.num_key_value_heads = 128
    config.n_routed_experts = 256  # DeepSeek-V3 uses n_routed_experts, not num_experts
    config.q_lora_rank = 1536
    config.kv_lora_rank = 512
    config.kv_channels = 128
    # Block-wise FP8 quantization config (same as Slime/SGLang)
    config.quantization_config = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    return config


def test_fp8_should_quantize():
    """Test _should_fp8_quantize correctly identifies parameters to quantize."""
    from awex.models.deepseek_v3 import McoreToHFWeightConverterDeepSeekV3

    # With FP8 enabled
    hf_config = create_mock_hf_config_with_fp8()
    rank_info = make_rank_info(tp_rank=0, tp_size=8)
    infer_conf = {"infer_atten_tp_size": 8, "router_dtype": "bf16"}
    converter = McoreToHFWeightConverterDeepSeekV3(hf_config, rank_info, infer_conf)

    # Should quantize
    assert converter._should_fp8_quantize("model.layers.0.self_attn.q_a_proj.weight")
    assert converter._should_fp8_quantize("model.layers.0.self_attn.o_proj.weight")
    assert converter._should_fp8_quantize("model.layers.0.mlp.experts.0.gate_proj.weight")
    assert converter._should_fp8_quantize("model.layers.0.mlp.experts.0.down_proj.weight")
    assert converter._should_fp8_quantize("model.layers.0.mlp.shared_experts.gate_proj.weight")

    # Should NOT quantize
    assert not converter._should_fp8_quantize("model.layers.0.input_layernorm.weight")
    assert not converter._should_fp8_quantize("model.layers.0.self_attn.q_a_layernorm.weight")
    assert not converter._should_fp8_quantize("model.embed_tokens.weight")
    assert not converter._should_fp8_quantize("model.norm.weight")
    assert not converter._should_fp8_quantize("model.layers.0.mlp.gate.weight")  # router weight


def test_fp8_should_quantize_disabled():
    """Test _should_fp8_quantize returns False when FP8 is disabled."""
    from awex.models.deepseek_v3 import McoreToHFWeightConverterDeepSeekV3

    # Without FP8
    hf_config = create_mock_hf_config()  # No quantization_config
    rank_info = make_rank_info(tp_rank=0, tp_size=8)
    infer_conf = {"infer_atten_tp_size": 8, "router_dtype": "bf16"}
    converter = McoreToHFWeightConverterDeepSeekV3(hf_config, rank_info, infer_conf)

    # All should return False when FP8 is disabled
    assert not converter._should_fp8_quantize("model.layers.0.self_attn.q_a_proj.weight")
    assert not converter._should_fp8_quantize("model.layers.0.mlp.experts.0.gate_proj.weight")


def test_fp8_use_ue8m0():
    """
    Test UE8M0 decision via should_use_ue8m0 from weights_converter.

    Note: The UE8M0 decision is now based on SGLang's runtime config
    (should_deepgemm_weight_requant_ue8m0), NOT hardcoded by parameter names.
    This matches Slime's behavior exactly.
    """
    from awex.converter.weights_converter import should_use_ue8m0

    # Test with block size - the actual UE8M0 decision depends on SGLang's runtime config
    # In most cases, should_use_ue8m0 will return what SGLang decides
    weight_block_size = [128, 128]
    result = should_use_ue8m0(weight_block_size)
    # Result depends on SGLang's configuration, just verify it returns a boolean
    assert isinstance(result, bool)

    # Test without block size - should always return False
    assert not should_use_ue8m0(None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FP8 test")
def test_fp8_quantization_output():
    """Test FP8 quantization produces correct output format."""
    from awex.models.deepseek_v3 import McoreToHFWeightConverterDeepSeekV3

    hf_config = create_mock_hf_config_with_fp8()
    rank_info = make_rank_info(tp_rank=0, tp_size=8)
    infer_conf = {"infer_atten_tp_size": 8, "router_dtype": "bf16"}
    converter = McoreToHFWeightConverterDeepSeekV3(hf_config, rank_info, infer_conf)

    # Create test input
    test_params = [
        ("model.layers.0.self_attn.q_a_proj.weight", torch.randn(1536, 7168, device="cuda")),
    ]

    result = converter._apply_fp8_quantization(test_params)

    # Should have 2 outputs: weight + scale
    assert len(result) == 2

    # Check weight
    weight_name, weight = result[0]
    assert weight_name == "model.layers.0.self_attn.q_a_proj.weight"
    assert weight.dtype == torch.float8_e4m3fn
    assert weight.shape == (1536, 7168)

    # Check scale
    scale_name, scale = result[1]
    assert scale_name == "model.layers.0.self_attn.q_a_proj.weight_scale_inv"
    # Scale shape: (ceil(1536/128), ceil(7168/128)) = (12, 56)
    assert scale.shape == (12, 56)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FP8 test")
def test_fp8_quantization_layernorm_passthrough():
    """Test LayerNorm parameters pass through FP8 quantization unchanged."""
    from awex.models.deepseek_v3 import McoreToHFWeightConverterDeepSeekV3

    hf_config = create_mock_hf_config_with_fp8()
    rank_info = make_rank_info(tp_rank=0, tp_size=8)
    infer_conf = {"infer_atten_tp_size": 8, "router_dtype": "bf16"}
    converter = McoreToHFWeightConverterDeepSeekV3(hf_config, rank_info, infer_conf)

    # Create LayerNorm input (should NOT be quantized)
    original_param = torch.randn(7168, device="cuda")
    test_params = [
        ("model.layers.0.input_layernorm.weight", original_param),
    ]

    result = converter._apply_fp8_quantization(test_params)

    # Should have 1 output (unchanged)
    assert len(result) == 1
    name, tensor = result[0]
    assert name == "model.layers.0.input_layernorm.weight"
    assert tensor is original_param  # Same object


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FP8 test")
def test_fp8_quantization_scale_range():
    """Test FP8 scale values are in reasonable range."""
    from awex.converter.weights_converter import per_block_cast_to_fp8

    # Create test tensor with known range
    test_tensor = torch.randn(256, 256, device="cuda") * 10  # ~[-30, 30]
    max_val = test_tensor.abs().max().item()

    qw, scale = per_block_cast_to_fp8(test_tensor, scale_ue8m0=False)

    # FP8 E4M3 max is 448, so scale should be approximately max_val / 448
    expected_scale_order = max_val / 448.0

    # Scale should be in reasonable range
    assert scale.min().item() > 0
    assert scale.max().item() < max_val  # Scale should be less than max value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FP8 test")
def test_fp8_quantization_ue8m0_format():
    """Test UE8M0 format produces power-of-2 scales."""
    from awex.converter.weights_converter import per_block_cast_to_fp8

    test_tensor = torch.randn(256, 256, device="cuda")

    qw, scale = per_block_cast_to_fp8(test_tensor, scale_ue8m0=True)

    # UE8M0 scales should be powers of 2
    # log2 of power-of-2 is integer
    log2_scales = torch.log2(scale)
    is_power_of_2 = torch.allclose(log2_scales, log2_scales.round(), atol=1e-5)
    assert is_power_of_2, "UE8M0 scales should be powers of 2"


# ----------------------
# TEST: DP Attention Mode Sharding
# ----------------------


def test_dp_attention_sharding_shared_experts():
    """Test that shared_experts use attn_tp_size in DP attention mode WITHOUT deepep.

    In DeepSeek-V3 with DP attention and moe_a2a_backend="none":
    - tp_size=64, attn_tp_size=8, ep_size=64
    - shared_experts should be sharded like dense layers (attn_tp_size=8)
    - Regular experts use EP sharding (ep_size=64)

    Note: When using deepep/mooncake backend, shared_experts are NOT sharded.
    See test_deepep_sharding_shared_experts for that case.
    """
    from awex.models.deepseek_v3 import DeepSeekV3ShardingStrategy

    rank_info = make_rank_info(
        tp_size=64,
        attn_tp_size=8,
        ep_size=64,
        ep_tp_size=1,
    )

    strategy = DeepSeekV3ShardingStrategy(
        engine_name="sglang",
        enable_dp_attention=True,
        enable_dp_lm_head=True,
        moe_dense_tp_size=1,
        tp_size=64,
        ep_size=64,
        ep_tp_size=1,
        rank_info=rank_info,
        moe_a2a_backend="none",  # No deepep, so shared_experts are TP-sharded
    )

    # shared_experts should use DP_TP_SHARDING with attn_tp_size=8
    for param_name in [
        "model.layers.10.mlp.shared_experts.gate_proj.weight",
        "model.layers.10.mlp.shared_experts.up_proj.weight",
        "model.layers.10.mlp.shared_experts.down_proj.weight",
    ]:
        sharding_type, sharding_dim, num_shards = strategy.get_sharding_strategy(param_name)
        assert sharding_type == ShardingType.DP_TP_SHARDING, \
            f"{param_name}: expected DP_TP_SHARDING, got {sharding_type}"
        assert num_shards == 8, \
            f"{param_name}: expected num_shards=8 (attn_tp_size), got {num_shards}"


def test_deepep_sharding_shared_experts():
    """Test that shared_experts are NOT sharded when using deepep/mooncake backend.

    In SGLang with deepep/mooncake backend, shared_experts use tp_size=1 (not TP-sharded).
    This is because SGLang creates shared_experts with dict(tp_rank=0, tp_size=1) when
    using deepep/mooncake (see deepseek_v2.py:692-706 in SGLang).

    This is the typical colocate mode configuration:
    - tp_size=64, attn_tp_size=8, ep_size=64
    - moe_a2a_backend="deepep"
    - shared_experts: NO_SHARDING (tp_size=1, fully replicated)
    """
    from awex.models.deepseek_v3 import DeepSeekV3ShardingStrategy

    rank_info = make_rank_info(
        tp_size=64,
        attn_tp_size=8,
        ep_size=64,
        ep_tp_size=1,
    )

    strategy = DeepSeekV3ShardingStrategy(
        engine_name="sglang",
        enable_dp_attention=True,
        enable_dp_lm_head=True,
        moe_dense_tp_size=1,
        tp_size=64,
        ep_size=64,
        ep_tp_size=1,
        rank_info=rank_info,
        moe_a2a_backend="deepep",  # With deepep, shared_experts are NOT TP-sharded
    )

    # shared_experts should use NO_SHARDING (fully replicated)
    for param_name in [
        "model.layers.10.mlp.shared_experts.gate_proj.weight",
        "model.layers.10.mlp.shared_experts.up_proj.weight",
        "model.layers.10.mlp.shared_experts.down_proj.weight",
    ]:
        sharding_type, sharding_dim, num_shards = strategy.get_sharding_strategy(param_name)
        assert sharding_type == ShardingType.NO_SHARDING, \
            f"{param_name}: expected NO_SHARDING with deepep, got {sharding_type}"
        assert num_shards == 1, \
            f"{param_name}: expected num_shards=1 (no sharding), got {num_shards}"


def test_dp_attention_sharding_embedding_lm_head():
    """Test that embed_tokens and lm_head use attn_tp_size in DP attention mode.

    embed_tokens doesn't contain 'embedding' in its name, so it needs explicit handling.
    """
    from awex.models.deepseek_v3 import DeepSeekV3ShardingStrategy

    rank_info = make_rank_info(
        tp_size=64,
        attn_tp_size=8,
        ep_size=64,
        ep_tp_size=1,
    )

    strategy = DeepSeekV3ShardingStrategy(
        engine_name="sglang",
        enable_dp_attention=True,
        enable_dp_lm_head=True,
        moe_dense_tp_size=1,
        tp_size=64,
        ep_size=64,
        ep_tp_size=1,
        rank_info=rank_info,
    )

    # embed_tokens: NO_SHARDING when enable_dp_attention=True
    # (SGLang's VocabParallelEmbedding uses enable_tp=not is_dp_attention_enabled())
    sharding_type, sharding_dim, num_shards = strategy.get_sharding_strategy("model.embed_tokens.weight")
    assert sharding_type == ShardingType.NO_SHARDING, \
        f"embed_tokens: expected NO_SHARDING, got {sharding_type}"
    assert num_shards == 1, \
        f"embed_tokens: expected num_shards=1, got {num_shards}"

    # lm_head: DP_TP_SHARDING with attn_tp_size=8 when enable_dp_lm_head=True
    # (SGLang's ParallelLMHead uses use_attn_tp_group=enable_dp_lm_head)
    sharding_type, sharding_dim, num_shards = strategy.get_sharding_strategy("lm_head.weight")
    assert sharding_type == ShardingType.DP_TP_SHARDING, \
        f"lm_head: expected DP_TP_SHARDING, got {sharding_type}"
    assert num_shards == 8, \
        f"lm_head: expected num_shards=8 (attn_tp_size), got {num_shards}"


def test_dp_attention_sharding_regular_experts():
    """Test that regular MoE experts still use EP_SHARDING in DP attention mode."""
    from awex.models.deepseek_v3 import DeepSeekV3ShardingStrategy

    rank_info = make_rank_info(
        tp_size=64,
        attn_tp_size=8,
        ep_size=64,
        ep_tp_size=1,
    )

    strategy = DeepSeekV3ShardingStrategy(
        engine_name="sglang",
        enable_dp_attention=True,
        enable_dp_lm_head=True,
        moe_dense_tp_size=1,
        tp_size=64,
        ep_size=64,
        ep_tp_size=1,
        rank_info=rank_info,
    )

    # Regular experts should use EP_SHARDING with ep_size=64
    for param_name in [
        "model.layers.10.mlp.experts.w13_weight",
        "model.layers.10.mlp.experts.w2_weight",
    ]:
        sharding_type, sharding_dim, num_shards = strategy.get_sharding_strategy(param_name)
        assert sharding_type == ShardingType.EP_SHARDING, \
            f"{param_name}: expected EP_SHARDING, got {sharding_type}"
        assert num_shards == 64, \
            f"{param_name}: expected num_shards=64 (ep_size), got {num_shards}"


def test_mcore_sharding_shared_experts():
    """Test that shared_experts use TP_SHARDING with tp_size in training (mcore) mode.

    In training (mcore) mode:
    - enable_dp_attention=False (always)
    - shared_experts should use TP_SHARDING with tp_size (e.g., 8)
    - NOT EP_SHARDING like regular MoE experts
    """
    from awex.models.deepseek_v3 import DeepSeekV3ShardingStrategy

    # Training setup: TP=8, PP=4, CP=4, EP=32
    rank_info = make_rank_info(
        tp_size=8,
        attn_tp_size=8,  # Same as tp_size for training
        ep_size=32,
        ep_tp_size=1,
    )

    strategy = DeepSeekV3ShardingStrategy(
        engine_name="mcore",
        enable_dp_attention=False,  # Always False for training
        enable_dp_lm_head=False,
        moe_dense_tp_size=8,
        tp_size=8,
        ep_size=32,
        ep_tp_size=1,
        rank_info=rank_info,
    )

    # shared_experts should use TP_SHARDING with tp_size=8
    for param_name in [
        "model.layers.10.mlp.shared_experts.gate_proj.weight",
        "model.layers.10.mlp.shared_experts.up_proj.weight",
        "model.layers.10.mlp.shared_experts.down_proj.weight",
    ]:
        sharding_type, sharding_dim, num_shards = strategy.get_sharding_strategy(param_name)
        assert sharding_type == ShardingType.TP_SHARDING, \
            f"{param_name}: expected TP_SHARDING, got {sharding_type}"
        assert num_shards == 8, \
            f"{param_name}: expected num_shards=8 (tp_size), got {num_shards}"


def test_dense_layer_mlp_sharding_with_deepep():
    """Test that Dense layer MLP (first 3 layers) use NO_SHARDING when moe_dense_tp_size=1.

    In SGLang with moe_dense_tp_size=1 (enable_moe_dense_fully_dp=True):
    - Dense layer MLP is fully replicated (tp_size=1)
    - This applies to layers 0, 1, 2 (first_k_dense_replace=3)

    This is the typical colocate mode configuration:
    - tp_size=64, attn_tp_size=8, ep_size=64
    - moe_dense_tp_size=1
    - Dense layer MLP: NO_SHARDING (fully replicated)
    """
    from awex.models.deepseek_v3 import DeepSeekV3ShardingStrategy

    rank_info = make_rank_info(
        tp_size=64,
        attn_tp_size=8,
        ep_size=64,
        ep_tp_size=1,
    )

    strategy = DeepSeekV3ShardingStrategy(
        engine_name="sglang",
        enable_dp_attention=True,
        enable_dp_lm_head=True,
        moe_dense_tp_size=1,  # Dense layers are fully replicated
        tp_size=64,
        ep_size=64,
        ep_tp_size=1,
        rank_info=rank_info,
        moe_a2a_backend="deepep",
    )

    # Dense layer MLP (layers 0, 1, 2) should use NO_SHARDING
    for layer_id in [0, 1, 2]:
        for param_name in [
            f"model.layers.{layer_id}.mlp.gate_proj.weight",
            f"model.layers.{layer_id}.mlp.up_proj.weight",
            f"model.layers.{layer_id}.mlp.down_proj.weight",
            f"model.layers.{layer_id}.mlp.gate_proj.weight_scale_inv",  # FP8 scale too
            f"model.layers.{layer_id}.mlp.down_proj.weight_scale_inv",
        ]:
            sharding_type, sharding_dim, num_shards = strategy.get_sharding_strategy(param_name)
            assert sharding_type == ShardingType.NO_SHARDING, \
                f"{param_name}: expected NO_SHARDING for dense layer MLP with moe_dense_tp_size=1, got {sharding_type}"
            assert num_shards == 1, \
                f"{param_name}: expected num_shards=1, got {num_shards}"


def test_moe_layer_mlp_not_affected_by_dense_sharding():
    """Test that MoE layer MLP (layer >= 3) is NOT affected by moe_dense_tp_size.

    MoE layers (layer_id >= first_k_dense_replace=3) should use their own sharding
    strategy, not the dense layer strategy. Even with moe_dense_tp_size=1, MoE layers
    should use EP sharding for experts.
    """
    from awex.models.deepseek_v3 import DeepSeekV3ShardingStrategy

    rank_info = make_rank_info(
        tp_size=64,
        attn_tp_size=8,
        ep_size=64,
        ep_tp_size=1,
    )

    strategy = DeepSeekV3ShardingStrategy(
        engine_name="sglang",
        enable_dp_attention=True,
        enable_dp_lm_head=True,
        moe_dense_tp_size=1,
        tp_size=64,
        ep_size=64,
        ep_tp_size=1,
        rank_info=rank_info,
        moe_a2a_backend="deepep",
    )

    # MoE layer experts (layer 10) should use EP_SHARDING, not be affected by moe_dense_tp_size
    for param_name in [
        "model.layers.10.mlp.experts.w13_weight",
        "model.layers.10.mlp.experts.w2_weight",
    ]:
        sharding_type, sharding_dim, num_shards = strategy.get_sharding_strategy(param_name)
        assert sharding_type == ShardingType.EP_SHARDING, \
            f"{param_name}: expected EP_SHARDING for MoE experts, got {sharding_type}"
        assert num_shards == 64, \
            f"{param_name}: expected num_shards=64 (ep_size), got {num_shards}"

    # MoE layer shared_experts (layer 10) should use NO_SHARDING with deepep
    for param_name in [
        "model.layers.10.mlp.shared_experts.gate_proj.weight",
        "model.layers.10.mlp.shared_experts.up_proj.weight",
    ]:
        sharding_type, sharding_dim, num_shards = strategy.get_sharding_strategy(param_name)
        assert sharding_type == ShardingType.NO_SHARDING, \
            f"{param_name}: expected NO_SHARDING for shared_experts with deepep, got {sharding_type}"


def test_mcore_dense_layer_uses_tp_sharding():
    """Test that Dense layer MLP uses TP_SHARDING in training (mcore) mode.

    In training (mcore) mode, moe_dense_tp_size equals tp_size, so dense layers
    use regular TP sharding.
    """
    from awex.models.deepseek_v3 import DeepSeekV3ShardingStrategy

    rank_info = make_rank_info(
        tp_size=8,
        attn_tp_size=8,
        ep_size=32,
        ep_tp_size=1,
    )

    strategy = DeepSeekV3ShardingStrategy(
        engine_name="mcore",
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=8,  # Same as tp_size for training
        tp_size=8,
        ep_size=32,
        ep_tp_size=1,
        rank_info=rank_info,
    )

    # Dense layer MLP (layer 0, 1, 2) should use TP_SHARDING in mcore mode
    for layer_id in [0, 1, 2]:
        for param_name in [
            f"model.layers.{layer_id}.mlp.gate_proj.weight",
            f"model.layers.{layer_id}.mlp.up_proj.weight",
            f"model.layers.{layer_id}.mlp.down_proj.weight",
        ]:
            sharding_type, sharding_dim, num_shards = strategy.get_sharding_strategy(param_name)
            assert sharding_type == ShardingType.TP_SHARDING, \
                f"{param_name}: expected TP_SHARDING for mcore dense layer, got {sharding_type}"
            assert num_shards == 8, \
                f"{param_name}: expected num_shards=8 (tp_size), got {num_shards}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
