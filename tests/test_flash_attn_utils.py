import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F

from src.models.flash_attn_utils import flash_attn_sdpa, is_fa_available
from src.utils import safely_eval_as_bool


def get_qkv(batch_size, seq_len, num_heads, head_dim):
    # Create random query, key, value tensors (in bfloat16, because e.g. fp32 not supported by FA)
    query = torch.randn(
        batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    key = torch.randn(
        batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    value = torch.randn(
        batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    return query, key, value


@pytest.mark.skipif(not is_fa_available(), reason="flash_attn is not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
@pytest.mark.skipif(
    safely_eval_as_bool(os.getenv("SKIP_GPU_TESTS", "false")),
    reason="Skipping GPU tests as per environment variable.",
)
def test_flash_attn_varlen_trivial_mask():
    # Set random seed for reproducibility
    torch.manual_seed(0)
    # Define dimensions
    batch_size = 2
    seq_len = 4
    num_heads = 2
    head_dim = 8
    # Create random query, key, value tensors
    query, key, value = get_qkv(batch_size, seq_len, num_heads, head_dim)
    attn_mask = query.new_ones(
        (
            batch_size,
            1,
            seq_len,
            1,
        )
    ).bool()
    # Compute outputs using flash_attn_sdpa with the attention mask
    output_flash_attn = flash_attn_sdpa(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
    )
    # Compute outputs using F.scaled_dot_product_attention without the attention mask
    output_scaled_dot_product = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
    )
    # Check that the outputs are the same
    assert torch.allclose(
        output_flash_attn, output_scaled_dot_product, atol=1e-3
    ), "Outputs do not match when using varlen attention!"


@pytest.mark.skipif(not is_fa_available(), reason="flash_attn is not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
@pytest.mark.skipif(
    safely_eval_as_bool(os.getenv("SKIP_GPU_TESTS", "false")),
    reason="Skipping GPU tests as per environment variable.",
)
def test_flash_attn_vs_scaled_dot_product_attention():
    # Set random seed for reproducibility
    torch.manual_seed(0)
    # Define dimensions
    batch_size = 2
    seq_len = 4
    num_heads = 2
    head_dim = 8
    # Create random query, key, value tensors
    query, key, value = get_qkv(batch_size, seq_len, num_heads, head_dim)
    # Compute outputs using both functions
    output_flash_attn = flash_attn_sdpa(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )
    # Compute output using PyTorch's scaled_dot_product_attention
    output_scaled_dot_product = F.scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    # Check that the outputs are the same
    assert torch.allclose(
        output_flash_attn, output_scaled_dot_product, atol=1e-3
    ), "Outputs do not match!"


def test_is_fa_available_when_none():
    """
    Test that is_fa_available() returns False when flash_attn is None.
    """
    with patch("src.models.flash_attn_utils.flash_attn", None):
        assert not is_fa_available()


def test_is_fa_available_when_not_none():
    """
    Test that is_fa_available() returns True when flash_attn is set.
    """
    mock_flash_attn = MagicMock()
    with patch("src.models.flash_attn_utils.flash_attn", mock_flash_attn):
        assert is_fa_available()


def test_fa_sdpa_basic():
    """
    Test that flash_attn_sdpa permutes the q, k, v tensors correctly before and after calling
    the flash attention function.
    """
    # Create fake q, k, v tensors with shape: [batch, heads, seq_len, head_dim]
    fake_q = torch.randn(2, 2, 4, 8)
    fake_k = torch.randn_like(fake_q)
    fake_v = torch.randn_like(fake_q)

    # Mock flash_attn_func to return a tensor in expected permuted shape.
    mock_flash_attn = MagicMock()
    mock_flash_attn_func = MagicMock(return_value=torch.randn(2, 4, 2, 8))
    with patch("src.models.flash_attn_utils.flash_attn", mock_flash_attn), patch(
        "src.models.flash_attn_utils.flash_attn.flash_attn_func", mock_flash_attn_func
    ):
        out = flash_attn_sdpa(fake_q, fake_k, fake_v, dropout_p=0.0, is_causal=False)
        # The output shape should match the original input shape.
        assert out.shape == fake_q.shape
        mock_flash_attn_func.assert_called_once()

        # Check that the inputs to flash_attn_func were permuted as expected.
        called_q, called_k, called_v, *_ = mock_flash_attn_func.call_args[0]
        assert called_q.shape == (2, 4, 2, 8)
        assert called_k.shape == (2, 4, 2, 8)
        assert called_v.shape == (2, 4, 2, 8)
