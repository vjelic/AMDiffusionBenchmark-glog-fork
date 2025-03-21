import warnings
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.models import attention


class FakeAttention:
    """
    A fake Attention class that simulates the interface of the diffusers' Attention.
    This allows us to test the attention processor without needing the full model.
    """

    def __init__(
        self, heads=2, norm_q=None, norm_k=None, norm_added_q=None, norm_added_k=None
    ):
        # Number of attention heads.
        self.heads = heads
        # Optional normalization layers (if any) for queries/keys.
        self.norm_q = norm_q
        self.norm_k = norm_k
        self.norm_added_q = norm_added_q
        self.norm_added_k = norm_added_k

        # Projection layers for query, key, and value.
        self.to_q = MagicMock()
        self.to_k = MagicMock()
        self.to_v = MagicMock()

        # Additional projections for cross-attention (encoder side).
        self.add_q_proj = MagicMock()
        self.add_k_proj = MagicMock()
        self.add_v_proj = MagicMock()

        # Output projection layers.
        # In real usage, to_out is typically a tuple: (linear layer, dropout layer).
        self.to_out = [MagicMock(), MagicMock()]
        # An additional output projection for the encoder branch.
        self.to_add_out = MagicMock()


def test_fa_flux_attn_init_no_torch_sdpa():
    """
    Test initialization failure when neither PyTorch's SDPA nor flash-attn is available.
    We simulate an older PyTorch by removing scaled_dot_product_attention.
    """

    class MockF:
        pass  # No scaled_dot_product_attention provided

    mock_F = MockF()

    with (
        patch("src.models.attention.F", mock_F),
        patch("src.models.attention.is_fa_available", return_value=False),
        pytest.raises(ImportError, match="requires PyTorch 2.0"),
    ):
        attention.FAFluxAttnProcessor2_0()


def test_fa_flux_attn_init_flash_true_not_available():
    """
    Test that if flash-attn is requested (substitute_sdpa_with_flash_attn=True)
    but flash_attn_func is not available, a ValueError is raised.
    """

    class MockF:
        scaled_dot_product_attention = MagicMock()

    mock_F = MockF()

    with (
        patch("src.models.attention.F", mock_F),
        patch("src.models.attention.is_fa_available", return_value=False),
        pytest.raises(ValueError, match="Flash-Attention is not available"),
    ):
        attention.FAFluxAttnProcessor2_0(substitute_sdpa_with_flash_attn=True)


def test_fa_flux_attn_init_no_issue():
    """
    Test successful initialization when both SDPA and flash-attn are available.
    """

    class MockF:
        scaled_dot_product_attention = MagicMock()

    mock_F = MockF()

    with (
        patch("src.models.attention.F", mock_F),
        patch("src.models.attention.is_fa_available", return_value=True),
    ):
        proc = attention.FAFluxAttnProcessor2_0(substitute_sdpa_with_flash_attn=True)
        assert proc.substitute_sdpa_with_flash_attn is True


def _make_fake_hidden_states(batch, seq, dim, dtype=torch.float32):
    """
    Helper to create a random tensor to simulate hidden states with shape [batch, seq_len, dim].
    """
    return torch.randn(batch, seq, dim, dtype=dtype)


def test_fa_flux_attn_call_basic():
    """
    Test the __call__ method for the attention processor when no encoder hidden states are provided.
    Flash-attn is used because the tensors are in a supported float16 dtype.

    Note: this tests the feature as used in FluxSingleTransformerBlock.
    """
    with (
        patch("src.models.attention.is_fa_available", return_value=True),
        patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa,
    ):
        proc = attention.FAFluxAttnProcessor2_0(substitute_sdpa_with_flash_attn=True)
        attn = FakeAttention(heads=2)

        # Create projections that return float16 tensors (compatible with flash-attn).
        attn.to_q.return_value = torch.randn(2 * 4, 16, dtype=torch.float16)
        attn.to_k.return_value = torch.randn(2 * 4, 16, dtype=torch.float16)
        attn.to_v.return_value = torch.randn(2 * 4, 16, dtype=torch.float16)

        # For this test, the output projections are not critical so we simply set them.
        attn.to_out[0].return_value = torch.randn(2 * 4, 16, dtype=torch.float16)
        attn.to_out[1].return_value = torch.randn(2 * 4, 16, dtype=torch.float16)

        hs = _make_fake_hidden_states(batch=2, seq=4, dim=16, dtype=torch.float16)

        # Mock flash_attn_func to return a tensor in expected permuted shape.
        mock_fa_sdpa = MagicMock(
            return_value=torch.randn(2, 2, 4, 8, dtype=torch.float16)
        )

        with patch("src.models.attention.sdpa", mock_fa_sdpa):
            out = proc(
                attn,
                hidden_states=hs,
                encoder_hidden_states=None,  # No cross-attention branch.
                attention_mask=None,
                image_rotary_emb=None,
            )

            # Verify that flash attention was used and SDPA was not called.
            mock_fa_sdpa.assert_called_once()
            mock_sdpa.assert_not_called()

            # Check that the output has the expected shape and dtype.
            assert isinstance(out, torch.Tensor)
            assert out.shape == (2, 4, 16)
            assert out.dtype == torch.float16


def test_fa_flux_attn_call_datatype_warning():
    """
    Test that when hidden_states are not in a flash-attn compatible dtype (using float32),
    the processor falls back to using SDPA and issues a warning.
    Also, test that the output projection layers are applied correctly.

    Note: When encoder_hidden_states are provided, the total sequence length is 10 (4 main + 6 encoder).
    """
    # Mock flash_attn_func to return a tensor in expected permuted shape.
    mock_sdpa = MagicMock(return_value=torch.randn(2, 2, 10, 8))

    with (
        patch("src.models.attention.is_fa_available", return_value=True),
        patch("torch.nn.functional.scaled_dot_product_attention", mock_sdpa),
        warnings.catch_warnings(record=True) as warn_log,
    ):
        warnings.simplefilter("always")

        proc = attention.FAFluxAttnProcessor2_0(substitute_sdpa_with_flash_attn=True)
        attn = FakeAttention(heads=2)

        # Create main projections with float32 (unsupported by flash-attn).
        attn.to_q.return_value = torch.randn(2 * 4, 16, dtype=torch.float32)
        attn.to_k.return_value = torch.randn(2 * 4, 16, dtype=torch.float32)
        attn.to_v.return_value = torch.randn(2 * 4, 16, dtype=torch.float32)

        # Create encoder projections.
        attn.add_q_proj.return_value = torch.randn(2 * 6, 16, dtype=torch.float32)
        attn.add_k_proj.return_value = torch.randn(2 * 6, 16, dtype=torch.float32)
        attn.add_v_proj.return_value = torch.randn(2 * 6, 16, dtype=torch.float32)

        # Instead of fixed outputs that lose shape, we use identity functions so that the shape remains intact.
        attn.to_out[0].side_effect = lambda x: x
        attn.to_out[1].side_effect = lambda x: x
        attn.to_add_out.side_effect = lambda x: x

        hs = _make_fake_hidden_states(batch=2, seq=4, dim=16, dtype=torch.float32)
        enc_hs = _make_fake_hidden_states(batch=2, seq=6, dim=16, dtype=torch.float32)

        # Process the inputs.
        out_hs, out_enc_hs = proc(attn, hs, encoder_hidden_states=enc_hs)

        # Verify that SDPA was used instead of flash-attn.
        mock_sdpa.assert_called_once()
        # Check that the output shapes are correct:
        # The main branch should have 4 tokens per batch, and the encoder branch 6 tokens.
        assert isinstance(out_hs, torch.Tensor)
        assert isinstance(out_enc_hs, torch.Tensor)
        assert out_hs.shape == (2, 4, 16)
        assert out_enc_hs.shape == (2, 6, 16)

        # Verify that a warning was issued regarding the unsupported datatype.
        assert len(warn_log) == 1
        assert "Disabling flash-attn because of unsupported datatype" in str(
            warn_log[0].message
        )


def test_fa_flux_attn_call_with_encoder():
    """
    Test the __call__ method when encoder_hidden_states are provided.
    This tests the concatenation of the normal projections with the encoder projections,
    and the subsequent splitting of the output into two branches.

    For a main hidden state with shape (2,4,16) and an encoder with shape (2,6,16),
    the concatenated sequence length is 10. The SDPA call returns a tensor with shape
    (2,2,10,8), which is later reshaped and split into:
      - main branch: (2,4,16)
      - encoder branch: (2,6,16)
    """
    # Mock flash_attn_func to return a tensor in expected permuted shape.
    mock_sdpa = MagicMock(return_value=torch.randn(2, 2, 10, 8))

    with (
        patch("torch.nn.functional.scaled_dot_product_attention", mock_sdpa),
        patch("src.models.attention.is_fa_available", return_value=False),
    ):
        proc = attention.FAFluxAttnProcessor2_0(substitute_sdpa_with_flash_attn=False)
        attn = FakeAttention(heads=2)

        # Set up normal hidden state projections.
        attn.to_q.return_value = torch.randn(2 * 4, 16)
        attn.to_k.return_value = torch.randn(2 * 4, 16)
        attn.to_v.return_value = torch.randn(2 * 4, 16)

        # Set up encoder hidden state projections.
        attn.add_q_proj.return_value = torch.randn(2 * 6, 16)
        attn.add_k_proj.return_value = torch.randn(2 * 6, 16)
        attn.add_v_proj.return_value = torch.randn(2 * 6, 16)

        # Use identity functions for output projections so that the input shape is preserved.
        attn.to_out[0].side_effect = lambda x: x
        attn.to_out[1].side_effect = lambda x: x
        attn.to_add_out.side_effect = lambda x: x

        hs = _make_fake_hidden_states(batch=2, seq=4, dim=16)
        enc_hs = _make_fake_hidden_states(batch=2, seq=6, dim=16)

        # Process the inputs.
        out_hs, out_enc_hs = proc(attn, hs, encoder_hidden_states=enc_hs)

        # Check that the outputs are tensors of the expected shapes.
        assert isinstance(out_hs, torch.Tensor)
        assert isinstance(out_enc_hs, torch.Tensor)
        # Main branch output should be of shape (2,4,16).
        assert out_hs.shape == (2, 4, 16)
        # Encoder branch output should be of shape (2,6,16).
        assert out_enc_hs.shape == (2, 6, 16)
