from typing import Optional
import warnings

import torch
from torch.nn import functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb

try:
    from flash_attn_interface import flash_attn_func
except ModuleNotFoundError:
    try:
        from flash_attn import flash_attn_func
    except ModuleNotFoundError:
        flash_attn_func = None


def is_fa_available():
    return flash_attn_func is not None


def fa_sdpa(q, k, v, dropout_p=0.0, is_causal=False):
    """Flash-Attention drop-in replacement of torch.nn.functional.scaled_dot_product_attention function"""
    q, k, v = [x.permute(0, 2, 1, 3) for x in [q, k, v]]
    out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=is_causal)
    return out.permute(0, 2, 1, 3)


# Based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py#L2254
class FAFluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, substitute_sdpa_with_flash_attn: bool = is_fa_available()):
        if not hasattr(F, "scaled_dot_product_attention") and not is_fa_available():
            raise ImportError(
                "FAFluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        if substitute_sdpa_with_flash_attn and not is_fa_available():
            raise ValueError("Flash-Attention is not available")

        self.substitute_sdpa_with_flash_attn = substitute_sdpa_with_flash_attn

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(
                    encoder_hidden_states_query_proj
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(
                    encoder_hidden_states_key_proj
                )

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        if self.substitute_sdpa_with_flash_attn and query.dtype in [
            torch.bfloat16,
            torch.float16,
        ]:
            hidden_states = fa_sdpa(query, key, value, dropout_p=0.0, is_causal=False)
        else:
            if self.substitute_sdpa_with_flash_attn and query.dtype not in [
                torch.bfloat16,
                torch.float16,
            ]:
                # Issue a user warning about preferring bf16 or fp16 for speed-ups
                warnings.warn(
                    f"Disabling flash-attn because of unsupported datatype {query.dtype}. "
                    "Consider using bf16 or fp16 quantization for speed-ups.",
                    UserWarning,
                )

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
