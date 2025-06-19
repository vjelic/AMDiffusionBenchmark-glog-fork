# source: https://github.com/huggingface/transformers/blob/343c8cb86f2ab6a51e7363ee11f69afb1c9e839e/src/transformers/modeling_flash_attention_utils.py

from typing import Optional, Tuple

import torch
from einops import rearrange

try:
    import flash_attn_interface as flash_attn
    from flash_attn_interface.bert_padding import index_first_axis, pad_input
except ModuleNotFoundError:
    try:
        import flash_attn
        from flash_attn.bert_padding import index_first_axis, pad_input

    except ModuleNotFoundError:
        flash_attn = None
        index_first_axis = None
        pad_input = None

torch._dynamo.config.capture_scalar_outputs = True


def is_fa_available():
    return flash_attn is not None


def _get_unpad_data(
    hidden_states,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Retrieves indexing data required to repad unpadded (ragged) tensors.

    Arguments:
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        indices (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input sequence.
        cu_seqlens (`torch.Tensor`):
            The cumulative sequence lengths, used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        max_seqlen_in_batch (`int`):
            Maximum sequence length in batch.
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cumsum_seqlens = torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32)
    padding = cumsum_seqlens.new_zeros((1,))
    cu_seqlens = torch.cat((padding, cumsum_seqlens), dim=0)
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _unpad_inputs(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor, int],
    Tuple[torch.Tensor, torch.Tensor, int],
]:
    """
    Unpads query, key, and values tensors, using an attention mask.

    Arguments:
        q (`torch.Tensor`):
            Query state with padding. Shape: (batch_size, query_length, num_heads, head_dim).
        k (`torch.Tensor`):
            Key state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        v (`torch.Tensor`):
            Value state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, q_max_seqlen, k_max_seqlen), 1 means valid and 0 means not valid.

    Return:
        q_flat (`torch.Tensor`):
            Query state without padding. Shape: (total_target_length, num_heads, head_dim).
        k_flat (`torch.Tensor`):
            Key state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        v_flat (`torch.Tensor`):
            Value state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        mask_info_q (`tuple[torch.Tensor]`):
            A tuple of tensors (q_indices, q_cu_seqlens, q_max_seqlen)
        mask_info_kv (`tuple[torch.Tensor]`):
            A tuple of tensors (k_indices, k_cu_seqlens, k_max_seqlen)
    """
    _, q_max_seqlen, k_max_seqlen = attn_mask.shape
    q_mask = attn_mask.any(dim=2)  # sum(dim=2).to(torch.bool)
    kv_mask = attn_mask.any(dim=1)  # sum(dim=1).to(torch.bool)
    q_flat, q_indices, q_cu_seqlens, q_max_seqlen = _get_unpad_data(q, q_mask)
    k_flat, k_indices, k_cu_seqlens, k_max_seqlen = _get_unpad_data(k, kv_mask)
    v_flat, *_ = _get_unpad_data(v, kv_mask)
    return (
        q_flat,
        k_flat,
        v_flat,
        (q_indices, q_cu_seqlens, q_max_seqlen),
        (k_indices, k_cu_seqlens, k_max_seqlen),
    )


def flash_attn_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention using the flash-attention implementation.

    This function serves as a drop-in replacement for PyTorch's scaled dot-product attention,
    leveraging the flash-attention library for improved performance on compatible hardware.

    Args:
        query (torch.Tensor): The query tensor of shape (batch_size, num_heads, seq_len, head_dim).
        key (torch.Tensor): The key tensor of shape (batch_size, num_heads, seq_len, head_dim).
        value (torch.Tensor): The value tensor of shape (batch_size, num_heads, seq_len, head_dim).
        attn_mask (Optional[torch.Tensor], optional): An optional attention mask of shape
            (batch_size, q_max_seqlen, k_max_seqlen). If provided, it should be a boolean tensor where `True` indicates
            valid positions and `False` indicates masked positions. Defaults to None.
        dropout_p (float, optional): The dropout probability. Defaults to 0.0.
        is_causal (bool, optional): If True, applies a causal mask to prevent attending to future positions.
            Defaults to False.

    Returns:
        torch.Tensor: The output tensor of shape (batch_size, num_heads, seq_len, head_dim).

    Notes:
        - The input tensors are expected to be in bfloat16 or float16 data type, as required by the
          flash-attention library.
        - The attention mask, if provided, is expected to have shape (batch_size, q_max_seqlen, k_max_seqlen).
          It will be automatically reshaped to be compatible with the flash-attention functions.
        - The function permutes the input tensors to match the expected input shape for the flash-attention
          functions and permutes the output back to the original shape.
    """
    query, key, value = [
        x.transpose(1, 2).contiguous() for x in [query, key, value]
    ]  # (bs, n, s, hd) -> (bs, s, n, hd)
    if attn_mask is not None:
        # Check if attn_mask is castable to bool
        try:
            attn_mask = attn_mask.to(torch.bool)
        except RuntimeError as e:
            raise ValueError("`attn_mask` must be castable to a boolean tensor.") from e

        # Check that attn_mask is squeezable to the expected shape (bs, q_max_seqlen, k_max_seqlen)
        # This is because F.scaled_dot_product_attention expects a shape
        # that can be broadcasted into query shape, e.g. (bs, num_attn_heads, q_max_seqlen, k_max_seqlen)
        # but flash-attention contradictorily expects shape (bs, q_max_seqlen, k_max_seqlen)
        batch_size, query_len, *_ = query.shape
        attn_mask = attn_mask.squeeze()
        if batch_size == 1:
            attn_mask = attn_mask.unsqueeze(0)

        seq_len = attn_mask.shape[1]
        if attn_mask.ndim == 2:
            attn_mask = attn_mask.unsqueeze(1).expand(
                batch_size, seq_len, seq_len
            )  # (bs, seq_len) -> (bs, seq_len, seq_len)
            attn_mask = (attn_mask & attn_mask.transpose(1, 2)).bool()

        if attn_mask.shape != (batch_size, seq_len, seq_len):
            raise ValueError(
                f"Unsupported `attn_mask` shape {attn_mask.shape} != {(batch_size, seq_len, seq_len)}"
            )

        query_states, key_states, value_states, mask_info_q, mask_info_kv = (
            _unpad_inputs(
                query,
                key,
                value,
                attn_mask,
            )
        )
        indices_q, cu_seqlens_q, max_seqlen_q = mask_info_q
        _, cu_seqlens_k, max_seqlen_k = mask_info_kv
        attn_output_unpad = flash_attn.flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            causal=is_causal,
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_len)
    else:
        attn_output = flash_attn.flash_attn_func(
            query, key, value, dropout_p=dropout_p, causal=is_causal
        )

    return attn_output.transpose(1, 2)  # transpose back to (bs, n, s, hd)
