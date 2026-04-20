from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache, StaticCache
try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError:
    ALL_ATTENTION_FUNCTIONS = None
try:
    from transformers.masking_utils import create_causal_mask as hf_create_causal_mask
except ImportError:
    hf_create_causal_mask = None
try:
    from transformers.masking_utils import create_sliding_window_causal_mask as hf_create_sliding_window_causal_mask
except ImportError:
    hf_create_sliding_window_causal_mask = None

from .cache_ops import (
    ensure_dynamic_cache,
    get_cache_layer_kv,
    make_static_cache,
    static_cache_to_dynamic,
)
from .pixel_change import decide_reuse_by_pixel_change


@dataclass
class PrefillOutput:
    cache: DynamicCache
    frame_kv_snapshots: List
    reuse_stats: List[Dict]
    next_token_logits: torch.Tensor
    latency_ms: Dict[str, float]
    profile_latency_ms: Dict[str, float]
    frame_latency_ms: List[Dict[str, float]]


TensorPair = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class FrameSnapshot:
    layer_kv: List[TensorPair]
    layer_hidden_states: List[torch.Tensor]
    layer_reuse_masks: List[torch.Tensor]


class _SelectiveLayerCache:
    """
    Minimal cache object for a single decoder layer.

    We pre-materialize prefix KV and reusable current-frame KV, then let the attention module
    write dirty-token KV at sparse `cache_position` indices.
    """

    def __init__(self, layer_idx: int, key_cache: torch.Tensor, value_cache: torch.Tensor):
        self.layer_idx = layer_idx
        self.key_cache = key_cache
        self.value_cache = value_cache

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx != self.layer_idx:
            raise RuntimeError(f"Selective layer cache expected layer {self.layer_idx}, got {layer_idx}")

        cache_kwargs = cache_kwargs or {}
        cache_position = cache_kwargs.get("cache_position")
        if cache_position is None:
            raise RuntimeError("Selective layer cache requires cache_position for sparse KV updates")

        try:
            self.key_cache.index_copy_(2, cache_position, key_states)
            self.value_cache.index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            self.key_cache[:, :, cache_position] = key_states
            self.value_cache[:, :, cache_position] = value_states

        return self.key_cache, self.value_cache


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _timer_start(device: torch.device, sync: bool = True) -> float:
    if sync:
        _sync_device(device)
    return perf_counter()


def _elapsed_ms(start_time: float, device: torch.device, sync: bool = True) -> float:
    if sync:
        _sync_device(device)
    return (perf_counter() - start_time) * 1000.0


@torch.inference_mode()
def _forward_span(language_model, span_embeds: torch.Tensor, cache, cache_position: Optional[torch.Tensor] = None):
    outputs = language_model(
        inputs_embeds=span_embeds,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
        cache_position=cache_position,
    )
    if isinstance(outputs.past_key_values, DynamicCache):
        outputs.past_key_values = ensure_dynamic_cache(outputs.past_key_values)
    return outputs


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return ((q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin))


def _build_layer_reuse_masks(base_mask: torch.Tensor, num_layers: int) -> List[torch.Tensor]:
    # We keep the per-layer plan explicit even when every layer currently shares the same reuse decision.
    return [base_mask.clone() for _ in range(num_layers)]


def _compute_frame_position_embeddings(
    decoder,
    hidden_states: torch.Tensor,
    cache_position: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    position_ids = cache_position.unsqueeze(0)

    if hasattr(decoder, "rotary_emb"):
        return decoder.rotary_emb(hidden_states, position_ids)

    first_attn = decoder.layers[0].self_attn
    if hasattr(first_attn, "rotary_emb"):
        return first_attn.rotary_emb(hidden_states, position_ids)

    raise RuntimeError("Could not locate a rotary embedding module for selective frame prefill")


def _slice_position_embeddings(
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    token_positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = position_embeddings
    return cos.index_select(1, token_positions), sin.index_select(1, token_positions)


def _copy_hidden_tokens(dst: torch.Tensor, token_positions: torch.Tensor, src: torch.Tensor) -> None:
    if token_positions.numel() == 0:
        return
    dst.index_copy_(1, token_positions, src)


def _clone_trace_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return value.detach().cpu().clone()


def _store_manual_trace_feature(
    layer_feature_trace: Optional[List[Dict[str, torch.Tensor]]],
    layer_idx: int,
    name: str,
    tensor: Optional[torch.Tensor],
) -> None:
    if layer_feature_trace is None or tensor is None:
        return
    if name in layer_feature_trace[layer_idx]:
        return
    layer_feature_trace[layer_idx][name] = _clone_trace_tensor(tensor)


def _store_manual_self_attn_trace_inputs(
    layer_feature_trace: Optional[List[Dict[str, torch.Tensor]]],
    layer_idx: int,
    normed_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
) -> None:
    _store_manual_trace_feature(layer_feature_trace, layer_idx, "self_attn_input", normed_states)
    _store_manual_trace_feature(layer_feature_trace, layer_idx, "self_attn_attention_mask", attention_mask)
    _store_manual_trace_feature(layer_feature_trace, layer_idx, "self_attn_cache_position", cache_position)
    cos, sin = position_embeddings
    _store_manual_trace_feature(layer_feature_trace, layer_idx, "self_attn_position_embeddings_cos", cos)
    _store_manual_trace_feature(layer_feature_trace, layer_idx, "self_attn_position_embeddings_sin", sin)


def _build_sparse_causal_mask(
    query_cache_positions: torch.Tensor,
    total_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if query_cache_positions.numel() == 0:
        return torch.zeros((1, 1, 0, total_len), device=query_cache_positions.device, dtype=dtype)

    key_positions = torch.arange(total_len, device=query_cache_positions.device)
    max_visible = query_cache_positions
    allowed = key_positions.unsqueeze(0) <= max_visible.unsqueeze(1)
    mask = torch.zeros((1, 1, query_cache_positions.numel(), total_len), device=query_cache_positions.device, dtype=dtype)
    return mask.masked_fill(~allowed.unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min)


def _build_decoder_attention_mask(
    config,
    inputs_embeds: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values,
) -> torch.Tensor:
    if hf_create_causal_mask is not None:
        attention_mask = hf_create_causal_mask(
            config=config,
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=cache_position.unsqueeze(0),
        )
        if attention_mask is not None:
            # Preserve the exact mask representation HF returns.
            # Casting here can turn bool / specialized mask objects into plain dense tensors,
            # which makes the selective path diverge from the dense teacher path under SDPA.
            return attention_mask

    total_len = past_key_values.max_cache_len
    return _build_sparse_causal_mask(
        query_cache_positions=cache_position,
        total_len=total_len,
        dtype=inputs_embeds.dtype,
    )


def _build_decoder_attention_mask_mapping(
    decoder,
    inputs_embeds: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values,
) -> Dict[str, torch.Tensor]:
    position_ids = cache_position.unsqueeze(0)
    attention_mask_mapping = {
        "full_attention": _build_decoder_attention_mask(
            config=decoder.config,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            past_key_values=past_key_values,
        ),
    }

    if getattr(decoder, "has_sliding_layers", False):
        if hf_create_sliding_window_causal_mask is None:
            raise RuntimeError("Sliding-window causal mask helper is unavailable for dense frame tracing")
        attention_mask_mapping["sliding_attention"] = hf_create_sliding_window_causal_mask(
            config=decoder.config,
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

    return attention_mask_mapping


def _build_decoder_layer_attention_mask(
    decoder,
    decoder_layer,
    inputs_embeds: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values,
    *,
    force_full_attention_mask: bool,
) -> torch.Tensor:
    if force_full_attention_mask:
        return _build_decoder_attention_mask(
            config=decoder.config,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )

    attention_type = getattr(decoder_layer, "attention_type", "full_attention")
    attention_mask_mapping = _build_decoder_attention_mask_mapping(
        decoder=decoder,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        past_key_values=past_key_values,
    )
    return attention_mask_mapping.get(attention_type, attention_mask_mapping["full_attention"])


def _extract_frame_kv_from_cache(
    cache,
    layer_idx: int,
    frame_start: int,
    frame_len: int,
) -> TensorPair:
    frame_end = frame_start + frame_len
    layer_key_cache, layer_value_cache = get_cache_layer_kv(cache, layer_idx)
    return (
        layer_key_cache[:, :, frame_start:frame_end, :].clone(),
        layer_value_cache[:, :, frame_start:frame_end, :].clone(),
    )


def _attention_shape_params(attn) -> Tuple[int, int, int]:
    num_heads = getattr(attn, "num_heads", None)
    if num_heads is None:
        num_heads = getattr(attn, "num_attention_heads", None)
    if num_heads is None and hasattr(attn, "config"):
        num_heads = getattr(attn.config, "num_attention_heads", None)
    if num_heads is None:
        raise RuntimeError("Could not infer attention head count for selective replay")

    num_key_value_heads = getattr(attn, "num_key_value_heads", None)
    if num_key_value_heads is None and hasattr(attn, "config"):
        num_key_value_heads = getattr(attn.config, "num_key_value_heads", None)
    if num_key_value_heads is None:
        groups = getattr(attn, "num_key_value_groups", None)
        if groups is None or groups <= 0:
            raise RuntimeError("Could not infer attention KV head count for selective replay")
        num_key_value_heads = int(num_heads) // int(groups)

    hidden_size = getattr(attn, "hidden_size", None)
    if hidden_size is None and hasattr(attn, "config"):
        hidden_size = getattr(attn.config, "hidden_size", None)
    if hidden_size is None and hasattr(attn, "o_proj"):
        hidden_size = int(attn.o_proj.out_features)
    if hidden_size is None:
        raise RuntimeError("Could not infer attention hidden size for selective replay")

    return int(num_heads), int(num_key_value_heads), int(hidden_size)


def _eager_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **_kwargs,
):
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = torch.dropout(attn_weights, p=dropout, train=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


@torch.inference_mode()
def _run_self_attn_with_module_backend(
    attn,
    normed_states: torch.Tensor,
    cache,
    cache_position: torch.Tensor,
    attention_mask: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    *,
    layer_feature_trace: Optional[List[Dict[str, torch.Tensor]]] = None,
) -> torch.Tensor:
    bsz, q_len, _ = normed_states.size()
    num_heads, num_key_value_heads, hidden_size = _attention_shape_params(attn)
    layer_idx = int(attn.layer_idx)

    _store_manual_self_attn_trace_inputs(
        layer_feature_trace=layer_feature_trace,
        layer_idx=layer_idx,
        normed_states=normed_states,
        attention_mask=attention_mask,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )

    query_states = attn.q_proj(normed_states)
    key_states = attn.k_proj(normed_states)
    value_states = attn.v_proj(normed_states)

    query_states = query_states.view(bsz, q_len, num_heads, attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, attn.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_key_value_heads, attn.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if cache is not None:
        key_states, value_states = cache.update(
            key_states,
            value_states,
            attn.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position},
        )

    attention_interface = _eager_attention_forward
    if ALL_ATTENTION_FUNCTIONS is not None:
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            getattr(attn.config, "_attn_implementation", None),
            _eager_attention_forward,
        )

    attn_output, _ = attention_interface(
        attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not attn.training else attn.attention_dropout,
        scaling=getattr(attn, "scaling", attn.head_dim**-0.5),
        sliding_window=getattr(attn, "sliding_window", None),
    )

    attn_output = attn_output.reshape(bsz, q_len, hidden_size).contiguous()
    attn_output = attn.o_proj(attn_output)
    _store_manual_trace_feature(layer_feature_trace, layer_idx, "self_attn_output", attn_output)
    return attn_output


@torch.inference_mode()
def _run_self_attn_with_sdpa_backend(
    attn,
    normed_states: torch.Tensor,
    cache,
    cache_position: torch.Tensor,
    attention_mask: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    bsz, q_len, _ = normed_states.size()
    num_heads, num_key_value_heads, hidden_size = _attention_shape_params(attn)
    query_states = attn.q_proj(normed_states)
    key_states = attn.k_proj(normed_states)
    value_states = attn.v_proj(normed_states)

    query_states = query_states.view(bsz, q_len, num_heads, attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, attn.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_key_value_heads, attn.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

    key_states, value_states = cache.update(
        key_states,
        value_states,
        attn.layer_idx,
        {"sin": sin, "cos": cos, "cache_position": cache_position},
    )

    key_states = _repeat_kv(key_states, attn.num_key_value_groups)
    value_states = _repeat_kv(value_states, attn.num_key_value_groups)

    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    attn_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=attn.attention_dropout if attn.training else 0.0,
        is_causal=False,
    )

    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, hidden_size)
    return attn.o_proj(attn_output)


@torch.inference_mode()
def _forward_decoder_layer_packed(
    decoder_layer,
    hidden_states: torch.Tensor,
    cache,
    cache_position: torch.Tensor,
    attention_mask: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    *,
    layer_feature_trace: Optional[List[Dict[str, torch.Tensor]]] = None,
) -> torch.Tensor:
    layer_idx = int(decoder_layer.self_attn.layer_idx)
    _store_manual_trace_feature(layer_feature_trace, layer_idx, "layer_input", hidden_states)

    residual = hidden_states
    hidden_states = decoder_layer.input_layernorm(hidden_states)
    hidden_states = _run_self_attn_with_module_backend(
        attn=decoder_layer.self_attn,
        normed_states=hidden_states,
        cache=cache,
        cache_position=cache_position,
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
        layer_feature_trace=layer_feature_trace,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
    hidden_states = decoder_layer.mlp(hidden_states)
    hidden_states = residual + hidden_states
    _store_manual_trace_feature(layer_feature_trace, layer_idx, "layer_output", hidden_states)
    return hidden_states


@torch.inference_mode()
def _forward_selective_decoder_layer(
    decoder,
    decoder_layer,
    global_cache: StaticCache,
    layer_input_states: torch.Tensor,
    prev_snapshot: Optional[FrameSnapshot],
    layer_idx: int,
    reuse_mask: torch.Tensor,
    frame_cache_positions: torch.Tensor,
    device: torch.device,
    profile_timing: bool,
    frame_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    force_full_attention_mask: bool,
    dense_attention_mask_mapping: Optional[Dict[str, torch.Tensor]] = None,
    manual_feature_trace: Optional[List[Dict[str, torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, TensorPair, Dict[str, float]]:
    """
    Partial decoder-layer update for Qwen2 causal attention.

    We map the user's "dirty-only gather/scatter" idea to a global-attention style block:
    - clean positions reuse previous-frame same-layer KV and post-layer hidden states
    - dirty positions run input LN + QKV projection only on the selected tokens
    - attention is computed for dirty queries against full mixed K/V (prefix + clean reused + dirty recomputed)
    - output projection and MLP are applied only to dirty positions, then scattered back
    """
    attn = decoder_layer.self_attn
    batch_size, frame_len, hidden_size = layer_input_states.shape
    if batch_size != 1:
        raise RuntimeError(f"Selective frame prefill expects batch_size=1, got {batch_size}")

    layer_stats = {
        "mode_partial_global": 1.0,
        "mode_decoder_forward": 1.0 if attn.__class__.__name__ != "Qwen2FlashAttention2" else 0.0,
        "mode_manual_fallback": 1.0 if attn.__class__.__name__ == "Qwen2FlashAttention2" else 0.0,
        "attn_backend_module": 1.0 if attn.__class__.__name__ != "Qwen2FlashAttention2" else 0.0,
        "attn_backend_sdpa_fallback": 1.0 if attn.__class__.__name__ == "Qwen2FlashAttention2" else 0.0,
        "clean_tokens": float(reuse_mask.sum().item()),
        "dirty_tokens": float((~reuse_mask).sum().item()),
        "copy_reuse": 0.0,
        "decoder_forward": 0.0,
        "kv_projection": 0.0,
        "attention": 0.0,
        "mlp": 0.0,
        "total": 0.0,
    }
    layer_start = _timer_start(device, sync=profile_timing)

    current_layer_output = torch.empty_like(layer_input_states)

    reuse_positions = reuse_mask.nonzero(as_tuple=False).flatten()
    recompute_positions = (~reuse_mask).nonzero(as_tuple=False).flatten()
    frame_start = int(frame_cache_positions[0].item()) if frame_len > 0 else 0
    layer_key_cache, layer_value_cache = get_cache_layer_kv(global_cache, layer_idx)
    full_recompute_without_reuse = reuse_positions.numel() == 0 and recompute_positions.numel() == frame_len

    step_start = _timer_start(device, sync=profile_timing)
    if reuse_positions.numel() > 0:
        if prev_snapshot is None:
            raise RuntimeError(f"Layer {layer_idx} requested KV reuse without a previous frame snapshot")
        prev_kv = prev_snapshot.layer_kv[layer_idx]
        prev_hidden = prev_snapshot.layer_hidden_states[layer_idx]
        reuse_cache_positions = frame_cache_positions.index_select(0, reuse_positions)
        reused_k = prev_kv[0].index_select(2, reuse_positions)
        reused_v = prev_kv[1].index_select(2, reuse_positions)
        reused_hidden = prev_hidden.index_select(1, reuse_positions)
        try:
            layer_key_cache.index_copy_(2, reuse_cache_positions, reused_k)
            layer_value_cache.index_copy_(2, reuse_cache_positions, reused_v)
        except NotImplementedError:
            layer_key_cache[:, :, reuse_cache_positions] = reused_k
            layer_value_cache[:, :, reuse_cache_positions] = reused_v
        _copy_hidden_tokens(current_layer_output, reuse_positions, reused_hidden)
    layer_stats["copy_reuse"] += _elapsed_ms(step_start, device, sync=profile_timing)

    if recompute_positions.numel() > 0:
        if full_recompute_without_reuse and not force_full_attention_mask and dense_attention_mask_mapping is not None:
            residual = layer_input_states
            dirty_cache_positions = frame_cache_positions
            dirty_position_embeddings = frame_position_embeddings
            attention_type = getattr(decoder_layer, "attention_type", "full_attention")
            attn_mask = dense_attention_mask_mapping.get(
                attention_type,
                dense_attention_mask_mapping["full_attention"],
            )
        else:
            residual = layer_input_states.index_select(1, recompute_positions)
            dirty_cache_positions = frame_cache_positions.index_select(0, recompute_positions)
            dirty_position_embeddings = _slice_position_embeddings(frame_position_embeddings, recompute_positions)
            attn_mask = _build_decoder_layer_attention_mask(
                decoder=decoder,
                decoder_layer=decoder_layer,
                inputs_embeds=residual,
                cache_position=dirty_cache_positions,
                past_key_values=global_cache,
                force_full_attention_mask=force_full_attention_mask,
            )
        if attn.__class__.__name__ != "Qwen2FlashAttention2":
            step_start = _timer_start(device, sync=profile_timing)
            hidden_states = _forward_decoder_layer_packed(
                decoder_layer=decoder_layer,
                hidden_states=residual,
                cache=global_cache,
                cache_position=dirty_cache_positions,
                attention_mask=attn_mask,
                position_embeddings=dirty_position_embeddings,
                layer_feature_trace=manual_feature_trace if full_recompute_without_reuse else None,
            )
            layer_stats["decoder_forward"] += _elapsed_ms(step_start, device, sync=profile_timing)
            if full_recompute_without_reuse:
                current_layer_output.copy_(hidden_states)
            else:
                _copy_hidden_tokens(current_layer_output, recompute_positions, hidden_states)
        else:
            step_start = _timer_start(device, sync=profile_timing)
            if full_recompute_without_reuse:
                _store_manual_trace_feature(manual_feature_trace, layer_idx, "layer_input", residual)
            normed_states = decoder_layer.input_layernorm(residual)
            if full_recompute_without_reuse:
                _store_manual_self_attn_trace_inputs(
                    layer_feature_trace=manual_feature_trace,
                    layer_idx=layer_idx,
                    normed_states=normed_states,
                    attention_mask=attn_mask,
                    cache_position=dirty_cache_positions,
                    position_embeddings=dirty_position_embeddings,
                )
            layer_stats["kv_projection"] += _elapsed_ms(step_start, device, sync=profile_timing)

            step_start = _timer_start(device, sync=profile_timing)
            attn_output = _run_self_attn_with_sdpa_backend(
                attn=attn,
                normed_states=normed_states,
                cache=global_cache,
                cache_position=dirty_cache_positions,
                attention_mask=attn_mask,
                position_embeddings=dirty_position_embeddings,
            )
            if full_recompute_without_reuse:
                _store_manual_trace_feature(manual_feature_trace, layer_idx, "self_attn_output", attn_output)
            layer_stats["attention"] += _elapsed_ms(step_start, device, sync=profile_timing)

            step_start = _timer_start(device, sync=profile_timing)
            hidden_states = residual + attn_output
            mlp_residual = hidden_states
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            hidden_states = decoder_layer.mlp(hidden_states)
            hidden_states = mlp_residual + hidden_states
            if full_recompute_without_reuse:
                current_layer_output.copy_(hidden_states)
            else:
                _copy_hidden_tokens(current_layer_output, recompute_positions, hidden_states)
            if full_recompute_without_reuse:
                _store_manual_trace_feature(manual_feature_trace, layer_idx, "layer_output", hidden_states)
            layer_stats["mlp"] += _elapsed_ms(step_start, device, sync=profile_timing)

    current_layer_kv = _extract_frame_kv_from_cache(
        cache=global_cache,
        layer_idx=layer_idx,
        frame_start=frame_start,
        frame_len=frame_len,
    )
    layer_stats["total"] = _elapsed_ms(layer_start, device, sync=profile_timing)
    return current_layer_output, current_layer_kv, layer_stats


@torch.inference_mode()
def _forward_frame_layerwise_with_selective_reuse(
    language_model,
    cache: StaticCache,
    frame_features: torch.Tensor,
    frame_cache_positions: torch.Tensor,
    prev_snapshot: Optional[FrameSnapshot],
    layer_reuse_masks: List[torch.Tensor],
    device: torch.device,
    profile_timing: bool,
    force_full_attention_mask: bool = False,
    manual_feature_trace: Optional[List[Dict[str, torch.Tensor]]] = None,
) -> Tuple[FrameSnapshot, Dict[str, float], List[Dict[str, float]], torch.Tensor]:
    decoder = language_model.model
    frame_hidden_states = frame_features.unsqueeze(0)
    frame_position_embeddings = _compute_frame_position_embeddings(
        decoder=decoder,
        hidden_states=frame_hidden_states,
        cache_position=frame_cache_positions,
    )
    dense_attention_mask_mapping = _build_decoder_attention_mask_mapping(
        decoder=decoder,
        inputs_embeds=frame_hidden_states,
        cache_position=frame_cache_positions,
        past_key_values=cache,
    )
    current_layer_kv: List[TensorPair] = []
    current_layer_outputs: List[torch.Tensor] = []
    layer_latency: List[Dict[str, float]] = []
    frame_latency = {
        "copy_reuse": 0.0,
        "decoder_forward": 0.0,
        "kv_projection": 0.0,
        "attention": 0.0,
        "mlp": 0.0,
        "layers_total": 0.0,
    }

    for layer_idx, decoder_layer in enumerate(decoder.layers):
        layer_output, layer_kv, layer_stats = _forward_selective_decoder_layer(
            decoder=decoder,
            decoder_layer=decoder_layer,
            global_cache=cache,
            layer_input_states=frame_hidden_states,
            prev_snapshot=prev_snapshot,
            layer_idx=layer_idx,
            reuse_mask=layer_reuse_masks[layer_idx],
            frame_cache_positions=frame_cache_positions,
            device=device,
            profile_timing=profile_timing,
            frame_position_embeddings=frame_position_embeddings,
            force_full_attention_mask=force_full_attention_mask,
            dense_attention_mask_mapping=dense_attention_mask_mapping,
            manual_feature_trace=manual_feature_trace,
        )
        current_layer_kv.append(layer_kv)
        current_layer_outputs.append(layer_output.clone())
        frame_hidden_states = layer_output
        layer_latency.append(
            {
                "layer_idx": float(layer_idx),
                **layer_stats,
            }
        )
        frame_latency["copy_reuse"] += layer_stats["copy_reuse"]
        frame_latency["decoder_forward"] += layer_stats["decoder_forward"]
        frame_latency["kv_projection"] += layer_stats["kv_projection"]
        frame_latency["attention"] += layer_stats["attention"]
        frame_latency["mlp"] += layer_stats["mlp"]
        frame_latency["layers_total"] += layer_stats["total"]

    frame_snapshot = FrameSnapshot(
        layer_kv=current_layer_kv,
        layer_hidden_states=current_layer_outputs,
        layer_reuse_masks=[m.clone() for m in layer_reuse_masks],
    )
    return frame_snapshot, frame_latency, layer_latency, frame_hidden_states


@torch.inference_mode()
def _forward_frame_layerwise_dense(
    language_model,
    cache: StaticCache,
    frame_features: torch.Tensor,
    frame_cache_positions: torch.Tensor,
) -> Tuple[FrameSnapshot, torch.Tensor]:
    decoder = language_model.model
    frame_cache = cache
    frame_hidden_outputs: List[torch.Tensor] = []
    hook_handles = []

    def _capture_layer_output(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        frame_hidden_outputs.append(hidden.detach().clone())

    for decoder_layer in decoder.layers:
        hook_handles.append(decoder_layer.register_forward_hook(_capture_layer_output))

    try:
        outputs = _forward_span(
            language_model,
            frame_features.unsqueeze(0),
            frame_cache,
            cache_position=frame_cache_positions,
        )
    finally:
        for handle in hook_handles:
            handle.remove()

    frame_len = int(frame_features.shape[0])
    frame_start = int(frame_cache_positions[0].item()) if frame_len > 0 else 0
    current_layer_kv: List[TensorPair] = []
    for layer_idx in range(len(decoder.layers)):
        layer_key_cache, layer_value_cache = get_cache_layer_kv(frame_cache, layer_idx)
        current_layer_kv.append(
            (
                layer_key_cache[:, :, frame_start : frame_start + frame_len, :].clone(),
                layer_value_cache[:, :, frame_start : frame_start + frame_len, :].clone(),
            )
        )

    zero_reuse_mask = torch.zeros(frame_len, dtype=torch.bool, device=frame_features.device)
    frame_snapshot = FrameSnapshot(
        layer_kv=current_layer_kv,
        layer_hidden_states=frame_hidden_outputs,
        layer_reuse_masks=[zero_reuse_mask.clone() for _ in range(len(decoder.layers))],
    )
    return frame_snapshot, outputs.last_hidden_state


@torch.inference_mode()
def baseline_prefill_from_embeds(
    model,
    prepared_prompt_or_inputs_embeds,
    profile_timing: bool = False,
) -> PrefillOutput:
    """
    Baseline prefill.

    - If given a raw `inputs_embeds` tensor, preserve the legacy one-shot dense prompt prefill.
    - If given a prepared prompt object, process prefix -> each frame -> suffix sequentially.

    The sequential mode matches the streaming-style assumption used by `kv_reuse_prefill`,
    but without any KV reuse.
    """
    if isinstance(prepared_prompt_or_inputs_embeds, torch.Tensor):
        inputs_embeds = prepared_prompt_or_inputs_embeds
        device = inputs_embeds.device
        start_time = _timer_start(device, sync=profile_timing)
        outputs = model.language_model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            return_dict=True,
        )
        total_ms = _elapsed_ms(start_time, device, sync=profile_timing)
        return PrefillOutput(
            cache=ensure_dynamic_cache(outputs.past_key_values),
            frame_kv_snapshots=[],
            reuse_stats=[],
            next_token_logits=outputs.logits[:, -1, :].detach(),
            latency_ms={},
            profile_latency_ms={
                "total": total_ms,
                "forward_total": total_ms,
            }
            if profile_timing
            else {},
            frame_latency_ms=[],
        )

    prepared_prompt = prepared_prompt_or_inputs_embeds
    language_model = model.language_model
    device = prepared_prompt.inputs_embeds.device
    total_start = _timer_start(device, sync=profile_timing)
    full_prompt_len = prepared_prompt.inputs_embeds.shape[1]
    cache = make_static_cache(
        config=model.config.text_config,
        batch_size=1,
        max_cache_len=full_prompt_len,
        device=device,
        dtype=prepared_prompt.inputs_embeds.dtype,
    )
    next_token_logits: Optional[torch.Tensor] = None
    frame_latency_ms: List[Dict[str, float]] = []
    reuse_stats: List[Dict] = []
    profile_latency_ms = {
        "prefix_forward": 0.0,
        "frame_forward": 0.0,
        "suffix_forward": 0.0,
    }

    video_positions = prepared_prompt.video_token_positions
    first_video_pos = int(video_positions[0].item())
    last_video_pos = int(video_positions[-1].item())
    suffix = prepared_prompt.inputs_embeds[:, last_video_pos + 1 :, :]

    if first_video_pos > 0:
        prefix = prepared_prompt.inputs_embeds[:, :first_video_pos, :]
        prefix_positions = torch.arange(first_video_pos, device=device, dtype=torch.long)
        step_start = _timer_start(device, sync=profile_timing)
        outputs = _forward_span(language_model, prefix, cache, cache_position=prefix_positions)
        profile_latency_ms["prefix_forward"] += _elapsed_ms(step_start, device, sync=profile_timing)
        cache = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :].detach()

    for frame_idx, (frame_features, frame_positions) in enumerate(
        zip(prepared_prompt.frame_features, prepared_prompt.frame_token_positions)
    ):
        frame_start = _timer_start(device, sync=profile_timing)
        outputs = _forward_span(
            language_model,
            frame_features.unsqueeze(0),
            cache,
            cache_position=frame_positions.to(device=device, dtype=torch.long),
        )
        frame_forward_ms = _elapsed_ms(frame_start, device, sync=profile_timing)
        profile_latency_ms["frame_forward"] += frame_forward_ms
        cache = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :].detach()
        frame_latency_ms.append(
            {
                "frame_idx": float(frame_idx),
                "frame_forward": frame_forward_ms,
                "total": frame_forward_ms,
            }
        )
        reuse_stats.append(
            {
                "frame_idx": int(frame_idx),
                "reused_tokens": 0,
                "recomputed_tokens": int(frame_features.shape[0]),
                "reuse_ratio": 0.0,
            }
        )

    if suffix.shape[1] > 0:
        suffix_positions = torch.arange(
            last_video_pos + 1,
            full_prompt_len,
            device=device,
            dtype=torch.long,
        )
        step_start = _timer_start(device, sync=profile_timing)
        outputs = _forward_span(language_model, suffix, cache, cache_position=suffix_positions)
        profile_latency_ms["suffix_forward"] += _elapsed_ms(step_start, device, sync=profile_timing)
        cache = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :].detach()

    if next_token_logits is None:
        raise RuntimeError("Failed to collect next-token logits during baseline prompt prefill.")

    profile_total_ms = _elapsed_ms(total_start, device, sync=profile_timing)
    profile_latency_ms["total"] = profile_total_ms
    if not profile_timing:
        profile_latency_ms = {}
        frame_latency_ms = []
    return PrefillOutput(
        cache=static_cache_to_dynamic(cache, seq_len=full_prompt_len),
        frame_kv_snapshots=[],
        reuse_stats=reuse_stats,
        next_token_logits=next_token_logits,
        latency_ms={},
        profile_latency_ms=profile_latency_ms,
        frame_latency_ms=frame_latency_ms,
    )


@torch.inference_mode()
def kv_reuse_prefill(
    model,
    prepared_prompt,
    raw_video_frames,
    pixel_threshold: float = 0.08,
    profile_timing: bool = False,
) -> PrefillOutput:
    """
    Layer-wise selective frame prefill.

    Each frame is processed in a single decoder-layer sweep:
    - reusable token positions copy previous-frame same-layer KV/hidden-state snapshots
    - recompute token positions run one batched attention/MLP pass per layer
    - the finalized current-frame KV block is appended to the global cache once per frame

    This keeps reuse/update decisions inside the patched decoder path instead of splitting the frame
    into multiple outer `forward()` calls.
    """
    language_model = model.language_model
    device = prepared_prompt.inputs_embeds.device
    total_start = _timer_start(device, sync=profile_timing)
    full_prompt_len = prepared_prompt.inputs_embeds.shape[1]
    cache = make_static_cache(
        config=model.config.text_config,
        batch_size=1,
        max_cache_len=full_prompt_len,
        device=device,
        dtype=prepared_prompt.inputs_embeds.dtype,
    )
    reuse_stats: List[Dict] = []
    frame_snapshots: List[FrameSnapshot] = []
    next_token_logits: Optional[torch.Tensor] = None
    frame_latency_ms: List[Dict[str, float]] = []
    prev_snapshot: Optional[FrameSnapshot] = None
    profile_latency_ms = {
        "prefix_forward": 0.0,
        "reuse_plan": 0.0,
        "layer_copy_reuse": 0.0,
        "layer_decoder_forward": 0.0,
        "layer_kv_projection": 0.0,
        "layer_attention": 0.0,
        "layer_mlp": 0.0,
        "layer_total": 0.0,
        "frame_inplace_write": 0.0,
        "frame_snapshot": 0.0,
        "suffix_forward": 0.0,
    }

    video_positions = prepared_prompt.video_token_positions
    first_video_pos = int(video_positions[0].item())
    last_video_pos = int(video_positions[-1].item())
    suffix = prepared_prompt.inputs_embeds[:, last_video_pos + 1 :, :]

    if first_video_pos > 0:
        prefix = prepared_prompt.inputs_embeds[:, :first_video_pos, :]
        prefix_positions = torch.arange(first_video_pos, device=device, dtype=torch.long)
        step_start = _timer_start(device, sync=profile_timing)
        outputs = _forward_span(language_model, prefix, cache, cache_position=prefix_positions)
        profile_latency_ms["prefix_forward"] += _elapsed_ms(step_start, device, sync=profile_timing)
        cache = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :].detach()

    for frame_idx, (frame_features, frame_positions) in enumerate(
        zip(prepared_prompt.frame_features, prepared_prompt.frame_token_positions)
    ):
        frame_total_start = _timer_start(device, sync=profile_timing)
        frame_positions = frame_positions.to(device=device, dtype=torch.long)
        frame_timing = {
            "frame_idx": float(frame_idx),
            "total": 0.0,
            "reuse_plan": 0.0,
            "layer_copy_reuse": 0.0,
            "layer_decoder_forward": 0.0,
            "layer_kv_projection": 0.0,
            "layer_attention": 0.0,
            "layer_mlp": 0.0,
            "layer_total": 0.0,
            "frame_inplace_write": 0.0,
            "frame_snapshot": 0.0,
        }

        if frame_idx == 0:
            base_reuse_mask = torch.zeros(frame_features.shape[0], dtype=torch.bool, device=frame_features.device)
            layer_reuse_masks = _build_layer_reuse_masks(
                base_reuse_mask,
                len(language_model.model.layers),
            )
        else:
            step_start = _timer_start(device, sync=profile_timing)
            decision = decide_reuse_by_pixel_change(
                raw_video_frames[frame_idx - 1],
                raw_video_frames[frame_idx],
                threshold=pixel_threshold,
                append_newline_token=(frame_idx == len(prepared_prompt.frame_features) - 1),
            )
            reuse_mask = decision.reuse_mask.to(frame_features.device)
            if reuse_mask.numel() != frame_features.shape[0]:
                raise RuntimeError(
                    f"Reuse mask length mismatch for frame {frame_idx}: mask={reuse_mask.numel()} vs frame_tokens={frame_features.shape[0]}"
                )
            if prev_snapshot is not None:
                prev_seq_len = prev_snapshot.layer_kv[0][0].shape[-2]
                if prev_seq_len < reuse_mask.numel():
                    reuse_mask = reuse_mask.clone()
                    reuse_mask[prev_seq_len:] = False
            base_reuse_mask = reuse_mask
            layer_reuse_masks = _build_layer_reuse_masks(reuse_mask, len(language_model.model.layers))
            plan_ms = _elapsed_ms(step_start, device, sync=profile_timing)
            profile_latency_ms["reuse_plan"] += plan_ms
            frame_timing["reuse_plan"] += plan_ms

        step_start = _timer_start(device, sync=profile_timing)
        frame_snapshot, layer_latency_totals, layer_latency_detail, final_frame_hidden = _forward_frame_layerwise_with_selective_reuse(
            language_model=language_model,
            cache=cache,
            frame_features=frame_features,
            frame_cache_positions=frame_positions,
            prev_snapshot=prev_snapshot,
            layer_reuse_masks=layer_reuse_masks,
            device=device,
            profile_timing=profile_timing,
        )
        layer_call_ms = _elapsed_ms(step_start, device, sync=profile_timing)

        profile_latency_ms["layer_copy_reuse"] += layer_latency_totals["copy_reuse"]
        profile_latency_ms["layer_decoder_forward"] += layer_latency_totals["decoder_forward"]
        profile_latency_ms["layer_kv_projection"] += layer_latency_totals["kv_projection"]
        profile_latency_ms["layer_attention"] += layer_latency_totals["attention"]
        profile_latency_ms["layer_mlp"] += layer_latency_totals["mlp"]
        profile_latency_ms["layer_total"] += layer_latency_totals["layers_total"]
        frame_timing["layer_copy_reuse"] += layer_latency_totals["copy_reuse"]
        frame_timing["layer_decoder_forward"] += layer_latency_totals["decoder_forward"]
        frame_timing["layer_kv_projection"] += layer_latency_totals["kv_projection"]
        frame_timing["layer_attention"] += layer_latency_totals["attention"]
        frame_timing["layer_mlp"] += layer_latency_totals["mlp"]
        frame_timing["layer_total"] += layer_call_ms

        step_start = _timer_start(device, sync=profile_timing)
        frame_snapshots.append(frame_snapshot)
        prev_snapshot = frame_snapshot
        snapshot_ms = _elapsed_ms(step_start, device, sync=profile_timing)
        profile_latency_ms["frame_snapshot"] += snapshot_ms
        frame_timing["frame_snapshot"] += snapshot_ms

        reused = int(layer_reuse_masks[0].sum().item())
        recomputed = int((~layer_reuse_masks[0]).sum().item())
        reuse_stats.append(
            {
                "frame_idx": int(frame_idx),
                "reused_tokens": reused,
                "recomputed_tokens": recomputed,
                "reuse_ratio": float(reused / max(1, base_reuse_mask.numel())),
                "layer_reuse": [
                    {
                        "layer_idx": int(layer_idx),
                        "mode": "partial_global",
                        "reused_tokens": int(mask.sum().item()),
                        "recomputed_tokens": int((~mask).sum().item()),
                    }
                    for layer_idx, mask in enumerate(layer_reuse_masks)
                ],
                "layer_latency_ms": layer_latency_detail,
            }
        )
        frame_timing["total"] = _elapsed_ms(frame_total_start, device, sync=profile_timing)
        frame_latency_ms.append(frame_timing)

        if suffix.shape[1] == 0:
            next_token_logits = language_model.lm_head(language_model.model.norm(final_frame_hidden))[:, -1, :].detach()

    if suffix.shape[1] > 0:
        suffix_positions = torch.arange(last_video_pos + 1, full_prompt_len, device=device, dtype=torch.long)
        step_start = _timer_start(device, sync=profile_timing)
        outputs = _forward_span(language_model, suffix, cache, cache_position=suffix_positions)
        profile_latency_ms["suffix_forward"] += _elapsed_ms(step_start, device, sync=profile_timing)
        cache = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :].detach()

    if next_token_logits is None:
        raise RuntimeError("Failed to collect next-token logits during prompt prefill.")

    profile_total_ms = _elapsed_ms(total_start, device, sync=profile_timing)
    profile_latency_ms["total"] = profile_total_ms
    if not profile_timing:
        profile_latency_ms = {}
        frame_latency_ms = []
    return PrefillOutput(
        cache=static_cache_to_dynamic(cache, seq_len=full_prompt_len),
        frame_kv_snapshots=frame_snapshots,
        reuse_stats=reuse_stats,
        next_token_logits=next_token_logits,
        latency_ms={},
        profile_latency_ms=profile_latency_ms,
        frame_latency_ms=frame_latency_ms,
    )
