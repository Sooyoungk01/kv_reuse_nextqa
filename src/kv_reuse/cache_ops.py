import copy
from typing import List, Tuple

import torch
from transformers.cache_utils import DynamicCache, StaticCache


TensorPair = Tuple[torch.Tensor, torch.Tensor]


def _has_legacy_cache_lists(cache) -> bool:
    return (
        hasattr(cache, "key_cache")
        and hasattr(cache, "value_cache")
        and isinstance(cache.key_cache, (list, tuple))
        and isinstance(cache.value_cache, (list, tuple))
    )


def _has_single_layer_cache_tensors(cache) -> bool:
    return (
        hasattr(cache, "key_cache")
        and hasattr(cache, "value_cache")
        and torch.is_tensor(cache.key_cache)
        and torch.is_tensor(cache.value_cache)
    )


def get_cache_layer_kv(cache, layer_idx: int) -> TensorPair:
    if hasattr(cache, "layers"):
        if layer_idx >= len(cache.layers):
            raise IndexError(f"Layer {layer_idx} is out of range for cache with {len(cache.layers)} layers")
        layer = cache.layers[layer_idx]
        if hasattr(layer, "keys") and hasattr(layer, "values"):
            return layer.keys, layer.values
        raise RuntimeError(f"Cache layer {layer_idx} does not expose keys/values tensors")

    if hasattr(cache, "keys") and hasattr(cache, "values"):
        return cache.keys, cache.values

    if _has_legacy_cache_lists(cache):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]

    if _has_single_layer_cache_tensors(cache):
        return cache.key_cache, cache.value_cache

    raise RuntimeError(f"Unsupported cache structure while reading layer {layer_idx} KV")


def ensure_dynamic_cache(cache) -> DynamicCache:
    if isinstance(cache, DynamicCache):
        return cache
    return DynamicCache.from_legacy_cache(cache)


def make_empty_cache(config) -> DynamicCache:
    """Create an empty DynamicCache for the Qwen2 text backbone."""
    try:
        return DynamicCache(config=config)
    except TypeError:
        return DynamicCache()


def make_static_cache(
    config,
    batch_size: int,
    max_cache_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> StaticCache:
    return StaticCache(
        config=config,
        batch_size=batch_size,
        max_cache_len=max_cache_len,
        device=device,
        dtype=dtype,
    )


def _maybe_update_seen_tokens(cache: DynamicCache, seq_len: int) -> None:
    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = seq_len
    if hasattr(cache, "seen_tokens"):
        try:
            cache.seen_tokens = seq_len
        except Exception:
            pass


def get_cache_seq_len(cache) -> int:
    if cache is None:
        return 0
    try:
        return int(cache.get_seq_length())
    except Exception:
        pass
    if hasattr(cache, "layers") and len(cache.layers) > 0:
        layer0 = cache.layers[0]
        try:
            return int(layer0.get_seq_length())
        except Exception:
            pass
        if hasattr(layer0, "keys") and layer0.keys.numel() > 0:
            return int(layer0.keys.shape[-2])
    if _has_legacy_cache_lists(cache) and len(cache.key_cache) > 0:
        return int(cache.key_cache[0].shape[-2])
    return 0


def append_kv_slice(cache: DynamicCache, kv_slices: List[TensorPair]) -> DynamicCache:
    """
    Append precomputed KV slices for each decoder layer directly into a DynamicCache.
    Each slice is (k, v) with shape:
      k: [bsz, n_kv_heads, seq, head_dim]
      v: [bsz, n_kv_heads, seq, head_dim]
    """
    if not isinstance(cache, DynamicCache):
        raise RuntimeError(f"append_kv_slice expects a DynamicCache, got {type(cache).__name__}")

    for layer_idx, (k, v) in enumerate(kv_slices):
        cache.update(k.clone(), v.clone(), layer_idx=layer_idx)

    _maybe_update_seen_tokens(cache, get_cache_seq_len(cache))
    return cache


def get_kv_block_seq_len(kv_pairs: List[TensorPair]) -> int:
    if not kv_pairs:
        return 0
    return int(kv_pairs[0][0].shape[-2])


def clone_kv_pairs(kv_pairs: List[TensorPair]) -> List[TensorPair]:
    return [(k.clone(), v.clone()) for k, v in kv_pairs]


def make_kv_buffer_like(kv_pairs: List[TensorPair], seq_len: int) -> List[TensorPair]:
    out: List[TensorPair] = []
    for k, v in kv_pairs:
        k_shape = list(k.shape)
        v_shape = list(v.shape)
        k_shape[-2] = seq_len
        v_shape[-2] = seq_len
        out.append((k.new_zeros(k_shape), v.new_zeros(v_shape)))
    return out


def slice_kv_pairs(kv_pairs: List[TensorPair], start: int, end: int) -> List[TensorPair]:
    if end < start:
        raise ValueError(f"Invalid KV pair slice: start={start}, end={end}")
    return [(k[:, :, start:end, :].clone(), v[:, :, start:end, :].clone()) for k, v in kv_pairs]


def overwrite_kv_slice(kv_buffer: List[TensorPair], start: int, kv_slice: List[TensorPair]) -> None:
    if len(kv_buffer) != len(kv_slice):
        raise ValueError(
            f"Layer mismatch while overwriting KV slice: buffer has {len(kv_buffer)} layers, "
            f"incoming has {len(kv_slice)} layers."
        )
    for (buf_k, buf_v), (k, v) in zip(kv_buffer, kv_slice):
        span_len = k.shape[-2]
        if span_len != v.shape[-2]:
            raise ValueError(f"Mismatched K/V slice lengths: k={k.shape[-2]} v={v.shape[-2]}")
        buf_k[:, :, start : start + span_len, :].copy_(k)
        buf_v[:, :, start : start + span_len, :].copy_(v)


def slice_cache(cache: DynamicCache, start: int, end: int) -> List[TensorPair]:
    """Extract [start:end) token KV slices from each decoder layer."""
    if end <= start:
        raise ValueError(f"Invalid cache slice: start={start}, end={end}")
    out: List[TensorPair] = []
    if hasattr(cache, "layers"):
        layer_indices = range(len(cache.layers))
    elif _has_legacy_cache_lists(cache):
        layer_indices = range(len(cache.key_cache))
    else:
        raise RuntimeError("Unsupported cache structure while slicing")
    for layer_idx in layer_indices:
        k, v = get_cache_layer_kv(cache, layer_idx)
        out.append((k[:, :, start:end, :].clone(), v[:, :, start:end, :].clone()))
    return out


def _clone_cache_layer(layer):
    new_layer = copy.copy(layer)
    if hasattr(layer, "keys"):
        new_layer.keys = layer.keys.clone() if layer.keys is not None else None
    if hasattr(layer, "values"):
        new_layer.values = layer.values.clone() if layer.values is not None else None
    if hasattr(layer, "_sliding_window_tensor"):
        tensor = layer._sliding_window_tensor
        new_layer._sliding_window_tensor = tensor.clone() if tensor is not None else None
    return new_layer


def clone_cache(cache):
    if hasattr(cache, "layers"):
        new_cache = copy.copy(cache)
        new_cache.layers = [_clone_cache_layer(layer) for layer in cache.layers]
        return new_cache

    if _has_legacy_cache_lists(cache):
        new_cache = copy.copy(cache)
        new_cache.key_cache = [x.clone() for x in cache.key_cache]
        new_cache.value_cache = [x.clone() for x in cache.value_cache]
        _maybe_update_seen_tokens(new_cache, get_cache_seq_len(new_cache))
        return new_cache

    cache = ensure_dynamic_cache(cache)
    new_cache = copy.copy(cache)
    new_cache.layers = [_clone_cache_layer(layer) for layer in cache.layers]
    _maybe_update_seen_tokens(new_cache, get_cache_seq_len(new_cache))
    return new_cache


def static_cache_to_dynamic(cache: StaticCache, seq_len: int) -> DynamicCache:
    if hasattr(cache, "layers"):
        out = DynamicCache()
        for layer_idx, layer in enumerate(cache.layers):
            if not getattr(layer, "is_initialized", False):
                continue
            if not hasattr(layer, "keys") or not hasattr(layer, "values"):
                raise RuntimeError(f"Static cache layer {layer_idx} does not expose keys/values tensors")
            out.update(
                layer.keys[:, :, :seq_len, :].clone(),
                layer.values[:, :, :seq_len, :].clone(),
                layer_idx=layer_idx,
            )
        _maybe_update_seen_tokens(out, seq_len)
        return out

    out = DynamicCache()
    out.key_cache = [layer[:, :, :seq_len, :].clone() for layer in cache.key_cache]
    out.value_cache = [layer[:, :, :seq_len, :].clone() for layer in cache.value_cache]
    _maybe_update_seen_tokens(out, seq_len)
    return out


def append_tokenwise_masked_slices(
    cache: DynamicCache,
    frame_prev_kv: List[TensorPair],
    reuse_mask: torch.Tensor,
) -> DynamicCache:
    """
    Append selected token positions from a previous frame cache snapshot.
    reuse_mask: [frame_seq_len] bool, True means reuse the token KV.
    """
    assert reuse_mask.dim() == 1
    selected = reuse_mask.nonzero(as_tuple=False).flatten()
    if selected.numel() == 0:
        return cache

    kv_slices: List[TensorPair] = []
    for k, v in frame_prev_kv:
        kv_slices.append((k.index_select(-2, selected), v.index_select(-2, selected)))
    return append_kv_slice(cache, kv_slices)


def contiguous_segments_from_bool(mask: torch.Tensor, target_value: bool):
    """Yield (start, end) segments where mask[start:end] == target_value."""
    mask = mask.tolist()
    segments = []
    start = None
    for i, flag in enumerate(mask):
        if flag == target_value and start is None:
            start = i
        elif flag != target_value and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(mask)))
    return segments
