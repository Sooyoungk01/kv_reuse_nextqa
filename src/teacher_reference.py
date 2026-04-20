from __future__ import annotations

from types import MethodType
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from runtime_config import normalize_teacher_reference
from kv_reuse.cache_ops import clone_cache, get_cache_layer_kv, static_cache_to_dynamic
from kv_reuse.pixel_change import decide_reuse_by_pixel_change
from kv_reuse.prefill import (
    FrameSnapshot,
    _build_layer_reuse_masks,
    _forward_frame_layerwise_dense,
    _forward_frame_layerwise_with_selective_reuse,
    _forward_span,
)


def average_policy_kl(
    teacher_logits_trace: List[torch.Tensor],
    counterfactual_logits_trace: List[torch.Tensor],
) -> float:
    length = min(len(teacher_logits_trace), len(counterfactual_logits_trace))
    if length == 0:
        return 0.0

    total_kl = 0.0
    for step_idx in range(length):
        teacher_logprobs = F.log_softmax(teacher_logits_trace[step_idx].float(), dim=-1)
        teacher_probs = teacher_logprobs.exp()
        cf_logprobs = F.log_softmax(counterfactual_logits_trace[step_idx].float(), dim=-1)
        step_kl = torch.sum(teacher_probs * (teacher_logprobs - cf_logprobs))
        total_kl += float(step_kl.item())

    return total_kl / length


def infer_token_spatial_grid(num_tokens: int) -> Tuple[int, int]:
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")

    root = int(num_tokens**0.5)
    for height in range(root, 0, -1):
        if num_tokens % height == 0:
            return height, num_tokens // height
    return 1, num_tokens


def compute_token_pixel_diff_scores(
    prev_frame,
    curr_frame,
    num_tokens: int,
) -> List[float]:
    grid_size = infer_token_spatial_grid(num_tokens)
    decision = decide_reuse_by_pixel_change(
        prev_frame=prev_frame,
        curr_frame=curr_frame,
        threshold=0.0,
        spatial_grid=grid_size,
        append_newline_token=False,
    )
    patch_change = decision.patch_change.detach().cpu()
    if int(patch_change.numel()) != num_tokens:
        raise RuntimeError(
            "Pixel diff token count mismatch: "
            f"expected {num_tokens}, got {int(patch_change.numel())}"
        )
    return [float(x) for x in patch_change.tolist()]


@torch.inference_mode()
def _forward_decode_span(
    model,
    cache,
    start_position: int,
    *,
    input_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
):
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Exactly one of input_ids or inputs_embeds must be provided")

    span = input_ids if input_ids is not None else inputs_embeds
    span_len = int(span.shape[1])
    cache_position = torch.arange(
        start_position,
        start_position + span_len,
        device=span.device,
        dtype=torch.long,
    )

    out = model.language_model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
        cache_position=cache_position,
        position_ids=cache_position.unsqueeze(0),
    )
    return out.past_key_values, out.logits[:, -1, :], start_position + span_len


@torch.inference_mode()
def _prepare_decode_prefix(
    model,
    live_cache,
    current_seq_len: int,
    newline_embed: torch.Tensor,
    suffix_input_ids: torch.Tensor,
):
    cache = static_cache_to_dynamic(live_cache, seq_len=current_seq_len)
    next_position = int(current_seq_len)

    cache, next_token_logits, next_position = _forward_decode_span(
        model=model,
        cache=cache,
        start_position=next_position,
        inputs_embeds=newline_embed.view(1, 1, -1),
    )

    if suffix_input_ids.shape[1] > 0:
        cache, next_token_logits, next_position = _forward_decode_span(
            model=model,
            cache=cache,
            start_position=next_position,
            input_ids=suffix_input_ids,
        )

    return cache, next_token_logits, next_position


@torch.inference_mode()
def collect_teacher_forced_policy_logits(
    model,
    live_cache,
    current_seq_len: int,
    newline_embed: torch.Tensor,
    suffix_input_ids: torch.Tensor,
    oracle_token_ids: List[int],
    device: torch.device,
) -> List[torch.Tensor]:
    cache, next_token_logits, next_position = _prepare_decode_prefix(
        model=model,
        live_cache=live_cache,
        current_seq_len=current_seq_len,
        newline_embed=newline_embed,
        suffix_input_ids=suffix_input_ids,
    )

    logits_trace: List[torch.Tensor] = []
    for token_id in oracle_token_ids:
        logits_trace.append(next_token_logits[0].detach().clone())
        step_ids = torch.tensor([[token_id]], device=device, dtype=torch.long)
        cache, next_token_logits, next_position = _forward_decode_span(
            model=model,
            cache=cache,
            start_position=next_position,
            input_ids=step_ids,
        )

    return logits_trace


@torch.inference_mode()
def generate_oracle_trace(
    model,
    tokenizer,
    live_cache,
    current_seq_len: int,
    newline_embed: torch.Tensor,
    suffix_input_ids: torch.Tensor,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> Dict[str, Any]:
    cache, next_token_logits, next_position = _prepare_decode_prefix(
        model=model,
        live_cache=live_cache,
        current_seq_len=current_seq_len,
        newline_embed=newline_embed,
        suffix_input_ids=suffix_input_ids,
    )

    stop_ids = {tokenizer.eos_token_id}
    if tokenizer.pad_token_id is not None:
        stop_ids.add(tokenizer.pad_token_id)

    token_ids: List[int] = []
    token_texts: List[str] = []
    token_logprobs: List[float] = []

    for _ in range(max_new_tokens):
        logprobs = F.log_softmax(next_token_logits, dim=-1)
        if do_sample:
            if temperature <= 0.0:
                raise ValueError("temperature must be > 0 when do_sample=True")
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        token_id = int(next_token.item())
        if token_id in stop_ids:
            break

        token_ids.append(token_id)
        token_texts.append(tokenizer.convert_ids_to_tokens([token_id])[0])
        token_logprobs.append(float(logprobs[0, token_id].item()))

        cache, next_token_logits, next_position = _forward_decode_span(
            model=model,
            cache=cache,
            start_position=next_position,
            input_ids=next_token,
        )

    oracle_text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    return {
        "oracle_text": oracle_text,
        "oracle_token_ids": token_ids,
        "oracle_token_texts": token_texts,
        "oracle_logprobs": token_logprobs,
        "oracle_steps": len(token_ids),
        "oracle_avg_nll": (-sum(token_logprobs) / len(token_logprobs)) if token_logprobs else 0.0,
    }


@torch.inference_mode()
def _build_dense_frame_cache(
    language_model,
    prefix_cache_template,
    frame_features: torch.Tensor,
    frame_positions: torch.Tensor,
):
    frame_cache = clone_cache(prefix_cache_template)
    outputs = _forward_span(
        language_model,
        frame_features.unsqueeze(0),
        frame_cache,
        cache_position=frame_positions,
    )
    return outputs.past_key_values


@torch.inference_mode()
def build_legacy_selective_zero_reuse_snapshot(
    language_model,
    prefix_cache_template,
    frame_features: torch.Tensor,
    frame_positions: torch.Tensor,
    device: torch.device,
) -> FrameSnapshot:
    frame_cache = clone_cache(prefix_cache_template)
    base_reuse_mask = torch.zeros(frame_features.shape[0], dtype=torch.bool, device=frame_features.device)
    layer_reuse_masks = _build_layer_reuse_masks(
        base_reuse_mask,
        len(language_model.model.layers),
    )
    frame_snapshot, _, _, _ = _forward_frame_layerwise_with_selective_reuse(
        language_model=language_model,
        cache=frame_cache,
        frame_features=frame_features,
        frame_cache_positions=frame_positions,
        prev_snapshot=None,
        layer_reuse_masks=layer_reuse_masks,
        device=device,
        profile_timing=False,
        force_full_attention_mask=True,
    )
    return frame_snapshot


@torch.inference_mode()
def build_full_frame_snapshot(
    language_model,
    prefix_cache_template,
    frame_features: torch.Tensor,
    frame_positions: torch.Tensor,
    device: torch.device,
) -> FrameSnapshot:
    frame_cache = clone_cache(prefix_cache_template)
    base_reuse_mask = torch.zeros(frame_features.shape[0], dtype=torch.bool, device=frame_features.device)
    layer_reuse_masks = _build_layer_reuse_masks(
        base_reuse_mask,
        len(language_model.model.layers),
    )
    frame_snapshot, _, _, _ = _forward_frame_layerwise_with_selective_reuse(
        language_model=language_model,
        cache=frame_cache,
        frame_features=frame_features,
        frame_cache_positions=frame_positions,
        prev_snapshot=None,
        layer_reuse_masks=layer_reuse_masks,
        device=device,
        profile_timing=False,
        force_full_attention_mask=False,
    )
    return frame_snapshot


@torch.inference_mode()
def build_dense_teacher_cache(
    model,
    prefix_cache_template,
    frame_features: torch.Tensor,
    frame_positions: torch.Tensor,
):
    return _build_dense_frame_cache(
        language_model=model.language_model,
        prefix_cache_template=prefix_cache_template,
        frame_features=frame_features,
        frame_positions=frame_positions,
    )


@torch.inference_mode()
def build_legacy_selective_counterfactual_cache(
    language_model,
    prefix_cache_template,
    frame_features: torch.Tensor,
    frame_positions: torch.Tensor,
    prev_full_snapshot,
    reuse_mask: torch.Tensor,
    device: torch.device,
):
    counterfactual_cache = clone_cache(prefix_cache_template)
    normalized_reuse_mask = reuse_mask.to(device=frame_features.device, dtype=torch.bool)
    layer_reuse_masks = _build_layer_reuse_masks(
        normalized_reuse_mask,
        len(language_model.model.layers),
    )
    _forward_frame_layerwise_with_selective_reuse(
        language_model=language_model,
        cache=counterfactual_cache,
        frame_features=frame_features,
        frame_cache_positions=frame_positions,
        prev_snapshot=prev_full_snapshot,
        layer_reuse_masks=layer_reuse_masks,
        device=device,
        profile_timing=False,
        force_full_attention_mask=True,
    )
    return counterfactual_cache


@torch.inference_mode()
def build_dense_teacher_snapshot(
    language_model,
    prefix_cache_template,
    frame_features: torch.Tensor,
    frame_positions: torch.Tensor,
):
    frame_cache = clone_cache(prefix_cache_template)
    decoder = language_model.model
    frame_hidden_outputs: List[torch.Tensor] = []
    hook_handles = []

    def _capture_layer_output(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        frame_hidden_outputs.append(hidden.detach().clone())

    for decoder_layer in decoder.layers:
        hook_handles.append(decoder_layer.register_forward_hook(_capture_layer_output))

    try:
        _forward_span(
            language_model,
            frame_features.unsqueeze(0),
            frame_cache,
            cache_position=frame_positions,
        )
    finally:
        for handle in hook_handles:
            handle.remove()

    frame_start = int(frame_positions[0].item()) if int(frame_positions.numel()) > 0 else 0
    frame_len = int(frame_features.shape[0])
    layer_kv = []
    for layer_idx in range(len(decoder.layers)):
        layer_key_cache, layer_value_cache = get_cache_layer_kv(frame_cache, layer_idx)
        layer_kv.append(
            (
                layer_key_cache[:, :, frame_start : frame_start + frame_len, :].clone(),
                layer_value_cache[:, :, frame_start : frame_start + frame_len, :].clone(),
            )
        )

    return FrameSnapshot(
        layer_kv=layer_kv,
        layer_hidden_states=frame_hidden_outputs,
        layer_reuse_masks=[
            torch.zeros(frame_len, dtype=torch.bool, device=frame_features.device)
            for _ in range(len(decoder.layers))
        ],
    )


FEATURE_TRACE_ORDER: List[str] = [
    "layer_input",
    "input_layernorm_output",
    "self_attn_input",
    "self_attn_attention_mask",
    "self_attn_cache_position",
    "self_attn_past_key_values_present",
    "self_attn_cache_read_success",
    "self_attn_cache_key_before",
    "self_attn_cache_value_before",
    "self_attn_cache_top_level_update_called",
    "self_attn_cache_layer_update_called",
    "self_attn_cache_key_update_input",
    "self_attn_cache_value_update_input",
    "self_attn_cache_key_after_update",
    "self_attn_cache_value_after_update",
    "self_attn_position_embeddings_cos",
    "self_attn_position_embeddings_sin",
    "q_proj_output",
    "k_proj_output",
    "v_proj_output",
    "o_proj_input",
    "o_proj_output",
    "self_attn_output",
    "post_attention_residual_input",
    "post_attention_layernorm_output",
    "mlp_input",
    "gate_proj_output",
    "up_proj_output",
    "down_proj_output",
    "mlp_output",
    "layer_output",
]


def _clone_trace_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return value.detach().cpu().clone()


def _extract_first_tensor(value) -> Optional[torch.Tensor]:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if torch.is_tensor(item):
                return item
    return None


def _slice_visible_cache_tensor(
    tensor: Optional[torch.Tensor],
    cache_position: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if torch.is_tensor(cache_position) and cache_position.numel() > 0:
        visible_cache_len = int(cache_position.max().item()) + 1
        return tensor[:, :, :visible_cache_len, :]
    return tensor


def _register_decoder_feature_trace_hooks(
    language_model,
) -> Tuple[List[Dict[str, torch.Tensor]], List[Any]]:
    decoder = language_model.model
    layer_feature_trace: List[Dict[str, torch.Tensor]] = [{} for _ in decoder.layers]
    hook_handles = []

    def _store_feature(layer_idx: int, name: str, tensor: Optional[torch.Tensor]) -> None:
        if tensor is None:
            return
        if name in layer_feature_trace[layer_idx]:
            return
        layer_feature_trace[layer_idx][name] = _clone_trace_tensor(tensor)

    def _store_scalar_feature(layer_idx: int, name: str, value: float) -> None:
        _store_feature(layer_idx, name, torch.tensor([value], dtype=torch.float32))

    for layer_idx, decoder_layer in enumerate(decoder.layers):
        def _capture_layer_input(_module, args, kwargs, *, _layer_idx=layer_idx):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = _extract_first_tensor(args[0])
            _store_feature(_layer_idx, "layer_input", hidden_states)

        def _capture_input_layernorm_output(_module, _args, output, *, _layer_idx=layer_idx):
            _store_feature(_layer_idx, "input_layernorm_output", _extract_first_tensor(output))

        def _capture_self_attn_input(_module, args, kwargs, *, _layer_idx=layer_idx):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = _extract_first_tensor(args[0])
            _store_feature(_layer_idx, "self_attn_input", hidden_states)
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is None and len(args) > 2:
                attention_mask = args[2]
            _store_feature(_layer_idx, "self_attn_attention_mask", attention_mask)
            cache_position = kwargs.get("cache_position")
            if cache_position is None and len(args) > 4:
                cache_position = args[4]
            _store_feature(_layer_idx, "self_attn_cache_position", cache_position)
            past_key_values = kwargs.get("past_key_values")
            if past_key_values is None and len(args) > 3:
                past_key_values = args[3]
            if past_key_values is not None:
                _store_scalar_feature(_layer_idx, "self_attn_past_key_values_present", 1.0)
                try:
                    cache_k, cache_v = get_cache_layer_kv(past_key_values, _layer_idx)
                    if torch.is_tensor(cache_position) and cache_position.numel() > 0:
                        visible_cache_len = int(cache_position.max().item()) + 1
                        cache_k = cache_k[:, :, :visible_cache_len, :]
                        cache_v = cache_v[:, :, :visible_cache_len, :]
                    _store_scalar_feature(_layer_idx, "self_attn_cache_read_success", 1.0)
                    _store_feature(_layer_idx, "self_attn_cache_key_before", cache_k)
                    _store_feature(_layer_idx, "self_attn_cache_value_before", cache_v)
                except Exception:
                    pass
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(args) > 1:
                position_embeddings = args[1]
            if isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) == 2:
                _store_feature(_layer_idx, "self_attn_position_embeddings_cos", position_embeddings[0])
                _store_feature(_layer_idx, "self_attn_position_embeddings_sin", position_embeddings[1])

        def _capture_q_proj_output(_module, _args, output, *, _layer_idx=layer_idx):
            _store_feature(_layer_idx, "q_proj_output", _extract_first_tensor(output))

        def _capture_k_proj_output(_module, _args, output, *, _layer_idx=layer_idx):
            _store_feature(_layer_idx, "k_proj_output", _extract_first_tensor(output))

        def _capture_v_proj_output(_module, _args, output, *, _layer_idx=layer_idx):
            _store_feature(_layer_idx, "v_proj_output", _extract_first_tensor(output))

        def _capture_o_proj_input(_module, args, _kwargs, *, _layer_idx=layer_idx):
            hidden_states = _extract_first_tensor(args[0]) if args else None
            _store_feature(_layer_idx, "o_proj_input", hidden_states)

        def _capture_o_proj_output(_module, _args, output, *, _layer_idx=layer_idx):
            _store_feature(_layer_idx, "o_proj_output", _extract_first_tensor(output))

        def _capture_self_attn_output(_module, _args, output, *, _layer_idx=layer_idx):
            _store_feature(_layer_idx, "self_attn_output", _extract_first_tensor(output))

        def _capture_post_attention_residual_input(_module, args, _kwargs, *, _layer_idx=layer_idx):
            hidden_states = _extract_first_tensor(args[0]) if args else None
            _store_feature(_layer_idx, "post_attention_residual_input", hidden_states)

        def _capture_post_attention_layernorm_output(_module, _args, output, *, _layer_idx=layer_idx):
            _store_feature(_layer_idx, "post_attention_layernorm_output", _extract_first_tensor(output))

        def _capture_mlp_input(_module, args, _kwargs, *, _layer_idx=layer_idx):
            hidden_states = _extract_first_tensor(args[0]) if args else None
            _store_feature(_layer_idx, "mlp_input", hidden_states)

        def _capture_gate_proj_output(_module, _args, output, *, _layer_idx=layer_idx):
            _store_feature(_layer_idx, "gate_proj_output", _extract_first_tensor(output))

        def _capture_up_proj_output(_module, _args, output, *, _layer_idx=layer_idx):
            _store_feature(_layer_idx, "up_proj_output", _extract_first_tensor(output))

        def _capture_down_proj_output(_module, _args, output, *, _layer_idx=layer_idx):
            _store_feature(_layer_idx, "down_proj_output", _extract_first_tensor(output))

        def _capture_mlp_output(_module, _args, output, *, _layer_idx=layer_idx):
            _store_feature(_layer_idx, "mlp_output", _extract_first_tensor(output))

        def _capture_layer_output(_module, _args, output, *, _layer_idx=layer_idx):
            _store_feature(_layer_idx, "layer_output", _extract_first_tensor(output))

        hook_handles.append(decoder_layer.register_forward_pre_hook(_capture_layer_input, with_kwargs=True))
        hook_handles.append(decoder_layer.input_layernorm.register_forward_hook(_capture_input_layernorm_output))
        hook_handles.append(decoder_layer.self_attn.register_forward_pre_hook(_capture_self_attn_input, with_kwargs=True))
        hook_handles.append(decoder_layer.self_attn.q_proj.register_forward_hook(_capture_q_proj_output))
        hook_handles.append(decoder_layer.self_attn.k_proj.register_forward_hook(_capture_k_proj_output))
        hook_handles.append(decoder_layer.self_attn.v_proj.register_forward_hook(_capture_v_proj_output))
        hook_handles.append(
            decoder_layer.self_attn.o_proj.register_forward_pre_hook(_capture_o_proj_input, with_kwargs=True)
        )
        hook_handles.append(decoder_layer.self_attn.o_proj.register_forward_hook(_capture_o_proj_output))
        hook_handles.append(decoder_layer.self_attn.register_forward_hook(_capture_self_attn_output))
        hook_handles.append(
            decoder_layer.post_attention_layernorm.register_forward_pre_hook(
                _capture_post_attention_residual_input,
                with_kwargs=True,
            )
        )
        hook_handles.append(
            decoder_layer.post_attention_layernorm.register_forward_hook(_capture_post_attention_layernorm_output)
        )
        hook_handles.append(decoder_layer.mlp.register_forward_pre_hook(_capture_mlp_input, with_kwargs=True))
        hook_handles.append(decoder_layer.mlp.gate_proj.register_forward_hook(_capture_gate_proj_output))
        hook_handles.append(decoder_layer.mlp.up_proj.register_forward_hook(_capture_up_proj_output))
        hook_handles.append(decoder_layer.mlp.down_proj.register_forward_hook(_capture_down_proj_output))
        hook_handles.append(decoder_layer.mlp.register_forward_hook(_capture_mlp_output))
        hook_handles.append(decoder_layer.register_forward_hook(_capture_layer_output))

    return layer_feature_trace, hook_handles


def _register_cache_update_trace_hooks(
    cache,
    layer_feature_trace: List[Dict[str, torch.Tensor]],
) -> List[Tuple[Any, str, Any]]:
    if not hasattr(cache, "update"):
        return []

    def _store_feature(layer_idx: int, name: str, tensor: Optional[torch.Tensor]) -> None:
        if tensor is None:
            return
        if name in layer_feature_trace[layer_idx]:
            return
        layer_feature_trace[layer_idx][name] = _clone_trace_tensor(tensor)

    def _store_scalar_feature(layer_idx: int, name: str, value: float) -> None:
        _store_feature(layer_idx, name, torch.tensor([value], dtype=torch.float32))

    original_update = cache.update

    def _trace_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None

        try:
            cache_k_before, cache_v_before = get_cache_layer_kv(self, layer_idx)
        except Exception:
            cache_k_before, cache_v_before = None, None

        _store_feature(
            layer_idx,
            "self_attn_cache_key_before",
            _slice_visible_cache_tensor(cache_k_before, cache_position),
        )
        _store_feature(
            layer_idx,
            "self_attn_cache_value_before",
            _slice_visible_cache_tensor(cache_v_before, cache_position),
        )
        _store_scalar_feature(layer_idx, "self_attn_cache_top_level_update_called", 1.0)
        _store_feature(layer_idx, "self_attn_cache_key_update_input", key_states)
        _store_feature(layer_idx, "self_attn_cache_value_update_input", value_states)

        updated_k, updated_v = original_update(key_states, value_states, layer_idx, cache_kwargs)

        _store_feature(
            layer_idx,
            "self_attn_cache_key_after_update",
            _slice_visible_cache_tensor(updated_k, cache_position),
        )
        _store_feature(
            layer_idx,
            "self_attn_cache_value_after_update",
            _slice_visible_cache_tensor(updated_v, cache_position),
        )
        return updated_k, updated_v

    cache.update = MethodType(_trace_update, cache)
    restorations: List[Tuple[Any, str, Any]] = [(cache, "update", original_update)]

    if hasattr(cache, "layers"):
        for layer_idx, layer_cache in enumerate(cache.layers):
            if not hasattr(layer_cache, "update"):
                continue

            original_layer_update = layer_cache.update

            def _trace_layer_update(
                self,
                key_states,
                value_states,
                cache_kwargs=None,
                *,
                _layer_idx=layer_idx,
                _original_layer_update=original_layer_update,
            ):
                cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
                cache_k_before = getattr(self, "keys", None)
                cache_v_before = getattr(self, "values", None)

                _store_feature(
                    _layer_idx,
                    "self_attn_cache_key_before",
                    _slice_visible_cache_tensor(cache_k_before, cache_position),
                )
                _store_feature(
                    _layer_idx,
                    "self_attn_cache_value_before",
                    _slice_visible_cache_tensor(cache_v_before, cache_position),
                )
                _store_scalar_feature(_layer_idx, "self_attn_cache_layer_update_called", 1.0)
                _store_feature(_layer_idx, "self_attn_cache_key_update_input", key_states)
                _store_feature(_layer_idx, "self_attn_cache_value_update_input", value_states)

                updated_k, updated_v = _original_layer_update(key_states, value_states, cache_kwargs)

                _store_feature(
                    _layer_idx,
                    "self_attn_cache_key_after_update",
                    _slice_visible_cache_tensor(updated_k, cache_position),
                )
                _store_feature(
                    _layer_idx,
                    "self_attn_cache_value_after_update",
                    _slice_visible_cache_tensor(updated_v, cache_position),
                )
                return updated_k, updated_v

            layer_cache.update = MethodType(_trace_layer_update, layer_cache)
            restorations.append((layer_cache, "update", original_layer_update))

    return restorations


@torch.inference_mode()
def build_dense_teacher_feature_trace(
    language_model,
    prefix_cache_template,
    frame_features: torch.Tensor,
    frame_positions: torch.Tensor,
) -> List[Dict[str, torch.Tensor]]:
    frame_cache = clone_cache(prefix_cache_template)
    layer_feature_trace, hook_handles = _register_decoder_feature_trace_hooks(language_model)
    cache_update_restorations = _register_cache_update_trace_hooks(frame_cache, layer_feature_trace)
    try:
        _forward_span(
            language_model,
            frame_features.unsqueeze(0),
            frame_cache,
            cache_position=frame_positions,
        )
    finally:
        for handle in hook_handles:
            handle.remove()
        for cache_obj, attr_name, original_method in cache_update_restorations:
            setattr(cache_obj, attr_name, original_method)
    return layer_feature_trace


@torch.inference_mode()
def build_selective_all_recompute_feature_trace(
    language_model,
    prefix_cache_template,
    frame_features: torch.Tensor,
    frame_positions: torch.Tensor,
    prev_full_snapshot,
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    frame_cache = clone_cache(prefix_cache_template)
    all_recompute_mask = torch.zeros(
        frame_features.shape[0],
        dtype=torch.bool,
        device=frame_features.device,
    )
    layer_reuse_masks = _build_layer_reuse_masks(
        all_recompute_mask,
        len(language_model.model.layers),
    )
    layer_feature_trace, hook_handles = _register_decoder_feature_trace_hooks(language_model)
    cache_update_restorations = _register_cache_update_trace_hooks(frame_cache, layer_feature_trace)
    try:
        _forward_frame_layerwise_with_selective_reuse(
            language_model=language_model,
            cache=frame_cache,
            frame_features=frame_features,
            frame_cache_positions=frame_positions,
            prev_snapshot=prev_full_snapshot,
            layer_reuse_masks=layer_reuse_masks,
            device=device,
            profile_timing=False,
            force_full_attention_mask=False,
            manual_feature_trace=layer_feature_trace,
        )
    finally:
        for handle in hook_handles:
            handle.remove()
        for cache_obj, attr_name, original_method in cache_update_restorations:
            setattr(cache_obj, attr_name, original_method)
    return layer_feature_trace


def _summarize_tensor_abs_diff(
    reference_tensor: Optional[torch.Tensor],
    probe_tensor: Optional[torch.Tensor],
) -> Dict[str, Any]:
    if reference_tensor is None and probe_tensor is None:
        return {
            "present_in_reference": False,
            "present_in_probe": False,
            "mean_abs_diff": 0.0,
            "max_abs_diff": 0.0,
        }
    if reference_tensor is None or probe_tensor is None:
        return {
            "present_in_reference": reference_tensor is not None,
            "present_in_probe": probe_tensor is not None,
            "reference_shape": list(reference_tensor.shape) if reference_tensor is not None else None,
            "probe_shape": list(probe_tensor.shape) if probe_tensor is not None else None,
            "mean_abs_diff": None,
            "max_abs_diff": None,
        }
    if tuple(reference_tensor.shape) != tuple(probe_tensor.shape):
        return {
            "present_in_reference": True,
            "present_in_probe": True,
            "reference_shape": list(reference_tensor.shape),
            "probe_shape": list(probe_tensor.shape),
            "mean_abs_diff": None,
            "max_abs_diff": None,
        }

    reference_float = reference_tensor.float()
    probe_float = probe_tensor.float()
    shared_inf_mask = torch.isinf(reference_float) & torch.isinf(probe_float)
    if shared_inf_mask.any():
        reference_float = torch.where(shared_inf_mask, torch.zeros_like(reference_float), reference_float)
        probe_float = torch.where(shared_inf_mask, torch.zeros_like(probe_float), probe_float)
    abs_diff = (reference_float - probe_float).abs()
    abs_diff = torch.nan_to_num(abs_diff, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
    return {
        "present_in_reference": True,
        "present_in_probe": True,
        "reference_shape": list(reference_tensor.shape),
        "probe_shape": list(probe_tensor.shape),
        "mean_abs_diff": float(abs_diff.mean().item()) if abs_diff.numel() > 0 else 0.0,
        "max_abs_diff": float(abs_diff.max().item()) if abs_diff.numel() > 0 else 0.0,
    }


def _collect_feature_present_layers(
    trace: List[Dict[str, torch.Tensor]],
    feature_name: str,
) -> List[int]:
    present_layers: List[int] = []
    for layer_idx, layer_features in enumerate(trace):
        if layer_features.get(feature_name) is not None:
            present_layers.append(int(layer_idx))
    return present_layers


def _summarize_feature_presence_layers(
    reference_trace: List[Dict[str, torch.Tensor]],
    probe_trace: List[Dict[str, torch.Tensor]],
    feature_names: List[str],
) -> Dict[str, Dict[str, Any]]:
    presence_summary: Dict[str, Dict[str, Any]] = {}
    for feature_name in feature_names:
        reference_layers = _collect_feature_present_layers(reference_trace, feature_name)
        probe_layers = _collect_feature_present_layers(probe_trace, feature_name)
        probe_layer_set = set(probe_layers)
        reference_layer_set = set(reference_layers)
        presence_summary[feature_name] = {
            "reference_layers": reference_layers,
            "probe_layers": probe_layers,
            "reference_count": len(reference_layers),
            "probe_count": len(probe_layers),
            "reference_only_layers": [layer_idx for layer_idx in reference_layers if layer_idx not in probe_layer_set],
            "probe_only_layers": [layer_idx for layer_idx in probe_layers if layer_idx not in reference_layer_set],
        }
    return presence_summary


def summarize_decoder_feature_trace_diff(
    reference_trace: List[Dict[str, torch.Tensor]],
    probe_trace: List[Dict[str, torch.Tensor]],
) -> Dict[str, Any]:
    compared_layers = min(len(reference_trace), len(probe_trace))
    layer_stage_diffs: List[Dict[str, Any]] = []
    first_divergence = None
    max_abs_diff_feature = None

    for layer_idx in range(compared_layers):
        reference_features = reference_trace[layer_idx]
        probe_features = probe_trace[layer_idx]
        feature_names = list(FEATURE_TRACE_ORDER)
        feature_names.extend(
            sorted(
                name
                for name in set(reference_features) | set(probe_features)
                if name not in FEATURE_TRACE_ORDER
            )
        )

        feature_mean_abs_diff: Dict[str, float | None] = {}
        feature_max_abs_diff: Dict[str, float | None] = {}
        feature_presence: Dict[str, Dict[str, Any]] = {}
        first_divergent_feature = None
        max_diff_feature = None
        max_diff_value = -1.0

        for feature_name in feature_names:
            diff_summary = _summarize_tensor_abs_diff(
                reference_features.get(feature_name),
                probe_features.get(feature_name),
            )
            feature_mean_abs_diff[feature_name] = diff_summary.get("mean_abs_diff")
            feature_max_abs_diff[feature_name] = diff_summary.get("max_abs_diff")
            feature_presence[feature_name] = {
                "present_in_reference": bool(diff_summary.get("present_in_reference", False)),
                "present_in_probe": bool(diff_summary.get("present_in_probe", False)),
                "reference_shape": diff_summary.get("reference_shape"),
                "probe_shape": diff_summary.get("probe_shape"),
            }

            feature_max = diff_summary.get("max_abs_diff")
            if feature_max is not None and feature_max > 0.0 and first_divergent_feature is None:
                first_divergent_feature = feature_name
            if feature_max is not None and feature_max > max_diff_value:
                max_diff_value = feature_max
                max_diff_feature = feature_name
            if (
                feature_max is not None
                and feature_max > 0.0
                and first_divergence is None
            ):
                first_divergence = {
                    "layer_idx": int(layer_idx),
                    "feature": feature_name,
                    "mean_abs_diff": diff_summary.get("mean_abs_diff"),
                    "max_abs_diff": feature_max,
                }
            if feature_max is not None and (
                max_abs_diff_feature is None or feature_max > float(max_abs_diff_feature["max_abs_diff"])
            ):
                max_abs_diff_feature = {
                    "layer_idx": int(layer_idx),
                    "feature": feature_name,
                    "mean_abs_diff": diff_summary.get("mean_abs_diff"),
                    "max_abs_diff": feature_max,
                }

        layer_stage_diffs.append(
            {
                "layer_idx": int(layer_idx),
                "feature_mean_abs_diff": feature_mean_abs_diff,
                "feature_max_abs_diff": feature_max_abs_diff,
                "feature_presence": feature_presence,
                "first_divergent_feature": first_divergent_feature,
                "max_diff_feature": max_diff_feature,
                "max_diff_value": max_diff_value if max_diff_value >= 0.0 else None,
            }
        )

    compact_presence_feature_names = [
        "self_attn_past_key_values_present",
        "self_attn_cache_read_success",
        "self_attn_cache_top_level_update_called",
        "self_attn_cache_layer_update_called",
        "self_attn_cache_key_before",
        "self_attn_cache_key_update_input",
        "self_attn_cache_key_after_update",
        "self_attn_cache_value_before",
        "self_attn_cache_value_update_input",
        "self_attn_cache_value_after_update",
    ]

    return {
        "reference_layers": len(reference_trace),
        "probe_layers": len(probe_trace),
        "compared_layers": compared_layers,
        "feature_order": FEATURE_TRACE_ORDER,
        "first_divergence": first_divergence,
        "max_abs_diff_feature": max_abs_diff_feature,
        "compact_feature_presence": _summarize_feature_presence_layers(
            reference_trace,
            probe_trace,
            compact_presence_feature_names,
        ),
        "layer_stage_diffs": layer_stage_diffs,
    }


@torch.inference_mode()
def build_counterfactual_cache(
    language_model,
    prefix_cache_template,
    frame_features: torch.Tensor,
    frame_positions: torch.Tensor,
    prev_full_snapshot,
    reuse_mask: torch.Tensor,
    device: torch.device,
):
    counterfactual_cache = clone_cache(prefix_cache_template)
    normalized_reuse_mask = reuse_mask.to(device=frame_features.device, dtype=torch.bool)
    layer_reuse_masks = _build_layer_reuse_masks(
        normalized_reuse_mask,
        len(language_model.model.layers),
    )
    _forward_frame_layerwise_with_selective_reuse(
        language_model=language_model,
        cache=counterfactual_cache,
        frame_features=frame_features,
        frame_cache_positions=frame_positions,
        prev_snapshot=prev_full_snapshot,
        layer_reuse_masks=layer_reuse_masks,
        device=device,
        profile_timing=False,
        force_full_attention_mask=False,
    )
    return counterfactual_cache


@torch.inference_mode()
def build_selective_all_recompute_cache(
    language_model,
    prefix_cache_template,
    frame_features: torch.Tensor,
    frame_positions: torch.Tensor,
    prev_full_snapshot,
    device: torch.device,
):
    all_recompute_mask = torch.zeros(
        frame_features.shape[0],
        dtype=torch.bool,
        device=frame_features.device,
    )
    return build_counterfactual_cache(
        language_model=language_model,
        prefix_cache_template=prefix_cache_template,
        frame_features=frame_features,
        frame_positions=frame_positions,
        prev_full_snapshot=prev_full_snapshot,
        reuse_mask=all_recompute_mask,
        device=device,
    )


@torch.inference_mode()
def build_counterfactual_outputs(
    model,
    prefix_cache_template,
    frame_features: torch.Tensor,
    frame_positions: torch.Tensor,
    prev_full_snapshot,
    reuse_mask: torch.Tensor,
    newline_embed: torch.Tensor,
    suffix_input_ids: torch.Tensor,
    oracle_token_ids: List[int],
    frame_seq_len: int,
    device: torch.device,
):
    counterfactual_cache = build_counterfactual_cache(
        language_model=model.language_model,
        prefix_cache_template=prefix_cache_template,
        frame_features=frame_features,
        frame_positions=frame_positions,
        prev_full_snapshot=prev_full_snapshot,
        reuse_mask=reuse_mask,
        device=device,
    )
    counterfactual_logits_trace = collect_teacher_forced_policy_logits(
        model=model,
        live_cache=counterfactual_cache,
        current_seq_len=frame_seq_len,
        newline_embed=newline_embed,
        suffix_input_ids=suffix_input_ids,
        oracle_token_ids=oracle_token_ids,
        device=device,
    )
    return counterfactual_cache, counterfactual_logits_trace


@torch.inference_mode()
def build_teacher_reference_cache(
    teacher_reference: str,
    *,
    model,
    prefix_cache_template,
    frame_features: torch.Tensor,
    frame_positions: torch.Tensor,
    prev_full_snapshot,
    device: torch.device,
):
    normalized_teacher_reference = normalize_teacher_reference(teacher_reference)
    if normalized_teacher_reference == "dense":
        return build_dense_teacher_cache(
            model=model,
            prefix_cache_template=prefix_cache_template,
            frame_features=frame_features,
            frame_positions=frame_positions,
        )

    return build_selective_all_recompute_cache(
        language_model=model.language_model,
        prefix_cache_template=prefix_cache_template,
        frame_features=frame_features,
        frame_positions=frame_positions,
        prev_full_snapshot=prev_full_snapshot,
        device=device,
    )


def explain_legacy_selective_all_recompute_divergence(language_model) -> Dict[str, Any]:
    layer_attention_types = [
        str(getattr(layer, "attention_type", "full_attention"))
        for layer in language_model.model.layers
    ]
    non_full_attention_layers = [
        {
            "layer_idx": int(layer_idx),
            "attention_type": attention_type,
        }
        for layer_idx, attention_type in enumerate(layer_attention_types)
        if attention_type != "full_attention"
    ]
    likely_root_cause = (
        "legacy selective all-recompute forced a dense-style mask while still using the partial replay loop; "
        "the current implementation keeps no-reuse / all-recompute on the same manual replay path, but feeds the "
        "exact dense mask, cache positions, and rotary embeddings into that shared layer loop."
    )
    return {
        "effective_impl": "selective_layerwise_manual_replay_with_dense_inputs_for_no_reuse",
        "legacy_impl": "selective_layerwise_partial_global_with_forced_full_attention_mask",
        "layer_attention_types": layer_attention_types,
        "non_full_attention_layers": non_full_attention_layers,
        "likely_root_cause": likely_root_cause,
    }
