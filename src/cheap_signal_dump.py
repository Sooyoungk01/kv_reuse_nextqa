from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

from kv_reuse.cache_ops import clone_cache, get_cache_layer_kv, static_cache_to_dynamic


EPS = 1e-12
TEMPORAL_WINDOW = 3
EARLY_MID_BLOCK_LAYER_RANGE = (10, 18)
TOPK_IMAGE_ATTENTION_OVERLAP_KS = (5, 10, 20)
PROJECTOR_TOPK_WEIGHT_K = 20
BASE_COMPARISON_SIGNAL_KEYS = [
    "hidden_state_mean_layer_l2_scores",
    "attention_cosine_distance_scores",
]
EXTRACTED_SIGNAL_KEYS = [
    "hidden_state_mean_layer_l2_scores",
    "attention_cosine_distance_scores",
    "text_token_decoder_hidden_delta_scores",
    "text_to_image_attention_cosine_distance_scores",
    "projector_output_delta_scores",
    "projector_cosine_distance_scores",
    "projector_topk_attention_weighted_delta_scores",
    "topk_image_attention_overlap_k5_scores",
    "topk_image_attention_overlap_k10_scores",
    "topk_image_attention_overlap_k20_scores",
    "cross_modal_gated_scores",
    "early_mid_decoder_block_hidden_delta_scores",
    "last_text_token_hidden_delta_scores",
    "high_attention_weighted_image_token_delta_scores",
    "hidden_state_mean_layer_l2_jump_scores",
    "hidden_state_mean_layer_l2_ratio_to_recent_mean_scores",
    "hidden_state_mean_layer_l2_recent_max_gap_scores",
    "attention_cosine_distance_jump_scores",
    "attention_cosine_distance_ratio_to_recent_mean_scores",
    "attention_cosine_distance_recent_max_gap_scores",
    "hidden_attention_product_scores",
    "hidden_attention_zsum_scores",
    "hidden_attention_zmax_scores",
]
RECOMMENDED_TOPK_SIGNAL_KEYS = [
    "text_token_decoder_hidden_delta_scores",
    "text_to_image_attention_cosine_distance_scores",
    "projector_output_delta_scores",
    "projector_cosine_distance_scores",
    "projector_topk_attention_weighted_delta_scores",
    "topk_image_attention_overlap_k10_scores",
    "cross_modal_gated_scores",
    "early_mid_decoder_block_hidden_delta_scores",
    "last_text_token_hidden_delta_scores",
    "high_attention_weighted_image_token_delta_scores",
    *BASE_COMPARISON_SIGNAL_KEYS,
]
RECOMMENDED_LOGISTIC_FEATURE_KEYS = [
    "hidden_state_mean_layer_l2_scores",
    "attention_cosine_distance_scores",
    "text_token_decoder_hidden_delta_scores",
    "text_to_image_attention_cosine_distance_scores",
    "projector_output_delta_scores",
    "projector_cosine_distance_scores",
    "projector_topk_attention_weighted_delta_scores",
    "topk_image_attention_overlap_k10_scores",
    "cross_modal_gated_scores",
    "early_mid_decoder_block_hidden_delta_scores",
    "last_text_token_hidden_delta_scores",
    "high_attention_weighted_image_token_delta_scores",
    "hidden_state_mean_layer_l2_jump_scores",
    "attention_cosine_distance_jump_scores",
    "hidden_state_mean_layer_l2_recent_max_gap_scores",
    "attention_cosine_distance_recent_max_gap_scores",
    "hidden_attention_product_scores",
    "hidden_attention_zmax_scores",
]
METRIC_DEFINITIONS: Dict[str, str] = {
    "hidden_state_mean_layer_l2_scores": (
        "Mean L2 distance across selective-all-recompute decoder layer outputs between previous-frame and current-frame image tokens."
    ),
    "attention_cosine_distance_scores": (
        "Mean cosine distance across selective-all-recompute decoder self-attention rows between previous-frame and current-frame image-token queries."
    ),
    "text_token_decoder_hidden_delta_scores": (
        "Text-token-only decoder hidden delta localized onto image tokens by weighting prompt-text hidden-state L2 changes with text-to-image attention mass."
    ),
    "text_to_image_attention_cosine_distance_scores": (
        "Per-image-token cosine distance between previous-frame and current-frame text-to-image attention columns over prompt-text queries and decoder layers."
    ),
    "projector_output_delta_scores": (
        "L2 distance between previous-frame and current-frame frame_features, using the exact frame-feature path consumed by run_drift_oracle.py."
    ),
    "projector_cosine_distance_scores": (
        "Per-image-token cosine distance between previous-frame and current-frame frame_features, giving a projector-side distribution-aware drift score."
    ),
    "projector_topk_attention_weighted_delta_scores": (
        "Projector-output L2 delta reweighted by the average prompt-text attention distribution and restricted to the top-20 most attended image tokens."
    ),
    "topk_image_attention_overlap_k5_scores": (
        "Per-image-token top-k attention-overlap instability for K=5, measured as how often the token enters or leaves prompt-text top-k image attention sets across layers."
    ),
    "topk_image_attention_overlap_k10_scores": (
        "Per-image-token top-k attention-overlap instability for K=10, measured as how often the token enters or leaves prompt-text top-k image attention sets across layers."
    ),
    "topk_image_attention_overlap_k20_scores": (
        "Per-image-token top-k attention-overlap instability for K=20, measured as how often the token enters or leaves prompt-text top-k image attention sets across layers."
    ),
    "cross_modal_gated_scores": (
        "Cross-modal gated score combining z-scored text-token hidden delta with a sigmoid gate from z-scored text-to-image attention cosine distance."
    ),
    "early_mid_decoder_block_hidden_delta_scores": (
        "Mean L2 hidden-state delta over selective-all-recompute decoder layers 10-18 when available."
    ),
    "last_text_token_hidden_delta_scores": (
        "Last prompt-text token hidden delta localized onto image tokens by weighting last-token hidden-state L2 changes with last-token text-to-image attention mass."
    ),
    "high_attention_weighted_image_token_delta_scores": (
        "Projector-output image-token delta weighted by combined previous/current prompt-text attention mass, emphasizing image tokens the prompt text focuses on."
    ),
    "hidden_state_mean_layer_l2_jump_scores": (
        "Current hidden_state_mean_layer_l2 minus an EMA-3 baseline from recent transitions, computed per token."
    ),
    "hidden_state_mean_layer_l2_ratio_to_recent_mean_scores": (
        "Current hidden_state_mean_layer_l2 divided by the mean of the last 3 transitions, computed per token."
    ),
    "hidden_state_mean_layer_l2_recent_max_gap_scores": (
        "Current hidden_state_mean_layer_l2 minus the maximum of the last 3 transitions, computed per token."
    ),
    "attention_cosine_distance_jump_scores": (
        "Current attention_cosine_distance minus an EMA-3 baseline from recent transitions, computed per token."
    ),
    "attention_cosine_distance_ratio_to_recent_mean_scores": (
        "Current attention_cosine_distance divided by the mean of the last 3 transitions, computed per token."
    ),
    "attention_cosine_distance_recent_max_gap_scores": (
        "Current attention_cosine_distance minus the maximum of the last 3 transitions, computed per token."
    ),
    "hidden_attention_product_scores": (
        "Cross-feature interaction: hidden_state_mean_layer_l2 multiplied by attention_cosine_distance, per token."
    ),
    "hidden_attention_zsum_scores": (
        "Cross-feature interaction: z-scored hidden_state_mean_layer_l2 plus z-scored attention_cosine_distance, per token."
    ),
    "hidden_attention_zmax_scores": (
        "Cross-feature interaction: max(z_hidden_state_mean_layer_l2, z_attention_cosine_distance), per token."
    ),
}


@dataclass
class CheapSignalState:
    frame_features: torch.Tensor
    image_hidden_layers: torch.Tensor
    image_attention_rows: torch.Tensor
    prompt_text_hidden_layers: torch.Tensor
    prompt_text_to_image_attention: torch.Tensor


def rounded_list(tensor: torch.Tensor, decimals: int = 8) -> list[Any]:
    return tensor.detach().cpu().to(dtype=torch.float32).numpy().round(decimals=decimals).tolist()


def normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.to(dtype=torch.float32)
    matrix = torch.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = matrix.clamp_min(0.0)
    denom = matrix.sum(dim=-1, keepdim=True).clamp_min(EPS)
    return matrix / denom


def cosine_distance_columns(prev_matrix: torch.Tensor, curr_matrix: torch.Tensor) -> torch.Tensor:
    prev_cols = prev_matrix.transpose(0, 1).to(dtype=torch.float32)
    curr_cols = curr_matrix.transpose(0, 1).to(dtype=torch.float32)
    similarity = F.cosine_similarity(prev_cols, curr_cols, dim=-1, eps=1e-8).clamp(-1.0, 1.0)
    return 1.0 - similarity


def cosine_distance_rows(prev_rows: torch.Tensor, curr_rows: torch.Tensor) -> torch.Tensor:
    prev_probs = normalize_rows(prev_rows)
    curr_probs = normalize_rows(curr_rows)
    similarity = F.cosine_similarity(prev_probs, curr_probs, dim=-1, eps=1e-8).clamp(-1.0, 1.0)
    return 1.0 - similarity


def hidden_state_l2_distance_layers(prev_hidden: torch.Tensor, curr_hidden: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(curr_hidden.to(dtype=torch.float32) - prev_hidden.to(dtype=torch.float32), dim=-1)


def combine_layer_scores(layer_scores: torch.Tensor) -> torch.Tensor:
    return layer_scores.mean(dim=0)


def recent_history(history: list[torch.Tensor], max_len: int = TEMPORAL_WINDOW) -> list[torch.Tensor]:
    return history[-max_len:]


def ema3_baseline(history: list[torch.Tensor], fallback: torch.Tensor) -> torch.Tensor:
    recent = recent_history(history)
    if not recent:
        return fallback
    stack = torch.stack([item.to(dtype=torch.float32) for item in recent], dim=0)
    weights = torch.tensor([0.25, 0.5, 1.0], dtype=torch.float32)[-stack.shape[0] :]
    weights = weights / weights.sum().clamp_min(EPS)
    return (stack * weights.view(-1, 1)).sum(dim=0)


def recent_mean_baseline(history: list[torch.Tensor], fallback: torch.Tensor) -> torch.Tensor:
    recent = recent_history(history)
    if not recent:
        return fallback
    stack = torch.stack([item.to(dtype=torch.float32) for item in recent], dim=0)
    return stack.mean(dim=0)


def recent_max_baseline(history: list[torch.Tensor], fallback: torch.Tensor) -> torch.Tensor:
    recent = recent_history(history)
    if not recent:
        return fallback
    stack = torch.stack([item.to(dtype=torch.float32) for item in recent], dim=0)
    return stack.max(dim=0).values


def standardize_scores(values: torch.Tensor) -> torch.Tensor:
    values = values.to(dtype=torch.float32)
    std = values.std(unbiased=False).clamp_min(EPS)
    return (values - values.mean()) / std


def _find_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> list[int] | None:
    if not subsequence:
        return None
    width = len(subsequence)
    for start_idx in range(len(sequence) - width + 1):
        if list(sequence[start_idx : start_idx + width]) == list(subsequence):
            return list(range(start_idx, start_idx + width))
    return None


def locate_prompt_text_token_positions(tokenizer, suffix_input_ids: torch.Tensor, text_prompt: str) -> torch.Tensor:
    suffix_ids = [int(x) for x in suffix_input_ids[0].detach().cpu().tolist()]
    prompt_ids = tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
    prompt_positions = _find_subsequence(suffix_ids, prompt_ids)
    if prompt_positions is None:
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        prompt_positions = [idx for idx, token_id in enumerate(suffix_ids) if int(token_id) not in special_ids]
    if not prompt_positions:
        raise RuntimeError("Could not locate prompt-text token positions inside suffix_input_ids.")
    return torch.tensor([idx + 1 for idx in prompt_positions], dtype=torch.long, device=suffix_input_ids.device)


def init_metric_history() -> dict[str, list[torch.Tensor]]:
    return {metric_key: [] for metric_key in BASE_COMPARISON_SIGNAL_KEYS}


def update_metric_history(metric_history: dict[str, list[torch.Tensor]], current_metrics: dict[str, torch.Tensor]) -> None:
    for metric_key in BASE_COMPARISON_SIGNAL_KEYS:
        history = metric_history[metric_key]
        history.append(current_metrics[metric_key].detach().to(dtype=torch.float32))
        if len(history) > TEMPORAL_WINDOW:
            del history[:-TEMPORAL_WINDOW]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
    if sin.dim() == 2:
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def _effective_decoder_layer_scores(layerwise_scores: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if layerwise_scores.shape[0] > 1:
        return layerwise_scores[1:], True
    return layerwise_scores, False


def _select_decoder_block(
    effective_layer_scores: torch.Tensor,
    requested_start: int,
    requested_end: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    num_layers = int(effective_layer_scores.shape[0])
    if num_layers == 0:
        raise RuntimeError("Cannot select a decoder block from zero effective layers.")
    actual_start = max(0, min(int(requested_start), num_layers - 1))
    actual_end = max(actual_start, min(int(requested_end), num_layers - 1))
    return effective_layer_scores[actual_start : actual_end + 1], {
        "requested": [int(requested_start), int(requested_end)],
        "actual": [int(actual_start), int(actual_end)],
        "count": int(actual_end - actual_start + 1),
    }


def _attention_mask_to_float(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    mask = mask.to(dtype=torch.float32)
    if mask.dim() == 2:
        return mask.unsqueeze(0).unsqueeze(0)
    if mask.dim() == 3:
        return mask.unsqueeze(1)
    return mask


def _materialize_frame_static_cache(prefix_cache_template, full_snapshot, frame_positions: torch.Tensor):
    frame_cache = clone_cache(prefix_cache_template)
    cache_positions = frame_positions.to(dtype=torch.long)
    for layer_idx, (layer_keys, layer_values) in enumerate(full_snapshot.layer_kv):
        cache_keys, cache_values = get_cache_layer_kv(frame_cache, layer_idx)
        try:
            cache_keys.index_copy_(2, cache_positions, layer_keys)
            cache_values.index_copy_(2, cache_positions, layer_values)
        except NotImplementedError:
            cache_keys[:, :, cache_positions] = layer_keys
            cache_values[:, :, cache_positions] = layer_values
    return frame_cache


def reconstruct_attention_rows(language_model, layer_feature_trace: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for layer_idx, layer_trace in enumerate(layer_feature_trace):
        attn = language_model.model.layers[layer_idx].self_attn
        q_proj = layer_trace.get("q_proj_output")
        cos = layer_trace.get("self_attn_position_embeddings_cos")
        sin = layer_trace.get("self_attn_position_embeddings_sin")
        key_cache = layer_trace.get("self_attn_cache_key_after_update")
        attn_mask = layer_trace.get("self_attn_attention_mask")
        if q_proj is None or cos is None or sin is None or key_cache is None:
            raise RuntimeError(
                "Selective-all-recompute feature trace is missing the tensors needed to reconstruct attention rows "
                f"at layer {layer_idx}."
            )

        q_proj = q_proj.to(dtype=torch.float32)
        cos = cos.to(dtype=torch.float32)
        sin = sin.to(dtype=torch.float32)
        key_cache = key_cache.to(dtype=torch.float32)
        attn_mask = _attention_mask_to_float(attn_mask)

        batch_size, query_len, _ = q_proj.shape
        num_heads = int(attn.num_heads)
        head_dim = int(attn.head_dim)
        num_key_value_groups = int(getattr(attn, "num_key_value_groups", 1))

        query_states = q_proj.view(batch_size, query_len, num_heads, head_dim).transpose(1, 2)
        query_states = _apply_rotary_pos_emb(query_states, cos, sin)

        if key_cache.shape[1] != num_heads:
            key_states = _repeat_kv(key_cache, num_key_value_groups)
        else:
            key_states = key_cache

        attn_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(float(head_dim))
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask.to(dtype=torch.float32)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0, posinf=0.0, neginf=0.0)
        rows.append(normalize_rows(attn_probs.mean(dim=1)[0]))

    return torch.stack(rows, dim=0)


def _collect_prompt_text_state(
    *,
    language_model,
    prefix_cache_template,
    full_snapshot,
    frame_positions: torch.Tensor,
    frame_seq_len: int,
    newline_embed: torch.Tensor,
    suffix_input_ids: torch.Tensor,
    prompt_text_token_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    frame_cache = _materialize_frame_static_cache(prefix_cache_template, full_snapshot, frame_positions)
    live_cache = static_cache_to_dynamic(frame_cache, seq_len=frame_seq_len)

    suffix_embeds = language_model.get_input_embeddings()(suffix_input_ids)
    prompt_span_embeds = torch.cat((newline_embed.view(1, 1, -1), suffix_embeds), dim=1)
    cache_position = torch.arange(
        frame_seq_len,
        frame_seq_len + prompt_span_embeds.shape[1],
        device=prompt_span_embeds.device,
        dtype=torch.long,
    )
    outputs = language_model(
        inputs_embeds=prompt_span_embeds,
        past_key_values=live_cache,
        use_cache=True,
        return_dict=True,
        cache_position=cache_position,
        position_ids=cache_position.unsqueeze(0),
        output_hidden_states=True,
        output_attentions=True,
    )
    if outputs.hidden_states is None or len(outputs.hidden_states) <= 1:
        raise RuntimeError("Prompt-text trace did not return decoder hidden states.")
    if outputs.attentions is None or any(attn is None for attn in outputs.attentions):
        raise RuntimeError("Prompt-text trace did not return decoder attentions.")

    text_positions_gpu = prompt_text_token_positions.to(device=prompt_span_embeds.device, dtype=torch.long)
    text_positions_cpu = text_positions_gpu.detach().cpu()
    text_hidden_layers = torch.stack(
        [layer_hidden[0].index_select(0, text_positions_gpu).detach().cpu().to(dtype=torch.float32) for layer_hidden in outputs.hidden_states[1:]],
        dim=0,
    )

    image_key_start = int(frame_positions[0].item()) if int(frame_positions.numel()) > 0 else 0
    image_key_len = int(frame_positions.numel())
    attn_layers: list[torch.Tensor] = []
    for layer_attn in outputs.attentions:
        layer_attn = layer_attn[0].detach().cpu().to(dtype=torch.float32).mean(dim=0)
        text_to_image = layer_attn.index_select(0, text_positions_cpu).narrow(1, image_key_start, image_key_len)
        attn_layers.append(normalize_rows(text_to_image))

    return text_hidden_layers, torch.stack(attn_layers, dim=0)


def build_cheap_signal_state(
    *,
    language_model,
    prefix_cache_template,
    frame_positions: torch.Tensor,
    frame_seq_len: int,
    frame_features: torch.Tensor,
    full_snapshot,
    selective_feature_trace: List[Dict[str, torch.Tensor]],
    newline_embed: torch.Tensor,
    suffix_input_ids: torch.Tensor,
    prompt_text_token_positions: torch.Tensor,
) -> CheapSignalState:
    image_hidden_layers = torch.stack(
        [layer_hidden[0].detach().cpu().to(dtype=torch.float32) for layer_hidden in full_snapshot.layer_hidden_states],
        dim=0,
    )
    prompt_text_hidden_layers, prompt_text_to_image_attention = _collect_prompt_text_state(
        language_model=language_model,
        prefix_cache_template=prefix_cache_template,
        full_snapshot=full_snapshot,
        frame_positions=frame_positions,
        frame_seq_len=frame_seq_len,
        newline_embed=newline_embed,
        suffix_input_ids=suffix_input_ids,
        prompt_text_token_positions=prompt_text_token_positions,
    )
    return CheapSignalState(
        frame_features=frame_features.detach().cpu().to(dtype=torch.float32),
        image_hidden_layers=image_hidden_layers,
        image_attention_rows=reconstruct_attention_rows(language_model, selective_feature_trace),
        prompt_text_hidden_layers=prompt_text_hidden_layers,
        prompt_text_to_image_attention=prompt_text_to_image_attention,
    )


def compute_temporal_metric_bundle(
    current_metrics: dict[str, torch.Tensor],
    metric_history: dict[str, list[torch.Tensor]],
) -> dict[str, torch.Tensor]:
    hidden_current = current_metrics["hidden_state_mean_layer_l2_scores"].to(dtype=torch.float32)
    attn_current = current_metrics["attention_cosine_distance_scores"].to(dtype=torch.float32)

    hidden_ema = ema3_baseline(metric_history["hidden_state_mean_layer_l2_scores"], hidden_current)
    hidden_recent_mean = recent_mean_baseline(metric_history["hidden_state_mean_layer_l2_scores"], hidden_current)
    hidden_recent_max = recent_max_baseline(metric_history["hidden_state_mean_layer_l2_scores"], hidden_current)
    attn_ema = ema3_baseline(metric_history["attention_cosine_distance_scores"], attn_current)
    attn_recent_mean = recent_mean_baseline(metric_history["attention_cosine_distance_scores"], attn_current)
    attn_recent_max = recent_max_baseline(metric_history["attention_cosine_distance_scores"], attn_current)

    return {
        "hidden_state_mean_layer_l2_jump_scores": hidden_current - hidden_ema,
        "hidden_state_mean_layer_l2_ratio_to_recent_mean_scores": hidden_current / hidden_recent_mean.clamp_min(EPS),
        "hidden_state_mean_layer_l2_recent_max_gap_scores": hidden_current - hidden_recent_max,
        "attention_cosine_distance_jump_scores": attn_current - attn_ema,
        "attention_cosine_distance_ratio_to_recent_mean_scores": attn_current / attn_recent_mean.clamp_min(EPS),
        "attention_cosine_distance_recent_max_gap_scores": attn_current - attn_recent_max,
    }


def compute_interaction_metric_bundle(current_metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    hidden_current = current_metrics["hidden_state_mean_layer_l2_scores"].to(dtype=torch.float32)
    attn_current = current_metrics["attention_cosine_distance_scores"].to(dtype=torch.float32)
    hidden_z = standardize_scores(hidden_current)
    attn_z = standardize_scores(attn_current)
    return {
        "hidden_attention_product_scores": hidden_current * attn_current,
        "hidden_attention_zsum_scores": hidden_z + attn_z,
        "hidden_attention_zmax_scores": torch.maximum(hidden_z, attn_z),
    }


def _topk_membership_change_scores(prev_rows: torch.Tensor, curr_rows: torch.Tensor, k: int) -> torch.Tensor:
    prev_rows = normalize_rows(prev_rows)
    curr_rows = normalize_rows(curr_rows)
    num_rows, num_tokens = prev_rows.shape
    effective_k = max(1, min(int(k), int(num_tokens)))
    scores = torch.zeros(num_tokens, dtype=torch.float32)
    for row_idx in range(num_rows):
        prev_top = torch.topk(prev_rows[row_idx], k=effective_k, largest=True).indices
        curr_top = torch.topk(curr_rows[row_idx], k=effective_k, largest=True).indices
        prev_mask = torch.zeros(num_tokens, dtype=torch.bool)
        curr_mask = torch.zeros(num_tokens, dtype=torch.bool)
        prev_mask[prev_top] = True
        curr_mask[curr_top] = True
        scores += torch.logical_xor(prev_mask, curr_mask).to(dtype=torch.float32)
    return scores / max(num_rows, 1)


def compute_exact_path_metric_bundle(
    prev_state: CheapSignalState,
    curr_state: CheapSignalState,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    layerwise_l2 = hidden_state_l2_distance_layers(prev_state.image_hidden_layers, curr_state.image_hidden_layers)
    effective_layers, dropped_input_embedding_layer = _effective_decoder_layer_scores(layerwise_l2)
    early_mid_block, block_meta = _select_decoder_block(
        effective_layers,
        requested_start=EARLY_MID_BLOCK_LAYER_RANGE[0],
        requested_end=EARLY_MID_BLOCK_LAYER_RANGE[1],
    )

    attention_layer_scores = cosine_distance_rows(
        prev_state.image_attention_rows.reshape(-1, prev_state.image_attention_rows.shape[-1]),
        curr_state.image_attention_rows.reshape(-1, curr_state.image_attention_rows.shape[-1]),
    ).view(curr_state.image_attention_rows.shape[0], curr_state.image_attention_rows.shape[1])

    prev_text_attn = prev_state.prompt_text_to_image_attention.to(dtype=torch.float32)
    curr_text_attn = curr_state.prompt_text_to_image_attention.to(dtype=torch.float32)
    prev_text_attn_rows = normalize_rows(prev_text_attn.reshape(-1, prev_text_attn.shape[-1])).view_as(prev_text_attn)
    curr_text_attn_rows = normalize_rows(curr_text_attn.reshape(-1, curr_text_attn.shape[-1])).view_as(curr_text_attn)
    combined_text_attn = normalize_rows((prev_text_attn_rows + curr_text_attn_rows).reshape(-1, curr_text_attn.shape[-1])).view_as(curr_text_attn)

    text_hidden_delta = hidden_state_l2_distance_layers(
        prev_state.prompt_text_hidden_layers,
        curr_state.prompt_text_hidden_layers,
    )
    last_text_hidden_delta = text_hidden_delta[:, -1]
    text_token_decoder_hidden_delta_scores = (
        text_hidden_delta.unsqueeze(-1) * combined_text_attn
    ).mean(dim=(0, 1))
    last_text_attention = normalize_rows(prev_text_attn_rows[:, -1, :] + curr_text_attn_rows[:, -1, :])
    last_text_token_hidden_delta_scores = (
        last_text_hidden_delta.unsqueeze(-1) * last_text_attention
    ).mean(dim=0)
    text_to_image_attention_cosine_distance_scores = cosine_distance_columns(
        prev_text_attn_rows.reshape(-1, prev_text_attn_rows.shape[-1]),
        curr_text_attn_rows.reshape(-1, curr_text_attn_rows.shape[-1]),
    )

    prev_frame_features = prev_state.frame_features.to(dtype=torch.float32)
    curr_frame_features = curr_state.frame_features.to(dtype=torch.float32)
    projector_output_delta_scores = torch.linalg.norm(curr_frame_features - prev_frame_features, dim=-1)
    projector_cosine_distance_scores = 1.0 - F.cosine_similarity(
        prev_frame_features,
        curr_frame_features,
        dim=-1,
        eps=1e-8,
    ).clamp(-1.0, 1.0)

    combined_text_attention_per_token = combined_text_attn.mean(dim=(0, 1))
    high_attention_weighted_image_token_delta_scores = combined_text_attention_per_token * projector_output_delta_scores

    topk = min(PROJECTOR_TOPK_WEIGHT_K, int(combined_text_attention_per_token.shape[0]))
    projector_topk_attention_weighted_delta_scores = torch.zeros_like(projector_output_delta_scores)
    if topk > 0:
        topk_values, topk_indices = torch.topk(combined_text_attention_per_token, k=topk, largest=True, sorted=False)
        topk_weights = torch.zeros_like(combined_text_attention_per_token)
        topk_weights[topk_indices] = topk_values
        topk_weights = topk_weights / topk_weights.sum().clamp_min(EPS)
        projector_topk_attention_weighted_delta_scores = projector_output_delta_scores * topk_weights

    cross_modal_gated_scores = standardize_scores(text_token_decoder_hidden_delta_scores) * torch.sigmoid(
        standardize_scores(text_to_image_attention_cosine_distance_scores)
    )

    metrics = {
        "hidden_state_mean_layer_l2_scores": effective_layers.mean(dim=0),
        "attention_cosine_distance_scores": combine_layer_scores(attention_layer_scores),
        "text_token_decoder_hidden_delta_scores": text_token_decoder_hidden_delta_scores,
        "text_to_image_attention_cosine_distance_scores": text_to_image_attention_cosine_distance_scores,
        "projector_output_delta_scores": projector_output_delta_scores,
        "projector_cosine_distance_scores": projector_cosine_distance_scores,
        "projector_topk_attention_weighted_delta_scores": projector_topk_attention_weighted_delta_scores,
        "topk_image_attention_overlap_k5_scores": _topk_membership_change_scores(
            prev_text_attn_rows.reshape(-1, prev_text_attn_rows.shape[-1]),
            curr_text_attn_rows.reshape(-1, curr_text_attn_rows.shape[-1]),
            k=TOPK_IMAGE_ATTENTION_OVERLAP_KS[0],
        ),
        "topk_image_attention_overlap_k10_scores": _topk_membership_change_scores(
            prev_text_attn_rows.reshape(-1, prev_text_attn_rows.shape[-1]),
            curr_text_attn_rows.reshape(-1, curr_text_attn_rows.shape[-1]),
            k=TOPK_IMAGE_ATTENTION_OVERLAP_KS[1],
        ),
        "topk_image_attention_overlap_k20_scores": _topk_membership_change_scores(
            prev_text_attn_rows.reshape(-1, prev_text_attn_rows.shape[-1]),
            curr_text_attn_rows.reshape(-1, curr_text_attn_rows.shape[-1]),
            k=TOPK_IMAGE_ATTENTION_OVERLAP_KS[2],
        ),
        "cross_modal_gated_scores": cross_modal_gated_scores,
        "early_mid_decoder_block_hidden_delta_scores": early_mid_block.mean(dim=0),
        "last_text_token_hidden_delta_scores": last_text_token_hidden_delta_scores,
        "high_attention_weighted_image_token_delta_scores": high_attention_weighted_image_token_delta_scores,
    }
    metadata = {
        "decoder_hidden_state_layer_count_total": int(layerwise_l2.shape[0]),
        "decoder_hidden_state_effective_layer_count": int(effective_layers.shape[0]),
        "decoder_hidden_state_dropped_input_embedding_layer": bool(dropped_input_embedding_layer),
        "early_mid_decoder_block_requested_layer_range": block_meta["requested"],
        "early_mid_decoder_block_actual_layer_range": block_meta["actual"],
        "early_mid_decoder_block_layer_count": block_meta["count"],
        "attention_row_layer_count": int(curr_state.image_attention_rows.shape[0]),
        "attention_row_key_length": int(curr_state.image_attention_rows.shape[-1]),
        "prompt_text_token_count": int(curr_state.prompt_text_hidden_layers.shape[1]),
        "prompt_text_hidden_layer_count": int(curr_state.prompt_text_hidden_layers.shape[0]),
        "prompt_text_to_image_attention_layer_count": int(curr_state.prompt_text_to_image_attention.shape[0]),
        "topk_image_attention_overlap_ks": [int(k) for k in TOPK_IMAGE_ATTENTION_OVERLAP_KS],
        "projector_topk_weight_k": int(PROJECTOR_TOPK_WEIGHT_K),
    }
    return metrics, metadata


def build_transition_record(
    *,
    sample,
    frame_idx: int,
    prompt_text: str,
    teacher_reference: str,
    oracle_scores: List[float],
    metric_tensors: dict[str, torch.Tensor],
    metric_metadata: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "sample_id": sample.sample_id,
        "video_path": sample.video_path,
        "frame_idx": int(frame_idx),
        "prev_frame_idx": int(frame_idx - 1),
        "transition": f"{frame_idx - 1}->{frame_idx}",
        "transition_id": f"{sample.sample_id}:{frame_idx - 1}->{frame_idx}",
        "status": "ok",
        "prompt_text": prompt_text,
        "original_question": sample.question,
        "choices": sample.choices,
        "gold_idx": sample.answer_idx,
        "teacher_reference": teacher_reference,
        "cheap_signal_source": "run_drift_oracle.py::selective_all_recompute",
        "oracle_semantic_drift_source_key": "token_importance_scores",
        "oracle_semantic_drift_scores": [float(x) for x in oracle_scores],
        "cheap_signal_keys": list(EXTRACTED_SIGNAL_KEYS),
        "recommended_topk_retrieval_signal_keys": list(RECOMMENDED_TOPK_SIGNAL_KEYS),
        "recommended_logistic_feature_keys": list(RECOMMENDED_LOGISTIC_FEATURE_KEYS),
        **metric_metadata,
    }
    for metric_key in EXTRACTED_SIGNAL_KEYS:
        record[metric_key] = rounded_list(metric_tensors[metric_key])
    return record
