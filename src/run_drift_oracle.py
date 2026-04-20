from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.video_io import get_video_frame_indices, iter_video_frames
from kv_reuse.cache_ops import clone_cache, get_cache_layer_kv, make_static_cache, static_cache_to_dynamic
from kv_reuse.pixel_change import decide_reuse_by_pixel_change
from kv_reuse.prefill import (
    FrameSnapshot,
    _build_layer_reuse_masks,
    _forward_frame_layerwise_dense,
    _forward_frame_layerwise_with_selective_reuse,
    _forward_span,
)
from kv_reuse.video_prompt import encode_streaming_video_frame, prepare_streaming_prompt_template
from runtime_config import (
    DEFAULT_TEACHER_REFERENCE,
    TEACHER_REFERENCE_CHOICES,
    load_cfg as shared_load_cfg,
    merge_cfg as shared_merge_cfg,
    normalize_teacher_reference,
    resolve_runtime as shared_resolve_runtime,
)
from teacher_reference import (
    average_policy_kl as shared_average_policy_kl,
    build_counterfactual_cache as shared_build_counterfactual_cache,
    build_counterfactual_outputs as shared_build_counterfactual_outputs,
    build_dense_teacher_cache as shared_build_dense_teacher_cache,
    build_dense_teacher_feature_trace as shared_build_dense_teacher_feature_trace,
    build_dense_teacher_snapshot as shared_build_dense_teacher_snapshot,
    build_full_frame_snapshot as shared_build_full_frame_snapshot,
    build_legacy_selective_counterfactual_cache as shared_build_legacy_selective_counterfactual_cache,
    build_legacy_selective_zero_reuse_snapshot as shared_build_legacy_selective_zero_reuse_snapshot,
    build_selective_all_recompute_feature_trace as shared_build_selective_all_recompute_feature_trace,
    build_selective_all_recompute_cache as shared_build_selective_all_recompute_cache,
    build_teacher_reference_cache,
    collect_teacher_forced_policy_logits as shared_collect_teacher_forced_policy_logits,
    compute_token_pixel_diff_scores as shared_compute_token_pixel_diff_scores,
    explain_legacy_selective_all_recompute_divergence,
    generate_oracle_trace as shared_generate_oracle_trace,
    infer_token_spatial_grid as shared_infer_token_spatial_grid,
    summarize_decoder_feature_trace_diff,
)


TEACHER_REFERENCE = DEFAULT_TEACHER_REFERENCE
HIERARCHICAL_ATTRIBUTION_GROUP_SIZES = [1, 4, 9, 16]
HIERARCHICAL_STAGE_CANDIDATE_RATIOS = {
    "singleton": 0.75,
    "size4": 0.50,
    "size9": 0.50,
}
HIERARCHICAL_FINAL_SCORE_AGGREGATION = "mean_available_stages"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--mode", type=str, choices=["baseline", "baseline_oracle", "drift_oracle"], required=True)
    ap.add_argument("--output_jsonl", type=str, default=None)
    ap.add_argument("--oracle_jsonl", type=str, default=None)
    ap.add_argument("--prompt_text", type=str, default="What is happening now?")
    ap.add_argument("--epsilon_budgets", type=str, default=None)
    ap.add_argument("--group_sizes", type=str, default="1,4,9,16")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--num_frames", type=int, default=None)
    ap.add_argument("--video_root", type=str, default=None)
    ap.add_argument("--dataset_name", type=str, default=None)
    ap.add_argument("--dataset_config_name", type=str, default=None)
    ap.add_argument("--dataset_split", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dtype", type=str, default=None)
    ap.add_argument("--attn_implementation", type=str, default=None)
    ap.add_argument("--teacher_reference", type=str, choices=list(TEACHER_REFERENCE_CHOICES), default=None)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--debug_counterfactual", action="store_true")
    ap.add_argument("--debug_counterfactual_topk", type=int, default=5)
    return ap.parse_args()


def load_cfg(path: str) -> Dict[str, Any]:
    return shared_load_cfg(path)


def merge_cfg(cfg: Dict[str, Any], args) -> Dict[str, Any]:
    return shared_merge_cfg(cfg, args)


def resolve_runtime(cfg: Dict[str, Any]) -> tuple[torch.device, torch.dtype]:
    return shared_resolve_runtime(cfg)


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_oracle_index(path: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("status") != "ok":
            continue
        out[(str(row["sample_id"]), int(row["frame_idx"]))] = row
    return out


def parse_group_sizes(arg_value: str | None) -> List[int]:
    if arg_value is None or not arg_value.strip():
        sizes = [1, 4, 9, 16]
    else:
        sizes = [int(x.strip()) for x in arg_value.split(",") if x.strip()]
    sizes = sorted({size for size in sizes if size > 0})
    if 1 not in sizes:
        sizes = [1, *sizes]
    return sizes


def _build_spatial_square_groups(
    num_tokens: int,
    group_size: int,
) -> List[Dict[str, Any]]:
    height, width = infer_token_spatial_grid(num_tokens)
    side = int(group_size**0.5)
    if side * side != group_size or min(height, width) <= 1:
        raise ValueError(f"group_size={group_size} is not a usable spatial square for {num_tokens} tokens.")

    groups: List[Dict[str, Any]] = []
    for row_start in range(0, height, side):
        row_end = min(row_start + side, height)
        for col_start in range(0, width, side):
            col_end = min(col_start + side, width)
            indices = [
                row_idx * width + col_idx
                for row_idx in range(row_start, row_end)
                for col_idx in range(col_start, col_end)
            ]
            groups.append(
                {
                    "group_size": int(group_size),
                    "start_idx": int(indices[0]),
                    "end_idx": int(indices[-1]),
                    "indices": indices,
                    "group_layout": "spatial_square",
                    "row_start": int(row_start),
                    "row_end": int(row_end - 1),
                    "col_start": int(col_start),
                    "col_end": int(col_end - 1),
                }
            )
    return groups


def build_group_specs(num_tokens: int, group_sizes: List[int]) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    for group_size in group_sizes:
        if group_size in {9, 16}:
            try:
                groups.extend(_build_spatial_square_groups(num_tokens=num_tokens, group_size=group_size))
                continue
            except ValueError:
                pass
        for start_idx in range(0, num_tokens, group_size):
            end_idx = min(start_idx + group_size, num_tokens)
            groups.append(
                {
                    "group_size": int(group_size),
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx - 1),
                    "indices": list(range(start_idx, end_idx)),
                    "group_layout": "contiguous_1d",
                }
            )
    return groups


def _topk_candidate_indices(
    candidates: List[int],
    scores: List[float | None],
    ratio: float,
) -> List[int]:
    if not candidates:
        return []
    keep_count = max(1, int(math.ceil(ratio * len(candidates))))
    ranked = sorted(
        candidates,
        key=lambda idx: (float(scores[idx]) if scores[idx] is not None else float("-inf"), -idx),
        reverse=True,
    )
    return [int(idx) for idx in ranked[:keep_count]]


def _build_fallback_window_group(
    token_idx: int,
    window_tokens: int,
    num_tokens: int,
) -> Dict[str, Any]:
    if window_tokens == 4:
        start_idx = max(0, token_idx - 1)
        end_idx = min(num_tokens, token_idx + 3)
    elif window_tokens == 9:
        start_idx = max(0, token_idx - 4)
        end_idx = min(num_tokens, token_idx + 5)
    elif window_tokens == 16:
        start_idx = max(0, token_idx - 8)
        end_idx = min(num_tokens, token_idx + 8)
    else:
        start_idx = token_idx
        end_idx = min(num_tokens, token_idx + 1)

    indices = list(range(start_idx, end_idx))
    return {
        "indices": indices,
        "group_size": int(len(indices)),
        "group_layout": "contiguous_1d",
        "start_idx": int(indices[0]),
        "end_idx": int(indices[-1]),
        "row_start": None,
        "row_end": None,
        "col_start": None,
        "col_end": None,
    }


def window_groups_containing_token(
    token_idx: int,
    window_tokens: int,
    num_tokens: int,
    height: int,
    width: int,
) -> List[Dict[str, Any]]:
    if window_tokens <= 1:
        return [
            {
                "indices": [int(token_idx)],
                "group_size": 1,
                "group_layout": "singleton",
                "start_idx": int(token_idx),
                "end_idx": int(token_idx),
                "row_start": None,
                "row_end": None,
                "col_start": None,
                "col_end": None,
            }
        ]

    side = int(window_tokens**0.5)
    spatial_possible = (
        side * side == window_tokens
        and height >= side
        and width >= side
        and min(height, width) > 1
    )
    if not spatial_possible:
        return [_build_fallback_window_group(token_idx=token_idx, window_tokens=window_tokens, num_tokens=num_tokens)]

    row_idx = token_idx // width
    col_idx = token_idx % width
    row_start_min = max(0, row_idx - side + 1)
    row_start_max = min(row_idx, height - side)
    col_start_min = max(0, col_idx - side + 1)
    col_start_max = min(col_idx, width - side)

    groups: List[Dict[str, Any]] = []
    for row_start in range(row_start_min, row_start_max + 1):
        row_end = row_start + side
        for col_start in range(col_start_min, col_start_max + 1):
            col_end = col_start + side
            indices = [
                row * width + col
                for row in range(row_start, row_end)
                for col in range(col_start, col_end)
            ]
            groups.append(
                {
                    "indices": indices,
                    "group_size": int(len(indices)),
                    "group_layout": "spatial_square",
                    "start_idx": int(indices[0]),
                    "end_idx": int(indices[-1]),
                    "row_start": int(row_start),
                    "row_end": int(row_end - 1),
                    "col_start": int(col_start),
                    "col_end": int(col_end - 1),
                }
            )
    return groups


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
    )
    return frame_snapshot


@torch.inference_mode()
def build_dense_teacher_cache(
    model,
    prefix_cache_template,
    frame_features: torch.Tensor,
    frame_positions: torch.Tensor,
):
    frame_cache = clone_cache(prefix_cache_template)
    outputs = _forward_span(
        model.language_model,
        frame_features.unsqueeze(0),
        frame_cache,
        cache_position=frame_positions,
    )
    return outputs.past_key_values


@torch.inference_mode()
def build_dense_teacher_snapshot(
    language_model,
    prefix_cache_template,
    frame_features: torch.Tensor,
    frame_positions: torch.Tensor,
):
    frame_cache = clone_cache(prefix_cache_template)
    frame_snapshot, _ = _forward_frame_layerwise_dense(
        language_model=language_model,
        cache=frame_cache,
        frame_features=frame_features,
        frame_cache_positions=frame_positions,
    )
    return frame_snapshot


def summarize_snapshot_vs_cache_diff(
    full_snapshot,
    cache,
    frame_start: int,
    frame_len: int,
) -> Dict[str, Any]:
    layerwise_mean_abs_diff: List[float] = []
    layerwise_max_abs_diff: List[float] = []

    for layer_idx, (selective_k, selective_v) in enumerate(full_snapshot.layer_kv):
        cache_k_full, cache_v_full = get_cache_layer_kv(cache, layer_idx)
        cache_k = cache_k_full[:, :, frame_start : frame_start + frame_len, :]
        cache_v = cache_v_full[:, :, frame_start : frame_start + frame_len, :]

        k_abs_diff = (selective_k - cache_k).abs()
        v_abs_diff = (selective_v - cache_v).abs()
        kv_abs_diff = torch.cat(
            [
                k_abs_diff.reshape(-1),
                v_abs_diff.reshape(-1),
            ]
        )
        layerwise_mean_abs_diff.append(float(kv_abs_diff.mean().item()))
        layerwise_max_abs_diff.append(float(kv_abs_diff.max().item()))

    overall_mean_abs_diff = float(sum(layerwise_mean_abs_diff) / len(layerwise_mean_abs_diff)) if layerwise_mean_abs_diff else 0.0
    overall_max_abs_diff = float(max(layerwise_max_abs_diff)) if layerwise_max_abs_diff else 0.0
    return {
        "layerwise_mean_abs_diff": layerwise_mean_abs_diff,
        "layerwise_max_abs_diff": layerwise_max_abs_diff,
        "mean_abs_diff": overall_mean_abs_diff,
        "max_abs_diff": overall_max_abs_diff,
    }


def summarize_cache_pair_diff(
    cache_a,
    cache_b,
    frame_start: int,
    frame_len: int,
) -> Dict[str, Any]:
    layerwise_mean_abs_diff: List[float] = []
    layerwise_max_abs_diff: List[float] = []

    num_layers = len(cache_a.key_cache)
    for layer_idx in range(num_layers):
        cache_a_k_full, cache_a_v_full = get_cache_layer_kv(cache_a, layer_idx)
        cache_b_k_full, cache_b_v_full = get_cache_layer_kv(cache_b, layer_idx)
        cache_a_k = cache_a_k_full[:, :, frame_start : frame_start + frame_len, :]
        cache_a_v = cache_a_v_full[:, :, frame_start : frame_start + frame_len, :]
        cache_b_k = cache_b_k_full[:, :, frame_start : frame_start + frame_len, :]
        cache_b_v = cache_b_v_full[:, :, frame_start : frame_start + frame_len, :]

        k_abs_diff = (cache_a_k - cache_b_k).abs()
        v_abs_diff = (cache_a_v - cache_b_v).abs()
        kv_abs_diff = torch.cat([k_abs_diff.reshape(-1), v_abs_diff.reshape(-1)])
        layerwise_mean_abs_diff.append(float(kv_abs_diff.mean().item()))
        layerwise_max_abs_diff.append(float(kv_abs_diff.max().item()))

    overall_mean_abs_diff = float(sum(layerwise_mean_abs_diff) / len(layerwise_mean_abs_diff)) if layerwise_mean_abs_diff else 0.0
    overall_max_abs_diff = float(max(layerwise_max_abs_diff)) if layerwise_max_abs_diff else 0.0
    return {
        "layerwise_mean_abs_diff": layerwise_mean_abs_diff,
        "layerwise_max_abs_diff": layerwise_max_abs_diff,
        "mean_abs_diff": overall_mean_abs_diff,
        "max_abs_diff": overall_max_abs_diff,
    }


def summarize_frame_snapshot_diff(
    reference_snapshot: FrameSnapshot,
    probe_snapshot: FrameSnapshot,
    *,
    layer_attention_types: List[str],
) -> Dict[str, Any]:
    compared_layers = min(
        len(reference_snapshot.layer_hidden_states),
        len(probe_snapshot.layer_hidden_states),
        len(reference_snapshot.layer_kv),
        len(probe_snapshot.layer_kv),
    )
    layerwise_hidden_mean_abs_diff: List[float] = []
    layerwise_hidden_max_abs_diff: List[float] = []
    layerwise_kv_mean_abs_diff: List[float] = []
    layerwise_kv_max_abs_diff: List[float] = []

    for layer_idx in range(compared_layers):
        reference_hidden = reference_snapshot.layer_hidden_states[layer_idx]
        probe_hidden = probe_snapshot.layer_hidden_states[layer_idx]
        hidden_abs_diff = (reference_hidden - probe_hidden).abs()
        layerwise_hidden_mean_abs_diff.append(float(hidden_abs_diff.mean().item()))
        layerwise_hidden_max_abs_diff.append(float(hidden_abs_diff.max().item()))

        reference_k, reference_v = reference_snapshot.layer_kv[layer_idx]
        probe_k, probe_v = probe_snapshot.layer_kv[layer_idx]
        kv_abs_diff = torch.cat(
            [
                (reference_k - probe_k).abs().reshape(-1),
                (reference_v - probe_v).abs().reshape(-1),
            ]
        )
        layerwise_kv_mean_abs_diff.append(float(kv_abs_diff.mean().item()))
        layerwise_kv_max_abs_diff.append(float(kv_abs_diff.max().item()))

    def _first_nonzero_layer(values: List[float]) -> int | None:
        for idx, value in enumerate(values):
            if value > 0.0:
                return idx
        return None

    def _max_layer(values: List[float]) -> int | None:
        if not values:
            return None
        return int(max(range(len(values)), key=lambda idx: values[idx]))

    first_hidden_divergence_layer = _first_nonzero_layer(layerwise_hidden_max_abs_diff)
    first_kv_divergence_layer = _first_nonzero_layer(layerwise_kv_max_abs_diff)
    max_hidden_diff_layer = _max_layer(layerwise_hidden_max_abs_diff)
    max_kv_diff_layer = _max_layer(layerwise_kv_max_abs_diff)

    return {
        "reference_layers": len(reference_snapshot.layer_hidden_states),
        "probe_layers": len(probe_snapshot.layer_hidden_states),
        "compared_layers": compared_layers,
        "layer_attention_types": layer_attention_types[:compared_layers],
        "layerwise_hidden_mean_abs_diff": layerwise_hidden_mean_abs_diff,
        "layerwise_hidden_max_abs_diff": layerwise_hidden_max_abs_diff,
        "layerwise_kv_mean_abs_diff": layerwise_kv_mean_abs_diff,
        "layerwise_kv_max_abs_diff": layerwise_kv_max_abs_diff,
        "hidden_mean_abs_diff": float(sum(layerwise_hidden_mean_abs_diff) / len(layerwise_hidden_mean_abs_diff))
        if layerwise_hidden_mean_abs_diff
        else 0.0,
        "hidden_max_abs_diff": float(max(layerwise_hidden_max_abs_diff)) if layerwise_hidden_max_abs_diff else 0.0,
        "kv_mean_abs_diff": float(sum(layerwise_kv_mean_abs_diff) / len(layerwise_kv_mean_abs_diff))
        if layerwise_kv_mean_abs_diff
        else 0.0,
        "kv_max_abs_diff": float(max(layerwise_kv_max_abs_diff)) if layerwise_kv_max_abs_diff else 0.0,
        "first_hidden_divergence_layer": first_hidden_divergence_layer,
        "first_hidden_divergence_attention_type": (
            layer_attention_types[first_hidden_divergence_layer]
            if first_hidden_divergence_layer is not None and first_hidden_divergence_layer < len(layer_attention_types)
            else None
        ),
        "first_kv_divergence_layer": first_kv_divergence_layer,
        "first_kv_divergence_attention_type": (
            layer_attention_types[first_kv_divergence_layer]
            if first_kv_divergence_layer is not None and first_kv_divergence_layer < len(layer_attention_types)
            else None
        ),
        "max_hidden_diff_layer": max_hidden_diff_layer,
        "max_hidden_diff_attention_type": (
            layer_attention_types[max_hidden_diff_layer]
            if max_hidden_diff_layer is not None and max_hidden_diff_layer < len(layer_attention_types)
            else None
        ),
        "max_kv_diff_layer": max_kv_diff_layer,
        "max_kv_diff_attention_type": (
            layer_attention_types[max_kv_diff_layer]
            if max_kv_diff_layer is not None and max_kv_diff_layer < len(layer_attention_types)
            else None
        ),
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


# Shared helper implementations live in dedicated modules so baseline/drift
# generation and the plotting script stay on the exact same teacher path.
average_policy_kl = shared_average_policy_kl
infer_token_spatial_grid = shared_infer_token_spatial_grid
compute_token_pixel_diff_scores = shared_compute_token_pixel_diff_scores
collect_teacher_forced_policy_logits = shared_collect_teacher_forced_policy_logits
generate_oracle_trace = shared_generate_oracle_trace
build_full_frame_snapshot = shared_build_full_frame_snapshot
build_dense_teacher_cache = shared_build_dense_teacher_cache
build_dense_teacher_feature_trace = shared_build_dense_teacher_feature_trace
build_dense_teacher_snapshot = shared_build_dense_teacher_snapshot
build_legacy_selective_counterfactual_cache = shared_build_legacy_selective_counterfactual_cache
build_legacy_selective_zero_reuse_snapshot = shared_build_legacy_selective_zero_reuse_snapshot
build_counterfactual_cache = shared_build_counterfactual_cache
build_selective_all_recompute_feature_trace = shared_build_selective_all_recompute_feature_trace
build_selective_all_recompute_cache = shared_build_selective_all_recompute_cache
build_counterfactual_outputs = shared_build_counterfactual_outputs


def summarize_logits_trace_diff(
    reference_logits_trace: List[torch.Tensor],
    probe_logits_trace: List[torch.Tensor],
    *,
    topk: int = 5,
) -> Dict[str, Any]:
    compared_steps = min(len(reference_logits_trace), len(probe_logits_trace))
    if compared_steps == 0:
        return {
            "reference_steps": len(reference_logits_trace),
            "probe_steps": len(probe_logits_trace),
            "compared_steps": 0,
            "avg_policy_kl": 0.0,
            "mean_abs_logit_diff": 0.0,
            "max_abs_logit_diff": 0.0,
            "step_mean_abs_logit_diff": [],
            "step_max_abs_logit_diff": [],
            "step_reference_argmax_token_ids": [],
            "step_probe_argmax_token_ids": [],
            "argmax_changed_steps": 0,
            "max_diff_step": None,
            "max_diff_top_token_ids": [],
            "max_diff_top_abs": [],
            "max_diff_reference_logits": [],
            "max_diff_probe_logits": [],
        }

    step_mean_abs_logit_diff: List[float] = []
    step_max_abs_logit_diff: List[float] = []
    step_reference_argmax_token_ids: List[int] = []
    step_probe_argmax_token_ids: List[int] = []
    argmax_changed_steps = 0

    max_diff_step = 0
    max_step_mean_abs_logit_diff = -1.0
    max_diff_top_token_ids: List[int] = []
    max_diff_top_abs: List[float] = []
    max_diff_reference_logits: List[float] = []
    max_diff_probe_logits: List[float] = []

    for step_idx in range(compared_steps):
        reference_logits = reference_logits_trace[step_idx].float()
        probe_logits = probe_logits_trace[step_idx].float()
        abs_diff = (reference_logits - probe_logits).abs()
        step_mean = float(abs_diff.mean().item())
        step_max = float(abs_diff.max().item())
        reference_argmax = int(reference_logits.argmax().item())
        probe_argmax = int(probe_logits.argmax().item())

        step_mean_abs_logit_diff.append(step_mean)
        step_max_abs_logit_diff.append(step_max)
        step_reference_argmax_token_ids.append(reference_argmax)
        step_probe_argmax_token_ids.append(probe_argmax)
        if reference_argmax != probe_argmax:
            argmax_changed_steps += 1

        if step_mean > max_step_mean_abs_logit_diff:
            max_step_mean_abs_logit_diff = step_mean
            max_diff_step = step_idx
            top_count = min(max(int(topk), 1), int(abs_diff.numel()))
            top_abs, top_indices = torch.topk(abs_diff, k=top_count)
            max_diff_top_token_ids = [int(idx) for idx in top_indices.tolist()]
            max_diff_top_abs = [float(val) for val in top_abs.tolist()]
            max_diff_reference_logits = [float(reference_logits[idx].item()) for idx in top_indices.tolist()]
            max_diff_probe_logits = [float(probe_logits[idx].item()) for idx in top_indices.tolist()]

    return {
        "reference_steps": len(reference_logits_trace),
        "probe_steps": len(probe_logits_trace),
        "compared_steps": compared_steps,
        "avg_policy_kl": float(average_policy_kl(reference_logits_trace, probe_logits_trace)),
        "mean_abs_logit_diff": float(sum(step_mean_abs_logit_diff) / len(step_mean_abs_logit_diff)),
        "max_abs_logit_diff": float(max(step_max_abs_logit_diff)),
        "step_mean_abs_logit_diff": step_mean_abs_logit_diff,
        "step_max_abs_logit_diff": step_max_abs_logit_diff,
        "step_reference_argmax_token_ids": step_reference_argmax_token_ids,
        "step_probe_argmax_token_ids": step_probe_argmax_token_ids,
        "argmax_changed_steps": int(argmax_changed_steps),
        "max_diff_step": int(max_diff_step),
        "max_diff_top_token_ids": max_diff_top_token_ids,
        "max_diff_top_abs": max_diff_top_abs,
        "max_diff_reference_logits": max_diff_reference_logits,
        "max_diff_probe_logits": max_diff_probe_logits,
    }


@torch.inference_mode()
def evaluate_counterfactual_action_kl(
    model,
    prefix_cache_template,
    frame_features: torch.Tensor,
    frame_positions: torch.Tensor,
    prev_full_snapshot,
    reuse_mask: torch.Tensor,
    teacher_logits_trace: List[torch.Tensor],
    newline_embed: torch.Tensor,
    suffix_input_ids: torch.Tensor,
    oracle_token_ids: List[int],
    frame_seq_len: int,
    device: torch.device,
) -> float:
    _, counterfactual_logits_trace = build_counterfactual_outputs(
        model=model,
        prefix_cache_template=prefix_cache_template,
        frame_features=frame_features,
        frame_positions=frame_positions,
        prev_full_snapshot=prev_full_snapshot,
        reuse_mask=reuse_mask,
        newline_embed=newline_embed,
        suffix_input_ids=suffix_input_ids,
        oracle_token_ids=oracle_token_ids,
        frame_seq_len=frame_seq_len,
        device=device,
    )
    return average_policy_kl(teacher_logits_trace, counterfactual_logits_trace)


@torch.inference_mode()
def _compute_hierarchical_stage_scores(
    *,
    candidate_tokens: List[int],
    stage_name: str,
    window_tokens: int,
    num_tokens: int,
    height: int,
    width: int,
    group_value_fn,
    group_action_delta_scores: List[Dict[str, Any]],
) -> List[float | None]:
    stage_scores: List[float | None] = [None] * num_tokens
    for token_idx in candidate_tokens:
        groups = window_groups_containing_token(
            token_idx=token_idx,
            window_tokens=window_tokens,
            num_tokens=num_tokens,
            height=height,
            width=width,
        )
        marginal_scores: List[float] = []
        for group_spec in groups:
            group_indices = [int(idx) for idx in group_spec["indices"]]
            group_without_token = [idx for idx in group_indices if idx != token_idx]
            group_action_kl = float(group_value_fn(group_indices))
            group_without_token_action_kl = float(group_value_fn(group_without_token))
            loo_marginal = float(group_action_kl - group_without_token_action_kl)
            marginal_scores.append(loo_marginal)
            group_action_delta_scores.append(
                {
                    "stage": stage_name,
                    "token_idx": int(token_idx),
                    "window_tokens": int(window_tokens),
                    "group_size": int(group_spec["group_size"]),
                    "group_layout": str(group_spec["group_layout"]),
                    "start_idx": int(group_spec["start_idx"]),
                    "end_idx": int(group_spec["end_idx"]),
                    "row_start": group_spec.get("row_start"),
                    "row_end": group_spec.get("row_end"),
                    "col_start": group_spec.get("col_start"),
                    "col_end": group_spec.get("col_end"),
                    "group_indices": group_indices,
                    "group_without_token_indices": group_without_token,
                    "action_kl": group_action_kl,
                    "action_kl_without_token": group_without_token_action_kl,
                    "loo_marginal": loo_marginal,
                }
            )
        stage_scores[token_idx] = float(sum(marginal_scores) / len(marginal_scores)) if marginal_scores else 0.0
    return stage_scores


@torch.inference_mode()
def compute_hierarchical_token_importance_scores(
    *,
    model,
    prefix_cache_template,
    frame_features: torch.Tensor,
    frame_positions: torch.Tensor,
    prev_full_snapshot,
    teacher_logits_trace: List[torch.Tensor],
    newline_embed: torch.Tensor,
    suffix_input_ids: torch.Tensor,
    oracle_token_ids: List[int],
    frame_seq_len: int,
    device: torch.device,
    all_recompute_action_kl: float,
) -> Dict[str, Any]:
    num_visual_tokens = int(frame_features.shape[0])
    height, width = infer_token_spatial_grid(num_visual_tokens)
    group_action_delta_scores: List[Dict[str, Any]] = []
    group_value_cache: Dict[Tuple[int, ...], float] = {
        tuple(): float(all_recompute_action_kl),
    }

    def group_value_fn(group_indices: List[int] | Tuple[int, ...]) -> float:
        key = tuple(sorted({int(idx) for idx in group_indices}))
        cached_value = group_value_cache.get(key)
        if cached_value is not None:
            return float(cached_value)

        reuse_mask = torch.zeros(
            num_visual_tokens,
            dtype=torch.bool,
            device=frame_features.device,
        )
        if key:
            reuse_mask[list(key)] = True
        value = evaluate_counterfactual_action_kl(
            model=model,
            prefix_cache_template=prefix_cache_template,
            frame_features=frame_features,
            frame_positions=frame_positions,
            prev_full_snapshot=prev_full_snapshot,
            reuse_mask=reuse_mask,
            teacher_logits_trace=teacher_logits_trace,
            newline_embed=newline_embed,
            suffix_input_ids=suffix_input_ids,
            oracle_token_ids=oracle_token_ids,
            frame_seq_len=frame_seq_len,
            device=device,
        )
        group_value_cache[key] = float(value)
        return float(value)

    single_token_scores = [float(group_value_fn((token_idx,))) for token_idx in range(num_visual_tokens)]
    all_token_indices = list(range(num_visual_tokens))
    candidate_stage1 = _topk_candidate_indices(
        candidates=all_token_indices,
        scores=single_token_scores,
        ratio=HIERARCHICAL_STAGE_CANDIDATE_RATIOS["singleton"],
    )

    size4_scores = _compute_hierarchical_stage_scores(
        candidate_tokens=candidate_stage1,
        stage_name="size4",
        window_tokens=4,
        num_tokens=num_visual_tokens,
        height=height,
        width=width,
        group_value_fn=group_value_fn,
        group_action_delta_scores=group_action_delta_scores,
    )
    candidate_stage2 = _topk_candidate_indices(
        candidates=candidate_stage1,
        scores=size4_scores,
        ratio=HIERARCHICAL_STAGE_CANDIDATE_RATIOS["size4"],
    )

    size9_scores = _compute_hierarchical_stage_scores(
        candidate_tokens=candidate_stage2,
        stage_name="size9",
        window_tokens=9,
        num_tokens=num_visual_tokens,
        height=height,
        width=width,
        group_value_fn=group_value_fn,
        group_action_delta_scores=group_action_delta_scores,
    )
    candidate_stage3 = _topk_candidate_indices(
        candidates=candidate_stage2,
        scores=size9_scores,
        ratio=HIERARCHICAL_STAGE_CANDIDATE_RATIOS["size9"],
    )

    size16_scores = _compute_hierarchical_stage_scores(
        candidate_tokens=candidate_stage3,
        stage_name="size16",
        window_tokens=16,
        num_tokens=num_visual_tokens,
        height=height,
        width=width,
        group_value_fn=group_value_fn,
        group_action_delta_scores=group_action_delta_scores,
    )

    final_scores: List[float] = []
    for token_idx in all_token_indices:
        available_stage_scores = [float(single_token_scores[token_idx])]
        if size4_scores[token_idx] is not None:
            available_stage_scores.append(float(size4_scores[token_idx]))
        if size9_scores[token_idx] is not None:
            available_stage_scores.append(float(size9_scores[token_idx]))
        if size16_scores[token_idx] is not None:
            available_stage_scores.append(float(size16_scores[token_idx]))
        final_scores.append(float(sum(available_stage_scores) / len(available_stage_scores)))

    greedy_rank = sorted(
        all_token_indices,
        key=lambda idx: final_scores[idx],
        reverse=True,
    )
    return {
        "single_token_scores": single_token_scores,
        "size4_scores": size4_scores,
        "size9_scores": size9_scores,
        "size16_scores": size16_scores,
        "final_scores": final_scores,
        "candidate_stage1": candidate_stage1,
        "candidate_stage2": candidate_stage2,
        "candidate_stage3": candidate_stage3,
        "group_action_delta_scores": group_action_delta_scores,
        "greedy_rank": greedy_rank,
        "spatial_grid_height": int(height),
        "spatial_grid_width": int(width),
        "group_value_cache_size": int(len(group_value_cache)),
        "final_score_aggregation": HIERARCHICAL_FINAL_SCORE_AGGREGATION,
    }


def make_record_base(sample, frame_idx: int, prompt_text: str) -> Dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "video_path": sample.video_path,
        "frame_idx": int(frame_idx),
        "prompt_text": prompt_text,
        "original_question": sample.question,
        "choices": sample.choices,
        "gold_idx": sample.answer_idx,
    }


def main():
    args = parse_args()
    if args.mode == "baseline":
        args.mode = "baseline_oracle"
    cfg = merge_cfg(load_cfg(args.config), args)
    from hf_patch.apply_runtime_patch import apply_runtime_patch

    apply_runtime_patch()
    teacher_reference = normalize_teacher_reference(cfg.get("teacher_reference"))

    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

    device, dtype = resolve_runtime(cfg)
    output_dir = Path(cfg.get("output_dir", str(ROOT / "results")))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_jsonl is not None:
        output_jsonl = args.output_jsonl
    elif args.mode == "baseline_oracle":
        output_jsonl = str(output_dir / "baseline_oracle.jsonl")
    else:
        output_jsonl = str(output_dir / "drift_oracle.jsonl")

    generation_cfg = cfg.get("generation", {})
    max_new_tokens = int(cfg.get("max_new_tokens", 32))
    do_sample = bool(generation_cfg.get("do_sample", False))
    temperature = float(generation_cfg.get("temperature", 0.0))
    group_sizes = list(HIERARCHICAL_ATTRIBUTION_GROUP_SIZES)

    if args.mode == "drift_oracle":
        oracle_jsonl = args.oracle_jsonl or str(output_dir / "baseline_oracle.jsonl")
        oracle_index = load_oracle_index(oracle_jsonl)
    else:
        oracle_index = {}

    model_name = cfg["model_name"]
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = "left"

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=str(cfg.get("attn_implementation", "sdpa")),
    ).to(device)
    model.eval()

    from data.nextqa_dataset import load_nextqa_samples

    samples = load_nextqa_samples(
        dataset_name=cfg["dataset_name"],
        dataset_config_name=cfg.get("dataset_config_name"),
        split=cfg.get("dataset_split", "test"),
        video_root=cfg.get("video_root"),
        max_samples=cfg.get("max_samples"),
    )

    rows_written = 0
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for sample in tqdm(samples, desc=args.mode):
            if sample.video_path is None or not os.path.exists(sample.video_path):
                fout.write(
                    json.dumps(
                        {
                            "sample_id": sample.sample_id,
                            "video_path": sample.video_path,
                            "status": "skipped_no_video",
                            "mode": args.mode,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            try:
                frame_indices = get_video_frame_indices(sample.video_path, num_frames=cfg["num_frames"])
                frame_iter = iter_video_frames(sample.video_path, num_frames=cfg["num_frames"])
                if not frame_indices:
                    raise RuntimeError(f"Selected zero frames for video: {sample.video_path}")

                first_frame = next(frame_iter)
                stream_template, first_frame_features = prepare_streaming_prompt_template(
                    model=model,
                    processor=processor,
                    first_frame=first_frame,
                    text_prompt=args.prompt_text,
                    device=device,
                    dtype=dtype,
                )

                prefix_len = int(stream_template.prefix_embeds.shape[1])
                frame_token_length = int(stream_template.frame_token_length)
                frame_seq_len = prefix_len + frame_token_length

                prefix_cache = make_static_cache(
                    config=model.config.text_config,
                    batch_size=1,
                    max_cache_len=frame_seq_len,
                    device=device,
                    dtype=stream_template.prefix_embeds.dtype,
                )
                if prefix_len > 0:
                    prefix_positions = torch.arange(prefix_len, device=device, dtype=torch.long)
                    prefix_outputs = _forward_span(
                        model.language_model,
                        stream_template.prefix_embeds,
                        prefix_cache,
                        cache_position=prefix_positions,
                    )
                    prefix_cache = prefix_outputs.past_key_values
                prefix_cache_template = clone_cache(prefix_cache)

                prev_raw_frame = None
                prev_full_snapshot = None

                for frame_idx in range(len(frame_indices)):
                    raw_frame = first_frame if frame_idx == 0 else next(frame_iter)
                    if frame_idx == 0:
                        frame_features = first_frame_features
                    else:
                        frame_features = encode_streaming_video_frame(
                            model=model,
                            processor=processor,
                            frame=raw_frame,
                            device=device,
                            dtype=stream_template.prefix_embeds.dtype,
                        )

                    frame_positions = torch.arange(
                        prefix_len,
                        prefix_len + frame_token_length,
                        device=device,
                        dtype=torch.long,
                    )

                    full_snapshot = build_full_frame_snapshot(
                        language_model=model.language_model,
                        prefix_cache_template=prefix_cache_template,
                        frame_features=frame_features,
                        frame_positions=frame_positions,
                        device=device,
                    )

                    if args.mode == "baseline_oracle":
                        if frame_idx == 0 or prev_full_snapshot is None:
                            prev_full_snapshot = full_snapshot
                            continue

                        frame_prefill_start = perf_counter()
                        teacher_cache = build_teacher_reference_cache(
                            teacher_reference=teacher_reference,
                            model=model,
                            prefix_cache_template=prefix_cache_template,
                            frame_features=frame_features,
                            frame_positions=frame_positions,
                            prev_full_snapshot=prev_full_snapshot,
                            device=device,
                        )
                        frame_prefill_ms = (perf_counter() - frame_prefill_start) * 1000.0

                        generate_start = perf_counter()
                        trace = generate_oracle_trace(
                            model=model,
                            tokenizer=processor.tokenizer,
                            live_cache=teacher_cache,
                            current_seq_len=frame_seq_len,
                            newline_embed=stream_template.newline_embed,
                            suffix_input_ids=stream_template.suffix_input_ids,
                            max_new_tokens=max_new_tokens,
                            do_sample=do_sample,
                            temperature=temperature,
                        )
                        frame_generate_ms = (perf_counter() - generate_start) * 1000.0

                        debug_counterfactual = None
                        if args.debug_counterfactual:
                            debug_topk = max(int(args.debug_counterfactual_topk), 1)
                            layer_attention_types = [
                                str(getattr(layer, "attention_type", "full_attention"))
                                for layer in model.language_model.model.layers
                            ]

                            dense_teacher_cache = (
                                teacher_cache
                                if teacher_reference == "dense"
                                else build_dense_teacher_cache(
                                    model=model,
                                    prefix_cache_template=prefix_cache_template,
                                    frame_features=frame_features,
                                    frame_positions=frame_positions,
                                )
                            )
                            selective_all_recompute_cache = (
                                teacher_cache
                                if teacher_reference == "selective_all_recompute"
                                else build_selective_all_recompute_cache(
                                    language_model=model.language_model,
                                    prefix_cache_template=prefix_cache_template,
                                    frame_features=frame_features,
                                    frame_positions=frame_positions,
                                    prev_full_snapshot=prev_full_snapshot,
                                    device=device,
                                )
                            )
                            dense_teacher_snapshot = build_dense_teacher_snapshot(
                                language_model=model.language_model,
                                prefix_cache_template=prefix_cache_template,
                                frame_features=frame_features,
                                frame_positions=frame_positions,
                            )
                            dense_teacher_feature_trace = build_dense_teacher_feature_trace(
                                language_model=model.language_model,
                                prefix_cache_template=prefix_cache_template,
                                frame_features=frame_features,
                                frame_positions=frame_positions,
                            )
                            selective_all_recompute_feature_trace = build_selective_all_recompute_feature_trace(
                                language_model=model.language_model,
                                prefix_cache_template=prefix_cache_template,
                                frame_features=frame_features,
                                frame_positions=frame_positions,
                                prev_full_snapshot=prev_full_snapshot,
                                device=device,
                            )

                            dense_trace = (
                                trace
                                if teacher_reference == "dense"
                                else generate_oracle_trace(
                                    model=model,
                                    tokenizer=processor.tokenizer,
                                    live_cache=dense_teacher_cache,
                                    current_seq_len=frame_seq_len,
                                    newline_embed=stream_template.newline_embed,
                                    suffix_input_ids=stream_template.suffix_input_ids,
                                    max_new_tokens=max_new_tokens,
                                    do_sample=do_sample,
                                    temperature=temperature,
                                )
                            )
                            selective_trace = (
                                trace
                                if teacher_reference == "selective_all_recompute"
                                else generate_oracle_trace(
                                    model=model,
                                    tokenizer=processor.tokenizer,
                                    live_cache=selective_all_recompute_cache,
                                    current_seq_len=frame_seq_len,
                                    newline_embed=stream_template.newline_embed,
                                    suffix_input_ids=stream_template.suffix_input_ids,
                                    max_new_tokens=max_new_tokens,
                                    do_sample=do_sample,
                                    temperature=temperature,
                                )
                            )

                            dense_oracle_token_ids = [int(x) for x in dense_trace.get("oracle_token_ids", [])]
                            dense_teacher_logits_trace = collect_teacher_forced_policy_logits(
                                model=model,
                                live_cache=dense_teacher_cache,
                                current_seq_len=frame_seq_len,
                                newline_embed=stream_template.newline_embed,
                                suffix_input_ids=stream_template.suffix_input_ids,
                                oracle_token_ids=dense_oracle_token_ids,
                                device=device,
                            )
                            selective_forced_logits_trace = collect_teacher_forced_policy_logits(
                                model=model,
                                live_cache=selective_all_recompute_cache,
                                current_seq_len=frame_seq_len,
                                newline_embed=stream_template.newline_embed,
                                suffix_input_ids=stream_template.suffix_input_ids,
                                oracle_token_ids=dense_oracle_token_ids,
                                device=device,
                            )

                            dense_cache_diff = summarize_cache_pair_diff(
                                cache_a=dense_teacher_cache,
                                cache_b=selective_all_recompute_cache,
                                frame_start=prefix_len,
                                frame_len=frame_token_length,
                            )
                            dense_snapshot_vs_cache_diff = summarize_snapshot_vs_cache_diff(
                                full_snapshot=dense_teacher_snapshot,
                                cache=dense_teacher_cache,
                                frame_start=prefix_len,
                                frame_len=frame_token_length,
                            )
                            selective_snapshot_vs_cache_diff = summarize_snapshot_vs_cache_diff(
                                full_snapshot=full_snapshot,
                                cache=selective_all_recompute_cache,
                                frame_start=prefix_len,
                                frame_len=frame_token_length,
                            )
                            dense_logits_diff = summarize_logits_trace_diff(
                                dense_teacher_logits_trace,
                                selective_forced_logits_trace,
                                topk=debug_topk,
                            )
                            debug_counterfactual = {
                                "debug_topk": debug_topk,
                                "teacher_reference": teacher_reference,
                                "selective_all_recompute_equivalence_debug": explain_legacy_selective_all_recompute_divergence(
                                    model.language_model
                                ),
                                "dense_teacher_oracle": dense_trace,
                                "selective_all_recompute_oracle": selective_trace,
                                "dense_teacher_vs_selective_all_recompute_layer_diff": summarize_frame_snapshot_diff(
                                    reference_snapshot=dense_teacher_snapshot,
                                    probe_snapshot=full_snapshot,
                                    layer_attention_types=layer_attention_types,
                                ),
                                "dense_teacher_vs_selective_all_recompute_feature_trace_diff": summarize_decoder_feature_trace_diff(
                                    reference_trace=dense_teacher_feature_trace,
                                    probe_trace=selective_all_recompute_feature_trace,
                                ),
                                "dense_teacher_snapshot_vs_dense_cache_diff": dense_snapshot_vs_cache_diff,
                                "selective_all_recompute_snapshot_vs_cache_diff": selective_snapshot_vs_cache_diff,
                                "dense_teacher_vs_selective_all_recompute_cache_diff": dense_cache_diff,
                                "dense_teacher_vs_selective_all_recompute_logits_diff_on_dense_oracle": dense_logits_diff,
                                "dense_teacher_vs_selective_all_recompute_exact_match": (
                                    float(dense_cache_diff.get("max_abs_diff", 0.0)) == 0.0
                                    and float(dense_logits_diff.get("max_abs_logit_diff", 0.0)) == 0.0
                                ),
                            }

                        record = {
                            **make_record_base(sample, frame_idx, args.prompt_text),
                            "status": "ok",
                            "mode": args.mode,
                            "teacher_reference": teacher_reference,
                            "frame_prefill_ms": frame_prefill_ms,
                            "frame_generate_ms": frame_generate_ms,
                            **trace,
                        }
                        if debug_counterfactual is not None:
                            record["debug_counterfactual"] = debug_counterfactual
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fout.flush()
                        rows_written += 1
                        prev_full_snapshot = full_snapshot
                    else:
                        if frame_idx == 0:
                            prev_full_snapshot = full_snapshot
                            prev_raw_frame = raw_frame
                            continue

                        oracle_row = oracle_index.get((str(sample.sample_id), int(frame_idx)))
                        if oracle_row is None:
                            fout.write(
                                json.dumps(
                                    {
                                        **make_record_base(sample, frame_idx, args.prompt_text),
                                        "status": "missing_oracle",
                                        "mode": args.mode,
                                        "teacher_reference": teacher_reference,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            fout.flush()
                            prev_full_snapshot = full_snapshot
                            prev_raw_frame = raw_frame
                            continue

                        oracle_token_ids = [int(x) for x in oracle_row.get("oracle_token_ids", [])]
                        num_visual_tokens = int(frame_features.shape[0])

                        if prev_full_snapshot is None:
                            prev_full_snapshot = full_snapshot
                            prev_raw_frame = raw_frame
                            continue

                        token_pixel_diff_scores = compute_token_pixel_diff_scores(
                            prev_frame=prev_raw_frame,
                            curr_frame=raw_frame,
                            num_tokens=num_visual_tokens,
                        )
                        zero_reuse_mask = torch.zeros(
                            num_visual_tokens,
                            dtype=torch.bool,
                            device=frame_features.device,
                        )

                        dense_teacher_cache = None
                        dense_teacher_logits_trace = None
                        dense_teacher_snapshot = None
                        dense_teacher_feature_trace = None
                        selective_all_recompute_feature_trace = None
                        legacy_selective_zero_reuse_snapshot = None
                        legacy_selective_all_recompute_cache = None
                        legacy_selective_all_recompute_logits_trace = None
                        layer_attention_types: List[str] = []
                        if args.debug_counterfactual or teacher_reference == "dense":
                            dense_teacher_cache = build_dense_teacher_cache(
                                model=model,
                                prefix_cache_template=prefix_cache_template,
                                frame_features=frame_features,
                                frame_positions=frame_positions,
                            )
                            dense_teacher_logits_trace = collect_teacher_forced_policy_logits(
                                model=model,
                                live_cache=dense_teacher_cache,
                                current_seq_len=frame_seq_len,
                                newline_embed=stream_template.newline_embed,
                                suffix_input_ids=stream_template.suffix_input_ids,
                                oracle_token_ids=oracle_token_ids,
                                device=device,
                            )
                            if args.debug_counterfactual:
                                dense_teacher_snapshot = build_dense_teacher_snapshot(
                                    language_model=model.language_model,
                                    prefix_cache_template=prefix_cache_template,
                                    frame_features=frame_features,
                                    frame_positions=frame_positions,
                                )
                                dense_teacher_feature_trace = build_dense_teacher_feature_trace(
                                    language_model=model.language_model,
                                    prefix_cache_template=prefix_cache_template,
                                    frame_features=frame_features,
                                    frame_positions=frame_positions,
                                )
                                selective_all_recompute_feature_trace = build_selective_all_recompute_feature_trace(
                                    language_model=model.language_model,
                                    prefix_cache_template=prefix_cache_template,
                                    frame_features=frame_features,
                                    frame_positions=frame_positions,
                                    prev_full_snapshot=prev_full_snapshot,
                                    device=device,
                                )
                                layer_attention_types = [
                                    str(getattr(layer, "attention_type", "full_attention"))
                                    for layer in model.language_model.model.layers
                                ]
                                legacy_selective_zero_reuse_snapshot = build_legacy_selective_zero_reuse_snapshot(
                                    language_model=model.language_model,
                                    prefix_cache_template=prefix_cache_template,
                                    frame_features=frame_features,
                                    frame_positions=frame_positions,
                                    device=device,
                                )
                                legacy_selective_all_recompute_cache = build_legacy_selective_counterfactual_cache(
                                    language_model=model.language_model,
                                    prefix_cache_template=prefix_cache_template,
                                    frame_features=frame_features,
                                    frame_positions=frame_positions,
                                    prev_full_snapshot=prev_full_snapshot,
                                    reuse_mask=zero_reuse_mask,
                                    device=device,
                                )
                                legacy_selective_all_recompute_logits_trace = collect_teacher_forced_policy_logits(
                                    model=model,
                                    live_cache=legacy_selective_all_recompute_cache,
                                    current_seq_len=frame_seq_len,
                                    newline_embed=stream_template.newline_embed,
                                    suffix_input_ids=stream_template.suffix_input_ids,
                                    oracle_token_ids=oracle_token_ids,
                                    device=device,
                                )
                        all_reuse_mask = torch.ones(num_visual_tokens, dtype=torch.bool, device=frame_features.device)
                        all_reuse_cache, all_reuse_logits_trace = build_counterfactual_outputs(
                            model=model,
                            prefix_cache_template=prefix_cache_template,
                            frame_features=frame_features,
                            frame_positions=frame_positions,
                            prev_full_snapshot=prev_full_snapshot,
                            reuse_mask=all_reuse_mask,
                            newline_embed=stream_template.newline_embed,
                            suffix_input_ids=stream_template.suffix_input_ids,
                            oracle_token_ids=oracle_token_ids,
                            frame_seq_len=frame_seq_len,
                            device=device,
                        )
                        all_recompute_cache, all_recompute_logits_trace = build_counterfactual_outputs(
                            model=model,
                            prefix_cache_template=prefix_cache_template,
                            frame_features=frame_features,
                            frame_positions=frame_positions,
                            prev_full_snapshot=prev_full_snapshot,
                            reuse_mask=zero_reuse_mask,
                            newline_embed=stream_template.newline_embed,
                            suffix_input_ids=stream_template.suffix_input_ids,
                            oracle_token_ids=oracle_token_ids,
                            frame_seq_len=frame_seq_len,
                            device=device,
                        )
                        if teacher_reference == "dense":
                            if dense_teacher_cache is None or dense_teacher_logits_trace is None:
                                raise RuntimeError("dense teacher outputs were not prepared for teacher_reference=dense")
                            teacher_cache = dense_teacher_cache
                            teacher_logits_trace = dense_teacher_logits_trace
                        else:
                            teacher_cache = all_recompute_cache
                            teacher_logits_trace = all_recompute_logits_trace
                        all_recompute_cache_diff = summarize_snapshot_vs_cache_diff(
                            full_snapshot=full_snapshot,
                            cache=all_recompute_cache,
                            frame_start=prefix_len,
                            frame_len=frame_token_length,
                        )
                        all_reuse_action_kl = average_policy_kl(teacher_logits_trace, all_reuse_logits_trace)
                        all_recompute_action_kl = average_policy_kl(teacher_logits_trace, all_recompute_logits_trace)

                        hierarchical_scores = compute_hierarchical_token_importance_scores(
                            model=model,
                            prefix_cache_template=prefix_cache_template,
                            frame_features=frame_features,
                            frame_positions=frame_positions,
                            prev_full_snapshot=prev_full_snapshot,
                            teacher_logits_trace=teacher_logits_trace,
                            newline_embed=stream_template.newline_embed,
                            suffix_input_ids=stream_template.suffix_input_ids,
                            oracle_token_ids=oracle_token_ids,
                            frame_seq_len=frame_seq_len,
                            device=device,
                            all_recompute_action_kl=float(all_recompute_action_kl),
                        )
                        single_token_scores = hierarchical_scores["single_token_scores"]
                        size4_scores = hierarchical_scores["size4_scores"]
                        size9_scores = hierarchical_scores["size9_scores"]
                        size16_scores = hierarchical_scores["size16_scores"]
                        token_importance_scores = hierarchical_scores["final_scores"]
                        group_action_delta_scores = hierarchical_scores["group_action_delta_scores"]
                        greedy_rank = hierarchical_scores["greedy_rank"]

                        debug_counterfactual = None
                        if args.debug_counterfactual:
                            debug_topk = max(int(args.debug_counterfactual_topk), 1)
                            debug_counterfactual = {
                                "debug_topk": debug_topk,
                                "teacher_reference": teacher_reference,
                                "selective_all_recompute_equivalence_debug": explain_legacy_selective_all_recompute_divergence(
                                    model.language_model
                                ),
                                "full_snapshot_vs_all_recompute_cache_diff": summarize_snapshot_vs_cache_diff(
                                    full_snapshot=full_snapshot,
                                    cache=all_recompute_cache,
                                    frame_start=prefix_len,
                                    frame_len=frame_token_length,
                                ),
                                "teacher_vs_all_recompute_cache_diff": summarize_cache_pair_diff(
                                    cache_a=teacher_cache,
                                    cache_b=all_recompute_cache,
                                    frame_start=prefix_len,
                                    frame_len=frame_token_length,
                                ),
                                "teacher_vs_all_reuse_cache_diff": summarize_cache_pair_diff(
                                    cache_a=teacher_cache,
                                    cache_b=all_reuse_cache,
                                    frame_start=prefix_len,
                                    frame_len=frame_token_length,
                                ),
                                "all_reuse_vs_all_recompute_cache_diff": summarize_cache_pair_diff(
                                    cache_a=all_reuse_cache,
                                    cache_b=all_recompute_cache,
                                    frame_start=prefix_len,
                                    frame_len=frame_token_length,
                                ),
                                "teacher_vs_all_recompute_logits_diff": summarize_logits_trace_diff(
                                    teacher_logits_trace,
                                    all_recompute_logits_trace,
                                    topk=debug_topk,
                                ),
                                "dense_teacher_vs_selective_all_recompute_layer_diff": summarize_frame_snapshot_diff(
                                    reference_snapshot=dense_teacher_snapshot,
                                    probe_snapshot=full_snapshot,
                                    layer_attention_types=layer_attention_types,
                                ),
                                "dense_teacher_vs_selective_all_recompute_feature_trace_diff": (
                                    summarize_decoder_feature_trace_diff(
                                        reference_trace=dense_teacher_feature_trace,
                                        probe_trace=selective_all_recompute_feature_trace,
                                    )
                                    if dense_teacher_feature_trace is not None
                                    and selective_all_recompute_feature_trace is not None
                                    else None
                                ),
                                "teacher_vs_all_reuse_logits_diff": summarize_logits_trace_diff(
                                    teacher_logits_trace,
                                    all_reuse_logits_trace,
                                    topk=debug_topk,
                                ),
                                "all_reuse_vs_all_recompute_logits_diff": summarize_logits_trace_diff(
                                    all_reuse_logits_trace,
                                    all_recompute_logits_trace,
                                    topk=debug_topk,
                                ),
                            }
                            if dense_teacher_cache is not None and dense_teacher_logits_trace is not None:
                                dense_cache_diff = summarize_cache_pair_diff(
                                    cache_a=dense_teacher_cache,
                                    cache_b=all_recompute_cache,
                                    frame_start=prefix_len,
                                    frame_len=frame_token_length,
                                )
                                dense_logits_diff = summarize_logits_trace_diff(
                                    dense_teacher_logits_trace,
                                    all_recompute_logits_trace,
                                    topk=debug_topk,
                                )
                                debug_counterfactual["dense_teacher_vs_selective_all_recompute_cache_diff"] = dense_cache_diff
                                debug_counterfactual["dense_teacher_vs_selective_all_recompute_logits_diff"] = dense_logits_diff
                                debug_counterfactual["dense_teacher_vs_selective_all_recompute_exact_match"] = (
                                    float(dense_cache_diff.get("max_abs_diff", 0.0)) == 0.0
                                    and float(dense_logits_diff.get("max_abs_logit_diff", 0.0)) == 0.0
                                )
                            if (
                                dense_teacher_cache is not None
                                and dense_teacher_logits_trace is not None
                                and dense_teacher_snapshot is not None
                                and legacy_selective_zero_reuse_snapshot is not None
                                and legacy_selective_all_recompute_cache is not None
                                and legacy_selective_all_recompute_logits_trace is not None
                            ):
                                debug_counterfactual["legacy_selective_zero_reuse_snapshot_vs_dense_layer_diff"] = summarize_frame_snapshot_diff(
                                    reference_snapshot=dense_teacher_snapshot,
                                    probe_snapshot=legacy_selective_zero_reuse_snapshot,
                                    layer_attention_types=layer_attention_types,
                                )
                                debug_counterfactual["legacy_selective_all_recompute_vs_dense_cache_diff"] = summarize_cache_pair_diff(
                                    cache_a=dense_teacher_cache,
                                    cache_b=legacy_selective_all_recompute_cache,
                                    frame_start=prefix_len,
                                    frame_len=frame_token_length,
                                )
                                debug_counterfactual["legacy_selective_all_recompute_vs_dense_logits_diff"] = summarize_logits_trace_diff(
                                    dense_teacher_logits_trace,
                                    legacy_selective_all_recompute_logits_trace,
                                    topk=debug_topk,
                                )
                                debug_counterfactual["legacy_selective_all_recompute_vs_effective_all_recompute_cache_diff"] = summarize_cache_pair_diff(
                                    cache_a=legacy_selective_all_recompute_cache,
                                    cache_b=all_recompute_cache,
                                    frame_start=prefix_len,
                                    frame_len=frame_token_length,
                                )
                                debug_counterfactual["legacy_selective_all_recompute_vs_effective_all_recompute_logits_diff"] = summarize_logits_trace_diff(
                                    legacy_selective_all_recompute_logits_trace,
                                    all_recompute_logits_trace,
                                    topk=debug_topk,
                                )

                        record = {
                            **make_record_base(sample, frame_idx, args.prompt_text),
                            "status": "ok",
                            "mode": args.mode,
                            "prev_frame_idx": int(frame_idx - 1),
                            "transition": f"{frame_idx - 1}->{frame_idx}",
                            "num_visual_tokens": num_visual_tokens,
                            "teacher_reference": teacher_reference,
                            "teacher_action_metric": "avg_decode_step_kl",
                            "oracle_text": oracle_row.get("oracle_text", ""),
                            "oracle_token_ids": oracle_token_ids,
                            "oracle_token_texts": oracle_row.get("oracle_token_texts", []),
                            "oracle_logprobs": oracle_row.get("oracle_logprobs", []),
                            "oracle_avg_nll": oracle_row.get("oracle_avg_nll", 0.0),
                            "group_sizes": group_sizes,
                            "attribution_method": "hierarchical_leave_one_out",
                            "hierarchical_final_score_aggregation": hierarchical_scores["final_score_aggregation"],
                            "hierarchical_candidate_ratios": HIERARCHICAL_STAGE_CANDIDATE_RATIOS,
                            "hierarchical_spatial_grid": {
                                "height": hierarchical_scores["spatial_grid_height"],
                                "width": hierarchical_scores["spatial_grid_width"],
                            },
                            "hierarchical_group_value_cache_size": hierarchical_scores["group_value_cache_size"],
                            "hierarchical_candidate_stage1": hierarchical_scores["candidate_stage1"],
                            "hierarchical_candidate_stage2": hierarchical_scores["candidate_stage2"],
                            "hierarchical_candidate_stage3": hierarchical_scores["candidate_stage3"],
                            "all_reuse_action_kl": float(all_reuse_action_kl),
                            "all_recompute_action_kl": float(all_recompute_action_kl),
                            "all_recompute_cache_mean_abs_diff": float(all_recompute_cache_diff["mean_abs_diff"]),
                            "all_recompute_cache_max_abs_diff": float(all_recompute_cache_diff["max_abs_diff"]),
                            "all_recompute_cache_layerwise_mean_abs_diff": all_recompute_cache_diff["layerwise_mean_abs_diff"],
                            "all_recompute_cache_layerwise_max_abs_diff": all_recompute_cache_diff["layerwise_max_abs_diff"],
                            "single_token_action_delta_scores": single_token_scores,
                            "size4_token_action_delta_scores": size4_scores,
                            "size9_token_action_delta_scores": size9_scores,
                            "size16_token_action_delta_scores": size16_scores,
                            "group_action_delta_scores": group_action_delta_scores,
                            "token_action_delta_scores": token_importance_scores,
                            "token_importance_scores": token_importance_scores,
                            "token_pixel_diff_score_metric": "raw_frame_patch_absdiff_mean",
                            "token_pixel_diff_scores": token_pixel_diff_scores,
                            "greedy_rank": greedy_rank,
                            "greedy_order": greedy_rank,
                        }
                        if debug_counterfactual is not None:
                            record["debug_counterfactual"] = debug_counterfactual
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fout.flush()
                        rows_written += 1

                        prev_full_snapshot = full_snapshot

                    prev_raw_frame = raw_frame

            except Exception as e:
                fout.write(
                    json.dumps(
                        {
                            "sample_id": sample.sample_id,
                            "video_path": sample.video_path,
                            "status": "error",
                            "mode": args.mode,
                            "error": repr(e),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                fout.flush()

    print(json.dumps({"path": output_jsonl, "rows_written": rows_written, "mode": args.mode}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
