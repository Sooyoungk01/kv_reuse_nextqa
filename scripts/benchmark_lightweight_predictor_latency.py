#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.nextqa_dataset import load_nextqa_samples
from data.video_io import get_video_frame_indices, iter_video_frames
from hf_patch.apply_runtime_patch import apply_runtime_patch
from kv_reuse.cache_ops import clone_cache, make_static_cache, static_cache_to_dynamic
from kv_reuse.prefill import _forward_span
from kv_reuse.scoring import rank_multiple_choices
from kv_reuse.video_prompt import (
    build_mc_prompt,
    encode_streaming_video_frame,
    prepare_streaming_prompt_template,
)
from runtime_config import load_cfg, merge_cfg, resolve_runtime

LETTERS = ["A", "B", "C", "D", "E"]


class LightweightTransformerPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        max_seq_len: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"predictor_hidden_dim ({hidden_dim}) must be divisible by predictor_num_heads ({num_heads})"
            )

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        if token_features.dim() != 3:
            raise ValueError(
                f"Expected predictor input with shape [batch, tokens, dim], got {tuple(token_features.shape)}"
            )

        if token_features.shape[1] > self.pos_embed.shape[1]:
            raise ValueError(
                f"Predictor max_seq_len={self.pos_embed.shape[1]} is smaller than token length={token_features.shape[1]}"
            )

        x = self.input_proj(token_features)
        x = x + self.pos_embed[:, : x.shape[1], :]
        x = self.encoder(x)
        return self.output_head(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark streaming per-frame latency for LLaVA-OneVision with and without "
            "a 4-layer lightweight transformer predictor using dummy token-wise inputs "
            "that match last-layer full attention-map rows plus full embedding-drift vectors."
        )
    )
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=str(ROOT / "results" / "lightweight_predictor_latency.jsonl"),
    )
    parser.add_argument(
        "--output_summary_json",
        type=str,
        default=str(ROOT / "results" / "lightweight_predictor_latency_summary.json"),
    )
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--video_root", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--benchmark_runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--predictor_hidden_dim", type=int, default=128)
    parser.add_argument("--predictor_num_layers", type=int, default=4)
    parser.add_argument("--predictor_num_heads", type=int, default=4)
    parser.add_argument("--predictor_ffn_dim", type=int, default=512)
    parser.add_argument("--predictor_max_seq_len", type=int, default=512)
    parser.add_argument("--predictor_dropout", type=float, default=0.0)
    parser.add_argument("--predictor_checkpoint", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _timer_start(device: torch.device | None = None, sync: bool = True) -> float:
    if sync and device is not None:
        _sync_device(device)
    return perf_counter()


def _elapsed_ms(start_time: float, device: torch.device | None = None, sync: bool = True) -> float:
    if sync and device is not None:
        _sync_device(device)
    return (perf_counter() - start_time) * 1000.0


def _summarize(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}

    ordered = sorted(float(x) for x in values)
    n = len(ordered)
    p50_idx = (n - 1) // 2
    p95_idx = min(n - 1, max(0, int(round((n - 1) * 0.95))))
    return {
        "avg": float(sum(ordered) / n),
        "p50": float(ordered[p50_idx]),
        "p95": float(ordered[p95_idx]),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def _mean(values) -> float:
    ordered = [float(v) for v in values]
    if not ordered:
        return 0.0
    return float(sum(ordered) / len(ordered))


def _build_dummy_predictor_inputs(
    *,
    batch_size: int,
    frame_token_length: int,
    attention_context_len: int,
    num_attention_heads: int,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    token_feature_dim = (num_attention_heads * attention_context_len) + hidden_size
    return torch.zeros(
        batch_size,
        frame_token_length,
        token_feature_dim,
        device=device,
        dtype=dtype,
    )


def _get_or_create_predictor(
    *,
    predictor_cache: Dict[int, LightweightTransformerPredictor],
    input_dim: int,
    args: argparse.Namespace,
    device: torch.device,
    predictor_dtype: torch.dtype,
) -> LightweightTransformerPredictor:
    predictor = predictor_cache.get(input_dim)
    if predictor is not None:
        return predictor

    predictor = LightweightTransformerPredictor(
        input_dim=input_dim,
        hidden_dim=int(args.predictor_hidden_dim),
        num_layers=int(args.predictor_num_layers),
        num_heads=int(args.predictor_num_heads),
        ffn_dim=int(args.predictor_ffn_dim),
        max_seq_len=int(args.predictor_max_seq_len),
        dropout=float(args.predictor_dropout),
    ).to(device=device, dtype=predictor_dtype)
    predictor.eval()

    if args.predictor_checkpoint:
        state = torch.load(args.predictor_checkpoint, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        predictor.load_state_dict(state, strict=True)

    predictor_cache[input_dim] = predictor
    return predictor


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
def _score_online_answer(
    model,
    tokenizer,
    live_cache,
    current_seq_len: int,
    newline_embed: torch.Tensor,
    suffix_input_ids: torch.Tensor,
    choice_letter_prefixes,
    device: torch.device,
    normalize_by_length: bool,
):
    score_cache = static_cache_to_dynamic(live_cache, seq_len=current_seq_len)
    next_position = int(current_seq_len)

    score_cache, next_token_logits, next_position = _forward_decode_span(
        model=model,
        cache=score_cache,
        start_position=next_position,
        inputs_embeds=newline_embed.view(1, 1, -1),
    )

    if suffix_input_ids.shape[1] > 0:
        score_cache, next_token_logits, next_position = _forward_decode_span(
            model=model,
            cache=score_cache,
            start_position=next_position,
            input_ids=suffix_input_ids,
        )

    pred_letter, scored = rank_multiple_choices(
        model=model,
        tokenizer=tokenizer,
        past_key_values=score_cache,
        initial_logits=next_token_logits,
        choice_letter_prefixes=choice_letter_prefixes,
        device=device,
        normalize_by_length=normalize_by_length,
    )
    return pred_letter, scored


@torch.inference_mode()
def run_single_stream(
    *,
    model,
    processor,
    predictor_cache: Dict[int, LightweightTransformerPredictor],
    predictor_args: argparse.Namespace,
    sample,
    cfg: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    predictor_dtype: torch.dtype,
    include_predictor: bool,
) -> Dict[str, Any]:
    num_frames = int(cfg["num_frames"])
    question_template = str(cfg["question_template"])
    normalize_scores = bool(cfg.get("scoring", {}).get("normalize_by_length", True))

    frame_indices = get_video_frame_indices(sample.video_path, num_frames=num_frames)
    if len(frame_indices) != num_frames:
        raise RuntimeError(
            f"Expected {num_frames} sampled frames, but got {len(frame_indices)} for {sample.video_path}"
        )

    prompt_start = perf_counter()
    prompt = build_mc_prompt(
        question=sample.question,
        choices=sample.choices,
        instruction=question_template,
    )
    prompt_build_ms = (perf_counter() - prompt_start) * 1000.0

    frame_iter = iter_video_frames(sample.video_path, num_frames=num_frames)
    first_frame = next(frame_iter)

    stream_prepare_start = _timer_start(device, sync=True)
    stream_template, first_frame_features = prepare_streaming_prompt_template(
        model=model,
        processor=processor,
        first_frame=first_frame,
        text_prompt=prompt,
        device=device,
        dtype=dtype,
    )
    stream_prepare_ms = _elapsed_ms(stream_prepare_start, device, sync=True)

    prefix_len = int(stream_template.prefix_embeds.shape[1])
    frame_token_length = int(stream_template.frame_token_length)
    frame_seq_len = prefix_len + frame_token_length
    frame_positions = torch.arange(
        prefix_len,
        prefix_len + frame_token_length,
        device=device,
        dtype=torch.long,
    )

    prefix_prefill_start = _timer_start(device, sync=True)
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
    prefix_prefill_ms = _elapsed_ms(prefix_prefill_start, device, sync=True)

    choice_letter_prefixes = [(letter, f" {letter}") for letter in LETTERS[: len(sample.choices)]]
    hidden_size = int(model.config.text_config.hidden_size)
    num_attention_heads = int(model.config.text_config.num_attention_heads)
    predictor_input_dim = (num_attention_heads * frame_seq_len) + hidden_size
    frame_metrics: List[Dict[str, Any]] = []
    final_pred_letter = None
    final_scored = None

    for frame_idx in range(num_frames):
        frame_total_start = _timer_start(device, sync=True)

        if frame_idx == 0:
            frame_features = first_frame_features
            frame_encode_ms = 0.0
        else:
            raw_frame = next(frame_iter)
            frame_encode_start = _timer_start(device, sync=True)
            frame_features = encode_streaming_video_frame(
                model=model,
                processor=processor,
                frame=raw_frame,
                device=device,
                dtype=stream_template.prefix_embeds.dtype,
            )
            frame_encode_ms = _elapsed_ms(frame_encode_start, device, sync=True)

        predictor_prepare_ms = 0.0
        predictor_forward_ms = 0.0
        predictor_output_mean = None
        predictor_output_std = None
        if include_predictor:
            predictor = _get_or_create_predictor(
                predictor_cache=predictor_cache,
                input_dim=predictor_input_dim,
                args=predictor_args,
                device=device,
                predictor_dtype=predictor_dtype,
            )
            predictor_prepare_start = _timer_start(device, sync=True)
            predictor_inputs = _build_dummy_predictor_inputs(
                batch_size=1,
                frame_token_length=frame_token_length,
                attention_context_len=frame_seq_len,
                num_attention_heads=num_attention_heads,
                hidden_size=hidden_size,
                device=device,
                dtype=predictor_dtype,
            )
            predictor_prepare_ms = _elapsed_ms(predictor_prepare_start, device, sync=True)

            predictor_forward_start = _timer_start(device, sync=True)
            predictor_output = predictor(predictor_inputs).detach()
            predictor_forward_ms = _elapsed_ms(predictor_forward_start, device, sync=True)
            predictor_output_mean = float(predictor_output.mean().item())
            predictor_output_std = float(predictor_output.std(unbiased=False).item())

        frame_cache = clone_cache(prefix_cache_template)
        prefill_start = _timer_start(device, sync=True)
        outputs = _forward_span(
            model.language_model,
            frame_features.unsqueeze(0),
            frame_cache,
            cache_position=frame_positions,
        )
        frame_cache = outputs.past_key_values
        prefill_ms = _elapsed_ms(prefill_start, device, sync=True)

        scoring_start = _timer_start(device, sync=True)
        pred_letter, scored = _score_online_answer(
            model=model,
            tokenizer=processor.tokenizer,
            live_cache=frame_cache,
            current_seq_len=frame_seq_len,
            newline_embed=stream_template.newline_embed,
            suffix_input_ids=stream_template.suffix_input_ids,
            choice_letter_prefixes=choice_letter_prefixes,
            device=device,
            normalize_by_length=normalize_scores,
        )
        scoring_ms = _elapsed_ms(scoring_start, device, sync=True)

        frame_total_ms = _elapsed_ms(frame_total_start, device, sync=True)
        pred_idx = LETTERS.index(pred_letter)
        frame_metrics.append(
            {
                "frame_idx": int(frame_idx),
                "pred_letter": pred_letter,
                "pred_idx": int(pred_idx),
                "correct": bool(pred_idx == sample.answer_idx),
                "frame_encode_ms": float(frame_encode_ms),
                "predictor_prepare_ms": float(predictor_prepare_ms),
                "predictor_forward_ms": float(predictor_forward_ms),
                "predictor_total_ms": float(predictor_prepare_ms + predictor_forward_ms),
                "prefill_ms": float(prefill_ms),
                "scoring_ms": float(scoring_ms),
                "total_ms": float(frame_total_ms),
                "predictor_output_mean": predictor_output_mean,
                "predictor_output_std": predictor_output_std,
            }
        )
        final_pred_letter = pred_letter
        final_scored = scored

    if final_pred_letter is None or final_scored is None:
        raise RuntimeError("Streaming benchmark produced no frame-level predictions")

    frame_total_values = [float(frame["total_ms"]) for frame in frame_metrics]
    predictor_total_values = [float(frame["predictor_total_ms"]) for frame in frame_metrics]
    predictor_forward_values = [float(frame["predictor_forward_ms"]) for frame in frame_metrics]

    setup_ms = float(prompt_build_ms + stream_prepare_ms + prefix_prefill_ms)
    online_total_ms = float(sum(frame_total_values))
    total_ms = float(setup_ms + online_total_ms)
    pred_idx = LETTERS.index(final_pred_letter)
    return {
        "pred_letter": final_pred_letter,
        "pred_idx": int(pred_idx),
        "correct": bool(pred_idx == sample.answer_idx),
        "prompt_build_ms": float(prompt_build_ms),
        "stream_prepare_ms": float(stream_prepare_ms),
        "prefix_prefill_ms": float(prefix_prefill_ms),
        "setup_ms": float(setup_ms),
        "setup_per_frame_ms": float(setup_ms / num_frames),
        "online_total_ms": float(online_total_ms),
        "total_ms": float(total_ms),
        "online_frame_total_ms_avg": float(online_total_ms / num_frames),
        "amortized_frame_total_ms_avg": float(total_ms / num_frames),
        "frame_encode_ms_avg": _mean(float(frame["frame_encode_ms"]) for frame in frame_metrics),
        "predictor_prepare_ms_avg": _mean(float(frame["predictor_prepare_ms"]) for frame in frame_metrics),
        "predictor_forward_ms_avg": _mean(predictor_forward_values),
        "predictor_total_ms_avg": _mean(predictor_total_values),
        "prefill_ms_avg": _mean(float(frame["prefill_ms"]) for frame in frame_metrics),
        "scoring_ms_avg": _mean(float(frame["scoring_ms"]) for frame in frame_metrics),
        "predictor_output_mean_avg": _mean(
            float(frame["predictor_output_mean"])
            for frame in frame_metrics
            if frame["predictor_output_mean"] is not None
        ),
        "predictor_output_std_avg": _mean(
            float(frame["predictor_output_std"])
            for frame in frame_metrics
            if frame["predictor_output_std"] is not None
        ),
        "predictor_input_dim": int(predictor_input_dim),
        "predictor_token_count": int(frame_token_length),
        "predictor_attention_context_len": int(frame_seq_len),
        "predictor_attention_heads": int(num_attention_heads),
        "frame_metrics": frame_metrics,
        "scores": {
            letter: {
                "score": float(obj.score),
                "avg_score": float(obj.avg_score),
            }
            for letter, obj in final_scored
        },
    }


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    baseline_amortized = [float(r["baseline_amortized_frame_total_ms_avg"]) for r in records]
    with_predictor_amortized = [float(r["with_predictor_amortized_frame_total_ms_avg"]) for r in records]
    predictor_totals = [float(r["predictor_frame_total_ms_avg"]) for r in records]
    predictor_forwards = [float(r["predictor_forward_frame_ms_avg"]) for r in records]
    overheads = [float(r["predictor_overhead_frame_ms_avg"]) for r in records]

    return {
        "num_records": len(records),
        "baseline_amortized_frame_total_ms": _summarize(baseline_amortized),
        "with_predictor_amortized_frame_total_ms": _summarize(with_predictor_amortized),
        "predictor_frame_total_ms": _summarize(predictor_totals),
        "predictor_forward_frame_ms": _summarize(predictor_forwards),
        "predictor_overhead_frame_ms": _summarize(overheads),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    cfg = merge_cfg(load_cfg(args.config), args)
    cfg["num_frames"] = int(args.num_frames)
    cfg["max_samples"] = int(args.max_samples)

    apply_runtime_patch()
    set_seed(args.seed)

    device, dtype = resolve_runtime(cfg)
    predictor_dtype = dtype if device.type == "cuda" else torch.float32

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json = Path(args.output_summary_json)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(cfg["model_name"])
    processor.tokenizer.padding_side = "left"

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        cfg["model_name"],
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=str(cfg.get("attn_implementation", "sdpa")),
    ).to(device)
    model.eval()

    predictor_cache: Dict[int, LightweightTransformerPredictor] = {}

    samples = load_nextqa_samples(
        dataset_name=cfg["dataset_name"],
        dataset_config_name=cfg.get("dataset_config_name"),
        split=cfg.get("dataset_split", "test"),
        video_root=cfg.get("video_root"),
        max_samples=cfg.get("max_samples"),
    )

    total_runs = int(args.warmup_runs) + int(args.benchmark_runs)
    if total_runs <= 0:
        raise ValueError("warmup_runs + benchmark_runs must be >= 1")
    if int(args.benchmark_runs) <= 0:
        raise ValueError("benchmark_runs must be >= 1")

    records: List[Dict[str, Any]] = []
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for sample in tqdm(samples, desc="predictor-latency"):
            if sample.video_path is None or not os.path.exists(sample.video_path):
                record = {
                    "sample_id": sample.sample_id,
                    "video_path": sample.video_path,
                    "status": "skipped_no_video",
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            try:
                baseline_runs: List[Dict[str, Any]] = []
                predictor_runs: List[Dict[str, Any]] = []

                for run_idx in range(total_runs):
                    baseline_result = run_single_stream(
                        model=model,
                        processor=processor,
                        predictor_cache=predictor_cache,
                        predictor_args=args,
                        sample=sample,
                        cfg=cfg,
                        device=device,
                        dtype=dtype,
                        predictor_dtype=predictor_dtype,
                        include_predictor=False,
                    )
                    predictor_result = run_single_stream(
                        model=model,
                        processor=processor,
                        predictor_cache=predictor_cache,
                        predictor_args=args,
                        sample=sample,
                        cfg=cfg,
                        device=device,
                        dtype=dtype,
                        predictor_dtype=predictor_dtype,
                        include_predictor=True,
                    )

                    if run_idx >= int(args.warmup_runs):
                        baseline_runs.append(baseline_result)
                        predictor_runs.append(predictor_result)

                baseline_amortized_values = [float(run["amortized_frame_total_ms_avg"]) for run in baseline_runs]
                with_predictor_amortized_values = [
                    float(run["amortized_frame_total_ms_avg"]) for run in predictor_runs
                ]
                predictor_total_values = [float(run["predictor_total_ms_avg"]) for run in predictor_runs]
                predictor_forward_values = [float(run["predictor_forward_ms_avg"]) for run in predictor_runs]
                overhead_values = [
                    float(pred["amortized_frame_total_ms_avg"] - base["amortized_frame_total_ms_avg"])
                    for base, pred in zip(baseline_runs, predictor_runs)
                ]

                record = {
                    "sample_id": sample.sample_id,
                    "video_path": sample.video_path,
                    "status": "ok",
                    "question": sample.question,
                    "num_frames": int(args.num_frames),
                    "gold_idx": int(sample.answer_idx),
                    "gold_letter": LETTERS[sample.answer_idx],
                    "benchmark_runs": int(args.benchmark_runs),
                    "warmup_runs": int(args.warmup_runs),
                    "baseline_pred_letter": baseline_runs[-1]["pred_letter"],
                    "with_predictor_pred_letter": predictor_runs[-1]["pred_letter"],
                    "baseline_correct": bool(baseline_runs[-1]["correct"]),
                    "with_predictor_correct": bool(predictor_runs[-1]["correct"]),
                    "baseline_amortized_frame_total_ms_avg": _mean(baseline_amortized_values),
                    "with_predictor_amortized_frame_total_ms_avg": _mean(with_predictor_amortized_values),
                    "predictor_frame_total_ms_avg": _mean(predictor_total_values),
                    "predictor_forward_frame_ms_avg": _mean(predictor_forward_values),
                    "predictor_overhead_frame_ms_avg": _mean(overhead_values),
                    "baseline_amortized_frame_total_ms_runs": baseline_amortized_values,
                    "with_predictor_amortized_frame_total_ms_runs": with_predictor_amortized_values,
                    "predictor_frame_total_ms_runs": predictor_total_values,
                    "predictor_forward_frame_ms_runs": predictor_forward_values,
                    "predictor_overhead_frame_ms_runs": overhead_values,
                    "baseline_component_avg_ms": {
                        key: _mean(float(run[key]) for run in baseline_runs)
                        for key in [
                            "prompt_build_ms",
                            "stream_prepare_ms",
                            "prefix_prefill_ms",
                            "setup_ms",
                            "setup_per_frame_ms",
                            "online_frame_total_ms_avg",
                            "frame_encode_ms_avg",
                            "prefill_ms_avg",
                            "scoring_ms_avg",
                        ]
                    },
                    "with_predictor_component_avg_ms": {
                        key: _mean(float(run[key]) for run in predictor_runs)
                        for key in [
                            "prompt_build_ms",
                            "stream_prepare_ms",
                            "prefix_prefill_ms",
                            "setup_ms",
                            "setup_per_frame_ms",
                            "online_frame_total_ms_avg",
                            "frame_encode_ms_avg",
                            "predictor_prepare_ms_avg",
                            "predictor_forward_ms_avg",
                            "predictor_total_ms_avg",
                            "prefill_ms_avg",
                            "scoring_ms_avg",
                        ]
                    },
                    "predictor_output_mean_avg": _mean(
                        float(run["predictor_output_mean_avg"]) for run in predictor_runs
                    ),
                    "predictor_output_std_avg": _mean(
                        float(run["predictor_output_std_avg"]) for run in predictor_runs
                    ),
                    "predictor_input_dim": int(predictor_runs[-1]["predictor_input_dim"]),
                    "predictor_token_count": int(predictor_runs[-1]["predictor_token_count"]),
                    "predictor_attention_context_len": int(predictor_runs[-1]["predictor_attention_context_len"]),
                    "predictor_attention_heads": int(predictor_runs[-1]["predictor_attention_heads"]),
                    "baseline_frame_metrics_last_run": baseline_runs[-1]["frame_metrics"],
                    "with_predictor_frame_metrics_last_run": predictor_runs[-1]["frame_metrics"],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                records.append(record)
            except Exception as exc:
                record = {
                    "sample_id": sample.sample_id,
                    "video_path": sample.video_path,
                    "status": "error",
                    "error": repr(exc),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

    summary = {
        "model_name": cfg["model_name"],
        "dataset_name": cfg["dataset_name"],
        "dataset_config_name": cfg.get("dataset_config_name"),
        "dataset_split": cfg.get("dataset_split"),
        "num_frames": int(args.num_frames),
        "max_samples": int(args.max_samples),
        "benchmark_runs": int(args.benchmark_runs),
        "warmup_runs": int(args.warmup_runs),
        "predictor": {
            "type": "streaming_tokenwise_transformer_encoder_dummy_inputs",
            "input_features": ["dummy_last_layer_attention_map", "dummy_embedding_drift"],
            "num_layers": int(args.predictor_num_layers),
            "hidden_dim": int(args.predictor_hidden_dim),
            "num_heads": int(args.predictor_num_heads),
            "ffn_dim": int(args.predictor_ffn_dim),
            "max_seq_len": int(args.predictor_max_seq_len),
            "dropout": float(args.predictor_dropout),
            "checkpoint": args.predictor_checkpoint,
        },
        "results": summarize_records(records),
        "output_jsonl": str(output_jsonl),
    }

    output_summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
