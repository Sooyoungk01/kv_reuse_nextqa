#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cheap_signal_dump import (
    EXTRACTED_SIGNAL_KEYS,
    RECOMMENDED_LOGISTIC_FEATURE_KEYS,
    RECOMMENDED_TOPK_SIGNAL_KEYS,
    build_cheap_signal_state,
    build_transition_record,
    compute_exact_path_metric_bundle,
    compute_interaction_metric_bundle,
    compute_temporal_metric_bundle,
    init_metric_history,
    locate_prompt_text_token_positions,
    update_metric_history,
)

RESULTS_DIR = ROOT / "results"
DEFAULT_DRIFT_JSONL = RESULTS_DIR / "drift_oracle.jsonl"
DEFAULT_OUTPUT_JSONL = RESULTS_DIR / "cheap_signals.jsonl"
DEFAULT_PROMPT_TEXT = "What is happening now?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay existing drift_oracle transitions with the same Llava-OneVision checkpoint and dump exact-path cheap signals "
            "without re-running hierarchical oracle attribution."
        )
    )
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--drift_jsonl", type=str, default=str(DEFAULT_DRIFT_JSONL))
    parser.add_argument("--output_jsonl", type=str, default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--prompt_text", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--video_root", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--teacher_reference", type=str, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def rows_by_sample(path: Path) -> OrderedDict[str, list[dict[str, Any]]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in read_jsonl(path):
        if row.get("status") != "ok":
            continue
        frame_idx = int(row.get("frame_idx", -1))
        if frame_idx <= 0:
            continue
        sample_id = str(row.get("sample_id"))
        if sample_id not in grouped:
            grouped[sample_id] = []
        grouped[sample_id].append(row)
    for sample_rows in grouped.values():
        sample_rows.sort(key=lambda row: int(row.get("frame_idx", -1)))
    return grouped


def require_oracle_scores(row: dict[str, Any], num_visual_tokens: int) -> list[float]:
    for key in ("token_importance_scores", "token_action_delta_scores", "oracle_semantic_drift_scores"):
        scores = row.get(key)
        if isinstance(scores, list) and len(scores) == num_visual_tokens:
            return [float(x) for x in scores]
    raise ValueError(
        "Could not recover oracle token scores aligned with the current visual token count for "
        f"sample_id={row.get('sample_id')} frame_idx={row.get('frame_idx')} num_visual_tokens={num_visual_tokens}."
    )


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

    from data.video_io import iter_video_frames
    from hf_patch.apply_runtime_patch import apply_runtime_patch
    from kv_reuse.cache_ops import clone_cache, make_static_cache
    from kv_reuse.prefill import _forward_span
    from kv_reuse.video_prompt import encode_streaming_video_frame, prepare_streaming_prompt_template
    from runtime_config import load_cfg, merge_cfg, normalize_teacher_reference, resolve_runtime
    from teacher_reference import build_full_frame_snapshot, build_selective_all_recompute_feature_trace

    cfg = merge_cfg(load_cfg(args.config), args)
    apply_runtime_patch()
    teacher_reference = normalize_teacher_reference(cfg.get("teacher_reference"))
    if teacher_reference != "selective_all_recompute":
        raise ValueError(
            "extract_cheap_signals.py expects teacher_reference=selective_all_recompute so the dumped features match "
            "the selective-all-recompute path."
        )

    drift_path = Path(args.drift_jsonl)
    output_path = Path(args.output_jsonl)
    sample_rows = rows_by_sample(drift_path)
    if args.max_samples is not None:
        sample_rows = OrderedDict(list(sample_rows.items())[: args.max_samples])

    device, dtype = resolve_runtime(cfg)
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    error_count = 0

    with output_path.open("w", encoding="utf-8") as fout:
        sample_progress = tqdm(
            sample_rows.items(),
            total=len(sample_rows),
            desc="cheap_signals",
            dynamic_ncols=True,
        )
        for sample_id, rows in sample_progress:
            sample_progress.set_postfix_str(str(sample_id))
            if not rows:
                continue

            first_row = rows[0]
            video_path = first_row.get("video_path")
            if not isinstance(video_path, str) or not os.path.exists(video_path):
                fout.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "video_path": video_path,
                            "status": "skipped_no_video",
                            "mode": "cheap_signals",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            try:
                prompt_text = str(first_row.get("prompt_text") or args.prompt_text or DEFAULT_PROMPT_TEXT)
                sample_num_frames = max(int(row.get("frame_idx", -1)) for row in rows) + 1
                row_index = {int(row["frame_idx"]): row for row in rows}

                frame_iter = iter_video_frames(video_path, num_frames=sample_num_frames)
                first_frame = next(frame_iter)
                stream_template, first_frame_features = prepare_streaming_prompt_template(
                    model=model,
                    processor=processor,
                    first_frame=first_frame,
                    text_prompt=prompt_text,
                    device=device,
                    dtype=dtype,
                )
                prompt_text_token_positions = locate_prompt_text_token_positions(
                    processor.tokenizer,
                    stream_template.suffix_input_ids,
                    prompt_text,
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

                prev_full_snapshot = None
                prev_cheap_state = None
                metric_history = init_metric_history()

                frame_progress = tqdm(
                    range(sample_num_frames),
                    total=sample_num_frames,
                    desc=f"{sample_id} frames",
                    leave=False,
                    dynamic_ncols=True,
                )
                for frame_idx in frame_progress:
                    frame_progress.set_postfix(frame=frame_idx)
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
                    selective_feature_trace = build_selective_all_recompute_feature_trace(
                        language_model=model.language_model,
                        prefix_cache_template=prefix_cache_template,
                        frame_features=frame_features,
                        frame_positions=frame_positions,
                        prev_full_snapshot=prev_full_snapshot,
                        device=device,
                    )
                    current_cheap_state = build_cheap_signal_state(
                        language_model=model.language_model,
                        prefix_cache_template=prefix_cache_template,
                        frame_positions=frame_positions,
                        frame_seq_len=frame_seq_len,
                        frame_features=frame_features,
                        full_snapshot=full_snapshot,
                        selective_feature_trace=selective_feature_trace,
                        newline_embed=stream_template.newline_embed,
                        suffix_input_ids=stream_template.suffix_input_ids,
                        prompt_text_token_positions=prompt_text_token_positions,
                    )

                    if frame_idx == 0 or prev_full_snapshot is None or prev_cheap_state is None:
                        prev_full_snapshot = full_snapshot
                        prev_cheap_state = current_cheap_state
                        continue

                    drift_row = row_index.get(frame_idx)
                    if drift_row is None:
                        prev_full_snapshot = full_snapshot
                        prev_cheap_state = current_cheap_state
                        continue

                    num_visual_tokens = int(frame_features.shape[0])
                    oracle_scores = require_oracle_scores(drift_row, num_visual_tokens)
                    base_metrics, metric_metadata = compute_exact_path_metric_bundle(prev_cheap_state, current_cheap_state)
                    temporal_metrics = compute_temporal_metric_bundle(base_metrics, metric_history)
                    interaction_metrics = compute_interaction_metric_bundle(base_metrics)
                    metric_tensors = {
                        **base_metrics,
                        **temporal_metrics,
                        **interaction_metrics,
                    }

                    sample_stub = SimpleNamespace(
                        sample_id=sample_id,
                        video_path=video_path,
                        question=str(drift_row.get("original_question", "")),
                        choices=drift_row.get("choices", []),
                        answer_idx=drift_row.get("gold_idx"),
                    )
                    record = build_transition_record(
                        sample=sample_stub,
                        frame_idx=frame_idx,
                        prompt_text=prompt_text,
                        teacher_reference=teacher_reference,
                        oracle_scores=oracle_scores,
                        metric_tensors=metric_tensors,
                        metric_metadata=metric_metadata,
                    )
                    record["mode"] = "cheap_signals"
                    record["cheap_signal_source"] = "extract_cheap_signals.py::selective_all_recompute"
                    record["oracle_jsonl"] = str(drift_path)
                    record["oracle_text"] = drift_row.get("oracle_text", "")
                    record["oracle_token_ids"] = drift_row.get("oracle_token_ids", [])
                    record["oracle_token_texts"] = drift_row.get("oracle_token_texts", [])
                    record["oracle_logprobs"] = drift_row.get("oracle_logprobs", [])
                    record["oracle_avg_nll"] = drift_row.get("oracle_avg_nll", 0.0)

                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    rows_written += 1
                    update_metric_history(metric_history, metric_tensors)
                    prev_full_snapshot = full_snapshot
                    prev_cheap_state = current_cheap_state

            except Exception as exc:
                error_count += 1
                fout.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "video_path": video_path,
                            "status": "error",
                            "mode": "cheap_signals",
                            "error": repr(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                fout.flush()

    print(
        json.dumps(
            {
                "path": str(output_path),
                "rows_written": rows_written,
                "mode": "cheap_signals",
                "drift_jsonl": str(drift_path),
                "teacher_reference": teacher_reference,
                "sample_count": len(sample_rows),
                "error_count": error_count,
                "cheap_signal_keys": list(EXTRACTED_SIGNAL_KEYS),
                "recommended_topk_retrieval_signal_keys": list(RECOMMENDED_TOPK_SIGNAL_KEYS),
                "recommended_logistic_feature_keys": list(RECOMMENDED_LOGISTIC_FEATURE_KEYS),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
