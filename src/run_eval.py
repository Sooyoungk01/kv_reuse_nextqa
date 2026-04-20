from __future__ import annotations

import argparse
import json
import os
import sys
from time import perf_counter
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from tqdm import tqdm
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.nextqa_dataset import load_nextqa_samples
from data.video_io import get_video_frame_indices, iter_video_frames
from eval.metrics import summarize_file
from hf_patch.apply_runtime_patch import apply_runtime_patch
from kv_reuse.cache_ops import clone_cache, make_static_cache, static_cache_to_dynamic
from kv_reuse.pixel_change import decide_reuse_by_pixel_change
from kv_reuse.prefill import _build_layer_reuse_masks, _forward_frame_layerwise_with_selective_reuse, _forward_span
from kv_reuse.scoring import rank_multiple_choices
from kv_reuse.video_prompt import build_mc_prompt, encode_streaming_video_frame, prepare_streaming_prompt_template


DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--mode", type=str, choices=["baseline", "kv_reuse"], required=True)
    ap.add_argument("--output_jsonl", type=str, default=None)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--num_frames", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--video_root", type=str, default=None)
    ap.add_argument("--dataset_name", type=str, default=None)
    ap.add_argument("--dataset_config_name", type=str, default=None)
    ap.add_argument("--dataset_split", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dtype", type=str, default=None)
    ap.add_argument("--attn_implementation", type=str, default=None)
    return ap.parse_args()


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_cfg(cfg: Dict[str, Any], args) -> Dict[str, Any]:
    out = dict(cfg)
    for key in [
        "max_samples",
        "num_frames",
        "threshold",
        "video_root",
        "dataset_name",
        "dataset_config_name",
        "dataset_split",
        "device",
        "dtype",
        "attn_implementation",
    ]:
        value = getattr(args, key)
        if value is not None:
            out[key] = value

    if not out.get("video_root"):
        env_video_root = os.environ.get("NEXTQA_VIDEO_ROOT")
        if env_video_root:
            out["video_root"] = env_video_root
    return out


def resolve_runtime(cfg: Dict[str, Any]) -> tuple[torch.device, torch.dtype]:
    requested_device = str(cfg.get("device", "cuda")).strip()
    requested_dtype = str(cfg.get("dtype", "bfloat16")).strip().lower()

    if requested_dtype not in DTYPE_MAP:
        valid = ", ".join(sorted(DTYPE_MAP))
        raise ValueError(f"Unsupported dtype '{requested_dtype}'. Expected one of: {valid}")

    dtype = DTYPE_MAP[requested_dtype]
    if requested_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            print("CUDA is unavailable; falling back to CPU because device=auto.", file=sys.stderr)
    else:
        device = torch.device(requested_device)

    if device.type == "cuda" and not torch.cuda.is_available():
        torch_cuda = torch.version.cuda or "unknown"
        raise RuntimeError(
            "Requested a CUDA device, but CUDA is unavailable in this environment.\n"
            f"Installed torch build: {torch.__version__} (CUDA {torch_cuda}).\n"
            "This usually means the NVIDIA driver is older than the CUDA runtime bundled with PyTorch.\n"
            "Fix one of the following before rerunning:\n"
            "  1. Update the NVIDIA driver.\n"
            "  2. Install a PyTorch build compiled for an older CUDA runtime supported by this machine.\n"
            "  3. Run with --device cpu (or set device: auto / cpu in the config) for debugging."
        )

    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32
        print("float16 is not supported well on CPU; using float32 instead.", file=sys.stderr)

    return device, dtype


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _timer_start(device: torch.device | None = None, sync: bool = False) -> float:
    if sync and device is not None:
        _sync_device(device)
    return perf_counter()


def _elapsed_ms(start_time: float, device: torch.device | None = None, sync: bool = False) -> float:
    if sync and device is not None:
        _sync_device(device)
    return (perf_counter() - start_time) * 1000.0


def _cuda_event_start(device: torch.device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    return start, end


def _cuda_event_end(event_pair) -> None:
    if event_pair is None:
        return
    _, end = event_pair
    end.record()


def _collect_cuda_event_times(event_pairs: Dict[str, object]) -> Dict[str, float]:
    realized = {name: pair for name, pair in event_pairs.items() if pair is not None}
    if not realized:
        return {}

    last_end = next(reversed(realized.values()))[1]
    last_end.synchronize()
    out: Dict[str, float] = {}
    for name, (start, end) in realized.items():
        out[name] = float(start.elapsed_time(end))
    return out


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
def main():
    args = parse_args()
    cfg = merge_cfg(load_cfg(args.config), args)
    apply_runtime_patch()

    device, dtype = resolve_runtime(cfg)
    output_dir = Path(cfg.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_jsonl or str(output_dir / f"{args.mode}_results.jsonl")

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

    samples = load_nextqa_samples(
        dataset_name=cfg["dataset_name"],
        dataset_config_name=cfg.get("dataset_config_name"),
        split=cfg.get("dataset_split", "test"),
        video_root=cfg.get("video_root"),
        max_samples=cfg.get("max_samples"),
    )

    letters = ["A", "B", "C", "D", "E"]
    normalize = bool(cfg.get("scoring", {}).get("normalize_by_length", True))
    profile_timing = bool(cfg.get("profile_timing", False))

    n_processed = 0
    n_correct = 0
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for sample in tqdm(samples, desc=f"eval[{args.mode}]"):
            if sample.video_path is None or not os.path.exists(sample.video_path):
                record = {
                    "sample_id": sample.sample_id,
                    "status": "skipped_no_video",
                    "video_path": sample.video_path,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            try:
                sample_total_start = _timer_start()
                sample_profile_start = _timer_start(device, sync=True) if profile_timing else None

                step_start = _timer_start()
                frame_indices = get_video_frame_indices(sample.video_path, num_frames=cfg["num_frames"])
                frame_iter = iter_video_frames(sample.video_path, num_frames=cfg["num_frames"])
                sample_video_ms = _elapsed_ms(step_start)
                if not frame_indices:
                    raise RuntimeError(f"Selected zero frames for video: {sample.video_path}")

                step_start = _timer_start()
                prompt = build_mc_prompt(
                    question=sample.question,
                    choices=sample.choices,
                    instruction=cfg["question_template"],
                )
                prompt_build_ms = _elapsed_ms(step_start)

                step_start = _timer_start(device, sync=profile_timing)
                prepare_event = _cuda_event_start(device)
                first_frame = next(frame_iter)
                stream_template, first_frame_features = prepare_streaming_prompt_template(
                    model=model,
                    processor=processor,
                    first_frame=first_frame,
                    text_prompt=prompt,
                    device=device,
                    dtype=dtype,
                )
                _cuda_event_end(prepare_event)
                prepare_inputs_profile_ms = _elapsed_ms(step_start, device, sync=profile_timing)

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

                choice_letter_prefixes = [(L, f" {L}") for L in letters[: len(sample.choices)]]
                step_start = _timer_start(device, sync=profile_timing)
                stream_frame_encode_profile_ms = 0.0
                stream_prefill_profile_ms = 0.0
                online_scoring_profile_ms = 0.0
                online_predictions = []
                reuse_stats = []
                prefill_frame_latency_ms = []
                prev_raw_frame = None
                prev_snapshot = None
                final_pred_letter = None
                final_scored = None

                for stream_frame_idx in range(len(frame_indices)):
                    raw_frame = first_frame if stream_frame_idx == 0 else next(frame_iter)
                    frame_cache = clone_cache(prefix_cache_template)
                    frame_encode_ms = 0.0
                    if stream_frame_idx == 0:
                        frame_features = first_frame_features
                    else:
                        encode_start = _timer_start(device, sync=profile_timing)
                        frame_features = encode_streaming_video_frame(
                            model=model,
                            processor=processor,
                            frame=raw_frame,
                            device=device,
                            dtype=stream_template.prefix_embeds.dtype,
                        )
                        frame_encode_ms = _elapsed_ms(encode_start, device, sync=profile_timing)
                        stream_frame_encode_profile_ms += frame_encode_ms

                    frame_positions = torch.arange(
                        prefix_len,
                        prefix_len + frame_token_length,
                        device=device,
                        dtype=torch.long,
                    )

                    frame_prefill_start = _timer_start(device, sync=profile_timing)
                    if args.mode == "baseline":
                        outputs = _forward_span(
                            model.language_model,
                            frame_features.unsqueeze(0),
                            frame_cache,
                            cache_position=frame_positions,
                        )
                        frame_cache = outputs.past_key_values
                        frame_prefill_ms = _elapsed_ms(frame_prefill_start, device, sync=profile_timing)
                        stream_prefill_profile_ms += frame_prefill_ms
                        reuse_stats.append(
                            {
                                "frame_idx": int(stream_frame_idx),
                                "reused_tokens": 0,
                                "recomputed_tokens": int(frame_features.shape[0]),
                                "reuse_ratio": 0.0,
                            }
                        )
                    else:
                        if stream_frame_idx == 0:
                            base_reuse_mask = torch.zeros(
                                frame_features.shape[0],
                                dtype=torch.bool,
                                device=frame_features.device,
                            )
                        else:
                            decision = decide_reuse_by_pixel_change(
                                prev_raw_frame,
                                raw_frame,
                                threshold=cfg["threshold"],
                                append_newline_token=False,
                            )
                            base_reuse_mask = decision.reuse_mask.to(frame_features.device)
                            if base_reuse_mask.numel() != frame_features.shape[0]:
                                raise RuntimeError(
                                    f"Streaming reuse mask length mismatch for frame {stream_frame_idx}: "
                                    f"mask={base_reuse_mask.numel()} frame_tokens={frame_features.shape[0]}"
                                )
                            if prev_snapshot is not None:
                                prev_seq_len = prev_snapshot.layer_kv[0][0].shape[-2]
                                if prev_seq_len < base_reuse_mask.numel():
                                    base_reuse_mask = base_reuse_mask.clone()
                                    base_reuse_mask[prev_seq_len:] = False

                        layer_reuse_masks = _build_layer_reuse_masks(
                            base_reuse_mask,
                            len(model.language_model.model.layers),
                        )
                        frame_snapshot, _, layer_latency_detail, _ = _forward_frame_layerwise_with_selective_reuse(
                            language_model=model.language_model,
                            cache=frame_cache,
                            frame_features=frame_features,
                            frame_cache_positions=frame_positions,
                            prev_snapshot=prev_snapshot,
                            layer_reuse_masks=layer_reuse_masks,
                            device=device,
                            profile_timing=profile_timing,
                        )
                        prev_snapshot = frame_snapshot
                        frame_prefill_ms = _elapsed_ms(frame_prefill_start, device, sync=profile_timing)
                        stream_prefill_profile_ms += frame_prefill_ms
                        reuse_stats.append(
                            {
                                "frame_idx": int(stream_frame_idx),
                                "reused_tokens": int(base_reuse_mask.sum().item()),
                                "recomputed_tokens": int((~base_reuse_mask).sum().item()),
                                "reuse_ratio": float(base_reuse_mask.float().mean().item()) if base_reuse_mask.numel() else 0.0,
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

                    score_start = _timer_start(device, sync=profile_timing)
                    pred_letter, scored = _score_online_answer(
                        model=model,
                        tokenizer=processor.tokenizer,
                        live_cache=frame_cache,
                        current_seq_len=frame_seq_len,
                        newline_embed=stream_template.newline_embed,
                        suffix_input_ids=stream_template.suffix_input_ids,
                        choice_letter_prefixes=choice_letter_prefixes,
                        device=device,
                        normalize_by_length=normalize,
                    )
                    frame_score_ms = _elapsed_ms(score_start, device, sync=profile_timing)
                    online_scoring_profile_ms += frame_score_ms

                    pred_idx_step = letters.index(pred_letter)
                    online_predictions.append(
                        {
                            "frame_idx": int(stream_frame_idx),
                            "pred_letter": pred_letter,
                            "pred_idx": int(pred_idx_step),
                            "correct": bool(pred_idx_step == sample.answer_idx),
                            "prefill_ms": frame_prefill_ms,
                            "score_ms": frame_score_ms,
                        }
                    )
                    prefill_frame_latency_ms.append(
                        {
                            "frame_idx": float(stream_frame_idx),
                            "frame_encode": frame_encode_ms,
                            "frame_prefill": frame_prefill_ms,
                            "frame_score": frame_score_ms,
                            "total": frame_prefill_ms + frame_score_ms,
                        }
                    )
                    final_pred_letter = pred_letter
                    final_scored = scored
                    prev_raw_frame = raw_frame

                prefill_call_profile_ms = stream_prefill_profile_ms
                actual_gpu_times = _collect_cuda_event_times(
                    {
                        "prepare_inputs": prepare_event,
                    }
                )
                total_e2e_ms = _elapsed_ms(sample_total_start)
                prepare_inputs_ms = actual_gpu_times.get("prepare_inputs", prepare_inputs_profile_ms)
                prefill_call_ms = prefill_call_profile_ms
                scoring_ms = online_scoring_profile_ms
                prefill_latency_ms = {"total": prefill_call_ms}
                if final_pred_letter is None or final_scored is None:
                    raise RuntimeError("Streaming online evaluation produced no predictions")
                pred_idx = letters.index(final_pred_letter)
                correct = pred_idx == sample.answer_idx
                n_processed += 1
                n_correct += int(correct)

                latency_profile_ms = {}
                if profile_timing and sample_profile_start is not None:
                    latency_profile_ms = {
                        "sample_video": sample_video_ms,
                        "prompt_build": prompt_build_ms,
                        "prepare_inputs": prepare_inputs_profile_ms,
                        "frame_encode": stream_frame_encode_profile_ms,
                        "prefill_call": prefill_call_profile_ms,
                        "scoring": online_scoring_profile_ms,
                        "total_e2e": _elapsed_ms(sample_profile_start, device, sync=True),
                    }

                record = {
                    "sample_id": sample.sample_id,
                    "status": "ok",
                    "video_path": sample.video_path,
                    "question": sample.question,
                    "choices": sample.choices,
                    "gold_idx": int(sample.answer_idx),
                    "gold_letter": letters[sample.answer_idx],
                    "pred_idx": int(pred_idx),
                    "pred_letter": final_pred_letter,
                    "correct": bool(correct),
                    "scores": {
                        letter: {
                            "score": obj.score,
                            "avg_score": obj.avg_score,
                            "token_ids": obj.token_ids,
                        }
                        for letter, obj in final_scored
                    },
                    "processed_frames": int(len(frame_indices)),
                    "online_predictions": online_predictions,
                    "reuse_stats": reuse_stats,
                    "latency_ms": {
                        "sample_video": sample_video_ms,
                        "prompt_build": prompt_build_ms,
                        "prepare_inputs": prepare_inputs_ms,
                        "frame_encode": stream_frame_encode_profile_ms,
                        "prefill_call": prefill_call_ms,
                        "scoring": scoring_ms,
                        "total_e2e": total_e2e_ms,
                    },
                    "latency_profile_ms": latency_profile_ms,
                    "prefill_latency_ms": prefill_latency_ms,
                    "prefill_profile_latency_ms": {},
                    "prefill_frame_latency_ms": prefill_frame_latency_ms,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

            except Exception as e:
                record = {
                    "sample_id": sample.sample_id,
                    "status": "error",
                    "video_path": sample.video_path,
                    "error": repr(e),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

    summary = summarize_file(output_jsonl)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if n_processed > 0:
        print(f"processed={n_processed} correct={n_correct} acc={n_correct / n_processed:.4f}")
    else:
        print("processed=0; no evaluable samples were scored.", file=sys.stderr)


if __name__ == "__main__":
    main()
