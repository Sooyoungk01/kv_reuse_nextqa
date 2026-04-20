from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from time import perf_counter
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
from hf_patch.apply_runtime_patch import apply_runtime_patch
from kv_reuse.cache_ops import clone_cache, make_static_cache, static_cache_to_dynamic
from kv_reuse.pixel_change import decide_reuse_by_pixel_change
from kv_reuse.prefill import _build_layer_reuse_masks, _forward_frame_layerwise_with_selective_reuse, _forward_span
from kv_reuse.video_prompt import encode_streaming_video_frame, prepare_streaming_prompt_template


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
    ap.add_argument("--mode", type=str, choices=["baseline", "kv_reuse"], default="baseline")
    ap.add_argument("--output_csv", type=str, default="results/baseline_frame_text.csv")
    ap.add_argument("--prompt_text", type=str, default="What is happening now?")
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
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=None)
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
        "max_new_tokens",
    ]:
        value = getattr(args, key)
        if value is not None:
            out[key] = value

    if args.temperature is not None:
        out.setdefault("generation", {})
        out["generation"]["temperature"] = args.temperature
    if args.do_sample:
        out.setdefault("generation", {})
        out["generation"]["do_sample"] = True

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)

    if device.type == "cuda" and not torch.cuda.is_available():
        torch_cuda = torch.version.cuda or "unknown"
        raise RuntimeError(
            "Requested a CUDA device, but CUDA is unavailable in this environment.\n"
            f"Installed torch build: {torch.__version__} (CUDA {torch_cuda})."
        )

    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32

    return device, dtype


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
def _generate_online_text(
    model,
    tokenizer,
    live_cache,
    current_seq_len: int,
    newline_embed: torch.Tensor,
    suffix_input_ids: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> str:
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

    stop_ids = {tokenizer.eos_token_id}
    if tokenizer.pad_token_id is not None:
        stop_ids.add(tokenizer.pad_token_id)

    generated_ids = []
    for _ in range(max_new_tokens):
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

        generated_ids.append(token_id)
        cache, next_token_logits, next_position = _forward_decode_span(
            model=model,
            cache=cache,
            start_position=next_position,
            input_ids=next_token,
        )

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    cfg = merge_cfg(load_cfg(args.config), args)
    apply_runtime_patch()

    device, dtype = resolve_runtime(cfg)
    output_dir = Path(cfg.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_csv or str(output_dir / f"{args.mode}_frame_text.csv")

    generation_cfg = cfg.get("generation", {})
    max_new_tokens = int(cfg.get("max_new_tokens", 32))
    do_sample = bool(generation_cfg.get("do_sample", False))
    temperature = float(generation_cfg.get("temperature", 0.0))

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

    fieldnames = [
        "sample_id",
        "video_path",
        "frame_idx",
        "status",
        "error",
        "frame_encode_ms",
        "frame_prefill_ms",
        "frame_generate_ms",
        "frame_e2e_ms",
        "prompt_text",
        "generated_text",
        "original_question",
        "choices_json",
        "gold_idx",
        "mode",
    ]

    n_rows = 0
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sample in tqdm(samples, desc=f"frame_text[{args.mode}]"):
            if sample.video_path is None or not os.path.exists(sample.video_path):
                writer.writerow(
                    {
                        "sample_id": sample.sample_id,
                        "video_path": sample.video_path,
                        "frame_idx": "",
                        "status": "skipped_no_video",
                        "error": "",
                        "frame_encode_ms": "",
                        "frame_prefill_ms": "",
                        "frame_generate_ms": "",
                        "frame_e2e_ms": "",
                        "prompt_text": args.prompt_text,
                        "generated_text": "",
                        "original_question": sample.question,
                        "choices_json": json.dumps(sample.choices, ensure_ascii=False),
                        "gold_idx": sample.answer_idx,
                        "mode": args.mode,
                    }
                )
                continue

            try:
                frame_indices = get_video_frame_indices(sample.video_path, num_frames=cfg["num_frames"])
                frame_iter = iter_video_frames(sample.video_path, num_frames=cfg["num_frames"])
                if not frame_indices:
                    raise RuntimeError(f"Selected zero frames for video: {sample.video_path}")

                first_frame = next(frame_iter)
                prepare_start = perf_counter()
                stream_template, first_frame_features = prepare_streaming_prompt_template(
                    model=model,
                    processor=processor,
                    first_frame=first_frame,
                    text_prompt=args.prompt_text,
                    device=device,
                    dtype=dtype,
                )
                first_frame_bootstrap_ms = (perf_counter() - prepare_start) * 1000.0

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
                prev_snapshot = None

                for stream_frame_idx in range(len(frame_indices)):
                    frame_total_start = perf_counter()
                    raw_frame = first_frame if stream_frame_idx == 0 else next(frame_iter)
                    frame_cache = clone_cache(prefix_cache_template)

                    if stream_frame_idx == 0:
                        frame_features = first_frame_features
                        frame_encode_ms = first_frame_bootstrap_ms
                    else:
                        encode_start = perf_counter()
                        frame_features = encode_streaming_video_frame(
                            model=model,
                            processor=processor,
                            frame=raw_frame,
                            device=device,
                            dtype=stream_template.prefix_embeds.dtype,
                        )
                        frame_encode_ms = (perf_counter() - encode_start) * 1000.0

                    frame_positions = torch.arange(
                        prefix_len,
                        prefix_len + frame_token_length,
                        device=device,
                        dtype=torch.long,
                    )

                    prefill_start = perf_counter()
                    if args.mode == "baseline":
                        outputs = _forward_span(
                            model.language_model,
                            frame_features.unsqueeze(0),
                            frame_cache,
                            cache_position=frame_positions,
                        )
                        frame_cache = outputs.past_key_values
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
                        frame_snapshot, _, _, _ = _forward_frame_layerwise_with_selective_reuse(
                            language_model=model.language_model,
                            cache=frame_cache,
                            frame_features=frame_features,
                            frame_cache_positions=frame_positions,
                            prev_snapshot=prev_snapshot,
                            layer_reuse_masks=layer_reuse_masks,
                            device=device,
                            profile_timing=False,
                        )
                        prev_snapshot = frame_snapshot
                    frame_prefill_ms = (perf_counter() - prefill_start) * 1000.0

                    generate_start = perf_counter()
                    generated_text = _generate_online_text(
                        model=model,
                        tokenizer=processor.tokenizer,
                        live_cache=frame_cache,
                        current_seq_len=frame_seq_len,
                        newline_embed=stream_template.newline_embed,
                        suffix_input_ids=stream_template.suffix_input_ids,
                        device=device,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                    )
                    frame_generate_ms = (perf_counter() - generate_start) * 1000.0
                    frame_e2e_ms = (perf_counter() - frame_total_start) * 1000.0

                    print(
                        f"[sample={sample.sample_id} frame={stream_frame_idx}] "
                        f"{generated_text.replace(chr(10), ' ')}"
                    )
                    writer.writerow(
                        {
                            "sample_id": sample.sample_id,
                            "video_path": sample.video_path,
                            "frame_idx": stream_frame_idx,
                            "status": "ok",
                            "error": "",
                            "frame_encode_ms": f"{frame_encode_ms:.3f}",
                            "frame_prefill_ms": f"{frame_prefill_ms:.3f}",
                            "frame_generate_ms": f"{frame_generate_ms:.3f}",
                            "frame_e2e_ms": f"{frame_e2e_ms:.3f}",
                            "prompt_text": args.prompt_text,
                            "generated_text": generated_text,
                            "original_question": sample.question,
                            "choices_json": json.dumps(sample.choices, ensure_ascii=False),
                            "gold_idx": sample.answer_idx,
                            "mode": args.mode,
                        }
                    )
                    f.flush()
                    n_rows += 1
                    prev_raw_frame = raw_frame

            except Exception as e:
                writer.writerow(
                    {
                        "sample_id": sample.sample_id,
                        "video_path": sample.video_path,
                        "frame_idx": "",
                        "status": "error",
                        "error": repr(e),
                        "frame_encode_ms": "",
                        "frame_prefill_ms": "",
                        "frame_generate_ms": "",
                        "frame_e2e_ms": "",
                        "prompt_text": args.prompt_text,
                        "generated_text": "",
                        "original_question": sample.question,
                        "choices_json": json.dumps(sample.choices, ensure_ascii=False),
                        "gold_idx": sample.answer_idx,
                        "mode": args.mode,
                    }
                )
                f.flush()
                print(f"[sample={sample.sample_id}] error: {repr(e)}", file=sys.stderr)

    print(json.dumps({"path": output_csv, "rows_written": n_rows, "mode": args.mode}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
