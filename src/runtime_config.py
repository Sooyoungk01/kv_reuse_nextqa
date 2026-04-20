from __future__ import annotations

import os
from typing import Any, Dict

import torch
import yaml


DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}

DEFAULT_TEACHER_REFERENCE = "selective_all_recompute"
TEACHER_REFERENCE_CHOICES = ("dense", "selective_all_recompute")


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_teacher_reference(value: str | None) -> str:
    raw_value = str(value or DEFAULT_TEACHER_REFERENCE).strip().lower()
    aliases = {
        "dense_teacher": "dense",
        "selective": "selective_all_recompute",
        "selective_all": "selective_all_recompute",
        "selective-recompute": "selective_all_recompute",
    }
    normalized = aliases.get(raw_value, raw_value)
    if normalized not in TEACHER_REFERENCE_CHOICES:
        valid = ", ".join(TEACHER_REFERENCE_CHOICES)
        raise ValueError(f"Unsupported teacher_reference '{value}'. Expected one of: {valid}")
    return normalized


def merge_cfg(cfg: Dict[str, Any], args) -> Dict[str, Any]:
    out = dict(cfg)
    for key in [
        "max_samples",
        "num_frames",
        "video_root",
        "dataset_name",
        "dataset_config_name",
        "dataset_split",
        "device",
        "dtype",
        "attn_implementation",
        "max_new_tokens",
        "teacher_reference",
    ]:
        value = getattr(args, key, None)
        if value is not None:
            out[key] = value

    if getattr(args, "temperature", None) is not None:
        out.setdefault("generation", {})
        out["generation"]["temperature"] = args.temperature
    if getattr(args, "do_sample", False):
        out.setdefault("generation", {})
        out["generation"]["do_sample"] = True

    if not out.get("video_root"):
        env_video_root = os.environ.get("NEXTQA_VIDEO_ROOT")
        if env_video_root:
            out["video_root"] = env_video_root

    out["teacher_reference"] = normalize_teacher_reference(out.get("teacher_reference"))
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
