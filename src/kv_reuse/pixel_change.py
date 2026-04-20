from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ReuseDecision:
    reuse_mask: torch.Tensor       # [tokens_per_frame] bool
    patch_change: torch.Tensor     # [196] float
    threshold: float


def _to_chw_float(frame: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(frame, np.ndarray):
        t = torch.from_numpy(frame)
    else:
        t = frame.detach().cpu()

    if t.dim() == 3 and t.shape[-1] in (1, 3):
        t = t.permute(2, 0, 1)
    if t.dim() != 3:
        raise ValueError(f"Expected frame with 3 dims, got shape={tuple(t.shape)}")
    return t.float() / 255.0 if t.max() > 1.0 else t.float()


def _patchify_abs_change(
    prev_frame: np.ndarray | torch.Tensor,
    curr_frame: np.ndarray | torch.Tensor,
    grid_size: Tuple[int, int] = (14, 14),
) -> torch.Tensor:
    prev_t = _to_chw_float(prev_frame).unsqueeze(0)
    curr_t = _to_chw_float(curr_frame).unsqueeze(0)

    if prev_t.shape != curr_t.shape:
        curr_t = F.interpolate(curr_t, size=prev_t.shape[-2:], mode="bilinear", align_corners=False)

    diff = (curr_t - prev_t).abs().mean(dim=1, keepdim=True)  # [1,1,H,W]
    pooled = F.adaptive_avg_pool2d(diff, grid_size).squeeze(0).squeeze(0)  # [gh, gw]
    return pooled.flatten()  # [196]


def decide_reuse_by_pixel_change(
    prev_frame: np.ndarray | torch.Tensor,
    curr_frame: np.ndarray | torch.Tensor,
    threshold: float = 0.08,
    spatial_grid: Tuple[int, int] = (14, 14),
    append_newline_token: bool = True,
) -> ReuseDecision:
    """
    Approximate reuse/recompute decision for one frame.

    - LLaVA-OneVision docs indicate video features are pooled to 196 tokens/frame.
    - We map those 196 tokens to a 14x14 grid over per-pixel absolute difference.
    - patch_change <= threshold => reuse
    - patch_change > threshold => recompute

    Returns a bool mask of length 196 or 197 (with newline token).
    """
    patch_change = _patchify_abs_change(prev_frame, curr_frame, grid_size=spatial_grid)
    spatial_reuse = patch_change <= threshold

    if append_newline_token:
        newline_reuse = torch.tensor([True], dtype=torch.bool)
        reuse_mask = torch.cat([spatial_reuse, newline_reuse], dim=0)
    else:
        reuse_mask = spatial_reuse

    return ReuseDecision(
        reuse_mask=reuse_mask,
        patch_change=patch_change,
        threshold=threshold,
    )
