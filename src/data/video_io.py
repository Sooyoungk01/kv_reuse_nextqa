from __future__ import annotations

import glob
import os
from typing import Iterator, List, Optional

import numpy as np

try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None
    cpu = None
    import imageio.v3 as iio


VIDEO_EXTS = [".mp4", ".webm", ".mkv", ".avi", ".mov"]


def _find_existing_path(candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _compute_frame_indices(total: int, num_frames: Optional[int]) -> List[int]:
    if total <= 0:
        return []
    if num_frames is None or num_frames <= 0:
        return np.arange(total).astype(int).tolist()
    if total <= num_frames:
        return np.linspace(0, total - 1, total).astype(int).tolist()
    return np.linspace(0, total - 1, num_frames).astype(int).tolist()


def _load_all_frames_imageio(video_path: str) -> List[np.ndarray]:
    frames = [np.asarray(frame) for frame in iio.imiter(video_path)]
    if not frames:
        raise RuntimeError(f"Empty video: {video_path}")
    return frames


def resolve_video_path(example: dict, video_root: Optional[str] = None) -> Optional[str]:
    for key in ["video_path", "video", "path", "file", "filepath"]:
        v = example.get(key)
        if isinstance(v, str) and os.path.exists(v):
            return v

    vid = None
    for key in ["video_id", "video", "id", "vid"]:
        v = example.get(key)
        if isinstance(v, (str, int)):
            vid = str(v)
            break

    if vid is None or video_root is None:
        return None

    candidates = []
    if os.path.splitext(vid)[1]:
        candidates.append(os.path.join(video_root, vid))
    else:
        for ext in VIDEO_EXTS:
            candidates.append(os.path.join(video_root, vid + ext))
            candidates.append(os.path.join(video_root, f"v_{vid}{ext}"))

    found = _find_existing_path(candidates)
    if found is not None:
        return found

    nested_candidates: List[str] = []
    if os.path.splitext(vid)[1]:
        nested_candidates.extend(glob.glob(os.path.join(video_root, "*", vid)))
    else:
        for ext in VIDEO_EXTS:
            nested_candidates.extend(glob.glob(os.path.join(video_root, "*", vid + ext)))
            nested_candidates.extend(glob.glob(os.path.join(video_root, "*", f"v_{vid}{ext}")))

    found = _find_existing_path(nested_candidates)
    if found is not None:
        return found

    return None


def sample_video_frames(video_path: str, num_frames: Optional[int] = 8) -> List[np.ndarray]:
    if VideoReader is None:
        frames = _load_all_frames_imageio(video_path)
        indices = _compute_frame_indices(len(frames), num_frames)
        return [frames[idx] for idx in indices]

    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total == 0:
        raise RuntimeError(f"Empty video: {video_path}")

    indices = _compute_frame_indices(total, num_frames)

    frames = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
    return [frames[i] for i in range(frames.shape[0])]


def get_video_frame_indices(video_path: str, num_frames: Optional[int] = 8) -> List[int]:
    if VideoReader is None:
        frames = _load_all_frames_imageio(video_path)
        return _compute_frame_indices(len(frames), num_frames)

    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total == 0:
        raise RuntimeError(f"Empty video: {video_path}")
    return _compute_frame_indices(total, num_frames)


def iter_video_frames(video_path: str, num_frames: Optional[int] = 8) -> Iterator[np.ndarray]:
    if VideoReader is None:
        frames = _load_all_frames_imageio(video_path)
        for idx in _compute_frame_indices(len(frames), num_frames):
            yield frames[idx]
        return

    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total == 0:
        raise RuntimeError(f"Empty video: {video_path}")

    for idx in _compute_frame_indices(total, num_frames):
        yield vr[idx].asnumpy()
