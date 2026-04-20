from __future__ import annotations

from transformers import LlavaOnevisionForConditionalGeneration

from kv_reuse.prefill import kv_reuse_prefill, baseline_prefill_from_embeds


def _prefill_with_video_kv_reuse(
    self,
    prepared_prompt,
    raw_video_frames,
    pixel_threshold: float = 0.08,
    profile_timing: bool = False,
):
    return kv_reuse_prefill(
        model=self,
        prepared_prompt=prepared_prompt,
        raw_video_frames=raw_video_frames,
        pixel_threshold=pixel_threshold,
        profile_timing=profile_timing,
    )


def _prefill_baseline_from_embeds(self, prepared_prompt_or_inputs_embeds, profile_timing: bool = False):
    return baseline_prefill_from_embeds(self, prepared_prompt_or_inputs_embeds, profile_timing=profile_timing)


def apply_runtime_patch():
    LlavaOnevisionForConditionalGeneration.prefill_with_video_kv_reuse = _prefill_with_video_kv_reuse
    LlavaOnevisionForConditionalGeneration.prefill_baseline_from_embeds = _prefill_baseline_from_embeds
    return LlavaOnevisionForConditionalGeneration
