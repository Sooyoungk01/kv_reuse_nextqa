from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch


@dataclass
class PreparedVideoPrompt:
    input_ids: torch.Tensor
    inputs_embeds: torch.Tensor
    pixel_values_videos: torch.Tensor
    image_sizes_videos: torch.Tensor | None
    video_token_positions: torch.Tensor
    frame_token_positions: List[torch.Tensor]
    frame_features: List[torch.Tensor]
    full_video_features: torch.Tensor
    prompt_text: str


@dataclass
class StreamingPromptTemplate:
    prefix_embeds: torch.Tensor
    suffix_input_ids: torch.Tensor
    frame_token_length: int
    newline_embed: torch.Tensor
    prompt_text: str


def build_mc_prompt(question: str, choices: Sequence[str], instruction: str) -> str:
    letters = ["A", "B", "C", "D", "E"]
    choice_lines = [f"{letters[i]}. {c}" for i, c in enumerate(choices)]
    return (
        f"{instruction.strip()}\n\n"
        f"Question: {question.strip()}\n"
        + "\n".join(choice_lines)
        + "\nAnswer:"
    )


def _make_conversation(text_prompt: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]


def _encode_single_frame_video_features(
    model,
    pixel_values_videos: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    pixel_values_videos = pixel_values_videos.to(device=device, dtype=dtype)
    actual_num_frames = pixel_values_videos.shape[1]
    if actual_num_frames != 1:
        raise RuntimeError(f"Expected exactly 1 frame for streaming encode, got {actual_num_frames}")

    video_features = model.get_video_features(
        pixel_values=pixel_values_videos,
        vision_feature_layer=model.config.vision_feature_layer,
        vision_feature_select_strategy=model.config.vision_feature_select_strategy,
    )
    tokens_per_frame = video_features.shape[1] // actual_num_frames
    if tokens_per_frame * actual_num_frames != video_features.shape[1]:
        raise RuntimeError("Video features do not divide evenly by the processed frame count")
    return video_features[0].to(device=device, dtype=dtype)


def prepare_streaming_prompt_template(
    model,
    processor,
    first_frame,
    text_prompt: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[StreamingPromptTemplate, torch.Tensor]:
    conversation = _make_conversation(text_prompt)
    prompt_text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    batch = processor(
        text=prompt_text,
        videos=[first_frame],
        return_tensors="pt",
    )

    input_ids = batch["input_ids"].to(device)
    pixel_values_videos = batch["pixel_values_videos"]
    video_token_positions = (input_ids[0] == model.config.video_token_index).nonzero(as_tuple=False).flatten()
    if video_token_positions.numel() == 0:
        raise RuntimeError("Streaming prompt template did not contain any video token placeholders")

    first_video_pos = int(video_token_positions[0].item())
    last_video_pos = int(video_token_positions[-1].item())
    prefix_input_ids = input_ids[:, :first_video_pos]
    suffix_input_ids = input_ids[:, last_video_pos + 1 :]
    prefix_embeds = model.get_input_embeddings()(prefix_input_ids)

    first_frame_features = _encode_single_frame_video_features(
        model=model,
        pixel_values_videos=pixel_values_videos,
        device=device,
        dtype=prefix_embeds.dtype,
    )
    if video_token_positions.numel() != first_frame_features.shape[0] + 1:
        raise RuntimeError(
            "Streaming prompt template token count mismatch: "
            f"placeholders={video_token_positions.numel()} frame_tokens+newline={first_frame_features.shape[0] + 1}"
        )

    newline_embed = model.image_newline.to(device=device, dtype=prefix_embeds.dtype)
    template = StreamingPromptTemplate(
        prefix_embeds=prefix_embeds,
        suffix_input_ids=suffix_input_ids,
        frame_token_length=int(first_frame_features.shape[0]),
        newline_embed=newline_embed,
        prompt_text=prompt_text,
    )
    return template, first_frame_features


def encode_streaming_video_frame(
    model,
    processor,
    frame,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # LlavaOnevisionVideoProcessor is a BaseImageProcessor subclass whose __call__
    # takes the video batch as the first positional argument rather than a
    # `videos=` keyword.
    batch = processor.video_processor(
        [frame],
        return_tensors="pt",
    )
    return _encode_single_frame_video_features(
        model=model,
        pixel_values_videos=batch["pixel_values_videos"],
        device=device,
        dtype=dtype,
    )


def prepare_video_inputs(
    model,
    processor,
    video_frames,
    text_prompt: str,
    num_frames: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
) -> PreparedVideoPrompt:
    conversation = _make_conversation(text_prompt)
    prompt_text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    batch = processor(
        text=prompt_text,
        videos=video_frames,
        return_tensors="pt",
    )

    # Keep ids as long, cast only floating tensors.
    input_ids = batch["input_ids"].to(device)
    pixel_values_videos = batch["pixel_values_videos"].to(device=device, dtype=dtype)
    image_sizes_videos = batch.get("image_sizes_videos")
    if image_sizes_videos is not None:
        image_sizes_videos = image_sizes_videos.to(device)

    inputs_embeds = model.get_input_embeddings()(input_ids)

    video_features = model.get_video_features(
        pixel_values=pixel_values_videos,
        vision_feature_layer=model.config.vision_feature_layer,
        vision_feature_select_strategy=model.config.vision_feature_select_strategy,
    )

    actual_num_frames = pixel_values_videos.shape[1]
    tokens_per_frame = video_features.shape[1] // actual_num_frames
    if tokens_per_frame * actual_num_frames != video_features.shape[1]:
        raise RuntimeError("Video features do not divide evenly by the processed frame count")

    image_newline = model.image_newline[None, None, :].repeat(video_features.shape[0], 1, 1).to(video_features.device)
    video_features_with_newline = torch.cat((video_features, image_newline), dim=1)
    full_video_features = video_features_with_newline.flatten(0, 1).to(
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
    )

    video_token_positions = (input_ids[0] == model.config.video_token_index).nonzero(as_tuple=False).flatten()
    if video_token_positions.numel() != full_video_features.shape[0]:
        raise RuntimeError(
            f"Video token count mismatch. placeholders={video_token_positions.numel()} vs features={full_video_features.shape[0]}"
        )

    inputs_embeds = inputs_embeds.clone()
    inputs_embeds[0, video_token_positions, :] = full_video_features

    frame_token_positions = []
    frame_features = []
    video_features = video_features[0].to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    for f in range(actual_num_frames):
        s = f * tokens_per_frame
        e = (f + 1) * tokens_per_frame
        frame_pos = video_token_positions[s:e]
        frame_feat = video_features[s:e]
        if f == actual_num_frames - 1:
            frame_pos = video_token_positions[s : e + 1]
            frame_feat = full_video_features[s : e + 1]
        frame_token_positions.append(frame_pos)
        frame_features.append(frame_feat)

    prompt_text = prompt_text

    return PreparedVideoPrompt(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        pixel_values_videos=pixel_values_videos,
        image_sizes_videos=image_sizes_videos,
        video_token_positions=video_token_positions,
        frame_token_positions=frame_token_positions,
        frame_features=frame_features,
        full_video_features=full_video_features,
        prompt_text=prompt_text,
    )
