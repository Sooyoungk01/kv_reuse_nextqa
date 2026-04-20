#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runtime_config import TEACHER_REFERENCE_CHOICES

RESULTS_DIR = ROOT / "results"
DEFAULT_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
DEFAULT_FONT_BOLD_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
TITLE_FONT_SIZE = 22
SUBTITLE_FONT_SIZE = 22
PANEL_TITLE_FONT_SIZE = 22
BODY_FONT_SIZE = 22


DEFAULT_BUDGET_PERCENTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
STRATEGY_ORDER = ["topk", "pixel_diff", "random"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate how closely current-frame outputs match the selected teacher output "
            "when the same recompute budget is allocated by token importance, raw pixel difference, "
            "or random selection."
        )
    )
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--drift_jsonl",
        type=str,
        default=str(RESULTS_DIR / "drift_oracle.jsonl"),
        help="JSONL with token_importance_scores / greedy_order and token_pixel_diff_scores for each transition.",
    )
    parser.add_argument(
        "--baseline_oracle_jsonl",
        type=str,
        default=None,
        help="Deprecated compatibility arg. This script now generates the selected teacher reference at runtime and ignores this path.",
    )
    parser.add_argument(
        "--budget_percents",
        type=str,
        default=",".join(str(x) for x in DEFAULT_BUDGET_PERCENTS),
        help="Comma-separated recompute budgets in percent, e.g. 0,5,10,30,50,70,90,100.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=str(RESULTS_DIR / "recompute_budget_similarity.jsonl"),
        help="Where to write per-transition per-budget results.",
    )
    parser.add_argument(
        "--output_summary_json",
        type=str,
        default=str(RESULTS_DIR / "recompute_budget_similarity_summary.json"),
        help="Where to write aggregated metrics.",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default=str(RESULTS_DIR / "recompute_budget_similarity.png"),
        help="Where to save the summary plot.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on the number of videos to evaluate from the oracle files.",
    )
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--video_root", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--teacher_reference", type=str, choices=list(TEACHER_REFERENCE_CHOICES), default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument(
        "--num_random_trials",
        type=int,
        default=1,
        help="Number of random-selection trials to run per budget. The random curve averages over these trials.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Base seed for deterministic random token selection.",
    )
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_budget_percents(raw: str) -> list[float]:
    values = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = float(piece)
        if value < 0.0 or value > 100.0:
            raise ValueError(f"Budget percent must be in [0, 100], got {value}.")
        values.append(value)
    if not values:
        raise ValueError("At least one recompute budget percent is required.")
    return sorted(set(values))


def budget_label(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value)}%"
    return f"{value:.1f}%"


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def token_match_ratio(reference_ids: list[int], candidate_ids: list[int]) -> float:
    denom = max(len(reference_ids), len(candidate_ids), 1)
    matches = sum(int(a == b) for a, b in zip(reference_ids, candidate_ids))
    return float(matches / denom)


def prefix_match_ratio(reference_ids: list[int], candidate_ids: list[int]) -> float:
    denom = max(len(reference_ids), len(candidate_ids), 1)
    common = 0
    for ref_token, cand_token in zip(reference_ids, candidate_ids):
        if ref_token != cand_token:
            break
        common += 1
    return float(common / denom)


def rows_by_sample(path: Path) -> OrderedDict[str, list[dict[str, Any]]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in read_jsonl(path):
        if int(row.get("frame_idx", -1)) <= 0:
            continue
        sample_id = str(row.get("sample_id"))
        if sample_id not in grouped:
            grouped[sample_id] = []
        grouped[sample_id].append(row)
    for sample_rows in grouped.values():
        sample_rows.sort(key=lambda row: int(row.get("frame_idx", -1)))
    return grouped


def make_ok_index(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "ok" or int(row.get("frame_idx", -1)) <= 0:
            continue
        out[(str(row["sample_id"]), int(row["frame_idx"]))] = row
    return out


def recompute_count_from_budget(num_visual_tokens: int, budget_percent: float) -> int:
    recompute_count = int(math.ceil(num_visual_tokens * budget_percent / 100.0))
    return min(num_visual_tokens, max(0, recompute_count))


def select_score_based_recompute_indices(
    row: dict[str, Any],
    num_visual_tokens: int,
    budget_percent: float,
    score_key: str,
    strategy_label: str,
) -> list[int]:
    scores = row.get(score_key)
    if not isinstance(scores, list) or len(scores) != num_visual_tokens:
        raise ValueError(
            f"Could not recover {strategy_label} scores for transition "
            f"{row.get('sample_id')} frame {row.get('frame_idx')}."
        )
    order = sorted(range(num_visual_tokens), key=lambda idx: float(scores[idx]), reverse=True)
    recompute_count = recompute_count_from_budget(num_visual_tokens, budget_percent)
    return order[:recompute_count]


def select_topk_recompute_indices(row: dict[str, Any], num_visual_tokens: int, budget_percent: float) -> list[int]:
    ranked = row.get("greedy_order")
    if isinstance(ranked, list) and len(ranked) == num_visual_tokens:
        order = [int(x) for x in ranked]
        recompute_count = recompute_count_from_budget(num_visual_tokens, budget_percent)
        return order[:recompute_count]

    return select_score_based_recompute_indices(
        row=row,
        num_visual_tokens=num_visual_tokens,
        budget_percent=budget_percent,
        score_key="token_importance_scores",
        strategy_label="token importance",
    )


def select_pixel_diff_recompute_indices(
    row: dict[str, Any],
    num_visual_tokens: int,
    budget_percent: float,
) -> list[int]:
    return select_score_based_recompute_indices(
        row=row,
        num_visual_tokens=num_visual_tokens,
        budget_percent=budget_percent,
        score_key="token_pixel_diff_scores",
        strategy_label="token pixel diff",
    )


def select_random_recompute_indices(
    num_visual_tokens: int,
    budget_percent: float,
    sample_id: str,
    frame_idx: int,
    trial_idx: int,
    random_seed: int,
) -> list[int]:
    recompute_count = recompute_count_from_budget(num_visual_tokens, budget_percent)
    if recompute_count <= 0:
        return []
    if recompute_count >= num_visual_tokens:
        return list(range(num_visual_tokens))

    seed_material = f"{random_seed}|{sample_id}|{frame_idx}|{budget_percent:.6f}|{trial_idx}"
    seed_bytes = hashlib.sha256(seed_material.encode("utf-8")).digest()[:8]
    local_seed = int.from_bytes(seed_bytes, byteorder="little", signed=False)
    rng = np.random.default_rng(local_seed)
    selected = rng.choice(num_visual_tokens, size=recompute_count, replace=False)
    return sorted(int(x) for x in selected.tolist())


def aggregate_results(result_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows_by_strategy_budget: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        rows_by_strategy_budget[(str(row["selection_strategy"]), float(row["budget_percent"]))].append(row)

    aggregated_strategies = []
    available_strategies = sorted(
        {strategy for strategy, _ in rows_by_strategy_budget},
        key=lambda x: STRATEGY_ORDER.index(x) if x in STRATEGY_ORDER else len(STRATEGY_ORDER),
    )
    for strategy in available_strategies:
        strategy_budgets = []
        budget_values = sorted({budget for s, budget in rows_by_strategy_budget if s == strategy})
        for budget_percent in budget_values:
            budget_rows = rows_by_strategy_budget[(strategy, budget_percent)]
            strategy_budgets.append(
                {
                    "selection_strategy": strategy,
                    "budget_percent": float(budget_percent),
                    "budget_label": budget_label(budget_percent),
                    "num_records": int(len(budget_rows)),
                    "mean_recompute_tokens": float(mean(row["num_recompute_tokens"] for row in budget_rows)),
                    "mean_recompute_ratio": float(mean(row["recompute_ratio"] for row in budget_rows)),
                    "exact_text_match_rate": float(mean(row["exact_text_match"] for row in budget_rows)),
                    "mean_token_match_ratio": float(mean(row["token_match_ratio"] for row in budget_rows)),
                    "mean_prefix_match_ratio": float(mean(row["prefix_match_ratio"] for row in budget_rows)),
                    "avg_policy_kl": float(mean(row["policy_kl"] for row in budget_rows)),
                }
            )
        aggregated_strategies.append(
            {
                "selection_strategy": strategy,
                "budgets": strategy_budgets,
            }
        )

    evaluated_transitions = sorted(
        {
            (str(row["sample_id"]), int(row["frame_idx"]))
            for row in result_rows
        }
    )
    return {
        "num_result_rows": int(len(result_rows)),
        "num_evaluated_transitions": int(len(evaluated_transitions)),
        "strategies": aggregated_strategies,
    }


def _draw_text_center(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=font, fill=fill)


def _map_linear(value: float, low: float, high: float, pixel_low: float, pixel_high: float) -> float:
    if high == low:
        return pixel_low
    return pixel_low + (value - low) * (pixel_high - pixel_low) / (high - low)


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    font_path = DEFAULT_FONT_BOLD_PATH if bold else DEFAULT_FONT_PATH
    try:
        return ImageFont.truetype(str(font_path), size=size)
    except OSError:
        return ImageFont.load_default()


def _draw_line_chart(
    draw: ImageDraw.ImageDraw,
    panel: tuple[int, int, int, int],
    title: str,
    x_values: list[float],
    series: list[dict[str, Any]],
    y_min: float,
    y_max: float,
    y_tick_values: list[float],
    y_format,
    x_labels: list[str],
    panel_title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
) -> None:
    px0, py0, px1, py1 = panel
    left = px0 + 132
    right = px1 - 40
    top = py0 + 112
    bottom = py1 - 188

    border = "#D7DEE4"
    grid = "#E6EDF3"
    text = "#1F2933"

    draw.rounded_rectangle(panel, radius=12, fill="#F7FAFC", outline=border, width=1)
    draw.text((px0 + 24, py0 + 24), title, font=panel_title_font, fill=text)
    draw.rectangle((left, top, right, bottom), outline=border, width=1)

    for tick in y_tick_values:
        y = _map_linear(tick, y_min, y_max, bottom, top)
        draw.line((left, y, right, y), fill=grid, width=1)
        tick_text = y_format(tick)
        bbox = draw.textbbox((0, 0), tick_text, font=body_font)
        draw.text(
            (left - 16 - (bbox[2] - bbox[0]), y - (bbox[3] - bbox[1]) / 2),
            tick_text,
            font=body_font,
            fill="#52606D",
        )

    for x_value, x_label in zip(x_values, x_labels):
        x = _map_linear(x_value, min(x_values), max(x_values), left, right)
        draw.line((x, bottom, x, bottom + 8), fill="#7B8794", width=1)
        bbox = draw.textbbox((0, 0), x_label, font=body_font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, bottom + 18), x_label, font=body_font, fill="#52606D")

    for series_spec in series:
        points = []
        for x_value, y_value in zip(x_values, series_spec["values"]):
            x = _map_linear(x_value, min(x_values), max(x_values), left, right)
            y = _map_linear(y_value, y_min, y_max, bottom, top)
            points.append((x, y))
        draw.line(points, fill=series_spec["color"], width=5)
        for point_x, point_y in points:
            draw.ellipse((point_x - 7, point_y - 7, point_x + 7, point_y + 7), fill=series_spec["color"])

    legend_x = px0 + 24
    legend_y = py1 - 148
    for idx, series_spec in enumerate(series):
        line_y = legend_y + idx * 40
        draw.line((legend_x, line_y + 13, legend_x + 32, line_y + 13), fill=series_spec["color"], width=5)
        draw.text((legend_x + 44, line_y), series_spec["label"], font=body_font, fill=text)


def plot_summary(summary: dict[str, Any], output_png: Path) -> None:
    strategies = summary["strategies"]
    if not strategies:
        raise ValueError("No aggregated strategy rows were available for plotting.")

    strategy_map = {
        strategy_block["selection_strategy"]: {
            float(row["budget_percent"]): row for row in strategy_block["budgets"]
        }
        for strategy_block in strategies
    }
    x_values = sorted(
        {
            float(row["budget_percent"])
            for strategy_block in strategies
            for row in strategy_block["budgets"]
        }
    )
    x_labels = [budget_label(x) for x in x_values]

    colors = {
        "topk": "#D1495B",
        "pixel_diff": "#F4A259",
        "random": "#2F6B7C",
    }
    label_prefix_map = {
        "topk": "Top-k Importance",
        "pixel_diff": "Pixel Diff",
        "random": "Random",
    }

    exact_series = []
    token_series = []
    kl_series = []
    for strategy in STRATEGY_ORDER:
        if strategy not in strategy_map:
            continue
        budget_rows = strategy_map[strategy]
        label_prefix = label_prefix_map.get(strategy, strategy.replace("_", " ").title())
        exact_series.append(
            {
                "label": f"{label_prefix} exact match",
                "color": colors[strategy],
                "values": [float(budget_rows[x]["exact_text_match_rate"]) for x in x_values],
            }
        )
        token_series.append(
            {
                "label": f"{label_prefix} token match",
                "color": colors[strategy],
                "values": [float(budget_rows[x]["mean_token_match_ratio"]) for x in x_values],
            }
        )
        kl_series.append(
            {
                "label": f"{label_prefix} policy KL",
                "color": colors[strategy],
                "values": [float(budget_rows[x]["avg_policy_kl"]) for x in x_values],
            }
        )

    kl_values = [value for series in kl_series for value in series["values"]]
    kl_max = max(kl_values) if kl_values else 0.0
    kl_top = kl_max * 1.15 if kl_max > 0 else 1.0
    kl_ticks = np.linspace(0.0, kl_top, 5).tolist()

    width, height = 3400, 1120
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _load_font(BODY_FONT_SIZE)
    title_font = _load_font(TITLE_FONT_SIZE, bold=True)
    subtitle_font = _load_font(SUBTITLE_FONT_SIZE)
    panel_title_font = _load_font(PANEL_TITLE_FONT_SIZE, bold=True)

    draw.rectangle((0, 0, width - 1, height - 1), outline="#E5E7EB")
    _draw_text_center(
        draw,
        (width / 2, 44),
        "Recompute Strategy Comparison Under the Same Budget",
        title_font,
        "#1F2933",
    )
    subtitle = (
        f"Transitions: {summary['num_evaluated_transitions']:,}    "
        f"Strategies: {', '.join(block['selection_strategy'] for block in strategies)}    "
        f"Budgets: {', '.join(x_labels)}"
    )
    _draw_text_center(draw, (width / 2, 88), subtitle, subtitle_font, "#52606D")

    panel_gap = 50
    panel_width = (width - 120 - panel_gap * 2) // 3
    left_panel = (60, 150, 60 + panel_width, 1060)
    middle_panel = (left_panel[2] + panel_gap, 150, left_panel[2] + panel_gap + panel_width, 1060)
    right_panel = (middle_panel[2] + panel_gap, 150, middle_panel[2] + panel_gap + panel_width, 1060)

    _draw_line_chart(
        draw=draw,
        panel=left_panel,
        title="Exact Output Match",
        x_values=x_values,
        series=exact_series,
        y_min=0.0,
        y_max=1.0,
        y_tick_values=[0.0, 0.25, 0.5, 0.75, 1.0],
        y_format=lambda value: f"{value:.2f}",
        x_labels=x_labels,
        panel_title_font=panel_title_font,
        body_font=font,
    )
    _draw_line_chart(
        draw=draw,
        panel=middle_panel,
        title="Token-Level Match",
        x_values=x_values,
        series=token_series,
        y_min=0.0,
        y_max=1.0,
        y_tick_values=[0.0, 0.25, 0.5, 0.75, 1.0],
        y_format=lambda value: f"{value:.2f}",
        x_labels=x_labels,
        panel_title_font=panel_title_font,
        body_font=font,
    )
    _draw_line_chart(
        draw=draw,
        panel=right_panel,
        title="Teacher-Forced Policy Distance",
        x_values=x_values,
        series=kl_series,
        y_min=0.0,
        y_max=kl_top,
        y_tick_values=kl_ticks,
        y_format=lambda value: f"{value:.4f}",
        x_labels=x_labels,
        panel_title_font=panel_title_font,
        body_font=font,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png)


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

    from data.video_io import iter_video_frames
    from hf_patch.apply_runtime_patch import apply_runtime_patch
    from kv_reuse.cache_ops import clone_cache, make_static_cache
    from kv_reuse.prefill import _forward_span
    from kv_reuse.video_prompt import encode_streaming_video_frame, prepare_streaming_prompt_template
    from runtime_config import (
        load_cfg,
        merge_cfg,
        normalize_teacher_reference,
        resolve_runtime,
    )
    from teacher_reference import (
        average_policy_kl,
        build_counterfactual_cache,
        build_full_frame_snapshot,
        build_teacher_reference_cache,
        collect_teacher_forced_policy_logits,
        generate_oracle_trace,
    )

    cfg = merge_cfg(load_cfg(args.config), args)
    apply_runtime_patch()
    teacher_reference = normalize_teacher_reference(cfg.get("teacher_reference"))

    budgets = parse_budget_percents(args.budget_percents)
    drift_path = Path(args.drift_jsonl)
    output_jsonl = Path(args.output_jsonl)
    output_summary_json = Path(args.output_summary_json)
    output_png = Path(args.output_png)

    drift_rows = list(read_jsonl(drift_path))
    drift_index = make_ok_index(drift_rows)
    sample_rows = rows_by_sample(drift_path)

    if args.max_samples is not None:
        sample_rows = OrderedDict(list(sample_rows.items())[: args.max_samples])
    if args.num_random_trials < 1:
        raise ValueError("--num_random_trials must be >= 1.")

    device, dtype = resolve_runtime(cfg)
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

    result_rows: list[dict[str, Any]] = []
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as fout:
        sample_progress = tqdm(
            sample_rows.items(),
            total=len(sample_rows),
            desc="Videos",
            dynamic_ncols=True,
        )
        for sample_id, rows in sample_progress:
            sample_progress.set_postfix_str(str(sample_id))
            if not rows:
                continue

            first_row = rows[0]
            video_path = first_row.get("video_path")
            if not isinstance(video_path, str) or not Path(video_path).exists():
                continue

            sample_num_frames = max(int(row.get("frame_idx", -1)) for row in rows) + 1
            if sample_num_frames <= 0:
                continue

            prompt_text = str(first_row.get("prompt_text", "What is happening now?"))
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

                if frame_idx == 0 or prev_full_snapshot is None:
                    prev_full_snapshot = full_snapshot
                    continue

                drift_row = drift_index.get((sample_id, frame_idx))
                if drift_row is None:
                    prev_full_snapshot = full_snapshot
                    continue

                num_visual_tokens = int(frame_features.shape[0])

                teacher_cache = build_teacher_reference_cache(
                    teacher_reference=teacher_reference,
                    model=model,
                    prefix_cache_template=prefix_cache_template,
                    frame_features=frame_features,
                    frame_positions=frame_positions,
                    prev_full_snapshot=prev_full_snapshot,
                    device=device,
                )
                teacher_generated = generate_oracle_trace(
                    model=model,
                    tokenizer=processor.tokenizer,
                    live_cache=clone_cache(teacher_cache),
                    current_seq_len=frame_seq_len,
                    newline_embed=stream_template.newline_embed,
                    suffix_input_ids=stream_template.suffix_input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                )
                teacher_text = str(teacher_generated.get("oracle_text", ""))
                teacher_token_ids = [int(x) for x in teacher_generated.get("oracle_token_ids", [])]

                teacher_logits_trace = collect_teacher_forced_policy_logits(
                    model=model,
                    live_cache=clone_cache(teacher_cache),
                    current_seq_len=frame_seq_len,
                    newline_embed=stream_template.newline_embed,
                    suffix_input_ids=stream_template.suffix_input_ids,
                    oracle_token_ids=teacher_token_ids,
                    device=device,
                )

                for budget_percent in budgets:
                    topk_recompute_indices = select_topk_recompute_indices(
                        row=drift_row,
                        num_visual_tokens=num_visual_tokens,
                        budget_percent=budget_percent,
                    )
                    pixel_diff_recompute_indices = select_pixel_diff_recompute_indices(
                        row=drift_row,
                        num_visual_tokens=num_visual_tokens,
                        budget_percent=budget_percent,
                    )
                    selection_jobs = [
                        ("topk", None, topk_recompute_indices),
                        ("pixel_diff", None, pixel_diff_recompute_indices),
                    ]
                    for random_trial_idx in range(args.num_random_trials):
                        random_recompute_indices = select_random_recompute_indices(
                            num_visual_tokens=num_visual_tokens,
                            budget_percent=budget_percent,
                            sample_id=sample_id,
                            frame_idx=frame_idx,
                            trial_idx=random_trial_idx,
                            random_seed=args.random_seed,
                        )
                        selection_jobs.append(("random", random_trial_idx, random_recompute_indices))

                    for selection_strategy, random_trial_idx, recompute_indices in selection_jobs:
                        reuse_mask = torch.ones(num_visual_tokens, dtype=torch.bool, device=frame_features.device)
                        if recompute_indices:
                            reuse_mask[torch.tensor(recompute_indices, device=frame_features.device)] = False

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
                            live_cache=clone_cache(counterfactual_cache),
                            current_seq_len=frame_seq_len,
                            newline_embed=stream_template.newline_embed,
                            suffix_input_ids=stream_template.suffix_input_ids,
                            oracle_token_ids=teacher_token_ids,
                            device=device,
                        )
                        policy_kl = average_policy_kl(teacher_logits_trace, counterfactual_logits_trace)

                        generated = generate_oracle_trace(
                            model=model,
                            tokenizer=processor.tokenizer,
                            live_cache=clone_cache(counterfactual_cache),
                            current_seq_len=frame_seq_len,
                            newline_embed=stream_template.newline_embed,
                            suffix_input_ids=stream_template.suffix_input_ids,
                            max_new_tokens=max_new_tokens,
                            do_sample=do_sample,
                            temperature=temperature,
                        )
                        generated_text = str(generated.get("oracle_text", ""))
                        generated_token_ids = [int(x) for x in generated.get("oracle_token_ids", [])]

                        row = {
                            "sample_id": sample_id,
                            "video_path": video_path,
                            "frame_idx": int(frame_idx),
                            "transition": f"{frame_idx - 1}->{frame_idx}",
                            "budget_percent": float(budget_percent),
                            "budget_label": budget_label(budget_percent),
                            "selection_strategy": selection_strategy,
                            "random_trial_idx": random_trial_idx,
                            "teacher_reference": teacher_reference,
                            "num_visual_tokens": int(num_visual_tokens),
                            "num_recompute_tokens": int(len(recompute_indices)),
                            "num_reuse_tokens": int(int(reuse_mask.sum().item())),
                            "recompute_ratio": float(len(recompute_indices) / max(1, num_visual_tokens)),
                            "policy_kl": float(policy_kl),
                            "exact_text_match": int(normalize_text(generated_text) == normalize_text(teacher_text)),
                            "token_match_ratio": float(token_match_ratio(teacher_token_ids, generated_token_ids)),
                            "prefix_match_ratio": float(prefix_match_ratio(teacher_token_ids, generated_token_ids)),
                            "teacher_text": teacher_text,
                            "generated_text": generated_text,
                            "teacher_token_ids": teacher_token_ids,
                            "generated_token_ids": generated_token_ids,
                            "recompute_indices": recompute_indices,
                        }
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                        fout.flush()
                        result_rows.append(row)

                prev_full_snapshot = full_snapshot

    if not result_rows:
        raise RuntimeError(
            "No result rows were produced. Check that drift_oracle.jsonl contains usable transitions "
            "and that the referenced video files exist."
        )

    summary = aggregate_results(result_rows)
    summary["budget_percents"] = budgets
    summary["num_random_trials"] = int(args.num_random_trials)
    summary["random_seed"] = int(args.random_seed)
    summary["config_path"] = str(args.config)
    summary["drift_jsonl"] = str(drift_path)
    summary["teacher_reference"] = teacher_reference

    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot_summary(summary, output_png)

    print(f"Saved detailed rows to: {output_jsonl}")
    print(f"Saved summary to: {output_summary_json}")
    print(f"Saved plot to: {output_png}")


if __name__ == "__main__":
    main()
