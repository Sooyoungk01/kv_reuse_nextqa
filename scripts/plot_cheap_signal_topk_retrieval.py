#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
DEFAULT_INPUT_JSONL = RESULTS_DIR / "cheap_signals.jsonl"
DEFAULT_OUTPUT_PNG = RESULTS_DIR / "cheap_signal_topk_retrieval.png"
DEFAULT_OUTPUT_SUMMARY_JSON = RESULTS_DIR / "cheap_signal_topk_retrieval_summary.json"
DEFAULT_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
DEFAULT_FONT_BOLD_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
BASE_COMPARISON_SIGNAL_KEYS = [
    "hidden_state_mean_layer_l2_scores",
    "attention_cosine_distance_scores",
]
RECOMMENDED_LOGISTIC_FEATURE_KEYS = [
    "hidden_state_mean_layer_l2_scores",
    "attention_cosine_distance_scores",
    "text_token_decoder_hidden_delta_scores",
    "text_to_image_attention_cosine_distance_scores",
    "projector_output_delta_scores",
    "projector_cosine_distance_scores",
    "projector_topk_attention_weighted_delta_scores",
    "topk_image_attention_overlap_k10_scores",
    "cross_modal_gated_scores",
    "early_mid_decoder_block_hidden_delta_scores",
    "last_text_token_hidden_delta_scores",
    "high_attention_weighted_image_token_delta_scores",
    "hidden_state_mean_layer_l2_jump_scores",
    "attention_cosine_distance_jump_scores",
    "hidden_state_mean_layer_l2_recent_max_gap_scores",
    "attention_cosine_distance_recent_max_gap_scores",
    "hidden_attention_product_scores",
    "hidden_attention_zmax_scores",
]
DEFAULT_SIGNAL_KEYS = [
    "text_token_decoder_hidden_delta_scores",
    "text_to_image_attention_cosine_distance_scores",
    "projector_output_delta_scores",
    "topk_image_attention_overlap_k10_scores",
    "cross_modal_gated_scores",
    "early_mid_decoder_block_hidden_delta_scores",
    "last_text_token_hidden_delta_scores",
    "high_attention_weighted_image_token_delta_scores",
    *BASE_COMPARISON_SIGNAL_KEYS,
]
SERIES_PALETTE = [
    (29, 78, 216),
    (190, 24, 93),
    (22, 163, 74),
    (245, 158, 11),
    (14, 165, 233),
    (124, 58, 237),
]
RANDOM_COLOR = (100, 116, 139)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare shortlisted cheap signals as top-k retrieval scores for oracle semantic drift, "
            "focusing on oracle top 5-10% token recovery."
        )
    )
    parser.add_argument("--input_jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output_png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--output_summary_json", type=Path, default=DEFAULT_OUTPUT_SUMMARY_JSON)
    parser.add_argument(
        "--top_percents",
        type=str,
        default="5,10",
        help="Comma-separated oracle top-percent budgets, e.g. 5,10 or 1,2,5,10.",
    )
    parser.add_argument(
        "--signal_keys",
        type=str,
        default=",".join(DEFAULT_SIGNAL_KEYS),
        help="Comma-separated cheap-signal score keys to evaluate.",
    )
    parser.add_argument("--sample_id", type=str, default=None, help="Optional sample_id filter.")
    return parser.parse_args()


def parse_top_percents(raw: str) -> list[float]:
    values: list[float] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = float(piece)
        if value <= 0.0 or value > 100.0:
            raise ValueError(f"top_percents must be within (0, 100], got {value}.")
        values.append(value)
    if not values:
        raise ValueError("At least one top-percent budget is required.")
    return sorted(set(values))


def parse_signal_keys(raw: str) -> list[str]:
    keys = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if not keys:
        raise ValueError("At least one signal key is required.")
    deduped: list[str] = []
    seen = set()
    for key in keys:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    font_path = DEFAULT_FONT_BOLD_PATH if bold else DEFAULT_FONT_PATH
    try:
        return ImageFont.truetype(str(font_path), size=size)
    except OSError:
        return ImageFont.load_default()


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= scores.size:
        return np.arange(scores.size, dtype=np.int64)
    order = np.argsort(-scores, kind="mergesort")
    return order[:k]


def summarize_metric(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
    }


def signal_label(signal_key: str) -> str:
    special_cases = {
        "hidden_state_mean_layer_l2_scores": "Mean L2",
        "attention_cosine_distance_scores": "Attn Cos",
        "text_token_decoder_hidden_delta_scores": "Text Hidden Delta",
        "text_to_image_attention_cosine_distance_scores": "Text->Image Attn Cos",
        "vision_attention_cosine_distance_scores": "Vision Attn Cos",
        "decoder_attention_cosine_distance_scores": "Decoder Attn Cos",
        "projector_output_delta_scores": "Projector Delta",
        "projector_cosine_distance_scores": "Projector Cos",
        "projector_topk_attention_weighted_delta_scores": "Proj Top-k Weighted",
        "topk_image_attention_overlap_k5_scores": "Top-k Overlap K5",
        "topk_image_attention_overlap_k10_scores": "Top-k Overlap K10",
        "topk_image_attention_overlap_k20_scores": "Top-k Overlap K20",
        "cross_modal_gated_scores": "Cross-modal Gate",
        "early_mid_decoder_block_hidden_delta_scores": "Mid Block L2",
        "last_text_token_hidden_delta_scores": "Last Text Delta",
        "high_attention_weighted_image_token_delta_scores": "Attn-weighted Img Delta",
        "hidden_state_mean_layer_l2_jump_scores": "Mean L2 Jump",
        "attention_cosine_distance_jump_scores": "Attn Cos Jump",
        "hidden_state_mean_layer_l2_ratio_to_recent_mean_scores": "Mean L2 / Recent",
        "attention_cosine_distance_ratio_to_recent_mean_scores": "Attn Cos / Recent",
        "hidden_state_mean_layer_l2_recent_max_gap_scores": "Mean L2 - Recent Max",
        "attention_cosine_distance_recent_max_gap_scores": "Attn Cos - Recent Max",
        "hidden_attention_product_scores": "Hidden x Attn",
        "hidden_attention_zsum_scores": "z(Hidden)+z(Attn)",
        "hidden_attention_zmax_scores": "max(zHidden,zAttn)",
        "logistic_combo_scores": "LogReg Combo",
    }
    if signal_key in special_cases:
        return special_cases[signal_key]
    stem = signal_key[:-7] if signal_key.endswith("_scores") else signal_key
    return stem.replace("_", " ").title()


def build_color_map(signal_keys: list[str]) -> dict[str, tuple[int, int, int]]:
    colors = {"random": RANDOM_COLOR}
    for idx, signal_key in enumerate(signal_keys):
        colors[signal_key] = SERIES_PALETTE[idx % len(SERIES_PALETTE)]
    return colors


def read_rows(path: Path, sample_id: str | None, signal_keys: list[str]) -> tuple[list[dict[str, Any]], str | None]:
    rows: list[dict[str, Any]] = []
    oracle_source_key = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            if sample_id is not None and str(row.get("sample_id")) != sample_id:
                continue
            oracle_scores = row.get("oracle_semantic_drift_scores")
            if not isinstance(oracle_scores, list) or len(oracle_scores) == 0:
                continue
            num_tokens = len(oracle_scores)
            if any(not isinstance(row.get(signal_key), list) or len(row.get(signal_key)) != num_tokens for signal_key in signal_keys):
                continue
            rows.append(row)
            oracle_source_key = row.get("oracle_semantic_drift_source_key", oracle_source_key)
    if not rows:
        raise ValueError(f"No usable rows were found in {str(path)!r} for the selected signal keys.")
    return rows, oracle_source_key


def evaluate_retrieval(
    rows: list[dict[str, Any]],
    top_percents: list[float],
    signal_keys: list[str],
    oracle_source_key: str | None,
) -> dict[str, Any]:
    per_percent: dict[str, Any] = {}

    for percent in top_percents:
        percent_key = f"{percent:g}%"
        signal_metrics: dict[str, dict[str, list[float]]] = {
            signal_key: {
                "recall": [],
                "jaccard": [],
                "oracle_mass_recall": [],
                "overlap_count": [],
                "hit_any": [],
            }
            for signal_key in signal_keys
        }
        pooled_overlap = {signal_key: 0 for signal_key in signal_keys}
        pooled_positive = 0
        k_values: list[int] = []

        for row in rows:
            oracle_scores = np.asarray(row["oracle_semantic_drift_scores"], dtype=np.float64)
            num_tokens = oracle_scores.size
            k = max(1, int(math.ceil(num_tokens * percent / 100.0)))
            oracle_top = topk_indices(oracle_scores, k)
            oracle_top_set = set(int(x) for x in oracle_top.tolist())
            oracle_top_mass = float(np.maximum(oracle_scores[oracle_top], 0.0).sum())
            pooled_positive += k
            k_values.append(k)

            for signal_key in signal_keys:
                scores = np.asarray(row[signal_key], dtype=np.float64)
                predicted_top = topk_indices(scores, k)
                predicted_set = set(int(x) for x in predicted_top.tolist())
                overlap = int(len(oracle_top_set & predicted_set))
                recall = float(overlap / k)
                jaccard = float(overlap / max((2 * k) - overlap, 1))
                oracle_mass_hit = float(np.maximum(oracle_scores[list(oracle_top_set & predicted_set)], 0.0).sum())
                oracle_mass_recall = float(oracle_mass_hit / oracle_top_mass) if oracle_top_mass > 0 else None

                pooled_overlap[signal_key] += overlap
                signal_metrics[signal_key]["recall"].append(recall)
                signal_metrics[signal_key]["jaccard"].append(jaccard)
                signal_metrics[signal_key]["overlap_count"].append(float(overlap))
                signal_metrics[signal_key]["hit_any"].append(float(overlap > 0))
                if oracle_mass_recall is not None:
                    signal_metrics[signal_key]["oracle_mass_recall"].append(oracle_mass_recall)

        percent_summary: dict[str, Any] = {
            "oracle_top_percent": percent,
            "random_baseline_recall": percent / 100.0,
            "num_transitions": len(rows),
            "oracle_semantic_drift_source_key": oracle_source_key,
            "k_stats": summarize_metric([float(k) for k in k_values]),
            "signals": {},
        }

        for signal_key in signal_keys:
            recall_values = signal_metrics[signal_key]["recall"]
            recall_mean = float(mean(recall_values))
            random_baseline = percent / 100.0
            percent_summary["signals"][signal_key] = {
                "label": signal_label(signal_key),
                "mean_recall_at_percent": recall_mean,
                "median_recall_at_percent": float(median(recall_values)),
                "std_recall_at_percent": float(np.std(recall_values, dtype=np.float64)),
                "pooled_recall_at_percent": float(pooled_overlap[signal_key] / max(pooled_positive, 1)),
                "mean_lift_vs_random": float(recall_mean / random_baseline) if random_baseline > 0 else None,
                "mean_jaccard": float(mean(signal_metrics[signal_key]["jaccard"])),
                "mean_overlap_count": float(mean(signal_metrics[signal_key]["overlap_count"])),
                "hit_any_rate": float(mean(signal_metrics[signal_key]["hit_any"])),
                "oracle_mass_recall_stats": summarize_metric(signal_metrics[signal_key]["oracle_mass_recall"]),
                "recall_stats": summarize_metric(recall_values),
            }

        per_percent[percent_key] = percent_summary

    return {
        "num_transitions": len(rows),
        "oracle_semantic_drift_source_key": oracle_source_key,
        "top_percent_budgets": top_percents,
        "signal_keys": signal_keys,
        "per_percent": per_percent,
    }


def bar_height(value: float, max_value: float, height: int) -> int:
    if max_value <= 0:
        return 0
    return int(round(height * value / max_value))


def draw_text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font, fill: str) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=font, fill=fill)


def draw_wrapped_legend(
    draw: ImageDraw.ImageDraw,
    box_left: int,
    box_right: int,
    start_y: int,
    series_names: list[str],
    series_labels: dict[str, str],
    colors: dict[str, tuple[int, int, int]],
) -> int:
    legend_font = load_font(15)
    cursor_x = box_left + 8
    cursor_y = start_y
    row_height = 24
    for series_name in series_names:
        label = series_labels[series_name]
        label_width = draw.textbbox((0, 0), label, font=legend_font)[2]
        item_width = 26 + label_width + 24
        if cursor_x + item_width > box_right - 12:
            cursor_x = box_left + 8
            cursor_y += row_height
        draw.rounded_rectangle((cursor_x, cursor_y, cursor_x + 18, cursor_y + 18), radius=4, fill=colors[series_name])
        draw.text((cursor_x + 26, cursor_y - 2), label, font=legend_font, fill="#334155")
        cursor_x += item_width
    return cursor_y + row_height


def draw_bar_panel(
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    box: tuple[int, int, int, int],
    title: str,
    y_label: str,
    percent_labels: list[str],
    series_names: list[str],
    series_labels: dict[str, str],
    values_by_series: dict[str, list[float]],
    colors: dict[str, tuple[int, int, int]],
    max_value: float,
) -> None:
    title_font = load_font(26, bold=True)
    axis_font = load_font(20)
    tick_font = load_font(16)
    value_font = load_font(14, bold=True)

    x0, y0, x1, y1 = box
    left = x0 + 95
    right = x1 - 35
    top = y0 + 55
    bottom = y1 - 90
    draw.rounded_rectangle(box, radius=26, fill="#ffffff", outline="#d7dde7", width=3)
    draw_text_center(draw, (0.5 * (x0 + x1), y0 + 24), title, title_font, "#102030")
    draw_text_center(draw, (0.5 * (left + right), y1 - 28), "Oracle Top-% Budget", axis_font, "#102030")

    label_bbox = draw.textbbox((0, 0), y_label, font=axis_font)
    label_image = Image.new(
        "RGBA",
        (label_bbox[2] - label_bbox[0] + 16, label_bbox[3] - label_bbox[1] + 16),
        (255, 255, 255, 0),
    )
    ImageDraw.Draw(label_image).text((8, 8), y_label, font=axis_font, fill="#102030")
    rotated = label_image.rotate(90, expand=True)
    image.alpha_composite(rotated, (x0 + 16, int(0.5 * (top + bottom) - rotated.size[1] / 2)))

    legend_bottom = draw_wrapped_legend(draw, left, right, top - 30, series_names, series_labels, colors)
    top = max(top + 12, legend_bottom + 8)

    for frac in np.linspace(0.0, 1.0, num=5):
        y_pix = bottom - frac * (bottom - top)
        draw.line((left, y_pix, right, y_pix), fill="#d6dce5", width=1)
        tick_value = frac * max_value
        draw_text_center(draw, (left - 38, y_pix), f"{tick_value:.2f}", tick_font, "#4b5563")

    draw.line((left, top, left, bottom), fill="#243447", width=2)
    draw.line((left, bottom, right, bottom), fill="#243447", width=2)

    num_groups = len(percent_labels)
    group_width = (right - left) / max(num_groups, 1)
    bar_width = group_width / max((2 * len(series_names) + 1), 3)

    for idx, percent_label in enumerate(percent_labels):
        group_center = left + group_width * (idx + 0.5)
        draw_text_center(draw, (group_center, bottom + 22), percent_label, tick_font, "#334155")
        total_bar_span = bar_width * (2 * len(series_names) - 1)
        start_x = group_center - 0.5 * total_bar_span
        for series_idx, series_name in enumerate(series_names):
            value = values_by_series[series_name][idx]
            height = bar_height(value, max_value, bottom - top)
            bar_left = start_x + (2 * bar_width * series_idx)
            bar_right = bar_left + bar_width
            bar_top = bottom - height
            draw.rounded_rectangle(
                (bar_left, bar_top, bar_right, bottom),
                radius=8,
                fill=colors[series_name],
                outline=None,
            )
            draw_text_center(draw, (0.5 * (bar_left + bar_right), bar_top - 12), f"{value:.2f}", value_font, "#111827")


def render_summary_plot(summary: dict[str, Any], output_png: Path) -> None:
    percent_labels = list(summary["per_percent"].keys())
    signal_keys = summary["signal_keys"]
    series_names = ["random", *signal_keys]
    series_labels = {"random": "Random"}
    series_labels.update({signal_key: signal_label(signal_key) for signal_key in signal_keys})
    colors = build_color_map(signal_keys)

    recall_values = {
        "random": [summary["per_percent"][label]["random_baseline_recall"] for label in percent_labels],
    }
    lift_values = {"random": [1.0 for _ in percent_labels]}
    for signal_key in signal_keys:
        recall_values[signal_key] = [
            summary["per_percent"][label]["signals"][signal_key]["mean_recall_at_percent"]
            for label in percent_labels
        ]
        lift_values[signal_key] = [
            summary["per_percent"][label]["signals"][signal_key]["mean_lift_vs_random"]
            for label in percent_labels
        ]

    width, height = 2280, 1060
    image = Image.new("RGBA", (width, height), "#f3f6fb")
    draw = ImageDraw.Draw(image)
    title_font = load_font(34, bold=True)
    subtitle_font = load_font(21)
    draw.text((62, 40), "Cheap Signal Top-k Retrieval", font=title_font, fill="#102030")
    draw.text(
        (62, 92),
        "Matched-budget retrieval: compare each shortlisted cheap signal against oracle-drift top q% tokens for each transition.",
        font=subtitle_font,
        fill="#405264",
    )

    draw_bar_panel(
        draw,
        image,
        box=(48, 165, 1120, 990),
        title="Mean Recall@q",
        y_label="Recovered oracle top-q fraction",
        percent_labels=percent_labels,
        series_names=series_names,
        series_labels=series_labels,
        values_by_series=recall_values,
        colors=colors,
        max_value=max(max(values) for values in recall_values.values()) * 1.18,
    )
    draw_bar_panel(
        draw,
        image,
        box=(1160, 165, 2232, 990),
        title="Lift vs Random",
        y_label="Recall / random baseline",
        percent_labels=percent_labels,
        series_names=series_names,
        series_labels=series_labels,
        values_by_series=lift_values,
        colors=colors,
        max_value=max(max(values) for values in lift_values.values()) * 1.18,
    )
    image.convert("RGB").save(output_png)


def main() -> None:
    args = parse_args()
    top_percents = parse_top_percents(args.top_percents)
    signal_keys = parse_signal_keys(args.signal_keys)
    rows, oracle_source_key = read_rows(args.input_jsonl, args.sample_id, signal_keys)
    summary = evaluate_retrieval(rows, top_percents, signal_keys, oracle_source_key)
    summary.update(
        {
            "input_jsonl": str(args.input_jsonl),
            "output_png": str(args.output_png),
            "output_summary_json": str(args.output_summary_json),
            "sample_id": args.sample_id,
            "signal_labels": {signal_key: signal_label(signal_key) for signal_key in signal_keys},
        }
    )

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    render_summary_plot(summary, args.output_png)
    args.output_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
