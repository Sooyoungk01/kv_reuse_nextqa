#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
DEFAULT_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
DEFAULT_FONT_BOLD_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
TITLE_FONT_SIZE = 22
SUBTITLE_FONT_SIZE = 22
PANEL_TITLE_FONT_SIZE = 22
BODY_FONT_SIZE = 22
STATS_FONT_SIZE = 22

DEFAULT_INPUT_JSONL = RESULTS_DIR / "drift_oracle.jsonl"
DEFAULT_OUTPUT_PNG = RESULTS_DIR / "token_importance_distribution.png"
DEFAULT_OUTPUT_SUMMARY_JSON = RESULTS_DIR / "token_importance_distribution_summary.json"
DEFAULT_SCORE_KEY = "token_importance_scores"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the distribution of per-token score fields from a drift oracle JSONL file."
    )
    parser.add_argument(
        "--input_jsonl",
        type=Path,
        default=DEFAULT_INPUT_JSONL,
        help="Path to the drift oracle JSONL file.",
    )
    parser.add_argument(
        "--output_png",
        type=Path,
        default=DEFAULT_OUTPUT_PNG,
        help="Where to save the output plot.",
    )
    parser.add_argument(
        "--output_summary_json",
        type=Path,
        default=DEFAULT_OUTPUT_SUMMARY_JSON,
        help="Where to save summary statistics.",
    )
    parser.add_argument(
        "--score_key",
        type=str,
        default=DEFAULT_SCORE_KEY,
        help="JSON field that contains per-token scores.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Number of histogram bins.",
    )
    return parser.parse_args()


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def _score_key_stem(score_key: str) -> str:
    return score_key[:-7] if score_key.endswith("_scores") else score_key


def score_label(score_key: str) -> str:
    special_cases = {
        "token_importance_scores": "Token Importance Score",
        "token_pixel_diff_scores": "Token Pixel Diff Score",
        "token_action_kl_scores": "Token Action KL Score",
        "token_action_kl_normalized_scores": "Normalized Token Action KL Score",
        "token_oracle_delta_nll_scores": "Token Oracle Delta NLL Score",
        "token_oracle_delta_nll_normalized_scores": "Normalized Token Oracle Delta NLL Score",
    }
    if score_key in special_cases:
        return special_cases[score_key]
    return _score_key_stem(score_key).replace("_", " ").title()


def score_axis_label(score_key: str) -> str:
    return _score_key_stem(score_key).replace("_", " ")


def default_output_paths_for_score_key(score_key: str) -> tuple[Path, Path]:
    if score_key == DEFAULT_SCORE_KEY:
        return DEFAULT_OUTPUT_PNG, DEFAULT_OUTPUT_SUMMARY_JSON

    stem = _score_key_stem(score_key)
    return (
        RESULTS_DIR / f"{stem}_distribution.png",
        RESULTS_DIR / f"{stem}_distribution_summary.json",
    )


def resolve_output_paths(args: argparse.Namespace) -> None:
    default_png, default_summary_json = default_output_paths_for_score_key(args.score_key)
    if args.output_png == DEFAULT_OUTPUT_PNG:
        args.output_png = default_png
    if args.output_summary_json == DEFAULT_OUTPUT_SUMMARY_JSON:
        args.output_summary_json = default_summary_json


def load_scores(path: Path, score_key: str) -> tuple[np.ndarray, list[float]]:
    flat_scores: list[float] = []
    per_transition_means: list[float] = []
    first_ok_row: dict | None = None

    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            if row.get("status") != "ok":
                continue

            if first_ok_row is None:
                first_ok_row = row

            scores = row.get(score_key)
            if not isinstance(scores, list) or not scores:
                continue

            numeric_scores = [float(score) for score in scores]
            flat_scores.extend(numeric_scores)
            per_transition_means.append(mean(numeric_scores))

    if not flat_scores:
        available_score_keys: list[str] = []
        if first_ok_row is not None:
            available_score_keys = sorted(key for key, value in first_ok_row.items() if isinstance(value, list) and key.endswith("_scores"))

        message = f"No scores were found in '{path}' using key '{score_key}'."
        if available_score_keys:
            message += f" Available *_scores keys in the file include: {', '.join(available_score_keys)}."
        message += " If you recently added this field, rerun run_drift_oracle.py to regenerate the JSONL."
        raise ValueError(message)

    return np.asarray(flat_scores, dtype=np.float64), per_transition_means


def build_summary(scores: np.ndarray, per_transition_means: list[float]) -> dict:
    positive_scores = scores[scores > 0]
    return {
        "num_token_scores": int(scores.size),
        "num_transitions": int(len(per_transition_means)),
        "num_positive_scores": int(positive_scores.size),
        "num_nonpositive_scores": int(scores.size - positive_scores.size),
        "positive_score_fraction": float(positive_scores.size / max(int(scores.size), 1)),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "std": float(scores.std()),
        "p90": percentile(scores, 90),
        "p95": percentile(scores, 95),
        "p99": percentile(scores, 99),
        "transition_mean_min": float(min(per_transition_means)),
        "transition_mean_max": float(max(per_transition_means)),
        "transition_mean_avg": float(mean(per_transition_means)),
        "transition_mean_median": float(median(per_transition_means)),
    }


def _log_ticks(min_value: float, max_value: float) -> list[float]:
    start_exp = math.floor(math.log10(min_value))
    end_exp = math.ceil(math.log10(max_value))
    ticks = [10**exp for exp in range(start_exp, end_exp + 1)]
    return [tick for tick in ticks if min_value <= tick <= max_value]


def _map_log_x(value: float, min_value: float, max_value: float, left: int, right: int) -> float:
    log_min = math.log10(min_value)
    log_max = math.log10(max_value)
    log_value = math.log10(value)
    if log_max == log_min:
        return float(left)
    return left + (log_value - log_min) * (right - left) / (log_max - log_min)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    dash: int = 8,
    gap: int = 6,
    width: int = 2,
) -> None:
    x0, y0 = start
    x1, y1 = end
    total = math.dist((x0, y0), (x1, y1))
    if total == 0:
        return

    dx = (x1 - x0) / total
    dy = (y1 - y0) / total
    drawn = 0.0
    while drawn < total:
        seg_end = min(drawn + dash, total)
        sx = x0 + dx * drawn
        sy = y0 + dy * drawn
        ex = x0 + dx * seg_end
        ey = y0 + dy * seg_end
        draw.line((sx, sy, ex, ey), fill=color, width=width)
        drawn += dash + gap


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


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    font_path = DEFAULT_FONT_BOLD_PATH if bold else DEFAULT_FONT_PATH
    try:
        return ImageFont.truetype(str(font_path), size=size)
    except OSError:
        return ImageFont.load_default()


def _draw_pillow_distribution(
    scores: np.ndarray,
    summary: dict,
    output_png: Path,
    bins: int,
    score_key: str,
    input_jsonl: Path,
) -> None:
    positive_scores = scores[scores > 0]
    if positive_scores.size == 0:
        raise ValueError("All collected scores are non-positive, so log-scale plotting is not possible.")

    score_title = score_label(score_key)
    score_axis = score_axis_label(score_key)

    sorted_scores = np.sort(positive_scores)
    ecdf = np.arange(1, sorted_scores.size + 1) / sorted_scores.size
    log_bins = np.logspace(np.log10(positive_scores.min()), np.log10(positive_scores.max()), bins)
    counts, edges = np.histogram(positive_scores, bins=log_bins)

    width, height = 2200, 1120
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _load_font(BODY_FONT_SIZE)
    title_font = _load_font(TITLE_FONT_SIZE, bold=True)
    subtitle_font = _load_font(SUBTITLE_FONT_SIZE)
    panel_title_font = _load_font(PANEL_TITLE_FONT_SIZE, bold=True)
    stats_font = _load_font(STATS_FONT_SIZE)

    bg_panel = "#F7FAFC"
    border = "#D7DEE4"
    text = "#1F2933"
    grid = "#E6EDF3"
    hist_bar = "#2F6B7C"
    ecdf_line = "#1B3A4B"
    ref_colors = {"mean": "#D1495B", "median": "#EDA400", "p95": "#4C956C"}

    draw.rectangle((0, 0, width - 1, height - 1), outline="#E5E7EB")
    _draw_text_center(
        draw,
        (width / 2, 42),
        f"Distribution of {score_title}s in {input_jsonl.name}",
        title_font,
        text,
    )
    _draw_text_center(
        draw,
        (width / 2, 86),
        f"score_key: {score_key}    positive-only log scale panels: {summary['num_positive_scores']:,}/{summary['num_token_scores']:,}",
        subtitle_font,
        "#52606D",
    )

    panel_top = 150
    panel_bottom = 1040
    panel_gap = 42
    panel_width = (width - 120 - panel_gap) // 2
    left_panel = (60, panel_top, 60 + panel_width, panel_bottom)
    right_panel = (left_panel[2] + panel_gap, panel_top, left_panel[2] + panel_gap + panel_width, panel_bottom)

    for panel in [left_panel, right_panel]:
        draw.rounded_rectangle(panel, radius=12, fill=bg_panel, outline=border, width=1)

    def draw_hist_panel(panel: tuple[int, int, int, int]) -> None:
        px0, py0, px1, py1 = panel
        left = px0 + 128
        right = px1 - 34
        top = py0 + 110
        bottom = py1 - 122
        draw.text((px0 + 24, py0 + 24), f"{score_title} Histogram", font=panel_title_font, fill=text)
        draw.text((px0 + 24, py0 + 62), "count", font=font, fill="#52606D")
        draw.rectangle((left, top, right, bottom), outline=border, width=1)

        max_count = int(counts.max()) if counts.size else 1
        for frac in [0.25, 0.5, 0.75]:
            y = bottom - frac * (bottom - top)
            draw.line((left, y, right, y), fill=grid, width=1)
            count_label = str(int(round(max_count * frac)))
            bbox = draw.textbbox((0, 0), count_label, font=font)
            draw.text((left - 16 - (bbox[2] - bbox[0]), y - (bbox[3] - bbox[1]) / 2), count_label, font=font, fill="#52606D")

        for count, edge_left, edge_right in zip(counts, edges[:-1], edges[1:]):
            x0 = _map_log_x(float(edge_left), float(positive_scores.min()), float(positive_scores.max()), left, right)
            x1 = _map_log_x(float(edge_right), float(positive_scores.min()), float(positive_scores.max()), left, right)
            y = bottom - (count / max_count) * (bottom - top)
            draw.rectangle((x0, y, max(x0 + 1, x1), bottom), fill=hist_bar)

        label_y = {"mean": top + 8, "median": top + 46, "p95": top + 84}
        for label_name, value in [("mean", summary["mean"]), ("median", summary["median"]), ("p95", summary["p95"])]:
            x = _map_log_x(float(value), float(positive_scores.min()), float(positive_scores.max()), left, right)
            _draw_dashed_line(draw, (x, top), (x, bottom), ref_colors[label_name], width=3)
            draw.text((x + 8, label_y[label_name]), f"{label_name} {value:.4g}", font=font, fill=ref_colors[label_name])

        ticks = _log_ticks(float(positive_scores.min()), float(positive_scores.max()))
        for tick in ticks:
            x = _map_log_x(float(tick), float(positive_scores.min()), float(positive_scores.max()), left, right)
            draw.line((x, bottom, x, bottom + 8), fill="#7B8794", width=1)
            tick_label = f"{tick:.0e}"
            bbox = draw.textbbox((0, 0), tick_label, font=font)
            draw.text((x - (bbox[2] - bbox[0]) / 2, bottom + 18), tick_label, font=font, fill="#52606D")

        _draw_text_center(draw, ((left + right) / 2, py1 - 44), f"{score_axis} (log scale)", font, "#52606D")

    def draw_ecdf_panel(panel: tuple[int, int, int, int]) -> None:
        px0, py0, px1, py1 = panel
        left = px0 + 128
        right = px1 - 34
        top = py0 + 110
        bottom = py1 - 122
        draw.text((px0 + 24, py0 + 24), f"{score_title} Empirical CDF", font=panel_title_font, fill=text)
        draw.text((px0 + 24, py0 + 62), "fraction of positive-score tokens", font=font, fill="#52606D")
        draw.rectangle((left, top, right, bottom), outline=border, width=1)

        for frac in [0.25, 0.5, 0.75]:
            y = bottom - frac * (bottom - top)
            draw.line((left, y, right, y), fill=grid, width=1)
            tick_text = f"{frac:.2f}"
            bbox = draw.textbbox((0, 0), tick_text, font=font)
            draw.text((left - 16 - (bbox[2] - bbox[0]), y - (bbox[3] - bbox[1]) / 2), tick_text, font=font, fill="#52606D")

        point_count = min(1600, sorted_scores.size)
        sample_indices = np.linspace(0, sorted_scores.size - 1, point_count, dtype=int)
        sampled_scores = sorted_scores[sample_indices]
        sampled_ecdf = ecdf[sample_indices]
        points = []
        for score, frac in zip(sampled_scores, sampled_ecdf):
            x = _map_log_x(float(score), float(positive_scores.min()), float(positive_scores.max()), left, right)
            y = bottom - float(frac) * (bottom - top)
            points.append((x, y))
        draw.line(points, fill=ecdf_line, width=4)

        ticks = _log_ticks(float(positive_scores.min()), float(positive_scores.max()))
        for tick in ticks:
            x = _map_log_x(float(tick), float(positive_scores.min()), float(positive_scores.max()), left, right)
            draw.line((x, bottom, x, bottom + 8), fill="#7B8794", width=1)
            tick_label = f"{tick:.0e}"
            bbox = draw.textbbox((0, 0), tick_label, font=font)
            draw.text((x - (bbox[2] - bbox[0]) / 2, bottom + 18), tick_label, font=font, fill="#52606D")

        stats_text = "\n".join(
            [
                f"tokens: {summary['num_token_scores']:,}",
                f"positive: {summary['num_positive_scores']:,}",
                f"transitions: {summary['num_transitions']:,}",
                f"min: {summary['min']:.4g}",
                f"mean: {summary['mean']:.4g}",
                f"median: {summary['median']:.4g}",
                f"p95: {summary['p95']:.4g}",
                f"p99: {summary['p99']:.4g}",
                f"max: {summary['max']:.4g}",
            ]
        )
        draw.rounded_rectangle((right - 360, top + 18, right - 12, top + 390), radius=10, fill="white", outline=border, width=1)
        draw.multiline_text((right - 336, top + 34), stats_text, font=stats_font, fill=text, spacing=10)
        _draw_text_center(draw, ((left + right) / 2, py1 - 44), f"{score_axis} (log scale)", font, "#52606D")

    draw_hist_panel(left_panel)
    draw_ecdf_panel(right_panel)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png)


def plot_distribution(
    scores: np.ndarray,
    summary: dict,
    output_png: Path,
    bins: int,
    score_key: str,
    input_jsonl: Path,
) -> None:
    positive_scores = scores[scores > 0]
    if positive_scores.size == 0:
        raise ValueError("All collected scores are non-positive, so log-scale plotting is not possible.")
    _draw_pillow_distribution(scores, summary, output_png, bins, score_key, input_jsonl)


def main() -> None:
    args = parse_args()
    resolve_output_paths(args)

    scores, per_transition_means = load_scores(args.input_jsonl, args.score_key)
    summary = build_summary(scores, per_transition_means)
    summary["score_key"] = args.score_key
    summary["score_label"] = score_label(args.score_key)
    summary["input_jsonl"] = str(args.input_jsonl)

    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    plot_distribution(
        scores,
        summary,
        args.output_png,
        bins=args.bins,
        score_key=args.score_key,
        input_jsonl=args.input_jsonl,
    )

    print(f"Saved plot to: {args.output_png}")
    print(f"Saved summary to: {args.output_summary_json}")


if __name__ == "__main__":
    main()
