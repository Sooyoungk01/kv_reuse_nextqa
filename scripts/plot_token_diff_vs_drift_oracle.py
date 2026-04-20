#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

DEFAULT_INPUT_JSONL = RESULTS_DIR / "drift_oracle.jsonl"
DEFAULT_OUTPUT_PNG = RESULTS_DIR / "token_diff_vs_drift_oracle.png"
DEFAULT_OUTPUT_SUMMARY_JSON = RESULTS_DIR / "token_diff_vs_drift_oracle_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the relationship between token-level pixel differences and drift oracle "
            "importance scores from a drift_oracle JSONL file."
        )
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
        help="Where to save the output figure.",
    )
    parser.add_argument(
        "--output_summary_json",
        type=Path,
        default=DEFAULT_OUTPUT_SUMMARY_JSON,
        help="Where to save summary statistics.",
    )
    parser.add_argument(
        "--x_score_key",
        type=str,
        default="token_pixel_diff_scores",
        help="JSON field used for the x-axis token score.",
    )
    parser.add_argument(
        "--y_score_key",
        type=str,
        default="token_importance_scores",
        help="JSON field used for the y-axis token score.",
    )
    parser.add_argument(
        "--transition_metric_key",
        type=str,
        default="all_reuse_action_kl",
        help="Transition-level metric to compare against summarized token scores.",
    )
    parser.add_argument(
        "--sample_id",
        type=str,
        default=None,
        help="Optional sample_id filter.",
    )
    parser.add_argument(
        "--max_annotated_points",
        type=int,
        default=20,
        help="Maximum number of transition points to annotate in the summary plots.",
    )
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def score_label(score_key: str) -> str:
    special_cases = {
        "token_pixel_diff_scores": "Token Pixel Diff Score",
        "token_importance_scores": "Token Importance Score",
        "token_action_delta_scores": "Token Action Delta Score",
    }
    if score_key in special_cases:
        return special_cases[score_key]
    stem = score_key[:-7] if score_key.endswith("_scores") else score_key
    return stem.replace("_", " ").title()


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    if x.size < 2 or np.allclose(x, x[0]):
        return None
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)


def safe_json_float(value: float) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def load_rows(
    path: Path,
    x_score_key: str,
    y_score_key: str,
    transition_metric_key: str,
    sample_id: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float | str | None]]]:
    token_x: list[float] = []
    token_y: list[float] = []
    token_transition_metric: list[float] = []
    transition_rows: list[dict[str, float | str | None]] = []

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

            x_scores = row.get(x_score_key)
            y_scores = row.get(y_score_key)
            metric_value = row.get(transition_metric_key)

            if not isinstance(x_scores, list) or not isinstance(y_scores, list):
                continue
            if len(x_scores) == 0 or len(x_scores) != len(y_scores):
                continue
            if metric_value is None:
                continue

            x_values = np.asarray([float(v) for v in x_scores], dtype=np.float64)
            y_values = np.asarray([float(v) for v in y_scores], dtype=np.float64)
            transition_metric = float(metric_value)

            token_x.extend(x_values.tolist())
            token_y.extend(y_values.tolist())
            token_transition_metric.extend([transition_metric] * len(x_values))

            transition_rows.append(
                {
                    "sample_id": str(row.get("sample_id")),
                    "transition": str(row.get("transition", f"{row.get('prev_frame_idx')}->{row.get('frame_idx')}")),
                    "frame_idx": int(row.get("frame_idx", -1)),
                    "transition_metric": transition_metric,
                    "mean_x": float(x_values.mean()),
                    "max_x": float(x_values.max()),
                    "mean_y": float(y_values.mean()),
                    "max_y": float(y_values.max()),
                    "token_corr": safe_json_float(pearson_corr(x_values, y_values)),
                }
            )

    if not token_x:
        raise ValueError(
            f"No usable rows were found in '{path}' for x='{x_score_key}', y='{y_score_key}', "
            f"metric='{transition_metric_key}'."
        )

    return (
        np.asarray(token_x, dtype=np.float64),
        np.asarray(token_y, dtype=np.float64),
        np.asarray(token_transition_metric, dtype=np.float64),
        transition_rows,
    )


def format_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value == 0:
        return "0"
    if abs_value >= 0.1:
        return f"{value:.3f}"
    if abs_value >= 0.01:
        return f"{value:.4f}"
    return f"{value:.2e}"


def map_linear(value: float, low: float, high: float, pixel_low: float, pixel_high: float) -> float:
    if high == low:
        return (pixel_low + pixel_high) / 2.0
    return pixel_low + (value - low) * (pixel_high - pixel_low) / (high - low)


def draw_text_center(
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


def lerp_color(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def metric_to_color(value: float, low: float, high: float) -> tuple[int, int, int]:
    if not math.isfinite(value) or high <= low:
        return (47, 107, 124)

    mid = low + 0.5 * (high - low)
    c0 = (59, 130, 246)
    c1 = (16, 185, 129)
    c2 = (245, 158, 11)
    if value <= mid:
        t = (value - low) / max(mid - low, 1e-12)
        return lerp_color(c0, c1, t)
    t = (value - mid) / max(high - mid, 1e-12)
    return lerp_color(c1, c2, t)


def compute_ticks(low: float, high: float, count: int = 5) -> list[float]:
    if not math.isfinite(low) or not math.isfinite(high):
        return [0.0]
    if high == low:
        return [low]
    return np.linspace(low, high, num=count).tolist()


def draw_colorbar(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    low: float,
    high: float,
    font: ImageFont.ImageFont,
    label: str,
) -> None:
    x0, y0, x1, y1 = box
    for y in range(y0, y1):
        t = (y - y0) / max(y1 - y0 - 1, 1)
        color = metric_to_color(high - t * (high - low), low, high)
        draw.line((x0, y, x1, y), fill=color, width=1)
    draw.rectangle((x0, y0, x1, y1), outline="#CBD2D9", width=1)
    draw.text((x0 - 6, y0 - 14), format_number(high), font=font, fill="#334E68")
    draw.text((x0 - 6, y1 + 2), format_number(low), font=font, fill="#334E68")
    draw.text((x0 - 4, y0 - 30), label, font=font, fill="#334E68")


def draw_scatter_panel(
    draw: ImageDraw.ImageDraw,
    panel: tuple[int, int, int, int],
    xs: np.ndarray,
    ys: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    dot_color: tuple[int, int, int] | None = None,
    color_values: np.ndarray | None = None,
    colorbar_label: str | None = None,
    labels: list[str] | None = None,
    max_annotated_points: int = 20,
) -> None:
    panel_x0, panel_y0, panel_x1, panel_y1 = panel
    draw.rounded_rectangle((panel_x0, panel_y0, panel_x1, panel_y1), radius=14, fill="#F8FAFC", outline="#D9E2EC", width=2)

    title_font = load_font(18)
    body_font = load_font(13)
    small_font = load_font(11)

    draw_text_center(draw, ((panel_x0 + panel_x1) / 2, panel_y0 + 22), title, font=title_font, fill="#102A43")

    left = panel_x0 + 78
    right = panel_x1 - 26
    if color_values is not None:
        right -= 42
    top = panel_y0 + 56
    bottom = panel_y1 - 66

    if xs.size == 0 or ys.size == 0:
        draw_text_center(draw, ((left + right) / 2, (top + bottom) / 2), "No data", font=body_font, fill="#52606D")
        return

    x_min = float(xs.min())
    x_max = float(xs.max())
    y_min = float(ys.min())
    y_max = float(ys.max())
    if x_max == x_min:
        x_min -= 0.5
        x_max += 0.5
    if y_max == y_min:
        y_min -= 0.5
        y_max += 0.5

    x_pad = 0.04 * (x_max - x_min)
    y_pad = 0.06 * (y_max - y_min)
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    x_ticks = compute_ticks(x_min, x_max, count=5)
    y_ticks = compute_ticks(y_min, y_max, count=5)

    for tick in x_ticks:
        x = map_linear(tick, x_min, x_max, left, right)
        draw.line((x, top, x, bottom), fill="#E5E7EB", width=1)
        draw.line((x, bottom, x, bottom + 5), fill="#7B8794", width=1)
        tick_text = format_number(tick)
        bbox = draw.textbbox((0, 0), tick_text, font=small_font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, bottom + 10), tick_text, font=small_font, fill="#52606D")

    for tick in y_ticks:
        y = map_linear(tick, y_min, y_max, bottom, top)
        draw.line((left, y, right, y), fill="#E5E7EB", width=1)
        draw.line((left - 5, y, left, y), fill="#7B8794", width=1)
        tick_text = format_number(tick)
        bbox = draw.textbbox((0, 0), tick_text, font=small_font)
        draw.text((left - 10 - (bbox[2] - bbox[0]), y - (bbox[3] - bbox[1]) / 2), tick_text, font=small_font, fill="#52606D")

    draw.line((left, top, left, bottom), fill="#334E68", width=2)
    draw.line((left, bottom, right, bottom), fill="#334E68", width=2)

    draw_text_center(draw, ((left + right) / 2, panel_y1 - 28), xlabel, font=body_font, fill="#102A43")
    draw.text((panel_x0 + 16, panel_y0 + 60), ylabel, font=body_font, fill="#102A43")

    color_low = float(color_values.min()) if color_values is not None else 0.0
    color_high = float(color_values.max()) if color_values is not None else 1.0

    radius = 2
    for idx, (x_value, y_value) in enumerate(zip(xs, ys)):
        px = map_linear(float(x_value), x_min, x_max, left, right)
        py = map_linear(float(y_value), y_min, y_max, bottom, top)
        if color_values is not None:
            color = metric_to_color(float(color_values[idx]), color_low, color_high)
        else:
            color = dot_color or (47, 107, 124)
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)

    fit = fit_line(xs, ys)
    if fit is not None:
        slope, intercept = fit
        x0 = x_min
        y0 = slope * x0 + intercept
        x1 = x_max
        y1 = slope * x1 + intercept
        px0 = map_linear(x0, x_min, x_max, left, right)
        py0 = map_linear(y0, y_min, y_max, bottom, top)
        px1 = map_linear(x1, x_min, x_max, left, right)
        py1 = map_linear(y1, y_min, y_max, bottom, top)
        draw.line((px0, py0, px1, py1), fill="#D1495B", width=2)

    if labels is not None and len(labels) <= max_annotated_points:
        for x_value, y_value, label in zip(xs, ys, labels):
            px = map_linear(float(x_value), x_min, x_max, left, right)
            py = map_linear(float(y_value), y_min, y_max, bottom, top)
            draw.text((px + 4, py - 10), label, font=small_font, fill="#334E68")

    if color_values is not None and colorbar_label is not None:
        colorbar_box = (right + 16, top + 12, right + 28, bottom - 12)
        draw_colorbar(draw, colorbar_box, color_low, color_high, small_font, colorbar_label)


def build_summary(
    input_jsonl: Path,
    output_png: Path,
    x_score_key: str,
    y_score_key: str,
    transition_metric_key: str,
    token_x: np.ndarray,
    token_y: np.ndarray,
    transition_rows: list[dict[str, float | str | None]],
) -> dict:
    transition_metric = np.asarray([float(row["transition_metric"]) for row in transition_rows], dtype=np.float64)
    transition_mean_x = np.asarray([float(row["mean_x"]) for row in transition_rows], dtype=np.float64)
    transition_max_x = np.asarray([float(row["max_x"]) for row in transition_rows], dtype=np.float64)

    token_corr = safe_json_float(pearson_corr(token_x, token_y))
    mean_corr = safe_json_float(pearson_corr(transition_mean_x, transition_metric))
    max_corr = safe_json_float(pearson_corr(transition_max_x, transition_metric))

    return {
        "input_jsonl": str(input_jsonl),
        "output_png": str(output_png),
        "x_score_key": x_score_key,
        "y_score_key": y_score_key,
        "transition_metric_key": transition_metric_key,
        "num_transitions": int(len(transition_rows)),
        "num_token_points": int(token_x.size),
        "token_level_pearson": token_corr,
        "transition_mean_x_vs_metric_pearson": mean_corr,
        "transition_max_x_vs_metric_pearson": max_corr,
        "transition_metric_min": float(transition_metric.min()),
        "transition_metric_max": float(transition_metric.max()),
        "transition_metric_mean": float(transition_metric.mean()),
        "x_score_min": float(token_x.min()),
        "x_score_max": float(token_x.max()),
        "x_score_mean": float(token_x.mean()),
        "y_score_min": float(token_y.min()),
        "y_score_max": float(token_y.max()),
        "y_score_mean": float(token_y.mean()),
        "transitions": transition_rows,
    }


def plot_relationships(
    input_jsonl: Path,
    output_png: Path,
    x_score_key: str,
    y_score_key: str,
    transition_metric_key: str,
    token_x: np.ndarray,
    token_y: np.ndarray,
    token_transition_metric: np.ndarray,
    transition_rows: list[dict[str, float | str | None]],
    max_annotated_points: int,
) -> dict:
    image = Image.new("RGB", (1980, 690), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(24)
    body_font = load_font(14)

    draw.text((32, 18), "Token Diff vs Drift Oracle", font=title_font, fill="#102A43")
    subtitle = (
        f"x={score_label(x_score_key)} | y={score_label(y_score_key)} | "
        f"transition metric={transition_metric_key}"
    )
    draw.text((32, 52), subtitle, font=body_font, fill="#486581")

    token_corr = pearson_corr(token_x, token_y)
    transition_metric = np.asarray([float(row["transition_metric"]) for row in transition_rows], dtype=np.float64)
    transition_mean_x = np.asarray([float(row["mean_x"]) for row in transition_rows], dtype=np.float64)
    transition_max_x = np.asarray([float(row["max_x"]) for row in transition_rows], dtype=np.float64)
    transition_labels = [f'{row["sample_id"]}:{row["transition"]}' for row in transition_rows]

    panel_width = 620
    gap = 24
    top = 92
    bottom = 654
    panels = [
        (32, top, 32 + panel_width, bottom),
        (32 + panel_width + gap, top, 32 + 2 * panel_width + gap, bottom),
        (32 + 2 * (panel_width + gap), top, 32 + 3 * panel_width + 2 * gap, bottom),
    ]

    token_title = "Token-Level Relation"
    if math.isfinite(token_corr):
        token_title += f" | Pearson={token_corr:.3f}"
    draw_scatter_panel(
        draw=draw,
        panel=panels[0],
        xs=token_x,
        ys=token_y,
        title=token_title,
        xlabel=score_label(x_score_key),
        ylabel=score_label(y_score_key),
        color_values=token_transition_metric,
        colorbar_label=transition_metric_key.replace("_", " "),
        max_annotated_points=max_annotated_points,
    )

    mean_corr = pearson_corr(transition_mean_x, transition_metric)
    mean_title = f"Transition Mean Diff vs {transition_metric_key}"
    if math.isfinite(mean_corr):
        mean_title += f" | Pearson={mean_corr:.3f}"
    draw_scatter_panel(
        draw=draw,
        panel=panels[1],
        xs=transition_mean_x,
        ys=transition_metric,
        title=mean_title,
        xlabel=f"Mean {score_label(x_score_key)}",
        ylabel=transition_metric_key.replace("_", " "),
        dot_color=(47, 107, 124),
        labels=transition_labels,
        max_annotated_points=max_annotated_points,
    )

    max_corr = pearson_corr(transition_max_x, transition_metric)
    max_title = f"Transition Max Diff vs {transition_metric_key}"
    if math.isfinite(max_corr):
        max_title += f" | Pearson={max_corr:.3f}"
    draw_scatter_panel(
        draw=draw,
        panel=panels[2],
        xs=transition_max_x,
        ys=transition_metric,
        title=max_title,
        xlabel=f"Max {score_label(x_score_key)}",
        ylabel=transition_metric_key.replace("_", " "),
        dot_color=(76, 149, 108),
        labels=transition_labels,
        max_annotated_points=max_annotated_points,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png)

    return build_summary(
        input_jsonl=input_jsonl,
        output_png=output_png,
        x_score_key=x_score_key,
        y_score_key=y_score_key,
        transition_metric_key=transition_metric_key,
        token_x=token_x,
        token_y=token_y,
        transition_rows=transition_rows,
    )


def main() -> None:
    args = parse_args()

    token_x, token_y, token_transition_metric, transition_rows = load_rows(
        path=args.input_jsonl,
        x_score_key=args.x_score_key,
        y_score_key=args.y_score_key,
        transition_metric_key=args.transition_metric_key,
        sample_id=args.sample_id,
    )

    summary = plot_relationships(
        input_jsonl=args.input_jsonl,
        output_png=args.output_png,
        x_score_key=args.x_score_key,
        y_score_key=args.y_score_key,
        transition_metric_key=args.transition_metric_key,
        token_x=token_x,
        token_y=token_y,
        token_transition_metric=token_transition_metric,
        transition_rows=transition_rows,
        max_annotated_points=args.max_annotated_points,
    )

    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved plot to {args.output_png}")
    print(f"Saved summary to {args.output_summary_json}")


if __name__ == "__main__":
    main()
