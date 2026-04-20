#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import torch

import plot_cheap_signal_topk_retrieval as topk_plot


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
DEFAULT_INPUT_JSONL = RESULTS_DIR / "cheap_signals.jsonl"
DEFAULT_OUTPUT_PNG = RESULTS_DIR / "cheap_signal_logistic_retrieval.png"
DEFAULT_OUTPUT_SUMMARY_JSON = RESULTS_DIR / "cheap_signal_logistic_retrieval_summary.json"
COMBO_SIGNAL_KEY = "logistic_combo_scores"
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit an out-of-fold logistic-regression combination of the shortlisted cheap signals and compare it "
            "against the raw single-signal retrieval baselines."
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
        "--feature_keys",
        type=str,
        default=",".join(topk_plot.RECOMMENDED_LOGISTIC_FEATURE_KEYS),
        help="Comma-separated cheap-signal feature keys used by the logistic regression combo.",
    )
    parser.add_argument(
        "--compare_keys",
        type=str,
        default=",".join(topk_plot.BASE_COMPARISON_SIGNAL_KEYS),
        help="Comma-separated baseline signal keys shown alongside the logistic combo in the retrieval plot.",
    )
    parser.add_argument("--sample_id", type=str, default=None, help="Optional sample_id filter.")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--regularization_c", type=float, default=1.0)
    return parser.parse_args()


def parse_feature_keys(raw: str) -> list[str]:
    return topk_plot.parse_signal_keys(raw)


def summarize_metric(values: list[float]) -> dict[str, float] | None:
    return topk_plot.summarize_metric(values)


def transition_group_id(row: dict[str, Any]) -> str:
    if row.get("transition_id"):
        return str(row["transition_id"])
    return f"{row.get('sample_id')}:{row.get('transition')}"


def build_binary_labels(oracle_scores: np.ndarray, percent: float) -> tuple[np.ndarray, int]:
    k = max(1, int(math.ceil(oracle_scores.size * percent / 100.0)))
    labels = np.zeros(oracle_scores.size, dtype=np.int64)
    labels[topk_plot.topk_indices(oracle_scores, k)] = 1
    return labels, k


def grouped_kfold_indices(groups: list[str], num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = list(dict.fromkeys(groups))
    if len(unique_groups) < 2:
        raise ValueError("Need at least two transitions for grouped logistic-regression evaluation.")
    n_splits = min(int(num_folds), len(unique_groups))
    group_to_indices: dict[str, list[int]] = {}
    for idx, group in enumerate(groups):
        group_to_indices.setdefault(group, []).append(idx)
    group_buckets = [[] for _ in range(n_splits)]
    for group_idx, group in enumerate(unique_groups):
        group_buckets[group_idx % n_splits].append(group)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for bucket_groups in group_buckets:
        test_indices = sorted(idx for group in bucket_groups for idx in group_to_indices[group])
        train_indices = sorted(idx for idx in range(len(groups)) if idx not in set(test_indices))
        splits.append((np.asarray(train_indices, dtype=np.int64), np.asarray(test_indices, dtype=np.int64)))
    return splits


def fit_weighted_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int,
    regularization_c: float,
) -> tuple[torch.nn.Linear | None, float]:
    positive_count = int(y_train.sum())
    negative_count = int(y_train.shape[0] - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None, float(y_train.mean())

    x_tensor = torch.from_numpy(x_train).to(dtype=torch.float32)
    y_tensor = torch.from_numpy(y_train).to(dtype=torch.float32)
    model = torch.nn.Linear(x_train.shape[1], 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()

    pos_weight = torch.tensor([negative_count / max(positive_count, 1)], dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=int(max_iter), line_search_fn="strong_wolfe")
    l2_strength = 0.0 if regularization_c <= 0 else 1.0 / float(regularization_c)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = model(x_tensor).squeeze(-1)
        loss = loss_fn(logits, y_tensor)
        if l2_strength > 0:
            loss = loss + 0.5 * l2_strength * (model.weight.square().sum() / x_tensor.shape[0])
        loss.backward()
        return loss

    optimizer.step(closure)
    return model, float(y_train.mean())


def predict_probabilities(model: torch.nn.Linear | None, constant_score: float, x_test: np.ndarray) -> np.ndarray:
    if model is None:
        return np.full(x_test.shape[0], constant_score, dtype=np.float64)
    with torch.no_grad():
        x_tensor = torch.from_numpy(x_test).to(dtype=torch.float32)
        logits = model(x_tensor).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
    return probs


def fit_oof_logistic_combo(
    rows: list[dict[str, Any]],
    feature_keys: list[str],
    percent: float,
    num_folds: int,
    max_iter: int,
    regularization_c: float,
) -> tuple[list[list[float]], dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    groups: list[str] = []
    k_values: list[int] = []
    for row in rows:
        oracle_scores = np.asarray(row["oracle_semantic_drift_scores"], dtype=np.float64)
        features = np.column_stack([np.asarray(row[key], dtype=np.float64) for key in feature_keys])
        labels, k = build_binary_labels(oracle_scores, percent)
        transitions.append(
            {
                "features": features,
                "labels": labels,
                "num_tokens": int(oracle_scores.size),
            }
        )
        groups.append(transition_group_id(row))
        k_values.append(k)

    splits = grouped_kfold_indices(groups, num_folds=num_folds)
    combo_scores: list[np.ndarray | None] = [None] * len(transitions)
    fold_coefficients: list[np.ndarray] = []
    fold_intercepts: list[float] = []
    fold_summaries: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        x_train = np.concatenate([transitions[idx]["features"] for idx in train_idx], axis=0)
        y_train = np.concatenate([transitions[idx]["labels"] for idx in train_idx], axis=0)
        mean_vec = x_train.mean(axis=0)
        std_vec = x_train.std(axis=0)
        std_vec = np.where(std_vec < EPS, 1.0, std_vec)
        x_train_norm = (x_train - mean_vec) / std_vec

        model, constant_score = fit_weighted_logistic_regression(
            x_train=x_train_norm,
            y_train=y_train,
            max_iter=max_iter,
            regularization_c=regularization_c,
        )
        if model is not None:
            fold_coefficients.append(model.weight.detach().cpu().numpy()[0].astype(np.float64))
            fold_intercepts.append(float(model.bias.detach().cpu().numpy()[0]))

        for idx in test_idx:
            x_test = transitions[idx]["features"]
            x_test_norm = (x_test - mean_vec) / std_vec
            combo_scores[idx] = predict_probabilities(model, constant_score, x_test_norm)

        fold_summaries.append(
            {
                "fold_index": fold_idx,
                "train_transition_count": int(len(train_idx)),
                "test_transition_count": int(len(test_idx)),
                "train_token_count": int(sum(transitions[idx]["num_tokens"] for idx in train_idx)),
                "test_token_count": int(sum(transitions[idx]["num_tokens"] for idx in test_idx)),
                "positive_rate_train": float(y_train.mean()),
                "used_constant_fallback": bool(model is None),
            }
        )

    if any(scores is None for scores in combo_scores):
        raise RuntimeError("Failed to produce logistic combo scores for every transition.")

    coefficient_mean_by_feature = None
    coefficient_std_by_feature = None
    intercept_mean = None
    intercept_std = None
    if fold_coefficients:
        coef_array = np.stack(fold_coefficients, axis=0)
        coefficient_mean_by_feature = {
            feature_key: float(value) for feature_key, value in zip(feature_keys, coef_array.mean(axis=0).tolist())
        }
        coefficient_std_by_feature = {
            feature_key: float(value) for feature_key, value in zip(feature_keys, coef_array.std(axis=0).tolist())
        }
        intercept_mean = float(np.mean(fold_intercepts))
        intercept_std = float(np.std(fold_intercepts))

    model_summary = {
        "model_type": "grouped_oof_logistic_regression",
        "optimizer": "torch_lbfgs",
        "grouping": "transition_id",
        "num_folds": int(len(splits)),
        "max_iter": int(max_iter),
        "regularization_c": float(regularization_c),
        "class_weight": "balanced",
        "feature_keys": feature_keys,
        "k_stats": summarize_metric([float(k) for k in k_values]),
        "fold_summaries": fold_summaries,
        "coefficient_mean_by_feature": coefficient_mean_by_feature,
        "coefficient_std_by_feature": coefficient_std_by_feature,
        "intercept_mean": intercept_mean,
        "intercept_std": intercept_std,
    }
    return [scores.tolist() for scores in combo_scores], model_summary


def evaluate_budget(rows: list[dict[str, Any]], signal_keys: list[str], percent: float, oracle_source_key: str | None) -> dict[str, Any]:
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
        oracle_top = topk_plot.topk_indices(oracle_scores, k)
        oracle_top_set = set(int(x) for x in oracle_top.tolist())
        oracle_top_mass = float(np.maximum(oracle_scores[oracle_top], 0.0).sum())
        pooled_positive += k
        k_values.append(k)

        for signal_key in signal_keys:
            scores = np.asarray(row[signal_key], dtype=np.float64)
            predicted_top = topk_plot.topk_indices(scores, k)
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
            "label": topk_plot.signal_label(signal_key),
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
    return percent_summary


def main() -> None:
    args = parse_args()
    top_percents = topk_plot.parse_top_percents(args.top_percents)
    feature_keys = parse_feature_keys(args.feature_keys)
    compare_keys = parse_feature_keys(args.compare_keys)
    required_keys = []
    for key in [*feature_keys, *compare_keys]:
        if key not in required_keys:
            required_keys.append(key)
    rows, oracle_source_key = topk_plot.read_rows(args.input_jsonl, args.sample_id, required_keys)

    signal_keys = [*compare_keys, COMBO_SIGNAL_KEY]
    summary: dict[str, Any] = {
        "num_transitions": len(rows),
        "oracle_semantic_drift_source_key": oracle_source_key,
        "top_percent_budgets": top_percents,
        "signal_keys": signal_keys,
        "feature_keys": feature_keys,
        "per_percent": {},
        "logistic_protocol": {
            "model_type": "grouped_oof_logistic_regression",
            "optimizer": "torch_lbfgs",
            "grouping": "transition_id",
            "num_folds": int(args.num_folds),
            "max_iter": int(args.max_iter),
            "regularization_c": float(args.regularization_c),
            "class_weight": "balanced",
            "budget_specific_model": True,
        },
    }

    for percent in top_percents:
        combo_scores, model_summary = fit_oof_logistic_combo(
            rows=rows,
            feature_keys=feature_keys,
            percent=percent,
            num_folds=args.num_folds,
            max_iter=args.max_iter,
            regularization_c=args.regularization_c,
        )
        enriched_rows: list[dict[str, Any]] = []
        for row, combo_score in zip(rows, combo_scores):
            copied_row = dict(row)
            copied_row[COMBO_SIGNAL_KEY] = combo_score
            enriched_rows.append(copied_row)

        percent_key = f"{percent:g}%"
        percent_summary = evaluate_budget(enriched_rows, signal_keys, percent, oracle_source_key)
        percent_summary["logistic_combo_model"] = model_summary
        summary["per_percent"][percent_key] = percent_summary

    summary.update(
        {
            "input_jsonl": str(args.input_jsonl),
            "output_png": str(args.output_png),
            "output_summary_json": str(args.output_summary_json),
            "sample_id": args.sample_id,
            "signal_labels": {signal_key: topk_plot.signal_label(signal_key) for signal_key in signal_keys},
        }
    )

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    topk_plot.render_summary_plot(summary, args.output_png)
    args.output_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
