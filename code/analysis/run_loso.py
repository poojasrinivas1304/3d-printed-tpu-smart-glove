#!/usr/bin/env python3
"""Reproduce the participant-independent LOSO recognition analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline


SEED = 42
N_ESTIMATORS = 700
LABELS = {
    "finger": ["Thumb", "Index", "Middle", "Ring", "Little"],
    "posture": ["Open", "Fist", "IndexPoint", "ThumbUp", "Pinch"],
}
PROTOCOLS = ("all_trials", "calibration_only", "calibration_to_evaluation")


def model_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    class_weight="balanced",
                    max_features="sqrt",
                    min_samples_leaf=1,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def wilson_interval(correct: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return centre - half, centre + half


def participant_bootstrap(values: pd.Series, n_boot: int = 20_000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    array = values.to_numpy(float)
    means = rng.choice(array, size=(n_boot, len(array)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def protocol_split(frame: pd.DataFrame, held_out: str, protocol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[frame.participant != held_out]
    test = frame[frame.participant == held_out]
    if protocol == "calibration_only":
        return train[train.session_type == "cal"], test[test.session_type == "cal"]
    if protocol == "calibration_to_evaluation":
        return train[train.session_type == "cal"], test[test.session_type == "evaluation"]
    if protocol == "all_trials":
        return train, test
    raise ValueError(f"Unknown protocol: {protocol}")


def evaluate(
    data: pd.DataFrame,
    task: str,
    protocol: str,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frame = data[data.task == task].copy()
    labels = LABELS[task]
    predictions: list[dict] = []
    folds: list[dict] = []

    for held_out in sorted(frame.participant.unique()):
        train, test = protocol_split(frame, held_out, protocol)
        model = model_pipeline()
        model.fit(train[feature_columns], train.target)
        predicted = model.predict(test[feature_columns])
        probabilities = model.predict_proba(test[feature_columns])
        classes = list(model.named_steps["classifier"].classes_)

        folds.append(
            {
                "task": task,
                "protocol": protocol,
                "held_out_participant": held_out,
                "train_participants": ", ".join(sorted(train.participant.unique())),
                "train_trials": len(train),
                "test_trials": len(test),
                "correct": int(np.sum(predicted == test.target.to_numpy())),
                "accuracy": accuracy_score(test.target, predicted),
                "balanced_accuracy": balanced_accuracy_score(test.target, predicted),
                "macro_f1": f1_score(test.target, predicted, labels=labels, average="macro", zero_division=0),
            }
        )

        for position, (_, row) in enumerate(test.iterrows()):
            record = {
                "task": task,
                "protocol": protocol,
                "held_out_participant": held_out,
                "session_type": row.session_type,
                "source_row": int(row.source_row),
                "true_label": row.target,
                "predicted_label": predicted[position],
                "correct": int(predicted[position] == row.target),
                "confidence": float(probabilities[position].max()),
                "source_session": row.source_session,
            }
            record.update({f"p_{name}": probabilities[position, index] for index, name in enumerate(classes)})
            predictions.append(record)

    prediction_frame = pd.DataFrame(predictions)
    fold_frame = pd.DataFrame(folds)
    correct = int(prediction_frame.correct.sum())
    total = len(prediction_frame)
    wilson_low, wilson_high = wilson_interval(correct, total)
    bootstrap_low, bootstrap_high = participant_bootstrap(fold_frame.accuracy)
    summary = {
        "task": task,
        "protocol": protocol,
        "participants": len(fold_frame),
        "trials": total,
        "correct": correct,
        "pooled_accuracy": correct / total,
        "pooled_accuracy_wilson_low": wilson_low,
        "pooled_accuracy_wilson_high": wilson_high,
        "mean_participant_accuracy": fold_frame.accuracy.mean(),
        "sd_participant_accuracy": fold_frame.accuracy.std(ddof=1),
        "participant_bootstrap_low": bootstrap_low,
        "participant_bootstrap_high": bootstrap_high,
        "pooled_balanced_accuracy": balanced_accuracy_score(
            prediction_frame.true_label, prediction_frame.predicted_label
        ),
        "pooled_macro_f1": f1_score(
            prediction_frame.true_label,
            prediction_frame.predicted_label,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
    }
    return prediction_frame, fold_frame, summary


def save_diagnostics(predictions: pd.DataFrame, task: str, protocol: str, output: Path) -> pd.DataFrame:
    labels = LABELS[task]
    matrix = pd.DataFrame(
        confusion_matrix(predictions.true_label, predictions.predicted_label, labels=labels),
        index=labels,
        columns=labels,
    )
    matrix.index.name = "true_label"
    matrix.to_csv(output / f"{task}_{protocol}_confusion_matrix.csv")

    report = classification_report(
        predictions.true_label,
        predictions.predicted_label,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    rows = [{"task": task, "protocol": protocol, "class": label, **report[label]} for label in labels]
    result = pd.DataFrame(rows)
    result.to_csv(output / f"{task}_{protocol}_class_metrics.csv", index=False)
    return result


def save_accuracy_figure(folds: pd.DataFrame, output: Path) -> None:
    primary = folds[folds.protocol == "all_trials"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, task, color in zip(axes, LABELS, ["#1f77b4", "#2a9d8f"]):
        frame = primary[primary.task == task]
        values = frame.accuracy * 100
        axis.bar(frame.held_out_participant, values, color=color)
        axis.axhline(20, color="#777777", linestyle="--", linewidth=1, label="Chance (20%)")
        axis.set(title=f"{task.capitalize()} LOSO accuracy", ylabel="Accuracy (%)", ylim=(0, 100))
        axis.legend(loc="lower right", frameon=False)
        for index, value in enumerate(values):
            axis.text(index, value + 2, f"{value:.0f}", ha="center", fontsize=9)
    figure.savefig(output / "loso_accuracy_by_participant.png", dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=repository / "data/derived/combined_selected_features.csv")
    parser.add_argument("--output", type=Path, default=repository / "reproduced_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input)
    feature_columns = [column for column in data.columns if "__" in column][:56]
    if len(feature_columns) != 56:
        raise ValueError(f"Expected 56 feature columns; found {len(feature_columns)}")
    if sorted(data.participant.unique()) != ["P01", "P02", "P03", "P04", "P05"]:
        raise ValueError("Expected de-identified participants P01-P05")

    all_predictions = []
    all_folds = []
    summaries = []
    class_metrics = []
    for task in LABELS:
        for protocol in PROTOCOLS:
            predictions, folds, summary = evaluate(data, task, protocol, feature_columns)
            predictions.to_csv(args.output / f"{task}_{protocol}_predictions.csv", index=False)
            all_predictions.append(predictions)
            all_folds.append(folds)
            summaries.append(summary)
            class_metrics.append(save_diagnostics(predictions, task, protocol, args.output))

    prediction_frame = pd.concat(all_predictions, ignore_index=True)
    fold_frame = pd.concat(all_folds, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    prediction_frame.to_csv(args.output / "all_loso_predictions.csv", index=False)
    fold_frame.to_csv(args.output / "loso_fold_metrics.csv", index=False)
    summary_frame.to_csv(args.output / "loso_summary.csv", index=False)
    pd.concat(class_metrics, ignore_index=True).to_csv(args.output / "all_class_metrics.csv", index=False)
    save_accuracy_figure(fold_frame, args.output)
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
