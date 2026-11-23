#!/usr/bin/env python3
"""Compare all seven channels with the five dorsal finger channels under LOSO."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


LABELS = {
    "finger": ["Thumb", "Index", "Middle", "Ring", "Little"],
    "posture": ["Open", "Fist", "IndexPoint", "ThumbUp", "Pinch"],
}


def pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=700,
                    class_weight="balanced",
                    max_features="sqrt",
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=repository / "data/derived/combined_selected_features.csv")
    parser.add_argument("--output", type=Path, default=repository / "reproduced_results/sensor_ablation_loso.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input)
    participants = sorted(data.participant.unique())
    all_features = [column for column in data.columns if "__" in column][:56]
    if len(all_features) != 56:
        raise ValueError(f"Expected 56 feature columns; found {len(all_features)}")
    feature_sets = {
        "seven_sensors": all_features,
        "five_dorsal_sensors": all_features[:40],
    }

    rows = []
    for task, labels in LABELS.items():
        task_data = data[data.task == task]
        for feature_set, columns in feature_sets.items():
            pooled_true = []
            pooled_predicted = []
            for held_out in participants:
                train = task_data[task_data.participant != held_out]
                test = task_data[task_data.participant == held_out]
                model = pipeline()
                model.fit(train[columns], train.target)
                predicted = model.predict(test[columns])
                rows.append(
                    {
                        "task": task,
                        "feature_set": feature_set,
                        "held_out_participant": held_out,
                        "feature_count": len(columns),
                        "test_trials": len(test),
                        "accuracy": accuracy_score(test.target, predicted),
                        "macro_f1": f1_score(test.target, predicted, labels=labels, average="macro", zero_division=0),
                    }
                )
                pooled_true.extend(test.target)
                pooled_predicted.extend(predicted)
            rows.append(
                {
                    "task": task,
                    "feature_set": feature_set,
                    "held_out_participant": "POOLED",
                    "feature_count": len(columns),
                    "test_trials": len(pooled_true),
                    "accuracy": accuracy_score(pooled_true, pooled_predicted),
                    "macro_f1": f1_score(pooled_true, pooled_predicted, labels=labels, average="macro", zero_division=0),
                }
            )

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result[result.held_out_participant == "POOLED"].to_string(index=False))


if __name__ == "__main__":
    main()
