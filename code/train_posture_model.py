#!/usr/bin/env python3
"""
Train posture-recognition model from stepwise posture training trials.

Expected input folder:
    ai_posture_data/posture_train_trials_*.csv

Outputs:
    posture_ai_model.joblib
    posture_ai_training_report.txt
    posture_ai_confusion_matrix.png
    posture_ai_training_features.csv

Run:
    python3 train_posture_model.py
"""

from __future__ import annotations

import glob
import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_DIR = os.path.dirname(SCRIPT_DIR)
ACQUISITION_DATA_DIR = os.path.join(SCRIPT_DIR, "ai_posture_data")
RELEASE_DATA_DIR = os.path.join(
    REPOSITORY_DIR,
    "data",
    "classification",
    "posture",
    "session_01",
)
DATA_DIR = os.environ.get(
    "GLOVE_POSTURE_DATA_DIR",
    RELEASE_DATA_DIR if os.path.isdir(RELEASE_DATA_DIR) else ACQUISITION_DATA_DIR,
)

POSTURES = ["Open", "Fist", "IndexPoint", "ThumbUp", "Pinch"]

MODEL_FILE = os.path.join(SCRIPT_DIR, "posture_ai_model.joblib")
REPORT_FILE = os.path.join(SCRIPT_DIR, "posture_ai_training_report.txt")
CM_FILE = os.path.join(SCRIPT_DIR, "posture_ai_confusion_matrix.png")
FEATURES_FILE = os.path.join(SCRIPT_DIR, "posture_ai_training_features.csv")


def load_trial_data() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "posture_train_trials_*.csv")))
    if not files:
        raise FileNotFoundError(f"No posture_train_trials_*.csv files found in {DATA_DIR}")
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = os.path.basename(f)
            frames.append(df)
            print(f"Loaded {f}: {df.shape}")
        except Exception as exc:
            print(f"Skipping unreadable file {f}: {exc}")
    if not frames:
        raise RuntimeError("No readable trial files found.")
    data = pd.concat(frames, ignore_index=True)
    data = data[data["status"].astype(str).str.lower() == "accepted"].copy()
    return data


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], accuracy: float) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.max()))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted class", fontweight="bold")
    ax.set_ylabel("True class", fontweight="bold")
    ax.set_title(f"Posture classifier confusion matrix (accuracy = {accuracy:.2f})", fontweight="bold")
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        for j in range(cm.shape[1]):
            value = cm[i, j]
            pct = 100 * value / row_sum if row_sum else 0
            text_color = "white" if value > cm.max() * 0.45 else "black"
            if value > 0:
                text = f"{value}\n{pct:.0f}%"
            else:
                text = "0"
            ax.text(j, i, text, ha="center", va="center", color=text_color, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of trials", fontweight="bold")
    fig.tight_layout()
    fig.savefig(CM_FILE, dpi=600)
    plt.close(fig)


def main() -> None:
    print("Posture model training")
    print("Data dir:", DATA_DIR)
    df = load_trial_data()
    if df.empty:
        raise RuntimeError("No accepted trials found.")

    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    if not feature_cols:
        raise RuntimeError("No feature columns found. Expected columns starting with feat_.")

    # Keep only known posture labels.
    df = df[df["target_label"].isin(POSTURES)].copy()
    X = df[feature_cols]
    y = df["target_label"].astype(str)

    print("\nClass counts:")
    print(y.value_counts().reindex(POSTURES).fillna(0).astype(int))

    min_count = int(y.value_counts().min())
    if min_count < 2:
        raise RuntimeError("Need at least 2 samples per class for cross-validation.")
    n_splits = min(5, min_count)

    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=700,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=42,
        )),
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    report = classification_report(y, y_pred, labels=POSTURES, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=POSTURES)
    accuracy = float(np.mean(y_pred == y))

    print("\nCross-validated classification report:")
    print(report)

    plot_confusion_matrix(cm, POSTURES, accuracy)

    # Fit final model on all accepted data.
    clf.fit(X, y)
    model_payload = {
        "pipeline": clf,
        "feature_names": feature_cols,
        "classes": POSTURES,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_trials": int(len(df)),
        "class_counts": y.value_counts().to_dict(),
        "cv_accuracy": accuracy,
    }
    joblib.dump(model_payload, MODEL_FILE)

    df.to_csv(FEATURES_FILE, index=False)
    with open(REPORT_FILE, "w") as f:
        f.write("Posture AI model training report\n")
        f.write("================================\n\n")
        f.write(f"Data directory: {DATA_DIR}\n")
        f.write(f"Accepted trials: {len(df)}\n")
        f.write(f"Cross-validation folds: {n_splits}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Class counts:\n")
        f.write(str(y.value_counts().reindex(POSTURES).fillna(0).astype(int)))
        f.write("\n\nClassification report:\n")
        f.write(report)
        f.write("\nConfusion matrix rows=true, columns=predicted:\n")
        f.write(pd.DataFrame(cm, index=POSTURES, columns=POSTURES).to_string())
        f.write("\n")

    print("\nSaved:")
    print(" ", MODEL_FILE)
    print(" ", REPORT_FILE)
    print(" ", CM_FILE)
    print(" ", FEATURES_FILE)


if __name__ == "__main__":
    main()
