import os
import glob
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_DIR = os.path.dirname(SCRIPT_DIR)
ACQUISITION_DATA_DIR = os.path.join(SCRIPT_DIR, "ai_stepwise_data")
RELEASE_DATA_DIR = os.path.join(
    REPOSITORY_DIR,
    "data",
    "classification",
    "folded_finger",
    "session_01",
)
DATA_DIR = os.environ.get(
    "GLOVE_STEPWISE_DATA_DIR",
    RELEASE_DATA_DIR if os.path.isdir(RELEASE_DATA_DIR) else ACQUISITION_DATA_DIR,
)
OUT_DIR = os.environ.get(
    "GLOVE_OUTPUT_DIR",
    os.path.join(REPOSITORY_DIR, "outputs", "folded_finger", "session_01"),
)

CHANNELS = ["Thumb", "Index", "Middle", "Ring", "Little", "ReverseHorizontal", "ReverseThumb"]
CLASSES = ["Thumb", "Index", "Middle", "Ring", "Little"]


def load_stepwise_samples():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "stepwise_train_samples_*.csv")))
    if not files:
        raise FileNotFoundError(f"No stepwise sample files found in {DATA_DIR}")
    frames = []
    for file in files:
        df = pd.read_csv(file)
        df["source_file"] = os.path.basename(file)
        frames.append(df)
    return pd.concat(frames, ignore_index=True), files


def slope_from_series(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    t = t[mask]
    y = y[mask]
    t = t - t.min()
    try:
        return float(np.polyfit(t, y, 1)[0])
    except Exception:
        return np.nan


def make_features(df):
    train = df[(df["phase"] == "fold") & (df["use_for_training"] == 1)].copy()
    if train.empty:
        raise ValueError("No training rows found. Check use_for_training column.")

    rows = []
    for (source_file, trial_id, true_label), g in train.groupby(["source_file", "trial_id", "true_label"]):
        if true_label not in CLASSES:
            continue
        feat = {
            "source_file": source_file,
            "trial_id": trial_id,
            "label": true_label,
        }
        for ch in CHANNELS:
            col = f"{ch}_local_dR_over_open_ref"
            y = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            t = pd.to_numeric(g["phase_elapsed_s"], errors="coerce").to_numpy(dtype=float)
            feat[f"{ch}_mean"] = np.nanmean(y)
            feat[f"{ch}_median"] = np.nanmedian(y)
            feat[f"{ch}_std"] = np.nanstd(y)
            feat[f"{ch}_min"] = np.nanmin(y)
            feat[f"{ch}_max"] = np.nanmax(y)
            feat[f"{ch}_range"] = np.nanmax(y) - np.nanmin(y)
            feat[f"{ch}_last"] = y[np.where(np.isfinite(y))[0][-1]] if np.isfinite(y).any() else np.nan
            feat[f"{ch}_slope"] = slope_from_series(t, y)
        rows.append(feat)
    feat_df = pd.DataFrame(rows)
    return feat_df


def main():
    print("Stepwise finger-folding model training")
    print(f"Data dir: {DATA_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)
    df, files = load_stepwise_samples()
    print("Using files:")
    for f in files:
        print("  ", os.path.basename(f))

    feat_df = make_features(df)
    feature_cols = [c for c in feat_df.columns if c not in ["source_file", "trial_id", "label"]]
    X = feat_df[feature_cols]
    y = feat_df["label"]
    groups = feat_df["source_file"]

    print("\nTrial counts by label:")
    print(y.value_counts().reindex(CLASSES).fillna(0).astype(int))
    print(f"Total trials: {len(feat_df)}")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=700,
            random_state=42,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=1,
        )),
    ])

    # Evaluation. Use leave-one-file-out if more than one file exists; otherwise stratified CV.
    if groups.nunique() >= 2:
        cv = LeaveOneGroupOut()
        y_pred = cross_val_predict(model, X, y, groups=groups, cv=cv)
        eval_name = "Leave-one-file-out"
    else:
        n_splits = min(5, y.value_counts().min())
        if n_splits < 2:
            raise ValueError("Need more trials per class for cross-validation.")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_pred = cross_val_predict(model, X, y, cv=cv)
        eval_name = f"Stratified {n_splits}-fold"

    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, labels=CLASSES, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=CLASSES)

    print(f"\n{eval_name} accuracy: {acc:.3f}")
    print(report)

    # Fit final model on all data.
    model.fit(X, y)

    model_file = os.path.join(OUT_DIR, "finger_ai_stepwise_model.joblib")
    bundle = {
        "model": model,
        "feature_names": feature_cols,
        "classes": CLASSES,
        "channels": CHANNELS,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "data_files": [os.path.basename(f) for f in files],
        "evaluation": eval_name,
        "accuracy": acc,
    }
    joblib.dump(bundle, model_file)

    feat_file = os.path.join(OUT_DIR, "finger_ai_stepwise_training_features.csv")
    feat_df.to_csv(feat_file, index=False)

    report_file = os.path.join(OUT_DIR, "finger_ai_stepwise_training_report.txt")
    with open(report_file, "w") as f:
        f.write("Stepwise finger-folding model\n")
        f.write(f"Evaluation: {eval_name}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Files:\n")
        for file in files:
            f.write(f"  {os.path.basename(file)}\n")
        f.write("\nClass counts:\n")
        f.write(y.value_counts().reindex(CLASSES).fillna(0).astype(int).to_string())
        f.write("\n\nClassification report:\n")
        f.write(report)
        f.write("\nConfusion matrix labels:\n")
        f.write(json.dumps(CLASSES))
        f.write("\nConfusion matrix:\n")
        f.write(np.array2string(cm))

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"Stepwise model confusion matrix\n{eval_name} accuracy = {acc:.1%}")
    fig.tight_layout()
    cm_file = os.path.join(OUT_DIR, "finger_ai_stepwise_confusion_matrix.png")
    fig.savefig(cm_file, dpi=300)

    print("\nSaved:")
    print("  ", model_file)
    print("  ", report_file)
    print("  ", cm_file)
    print("  ", feat_file)


if __name__ == "__main__":
    main()
