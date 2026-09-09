#!/usr/bin/env python3
"""
Plot the latest posture blind demo summary.

Input:
    posture_blind_summary_*.csv

Outputs:
    posture_blind_demo_<timestamp>_summary.png/pdf

Run:
    python3 plot_posture_blind_demo.py
"""

from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSTURES = ["Open", "Fist", "IndexPoint", "ThumbUp", "Pinch"]


def latest_summary_file() -> str:
    files = sorted(glob.glob(os.path.join(SCRIPT_DIR, "posture_blind_summary_*.csv")))
    if not files:
        raise FileNotFoundError("No posture_blind_summary_*.csv files found.")
    return files[-1]


def main() -> None:
    f = latest_summary_file()
    df = pd.read_csv(f)
    base = os.path.splitext(os.path.basename(f))[0]
    out_png = os.path.join(SCRIPT_DIR, base + "_plot.png")
    out_pdf = os.path.join(SCRIPT_DIR, base + "_plot.pdf")

    trials = df["trial_id"].to_numpy()
    label_to_y = {label: i + 1 for i, label in enumerate(POSTURES)}
    true_y = df["true_label"].map(label_to_y).to_numpy()
    top_y = df["top_label"].map(label_to_y).to_numpy()
    final_labels = df["final_prediction"].astype(str).tolist()
    final_y = [label_to_y.get(x, 0) for x in final_labels]

    accepted_correct = df["is_correct_accepted"].astype(int).to_numpy()
    forced_correct = df["is_correct_forced"].astype(int).to_numpy()
    accepted_mask = np.array([x in POSTURES for x in final_labels])

    accepted_n = int(accepted_mask.sum())
    accepted_correct_n = int(accepted_correct.sum())
    forced_correct_n = int(forced_correct.sum())
    total_n = len(df)

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), gridspec_kw={"height_ratios": [1.1, 2.0, 1.0]})

    # Panel 1: true vs final
    ax = axes[0]
    ax.set_title(
        f"Blind posture demo: accepted {accepted_correct_n}/{accepted_n} correct, forced-choice {forced_correct_n}/{total_n} correct",
        fontweight="bold",
    )
    ax.set_xlim(0.5, total_n + 0.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([1.5, 0.5])
    ax.set_yticklabels(["True label", "AI final"])
    ax.set_xticks(trials)
    ax.set_xlabel("Blind trial number", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    for idx, row in df.iterrows():
        x = row["trial_id"]
        true_label = row["true_label"]
        final = row["final_prediction"]
        ax.add_patch(plt.Rectangle((x - 0.5, 1.0), 1.0, 1.0, facecolor="#d8ecff", edgecolor="white"))
        if final == true_label:
            color = "#a8ddb5"
        elif final == "Uncertain":
            color = "#d9d9d9"
        else:
            color = "#f4a3a3"
        ax.add_patch(plt.Rectangle((x - 0.5, 0.0), 1.0, 1.0, facecolor=color, edgecolor="white"))
        ax.text(x, 1.5, true_label, ha="center", va="center", fontsize=9, fontweight="bold")
        display_final = "Unc." if final == "Uncertain" else final
        ax.text(x, 0.5, display_final, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(0.02, -0.28, "Green = accepted correct; grey = uncertain; red = wrong", transform=ax.transAxes, fontsize=9)

    # Panel 2: probability matrix
    ax = axes[1]
    prob_cols = [f"prob_{p}" for p in POSTURES]
    prob = df[prob_cols].to_numpy().T
    im = ax.imshow(prob, cmap="PuBuGn", vmin=0, vmax=1, aspect="auto")
    ax.set_title("Class probabilities for each blind trial", fontweight="bold")
    ax.set_yticks(np.arange(len(POSTURES)))
    ax.set_yticklabels(POSTURES)
    ax.set_xticks(np.arange(total_n))
    ax.set_xticklabels(trials)
    ax.set_ylabel("Model class", fontweight="bold")
    ax.set_xlabel("Blind trial number", fontweight="bold")
    for i in range(prob.shape[0]):
        for j in range(prob.shape[1]):
            if prob[i, j] >= 0.30:
                color = "white" if prob[i, j] > 0.60 else "black"
                ax.text(j, i, f"{prob[i,j]:.2f}", ha="center", va="center", fontsize=8, fontweight="bold", color=color)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Class probability", fontweight="bold")

    # Panel 3: confidence/margin
    ax = axes[2]
    ax.plot(trials, df["confidence"], marker="o", label="Top-class probability")
    ax.plot(trials, df["margin"], marker="s", label="Probability margin")
    ax.axhline(0.35, linestyle="--", linewidth=1, color="gray", label="Confidence threshold")
    ax.axhline(0.08, linestyle=":", linewidth=1, color="gray", label="Margin threshold")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.5, total_n + 0.5)
    ax.set_xticks(trials)
    ax.set_xlabel("Blind trial number", fontweight="bold")
    ax.set_ylabel("Probability", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    print("Saved:")
    print(" ", out_png)
    print(" ", out_pdf)


if __name__ == "__main__":
    main()
