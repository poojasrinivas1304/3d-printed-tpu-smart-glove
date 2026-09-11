#!/usr/bin/env python3
"""Generate the archived blind-session performance figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPOSITORY_DIR / "data" / "classification" / "archived_session_summary.csv"


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided 95% Wilson interval for a binomial proportion."""
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    half_width = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return 100 * (centre - half_width), 100 * (centre + half_width)


def plot_task(ax: plt.Axes, data: pd.DataFrame, title: str) -> None:
    rows = [
        ("Accepted accuracy", "correct_accepted", "accepted", "#117DAA"),
        ("Coverage", "accepted", "total", "#F28E3B"),
        ("Forced-choice accuracy", "forced_correct", "total", "#65717B"),
    ]
    markers = {"S1": "o", "S2": "s", "S3": "^"}
    offsets = {"S1": -0.14, "S2": 0.0, "S3": 0.14}

    for y, (label, numerator, denominator, colour) in zip([2, 1, 0], rows):
        for session in ["S1", "S2", "S3"]:
            row = data.loc[data["session"] == session].iloc[0]
            value = 100 * row[numerator] / row[denominator]
            ax.scatter(
                value,
                y + offsets[session],
                marker=markers[session],
                s=78,
                facecolors="white",
                edgecolors=colour,
                linewidths=1.8,
                zorder=3,
            )

        successes = int(data[numerator].sum())
        total = int(data[denominator].sum())
        pooled = 100 * successes / total
        low, high = wilson_interval(successes, total)
        ax.errorbar(
            pooled,
            y,
            xerr=np.array([[pooled - low], [high - pooled]]),
            fmt="D",
            markersize=8,
            color=colour,
            markerfacecolor=colour,
            markeredgecolor="white",
            linewidth=2.1,
            capsize=4,
            zorder=4,
        )
        ax.text(
            pooled + 2.2,
            y + 0.18,
            f"{pooled:.1f}%",
            color=colour,
            fontsize=11,
            fontweight="bold",
            va="bottom",
        )

    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=10)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([item[0] for item in rows], fontsize=11)
    ax.set_xlim(0, 105)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xlabel("Performance (%)", fontsize=11)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.grid(axis="x", color="#D7DDE2", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    fig, axes = plt.subplots(1, 2, figsize=(15.25, 6.0), constrained_layout=True)
    plot_task(axes[0], data[data["task"] == "Folded finger"], "(a) Folded-finger recognition")
    plot_task(axes[1], data[data["task"] == "Posture"], "(b) Posture recognition")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="#117DAA", markeredgewidth=1.8, label="Archived session S1"),
        plt.Line2D([], [], marker="s", linestyle="", markerfacecolor="white", markeredgecolor="#117DAA", markeredgewidth=1.8, label="Archived session S2"),
        plt.Line2D([], [], marker="^", linestyle="", markerfacecolor="white", markeredgecolor="#117DAA", markeredgewidth=1.8, label="Archived session S3"),
        plt.Line2D([], [], marker="D", linestyle="", markerfacecolor="#117DAA", markeredgecolor="white", label="Pooled proportion with 95% Wilson interval"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=10.5)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.91))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
