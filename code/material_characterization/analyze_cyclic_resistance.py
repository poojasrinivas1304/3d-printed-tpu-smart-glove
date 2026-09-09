import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

FILES = {
    "Only conductive TPU\n2.5 mm, 5 mm/s": "TPU_1000.csv",
    "Conductive TPU + backing\n2.5 mm, 5 mm/s": "TPU_backing_1000.csv",
    "Conductive TPU + backing\n5 mm, 10 mm/s": "TPU_backing_5mm_1000.csv",
}

BASELINE_SECONDS = 5.0

OUTPUT_PREFIX = "cyclic_characterization"

# ============================================================


def load_and_normalize(filename):
    df = pd.read_csv(filename)

    required = ["arduino_t_ms", "R_kOhm"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"{filename} is missing columns: {missing}")

    df["time_s"] = (df["arduino_t_ms"] - df["arduino_t_ms"].iloc[0]) / 1000.0

    R = pd.to_numeric(df["R_kOhm"], errors="coerce")
    df["R_kOhm"] = R

    df = df.dropna(subset=["time_s", "R_kOhm"]).reset_index(drop=True)

    baseline_mask = df["time_s"] <= BASELINE_SECONDS

    if baseline_mask.sum() < 5:
        baseline_mask = df.index < min(100, len(df))

    R0 = df.loc[baseline_mask, "R_kOhm"].median()

    df["dR_R0"] = (df["R_kOhm"] - R0) / R0

    return df, R0


def summarize_signal(label, df, R0):
    y = df["dR_R0"].values

    summary = {
        "test": label.replace("\n", " "),
        "duration_s": df["time_s"].iloc[-1],
        "samples": len(df),
        "sampling_rate_Hz": (len(df) - 1) / df["time_s"].iloc[-1],
        "R0_kOhm": R0,
        "R_min_kOhm": df["R_kOhm"].min(),
        "R_max_kOhm": df["R_kOhm"].max(),
        "dR_R0_min": np.min(y),
        "dR_R0_max": np.max(y),
        "dR_R0_peak_to_peak": np.max(y) - np.min(y),
        "dR_R0_mean": np.mean(y),
        "dR_R0_std": np.std(y),
    }

    return summary


def plot_full_response(data):
    plt.figure(figsize=(14, 7))

    for label, (df, R0) in data.items():
        plt.plot(df["time_s"], df["dR_R0"], linewidth=0.8, label=label)

    plt.axhline(0, linewidth=0.6)
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized resistance change, ΔR/R0")
    plt.title("Full cyclic response comparison")
    plt.legend()
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_full_response.png", dpi=250)
    plt.show()


def plot_zoom_window(data, start_s=None, duration_s=10):
    # Default: middle of the shortest recording
    if start_s is None:
        min_duration = min(df["time_s"].iloc[-1] for df, _ in data.values())
        start_s = 0.5 * min_duration

    end_s = start_s + duration_s

    plt.figure(figsize=(14, 7))

    for label, (df, R0) in data.items():
        mask = (df["time_s"] >= start_s) & (df["time_s"] <= end_s)

        plt.plot(
            df.loc[mask, "time_s"],
            df.loc[mask, "dR_R0"],
            linewidth=1.0,
            label=label,
        )

    plt.axhline(0, linewidth=0.6)
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized resistance change, ΔR/R0")
    plt.title(f"Zoomed cyclic response: {start_s:.1f}–{end_s:.1f} s")
    plt.legend()
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_zoom_response.png", dpi=250)
    plt.show()


def plot_first_vs_last(data, window_s=10):
    fig, axes = plt.subplots(
        len(data),
        2,
        figsize=(14, 3.4 * len(data)),
        sharey=False,
    )

    if len(data) == 1:
        axes = np.array([axes])

    for row, (label, (df, R0)) in enumerate(data.items()):
        t_end = df["time_s"].iloc[-1]

        first_mask = df["time_s"] <= window_s
        last_mask = df["time_s"] >= (t_end - window_s)

        ax1 = axes[row, 0]
        ax2 = axes[row, 1]

        ax1.plot(
            df.loc[first_mask, "time_s"],
            df.loc[first_mask, "dR_R0"],
            linewidth=0.9,
        )
        ax1.axhline(0, linewidth=0.6)
        ax1.set_title(f"{label}\nFirst {window_s} s")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("ΔR/R0")
        ax1.grid(True, linewidth=0.3)

        # Shift last segment time to start from zero for easier comparison
        last_time = df.loc[last_mask, "time_s"] - df.loc[last_mask, "time_s"].iloc[0]

        ax2.plot(
            last_time,
            df.loc[last_mask, "dR_R0"],
            linewidth=0.9,
        )
        ax2.axhline(0, linewidth=0.6)
        ax2.set_title(f"{label}\nLast {window_s} s")
        ax2.set_xlabel("Time within window (s)")
        ax2.set_ylabel("ΔR/R0")
        ax2.grid(True, linewidth=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_first_vs_last.png", dpi=250)
    plt.show()


def main():
    data = {}
    summaries = []

    for label, filename in FILES.items():
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Could not find {filename}. "
                "Make sure the CSV is in the same folder as this script."
            )

        df, R0 = load_and_normalize(filename)
        data[label] = (df, R0)
        summaries.append(summarize_signal(label, df, R0))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(f"{OUTPUT_PREFIX}_summary.csv", index=False)

    print()
    print("Summary:")
    print(summary_df.to_string(index=False))

    plot_full_response(data)
    plot_zoom_window(data, duration_s=10)
    plot_first_vs_last(data, window_s=10)

    print()
    print("Saved:")
    print(f"{OUTPUT_PREFIX}_summary.csv")
    print(f"{OUTPUT_PREFIX}_full_response.png")
    print(f"{OUTPUT_PREFIX}_zoom_response.png")
    print(f"{OUTPUT_PREFIX}_first_vs_last.png")


if __name__ == "__main__":
    main()