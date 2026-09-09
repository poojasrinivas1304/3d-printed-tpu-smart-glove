"""
collect_posture_protocol_fig2_style.py

Collects posture-response data for a Figure-2-style analysis:
  (a) raw normalized resistance traces during sequential posture holds
  (b) local-corrected posture response matrix
  (c) repeatability across repeated runs

Postures:
  Open, Fist, IndexPoint, ThumbUp, Pinch

Protocol per run:
  Initial open baseline, 10 s
  Open posture, 10 s
  Open, 10 s
  Fist, 10 s
  Open, 10 s
  Index point, 10 s
  Open, 10 s
  Thumb-up, 10 s
  Open, 10 s
  Pinch, 10 s
  Final open, 10 s

The final 5 s of each posture and its immediately preceding open period
are used for local baseline-corrected summaries.

Requirements:
  pip install bleak numpy matplotlib

Run:
  python3 collect_posture_protocol_fig2_style.py
"""

from __future__ import annotations

import asyncio
import csv
import os
import platform
import re
import subprocess
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# Matplotlib backend must be selected before pyplot.
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt

from bleak import BleakScanner, BleakClient


# ============================================================
# BLE SETTINGS
# ============================================================

DEVICE_NAMES = [
    "Glove_BLE_7_RAW",
    "TShirt_BLE_VP_RAW",
    "TShirt_BLE_10_RAW",
]

TX_CHARACTERISTIC = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"


# ============================================================
# CHANNEL ORDER
# Arduino order:
# GPIO36, GPIO39, GPIO34, GPIO35, GPIO32, GPIO33, GPIO25
# ============================================================

CHANNELS = [
    "Thumb",
    "Index",
    "Middle",
    "Ring",
    "Little",
    "ReverseHorizontal",
    "ReverseThumb",
]

CHANNEL_LABELS = [
    "Thumb",
    "Index",
    "Middle",
    "Ring",
    "Little",
    "Rev. horizontal",
    "Rev. thumb",
]

N_CHANNELS = len(CHANNELS)


# ============================================================
# VOLTAGE DIVIDER SETTINGS
# Wiring:
# 3.3 V ---- fixed resistor ---- ADC pin ---- sensor ---- GND
# Rs = Rf * ADC / (4095 - ADC)
# ============================================================

ADC_MAX = 4095.0

# Nominal resistor: 3.3 kOhm; measured value used for conversion: 3.4 kOhm.
R_FIXED_OHM = 3400.0
R_FIXED = np.array([R_FIXED_OHM] * N_CHANNELS, dtype=float)

VALID_ADC_MIN = 20

# For summary calculations, treat ADC >= 3900 as high/open-contact artifact.
# Raw ADC is still saved.
VALID_ADC_MAX_FOR_SUMMARY = 3900

# For raw trace calculation, allow higher ADC but avoid divide-by-zero blow-up.
VALID_ADC_MAX_FOR_TRACE = 4088

HIGH_ADC_WARN = 3600
HIGH_ADC_BAD = 3900


# ============================================================
# PROTOCOL SETTINGS
# ============================================================

OUTPUT_DIR = "posture_tests"

N_RUNS_DEFAULT = 3
START_RUN_DEFAULT = 1

EVENT_DURATION_S = 10.0
SUMMARY_LAST_S = 5.0

# This order is intended to make Figure-5 analog to Figure 2.
# Open is included as a posture class by comparing Open_Posture to InitialOpen.
POSTURE_DEFINITIONS = [
    ("Open", "Open hand. Relax."),
    ("Fist", "Make a fist. Hold."),
    ("IndexPoint", "Index point. Hold."),
    ("ThumbUp", "Thumb up. Hold."),
    ("Pinch", "Pinch thumb and index. Hold."),
]

POSTURE_DISPLAY = {
    "Open": "Open",
    "Fist": "Fist",
    "IndexPoint": "Index point",
    "ThumbUp": "Thumb-up",
    "Pinch": "Pinch",
}


# ============================================================
# LIVE PLOT SETTINGS
# ============================================================

SHOW_LIVE_PLOT = True
PLOT_HISTORY_SAMPLES = 700
PLOT_UPDATE_INTERVAL_S = 0.20
PLOT_FIGSIZE = (8.0, 5.0)

USE_FIXED_YLIM = True
TRACE_Y_LIMITS = (-0.45, 2.00)


# ============================================================
# VOICE SETTINGS
# ============================================================

VOICE_ENABLED = True
VOICE_RATE = 205
_voice_process = None


def speak(text: str) -> None:
    global _voice_process

    if not VOICE_ENABLED:
        return

    print(f"VOICE: {text}")

    if platform.system() == "Darwin":
        try:
            if _voice_process is not None and _voice_process.poll() is None:
                _voice_process.terminate()
            _voice_process = subprocess.Popen(
                ["say", "-r", str(VOICE_RATE), text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            print("Voice failed:", exc)
    else:
        print("\a")


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def sanitize_int_input(text: str, default_value: int) -> int:
    text = str(text).strip()
    if text == "":
        return default_value
    found = re.findall(r"\d+", text)
    if not found:
        raise ValueError("Please enter a number.")
    return int(found[0])


def parse_line(line: str) -> Optional[Tuple[int, List[int]]]:
    """Expected BLE line: t_ms,adc1,adc2,adc3,adc4,adc5,adc6,adc7"""
    line = line.strip()
    if not line:
        return None
    if line.startswith("t_ms"):
        return None

    parts = line.split(",")
    if len(parts) != N_CHANNELS + 1:
        return None

    try:
        t_ms = int(parts[0])
        adc_values = [int(x) for x in parts[1:]]
    except ValueError:
        return None

    return t_ms, adc_values


def adc_to_resistance(adc_values: List[int], valid_adc_max: int) -> np.ndarray:
    adc = np.array(adc_values, dtype=float)
    resistance = np.full(N_CHANNELS, np.nan, dtype=float)
    valid = (adc > VALID_ADC_MIN) & (adc < valid_adc_max)
    resistance[valid] = R_FIXED[valid] * adc[valid] / (ADC_MAX - adc[valid])
    return resistance


def calculate_normalized(resistance: np.ndarray, r0: Optional[np.ndarray]) -> np.ndarray:
    out = np.full(N_CHANNELS, np.nan, dtype=float)
    if r0 is None:
        return out
    valid = np.isfinite(resistance) & np.isfinite(r0) & (r0 != 0)
    out[valid] = (resistance[valid] - r0[valid]) / r0[valid]
    return out


def format_vector(values: np.ndarray, decimals: int = 2) -> str:
    parts = []
    for value in values:
        if np.isfinite(value):
            parts.append(f"{value:+.{decimals}f}")
        else:
            parts.append("nan")
    return "[" + ", ".join(parts) + "]"


def build_events() -> List[Dict[str, object]]:
    """Builds the full sequence for one run."""
    events: List[Dict[str, object]] = []

    events.append({
        "label": "InitialOpen",
        "posture": "Open",
        "phase": "initial_open",
        "duration_s": EVENT_DURATION_S,
        "instruction": "Open hand baseline. Keep relaxed.",
        "voice": "Open hand baseline. Keep relaxed.",
        "reference_for": "Open",
    })

    # Open posture class.
    events.append({
        "label": "Posture_Open",
        "posture": "Open",
        "phase": "posture",
        "duration_s": EVENT_DURATION_S,
        "instruction": "Open hand posture. Keep relaxed.",
        "voice": "Open hand. Hold.",
        "reference_label": "InitialOpen",
    })

    # Remaining posture classes, each preceded by an open reference.
    for posture, instruction in POSTURE_DEFINITIONS[1:]:
        open_label = f"Open_before_{posture}"
        target_label = f"Posture_{posture}"

        events.append({
            "label": open_label,
            "posture": "Open",
            "phase": "open_reference",
            "duration_s": EVENT_DURATION_S,
            "instruction": f"Open hand. Prepare for {POSTURE_DISPLAY[posture]}.",
            "voice": f"Open hand. Prepare for {POSTURE_DISPLAY[posture]}.",
            "reference_for": posture,
        })

        events.append({
            "label": target_label,
            "posture": posture,
            "phase": "posture",
            "duration_s": EVENT_DURATION_S,
            "instruction": instruction,
            "voice": instruction,
            "reference_label": open_label,
        })

    events.append({
        "label": "FinalOpen",
        "posture": "Open",
        "phase": "final_open",
        "duration_s": EVENT_DURATION_S,
        "instruction": "Final open hand. Relax.",
        "voice": "Final open hand. Relax.",
    })

    elapsed = 0.0
    for i, event in enumerate(events):
        event["event_index"] = i
        event["start_s"] = elapsed
        event["end_s"] = elapsed + float(event["duration_s"])
        elapsed = float(event["end_s"])

    return events


def get_current_event(events: List[Dict[str, object]], elapsed_s: float):
    for event in events:
        if float(event["start_s"]) <= elapsed_s < float(event["end_s"]):
            phase_elapsed_s = elapsed_s - float(event["start_s"])
            phase_remaining_s = float(event["end_s"]) - elapsed_s
            return event, phase_elapsed_s, phase_remaining_s
    return None, None, None


def rows_for_event(rows: List[Dict[str, object]], label: str, duration_s: float, last_s: float) -> List[Dict[str, object]]:
    cutoff = max(0.0, duration_s - last_s)
    return [
        row for row in rows
        if row["event_label"] == label and float(row["phase_elapsed_s"]) >= cutoff
    ]


def mean_drr_for_event(rows: List[Dict[str, object]], label: str, duration_s: float, r0: np.ndarray) -> np.ndarray:
    selected = rows_for_event(rows, label, duration_s, SUMMARY_LAST_S)
    if not selected:
        return np.full(N_CHANNELS, np.nan, dtype=float)
    values = [calculate_normalized(row["resistance_summary"], r0) for row in selected]
    return np.nanmean(np.array(values, dtype=float), axis=0)


def std_drr_for_event(rows: List[Dict[str, object]], label: str, duration_s: float, r0: np.ndarray) -> np.ndarray:
    selected = rows_for_event(rows, label, duration_s, SUMMARY_LAST_S)
    if not selected:
        return np.full(N_CHANNELS, np.nan, dtype=float)
    values = [calculate_normalized(row["resistance_summary"], r0) for row in selected]
    return np.nanstd(np.array(values, dtype=float), axis=0)


def high_adc_count_for_event(rows: List[Dict[str, object]], label: str, duration_s: float) -> int:
    selected = rows_for_event(rows, label, duration_s, SUMMARY_LAST_S)
    count = 0
    for row in selected:
        adc = np.array(row["adc_values"], dtype=float)
        count += int(np.any(adc >= HIGH_ADC_BAD))
    return count


# ============================================================
# BLE
# ============================================================

async def find_ble_device():
    print("Scanning for ESP32 BLE device...")
    devices = await BleakScanner.discover(timeout=8.0)

    selected = None
    for device in devices:
        name = device.name or "None"
        print(f"Found: {name}  {device.address}")
        if name in DEVICE_NAMES:
            selected = device
            break

    if selected is None:
        print("Could not find expected ESP32 BLE device.")
        print("Expected one of:")
        for name in DEVICE_NAMES:
            print("  ", name)
        return None

    print(f"Selected device: {selected.name}  {selected.address}")
    return selected


# ============================================================
# LIVE PLOT
# ============================================================

def setup_live_plot():
    if not SHOW_LIVE_PLOT:
        return None, None

    plt.ion()
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    try:
        fig.canvas.manager.set_window_title("Posture protocol live plot")
    except Exception:
        pass
    plt.show(block=False)
    return fig, ax


def update_live_plot(fig, ax, time_hist, norm_hist, resistance_hist, r0, event, phase_remaining_s):
    if fig is None or ax is None or len(time_hist) < 2:
        return

    ax.clear()
    x = np.array(time_hist, dtype=float)
    x = x - x[0]

    posture = str(event["posture"])
    instruction = str(event["instruction"])

    if r0 is None:
        data = np.array(resistance_hist, dtype=float)
        for i, label in enumerate(CHANNEL_LABELS):
            ax.plot(x, data[:, i], linewidth=1.2, label=label)
        ax.set_ylabel("Resistance (ohm)")
        ax.set_title(f"{instruction}\n{phase_remaining_s:.1f} s left | collecting R0")
    else:
        data = np.array(norm_hist, dtype=float)
        for i, label in enumerate(CHANNEL_LABELS):
            ax.plot(x, data[:, i], linewidth=1.4, label=label)
        ax.axhline(0, linewidth=1.0)
        ax.set_ylabel("Delta R / R0")
        ax.set_title(f"{instruction}\n{phase_remaining_s:.1f} s left | posture: {POSTURE_DISPLAY.get(posture, posture)}")
        if USE_FIXED_YLIM:
            ax.set_ylim(*TRACE_Y_LIMITS)

    ax.set_xlabel("Time in current view (s)")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()

    try:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)
    except Exception:
        pass


# ============================================================
# SAVE FUNCTIONS
# ============================================================

def full_csv_header() -> List[str]:
    header = [
        "wall_time_iso",
        "esp32_t_ms",
        "run_id",
        "run_elapsed_s",
        "event_index",
        "event_label",
        "posture",
        "phase",
        "phase_elapsed_s",
        "instruction",
        "use_for_summary",
        "any_adc_warn_ge_3600",
        "any_adc_bad_ge_3900",
    ]
    for ch in CHANNELS:
        header.append(f"{ch}_raw_ADC")
    for ch in CHANNELS:
        header.append(f"{ch}_R_ohm")
    for ch in CHANNELS:
        header.append(f"{ch}_dR_over_R0")
    return header


def save_full_csv(filename: str, rows: List[Dict[str, object]], r0: np.ndarray) -> None:
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(full_csv_header())

        for row in rows:
            norm = calculate_normalized(row["resistance_trace"], r0)
            adc = np.array(row["adc_values"], dtype=float)
            any_warn = int(np.any(adc >= HIGH_ADC_WARN))
            any_bad = int(np.any(adc >= HIGH_ADC_BAD))

            line = [
                row["wall_time_iso"],
                row["esp32_t_ms"],
                row["run_id"],
                f"{float(row['run_elapsed_s']):.4f}",
                row["event_index"],
                row["event_label"],
                row["posture"],
                row["phase"],
                f"{float(row['phase_elapsed_s']):.4f}",
                row["instruction"],
                row["use_for_summary"],
                any_warn,
                any_bad,
            ]

            line.extend(row["adc_values"])

            for value in row["resistance_trace"]:
                line.append(f"{value:.6f}" if np.isfinite(value) else "nan")

            for value in norm:
                line.append(f"{value:.6f}" if np.isfinite(value) else "nan")

            writer.writerow(line)


def save_local_summary_csv(filename: str, rows: List[Dict[str, object]], events: List[Dict[str, object]], r0: np.ndarray) -> None:
    event_by_label = {str(event["label"]): event for event in events}

    header = [
        "posture_label",
        "posture_display",
        "target_event_label",
        "reference_event_label",
        "used_last_seconds",
        "target_high_adc_ge_3900_samples",
        "reference_high_adc_ge_3900_samples",
    ]

    for ch in CHANNELS:
        header.append(f"{ch}_posture_mean_dR_over_R0")
    for ch in CHANNELS:
        header.append(f"{ch}_reference_mean_dR_over_R0")
    for ch in CHANNELS:
        header.append(f"{ch}_local_corrected_dR_over_R0")
    for ch in CHANNELS:
        header.append(f"{ch}_posture_sd_dR_over_R0")

    lines = []

    target_specs = [
        ("Open", "Posture_Open", "InitialOpen"),
        ("Fist", "Posture_Fist", "Open_before_Fist"),
        ("IndexPoint", "Posture_IndexPoint", "Open_before_IndexPoint"),
        ("ThumbUp", "Posture_ThumbUp", "Open_before_ThumbUp"),
        ("Pinch", "Posture_Pinch", "Open_before_Pinch"),
    ]

    for posture, target_label, ref_label in target_specs:
        target_event = event_by_label[target_label]
        ref_event = event_by_label[ref_label]

        target_mean = mean_drr_for_event(rows, target_label, float(target_event["duration_s"]), r0)
        ref_mean = mean_drr_for_event(rows, ref_label, float(ref_event["duration_s"]), r0)
        target_sd = std_drr_for_event(rows, target_label, float(target_event["duration_s"]), r0)
        local = target_mean - ref_mean

        target_high = high_adc_count_for_event(rows, target_label, float(target_event["duration_s"]))
        ref_high = high_adc_count_for_event(rows, ref_label, float(ref_event["duration_s"]))

        line = [
            posture,
            POSTURE_DISPLAY[posture],
            target_label,
            ref_label,
            SUMMARY_LAST_S,
            target_high,
            ref_high,
        ]

        for value in target_mean:
            line.append(f"{value:.6f}" if np.isfinite(value) else "nan")
        for value in ref_mean:
            line.append(f"{value:.6f}" if np.isfinite(value) else "nan")
        for value in local:
            line.append(f"{value:.6f}" if np.isfinite(value) else "nan")
        for value in target_sd:
            line.append(f"{value:.6f}" if np.isfinite(value) else "nan")

        lines.append(line)

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(lines)


def compute_r0(rows: List[Dict[str, object]]) -> Optional[np.ndarray]:
    selected = rows_for_event(rows, "InitialOpen", EVENT_DURATION_S, SUMMARY_LAST_S)
    if not selected:
        return None
    arr = np.array([row["resistance_summary"] for row in selected], dtype=float)
    return np.nanmedian(arr, axis=0)


def print_r0(r0: np.ndarray) -> None:
    print("\nBaseline R0 from final 5 s of InitialOpen:")
    for label, value in zip(CHANNEL_LABELS, r0):
        if np.isfinite(value):
            print(f"  {label:18s}: {value:10.2f} ohm")
        else:
            print(f"  {label:18s}: INVALID")
    print()


# ============================================================
# COLLECTION
# ============================================================

async def collect_one_run(run_id: int, client: BleakClient, pending_samples: deque) -> Tuple[List[Dict[str, object]], np.ndarray]:
    events = build_events()
    total_duration_s = float(events[-1]["end_s"])

    print("\n" + "=" * 80)
    print(f"STARTING POSTURE PROTOCOL RUN {run_id}")
    print("=" * 80)
    print(f"Total duration: {total_duration_s:.1f} s")
    print("Posture sequence: Open, Fist, Index point, Thumb-up, Pinch")
    print("Each step: 10 s. Summary uses final 5 s.")
    print("\nKeep the ESP32 and wire bundle fixed. Watch the live plot for large peaks.")
    print()

    speak(f"Starting posture run {run_id}. Open hand baseline.")

    rows: List[Dict[str, object]] = []
    r0: Optional[np.ndarray] = None

    time_hist = deque(maxlen=PLOT_HISTORY_SAMPLES)
    resistance_hist = deque(maxlen=PLOT_HISTORY_SAMPLES)
    norm_hist = deque(maxlen=PLOT_HISTORY_SAMPLES)

    fig, ax = setup_live_plot()

    last_event_index = None
    last_print_time = 0.0
    last_plot_time = 0.0

    run_start = time.time()

    while True:
        await asyncio.sleep(0.01)

        now = time.time()
        elapsed_s = now - run_start

        if elapsed_s >= total_duration_s:
            break

        event, phase_elapsed_s, phase_remaining_s = get_current_event(events, elapsed_s)
        if event is None:
            continue

        if event["event_index"] != last_event_index:
            last_event_index = event["event_index"]
            print("\n" + "-" * 80)
            print(f"Run {run_id} | Event {int(event['event_index']) + 1}/{len(events)}")
            print(f"Instruction: {event['instruction']}")
            print(f"Label:       {event['label']}")
            print("-" * 80)
            speak(str(event["voice"]))

        if now - last_print_time >= 1.0:
            last_print_time = now
            print(f"Run {run_id} | {event['instruction']} | {phase_remaining_s:4.1f} s left | total {elapsed_s:6.1f}/{total_duration_s:.1f} s")

        while pending_samples:
            esp32_t_ms, adc_values = pending_samples.popleft()
            sample_time = time.time()
            sample_elapsed_s = sample_time - run_start

            sample_event, sample_phase_elapsed_s, _ = get_current_event(events, sample_elapsed_s)
            if sample_event is None:
                continue

            adc_array = np.array(adc_values, dtype=float)
            any_bad = bool(np.any(adc_array >= HIGH_ADC_BAD))

            resistance_trace = adc_to_resistance(adc_values, VALID_ADC_MAX_FOR_TRACE)
            resistance_summary = adc_to_resistance(adc_values, VALID_ADC_MAX_FOR_SUMMARY)

            row = {
                "wall_time_iso": datetime.now().isoformat(timespec="milliseconds"),
                "esp32_t_ms": esp32_t_ms,
                "run_id": run_id,
                "run_elapsed_s": sample_elapsed_s,
                "event_index": sample_event["event_index"],
                "event_label": sample_event["label"],
                "posture": sample_event["posture"],
                "phase": sample_event["phase"],
                "phase_elapsed_s": sample_phase_elapsed_s,
                "instruction": sample_event["instruction"],
                "use_for_summary": int(sample_phase_elapsed_s >= (EVENT_DURATION_S - SUMMARY_LAST_S)),
                "adc_values": adc_values,
                "resistance_trace": resistance_trace,
                "resistance_summary": resistance_summary,
            }
            rows.append(row)

            # Calculate R0 once initial open baseline is complete.
            if r0 is None and sample_elapsed_s >= EVENT_DURATION_S:
                r0_tmp = compute_r0(rows)
                if r0_tmp is not None:
                    r0 = r0_tmp
                    print_r0(r0)
                    speak("Baseline ready.")

            if r0 is not None:
                norm = calculate_normalized(resistance_trace, r0)
            else:
                norm = np.full(N_CHANNELS, np.nan, dtype=float)

            time_hist.append(sample_elapsed_s)
            resistance_hist.append(resistance_trace)
            norm_hist.append(norm)

            if any_bad:
                bad_channels = [name for name, adc in zip(CHANNEL_LABELS, adc_values) if adc >= HIGH_ADC_BAD]
                print("WARNING high ADC >= 3900:", ", ".join(bad_channels))

        if SHOW_LIVE_PLOT and len(time_hist) > 3 and now - last_plot_time >= PLOT_UPDATE_INTERVAL_S:
            last_plot_time = now
            update_live_plot(fig, ax, time_hist, norm_hist, resistance_hist, r0, event, phase_remaining_s)

    if r0 is None:
        r0_tmp = compute_r0(rows)
        if r0_tmp is None:
            raise RuntimeError("Could not compute R0 from InitialOpen.")
        r0 = r0_tmp

    speak(f"Run {run_id} finished.")
    return rows, r0


async def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nPosture protocol collector for Figure-5-style analysis")
    print("This records raw traces and local summaries for:")
    print("  Open, Fist, IndexPoint, ThumbUp, Pinch")
    print("\nOutput folder:")
    print(" ", os.path.abspath(OUTPUT_DIR))
    print()

    n_runs_text = input(f"How many runs do you want to collect? [{N_RUNS_DEFAULT}]: ")
    start_run_text = input(f"Start run number? [{START_RUN_DEFAULT}]: ")

    n_runs = sanitize_int_input(n_runs_text, N_RUNS_DEFAULT)
    start_run = sanitize_int_input(start_run_text, START_RUN_DEFAULT)

    device = await find_ble_device()
    if device is None:
        return

    line_buffer = ""
    pending_samples: deque = deque()

    def notification_handler(sender, data):
        nonlocal line_buffer
        chunk = data.decode("utf-8", errors="ignore")
        line_buffer += chunk
        while "\n" in line_buffer:
            line, line_buffer = line_buffer.split("\n", 1)
            parsed = parse_line(line)
            if parsed is not None:
                pending_samples.append(parsed)

    print("\nConnecting...")
    async with BleakClient(device) as client:
        print("Connected.")
        await client.start_notify(TX_CHARACTERISTIC, notification_handler)

        print("\nPut the glove on. Keep hand open. Press ENTER to start.")
        input()

        for i in range(n_runs):
            run_id = start_run + i
            pending_samples.clear()

            rows, r0 = await collect_one_run(run_id, client, pending_samples)

            protocol_file = os.path.join(OUTPUT_DIR, f"posture_run{run_id}_protocol.csv")
            local_file = os.path.join(OUTPUT_DIR, f"posture_run{run_id}_local.csv")

            save_full_csv(protocol_file, rows, r0)
            save_local_summary_csv(local_file, rows, build_events(), r0)

            print("\nSaved:")
            print(" ", protocol_file)
            print(" ", local_file)

            if i < n_runs - 1:
                print("\nRest, keep hand open, and check the wires if needed.")
                speak("Take a short rest.")
                input(f"Press ENTER when ready to start run {run_id + 1}...")

        await client.stop_notify(TX_CHARACTERISTIC)

    print("\nAll posture runs complete.")
    speak("All posture runs complete.")
    print("\nUpload these files later for MATLAB Figure 5:")
    print("  posture_tests/posture_run1_protocol.csv")
    print("  posture_tests/posture_run1_local.csv")
    print("  ...")


if __name__ == "__main__":
    asyncio.run(main())
