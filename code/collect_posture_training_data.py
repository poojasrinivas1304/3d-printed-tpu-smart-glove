#!/usr/bin/env python3
"""
Collect stepwise posture-recognition training data for the TPU glove.

Posture classes:
    Open, Fist, IndexPoint, ThumbUp, Pinch

Each accepted trial:
    1. Open-hand reference, 4 s
       - final 2 s are used as local reference
    2. Target posture hold, 5 s
       - final 3 s are used for feature extraction

Outputs:
    ai_posture_data/posture_train_samples_<timestamp>.csv
    ai_posture_data/posture_train_trials_<timestamp>.csv

Run:
    python3 collect_posture_training_data.py
"""

from __future__ import annotations

import asyncio
import csv
import os
import platform
import random
import re
import subprocess
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from bleak import BleakClient, BleakScanner

# ---------------- BLE settings ----------------
DEVICE_NAMES = [
    "Glove_BLE_7_RAW",
    "TShirt_BLE_VP_RAW",
    "TShirt_BLE_10_RAW",
]
TX_CHARACTERISTIC = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

# ---------------- Channel order ----------------
# Arduino/ESP32 order: GPIO36, GPIO39, GPIO34, GPIO35, GPIO32, GPIO33, GPIO25
CHANNELS = [
    "Thumb",
    "Index",
    "Middle",
    "Ring",
    "Little",
    "ReverseHorizontal",
    "ReverseThumb",
]
N_CHANNELS = len(CHANNELS)

# ---------------- ADC/resistance settings ----------------
ADC_MAX = 4095.0
R_FIXED = np.array([3400.0] * N_CHANNELS, dtype=float)  # nominal 3.3 kOhm, measured around 3.4 kOhm
VALID_ADC_MIN = 20
VALID_ADC_MAX = 3900  # samples at/above this are treated as invalid for resistance conversion
HIGH_ADC_THRESHOLD = 3900

# ---------------- Trial timing ----------------
OPEN_REFERENCE_SECONDS = 4.0
OPEN_REFERENCE_USE_LAST_SECONDS = 2.0
POSTURE_SECONDS = 5.0
POSTURE_USE_LAST_SECONDS = 3.0

MAX_OPEN_HIGH_ADC_SAMPLES = 5
MAX_POSTURE_HIGH_ADC_SAMPLES = 20

# ---------------- Output ----------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "ai_posture_data")

# ---------------- Voice ----------------
VOICE_ENABLED = True
VOICE_RATE = 185
_voice_process = None

POSTURES: List[Tuple[str, str]] = [
    ("Open", "Keep your hand open and relaxed. Hold still."),
    ("Fist", "Make a fist. Hold it."),
    ("IndexPoint", "Make an index pointing posture. Keep index extended and fold the other fingers."),
    ("ThumbUp", "Make a thumb up posture. Hold it."),
    ("Pinch", "Make a pinch posture with thumb and index. Hold it."),
]


def speak(text: str) -> None:
    global _voice_process
    print(f"VOICE: {text}")
    if not VOICE_ENABLED:
        return
    if platform.system() == "Darwin":
        try:
            if _voice_process is not None and _voice_process.poll() is None:
                _voice_process.terminate()
            _voice_process = subprocess.Popen(
                ["say", "-r", str(VOICE_RATE), text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
    else:
        print("\a")


def parse_line(line: str) -> Optional[Tuple[int, List[int]]]:
    line = line.strip()
    if not line or line.startswith("t_ms"):
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


def adc_to_resistance(adc_values: List[int]) -> np.ndarray:
    adc = np.array(adc_values, dtype=float)
    resistance = np.full(N_CHANNELS, np.nan, dtype=float)
    valid = (adc > VALID_ADC_MIN) & (adc < VALID_ADC_MAX)
    resistance[valid] = R_FIXED[valid] * adc[valid] / (ADC_MAX - adc[valid])
    return resistance


def local_normalized(resistance: np.ndarray, open_ref_r: np.ndarray) -> np.ndarray:
    out = np.full(N_CHANNELS, np.nan, dtype=float)
    valid = np.isfinite(resistance) & np.isfinite(open_ref_r) & (open_ref_r != 0)
    out[valid] = (resistance[valid] - open_ref_r[valid]) / open_ref_r[valid]
    return out


def high_adc_count(rows: List[Dict]) -> int:
    count = 0
    for row in rows:
        adc = np.array(row["adc_values"], dtype=float)
        if np.any(adc >= HIGH_ADC_THRESHOLD):
            count += 1
    return count


def per_channel_high_counts(rows: List[Dict]) -> Dict[str, int]:
    counts = {ch: 0 for ch in CHANNELS}
    for row in rows:
        for ch, adc in zip(CHANNELS, row["adc_values"]):
            if adc >= HIGH_ADC_THRESHOLD:
                counts[ch] += 1
    return counts


def extract_features(local_values: np.ndarray, times: np.ndarray) -> Dict[str, float]:
    features: Dict[str, float] = {}
    if local_values.size == 0:
        local_values = np.empty((0, N_CHANNELS), dtype=float)
    for i, ch in enumerate(CHANNELS):
        y = local_values[:, i] if local_values.shape[0] else np.array([], dtype=float)
        valid = np.isfinite(y)
        yv = y[valid]
        tv = times[valid] if len(times) == len(y) else np.arange(len(yv), dtype=float)
        prefix = f"feat_{ch}_"
        if len(yv) == 0:
            features[prefix + "mean"] = np.nan
            features[prefix + "median"] = np.nan
            features[prefix + "std"] = np.nan
            features[prefix + "min"] = np.nan
            features[prefix + "max"] = np.nan
            features[prefix + "range"] = np.nan
            features[prefix + "last"] = np.nan
            features[prefix + "slope"] = np.nan
            continue
        features[prefix + "mean"] = float(np.mean(yv))
        features[prefix + "median"] = float(np.median(yv))
        features[prefix + "std"] = float(np.std(yv))
        features[prefix + "min"] = float(np.min(yv))
        features[prefix + "max"] = float(np.max(yv))
        features[prefix + "range"] = float(np.max(yv) - np.min(yv))
        features[prefix + "last"] = float(yv[-1])
        if len(yv) >= 3 and np.ptp(tv) > 0:
            features[prefix + "slope"] = float(np.polyfit(tv - tv[0], yv, 1)[0])
        else:
            features[prefix + "slope"] = 0.0
    return features


def sample_header() -> List[str]:
    header = [
        "wall_time_iso",
        "esp32_t_ms",
        "trial_id",
        "accepted_index",
        "target_label",
        "phase",
        "phase_elapsed_s",
        "sample_elapsed_s",
        "status",
    ]
    header += [f"{ch}_raw_ADC" for ch in CHANNELS]
    header += [f"{ch}_R_ohm" for ch in CHANNELS]
    header += [f"{ch}_local_dR_over_Ropen" for ch in CHANNELS]
    return header


def write_sample_row(writer: csv.writer, row: Dict, open_ref_r: Optional[np.ndarray], status: str) -> None:
    resistance = row["resistance"]
    if open_ref_r is None:
        local = np.full(N_CHANNELS, np.nan, dtype=float)
    else:
        local = local_normalized(resistance, open_ref_r)
    line = [
        row["wall_time_iso"],
        row["esp32_t_ms"],
        row["trial_id"],
        row["accepted_index"],
        row["target_label"],
        row["phase"],
        f"{row['phase_elapsed_s']:.4f}",
        f"{row['sample_elapsed_s']:.4f}",
        status,
    ]
    line += list(row["adc_values"])
    line += [f"{v:.6f}" if np.isfinite(v) else "nan" for v in resistance]
    line += [f"{v:.6f}" if np.isfinite(v) else "nan" for v in local]
    writer.writerow(line)


def trial_header(feature_names: List[str]) -> List[str]:
    header = [
        "trial_id",
        "accepted_index",
        "target_label",
        "status",
        "open_high_adc_total",
        "posture_high_adc_total",
        "open_ref_samples_used",
        "posture_samples_used",
    ]
    for ch in CHANNELS:
        header.append(f"open_high_adc_{ch}")
    for ch in CHANNELS:
        header.append(f"posture_high_adc_{ch}")
    header += feature_names
    return header


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
        print("Expected one of:", ", ".join(DEVICE_NAMES))
        return None
    print(f"Selected device: {selected.name}  {selected.address}")
    return selected


async def collect_phase(
    pending_samples: deque,
    duration_s: float,
    phase: str,
    trial_id: int,
    accepted_index: int,
    target_label: str,
    open_ref_r: Optional[np.ndarray],
    sample_writer: csv.writer,
    status: str,
) -> List[Dict]:
    rows: List[Dict] = []
    pending_samples.clear()
    start = time.time()
    last_print = 0.0
    while True:
        await asyncio.sleep(0.02)
        now = time.time()
        elapsed = now - start
        if elapsed >= duration_s:
            break
        if now - last_print >= 1.0:
            last_print = now
            print(f"{phase} | {duration_s - elapsed:.1f} s remaining")
        while pending_samples:
            esp32_t_ms, adc_values = pending_samples.popleft()
            sample_time = time.time()
            phase_elapsed = sample_time - start
            if phase_elapsed > duration_s:
                continue
            resistance = adc_to_resistance(adc_values)
            row = {
                "wall_time_iso": datetime.now().isoformat(timespec="milliseconds"),
                "esp32_t_ms": esp32_t_ms,
                "trial_id": trial_id,
                "accepted_index": accepted_index,
                "target_label": target_label,
                "phase": phase,
                "phase_elapsed_s": phase_elapsed,
                "sample_elapsed_s": sample_time,
                "adc_values": adc_values,
                "resistance": resistance,
            }
            rows.append(row)
            write_sample_row(sample_writer, row, open_ref_r, status=status)
    return rows


async def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Posture training data collector")
    print("Postures:", ", ".join([p[0] for p in POSTURES]))
    trials_text = input("Trials per posture? [10]: ").strip()
    trials_per_posture = int(trials_text) if trials_text else 10
    shuffle_text = input("Shuffle posture order? [y]: ").strip().lower()
    shuffle_order = shuffle_text != "n"

    planned: List[Tuple[str, str]] = []
    for label, instruction in POSTURES:
        planned += [(label, instruction)] * trials_per_posture
    if shuffle_order:
        random.shuffle(planned)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_file = os.path.join(OUTPUT_DIR, f"posture_train_samples_{timestamp}.csv")
    trial_file = os.path.join(OUTPUT_DIR, f"posture_train_trials_{timestamp}.csv")

    # Build feature header by creating dummy feature names once.
    dummy_features = extract_features(np.empty((0, N_CHANNELS), dtype=float), np.array([], dtype=float))
    feature_names = list(dummy_features.keys())

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

    print("\nKeep the ESP32 and wire bundle fixed.")
    print("If a trial is rejected due to high ADC, check the connection and repeat it.")
    input("Put on the glove, keep hand open, and press ENTER to start...")

    with open(sample_file, "w", newline="") as sf, open(trial_file, "w", newline="") as tf:
        sample_writer = csv.writer(sf)
        trial_writer = csv.writer(tf)
        sample_writer.writerow(sample_header())
        trial_writer.writerow(trial_header(feature_names))

        async with BleakClient(device) as client:
            print("Connected.")
            await client.start_notify(TX_CHARACTERISTIC, notification_handler)

            accepted_count = 0
            trial_id = 0
            for target_label, instruction in planned:
                accepted = False
                while not accepted:
                    trial_id += 1
                    accepted_index = accepted_count + 1
                    print("\n" + "=" * 80)
                    print(f"Training trial {accepted_index}/{len(planned)} | Target posture: {target_label}")
                    print("=" * 80)

                    speak("Open hand. Hold still for reference.")
                    open_rows = await collect_phase(
                        pending_samples, OPEN_REFERENCE_SECONDS, "open_reference",
                        trial_id, accepted_index, target_label, None, sample_writer, status="candidate",
                    )
                    open_high = high_adc_count(open_rows)
                    open_counts = per_channel_high_counts(open_rows)
                    usable_open = [
                        row["resistance"] for row in open_rows
                        if row["phase_elapsed_s"] >= OPEN_REFERENCE_SECONDS - OPEN_REFERENCE_USE_LAST_SECONDS
                    ]
                    if len(usable_open) == 0:
                        print("No open reference samples. Repeating trial.")
                        speak("Bad reference. Repeat.")
                        continue
                    open_ref_r = np.nanmedian(np.array(usable_open, dtype=float), axis=0)
                    if open_high > MAX_OPEN_HIGH_ADC_SAMPLES:
                        print(f"Rejected: open reference high-ADC samples = {open_high}")
                        print(open_counts)
                        speak("Bad open reference. Repeat.")
                        # Mark previous open rows as rejected in trial table only; samples remain candidate.
                        continue

                    speak(instruction)
                    posture_rows = await collect_phase(
                        pending_samples, POSTURE_SECONDS, "posture_hold",
                        trial_id, accepted_index, target_label, open_ref_r, sample_writer, status="accepted",
                    )
                    posture_high = high_adc_count(posture_rows)
                    posture_counts = per_channel_high_counts(posture_rows)
                    posture_local = []
                    posture_times = []
                    for row in posture_rows:
                        if row["phase_elapsed_s"] >= POSTURE_SECONDS - POSTURE_USE_LAST_SECONDS:
                            posture_local.append(local_normalized(row["resistance"], open_ref_r))
                            posture_times.append(row["phase_elapsed_s"])
                    if len(posture_local) == 0:
                        print("No posture feature samples. Repeating trial.")
                        speak("Bad posture. Repeat.")
                        continue
                    if posture_high > MAX_POSTURE_HIGH_ADC_SAMPLES:
                        print(f"Rejected: posture high-ADC samples = {posture_high}")
                        print(posture_counts)
                        speak("Bad posture signal. Repeat.")
                        continue

                    local_arr = np.array(posture_local, dtype=float)
                    time_arr = np.array(posture_times, dtype=float)
                    feats = extract_features(local_arr, time_arr)

                    line = [
                        trial_id,
                        accepted_index,
                        target_label,
                        "accepted",
                        open_high,
                        posture_high,
                        len(usable_open),
                        len(posture_local),
                    ]
                    line += [open_counts[ch] for ch in CHANNELS]
                    line += [posture_counts[ch] for ch in CHANNELS]
                    line += [feats[name] for name in feature_names]
                    trial_writer.writerow(line)
                    tf.flush()
                    sf.flush()

                    accepted_count += 1
                    accepted = True
                    print(f"Accepted trial {accepted_count}/{len(planned)} for {target_label}")

            await client.stop_notify(TX_CHARACTERISTIC)

    print("\nAll posture training trials complete.")
    print("Saved:")
    print(" ", sample_file)
    print(" ", trial_file)
    speak("All posture training trials complete.")


if __name__ == "__main__":
    asyncio.run(main())
