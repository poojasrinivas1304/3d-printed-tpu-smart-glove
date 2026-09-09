#!/usr/bin/env python3
"""
Run a blind stepwise posture-recognition demo using posture_ai_model.joblib.

Postures:
    Open, Fist, IndexPoint, ThumbUp, Pinch

The script does not tell you which posture to make. It asks you to make one
posture and hold it, predicts once from the final stable window, and then asks
you to enter the true posture for scoring.

Outputs:
    posture_blind_samples_<timestamp>.csv
    posture_blind_summary_<timestamp>.csv

Run:
    python3 run_posture_blind_demo.py
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

import joblib
import numpy as np
import pandas as pd
from bleak import BleakClient, BleakScanner

# ---------------- BLE settings ----------------
DEVICE_NAMES = ["Glove_BLE_7_RAW", "TShirt_BLE_VP_RAW", "TShirt_BLE_10_RAW"]
TX_CHARACTERISTIC = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

CHANNELS = ["Thumb", "Index", "Middle", "Ring", "Little", "ReverseHorizontal", "ReverseThumb"]
N_CHANNELS = len(CHANNELS)

ADC_MAX = 4095.0
R_FIXED = np.array([3400.0] * N_CHANNELS, dtype=float)
VALID_ADC_MIN = 20
VALID_ADC_MAX = 3900
HIGH_ADC_THRESHOLD = 3900

OPEN_REFERENCE_SECONDS = 4.0
OPEN_REFERENCE_USE_LAST_SECONDS = 2.0
POSTURE_SECONDS = 5.0
POSTURE_USE_LAST_SECONDS = 3.0

MAX_OPEN_HIGH_ADC_SAMPLES = 5
MAX_POSTURE_HIGH_ADC_SAMPLES = 20

MIN_CONFIDENCE = 0.35
MIN_MARGIN = 0.08

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(SCRIPT_DIR, "posture_ai_model.joblib")

POSTURES = ["Open", "Fist", "IndexPoint", "ThumbUp", "Pinch"]
ALIASES = {
    "open": "Open",
    "o": "Open",
    "fist": "Fist",
    "f": "Fist",
    "index": "IndexPoint",
    "indexpoint": "IndexPoint",
    "point": "IndexPoint",
    "i": "IndexPoint",
    "thumbup": "ThumbUp",
    "thumb": "ThumbUp",
    "t": "ThumbUp",
    "pinch": "Pinch",
    "ok": "Pinch",
    "p": "Pinch",
}

VOICE_ENABLED = True
VOICE_RATE = 185
_voice_process = None


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


def parse_true_label(text: str) -> Optional[str]:
    key = re.sub(r"[^a-zA-Z]", "", text).lower()
    return ALIASES.get(key)


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
            for name in ["mean", "median", "std", "min", "max", "range", "last", "slope"]:
                features[prefix + name] = np.nan
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
    h = ["wall_time_iso", "esp32_t_ms", "trial_id", "phase", "phase_elapsed_s", "true_label", "final_prediction"]
    h += [f"{ch}_raw_ADC" for ch in CHANNELS]
    h += [f"{ch}_R_ohm" for ch in CHANNELS]
    h += [f"{ch}_local_dR_over_Ropen" for ch in CHANNELS]
    return h


def write_sample_row(writer: csv.writer, row: Dict, open_ref_r: Optional[np.ndarray], true_label: str, final_prediction: str) -> None:
    resistance = row["resistance"]
    if open_ref_r is None:
        local = np.full(N_CHANNELS, np.nan, dtype=float)
    else:
        local = local_normalized(resistance, open_ref_r)
    line = [
        row["wall_time_iso"], row["esp32_t_ms"], row["trial_id"], row["phase"],
        f"{row['phase_elapsed_s']:.4f}", true_label, final_prediction,
    ]
    line += list(row["adc_values"])
    line += [f"{v:.6f}" if np.isfinite(v) else "nan" for v in resistance]
    line += [f"{v:.6f}" if np.isfinite(v) else "nan" for v in local]
    writer.writerow(line)


def summary_header(classes: List[str]) -> List[str]:
    h = [
        "trial_id", "true_label", "top_label", "final_prediction", "is_correct_accepted",
        "is_correct_forced", "confidence", "margin", "open_high_adc_total", "posture_high_adc_total",
    ]
    h += [f"prob_{c}" for c in classes]
    for ch in CHANNELS:
        h.append(f"local_mean_{ch}")
    return h


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
        return None
    print(f"Selected device: {selected.name}  {selected.address}")
    return selected


async def collect_phase(pending_samples: deque, duration_s: float, phase: str, trial_id: int) -> List[Dict]:
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
            rows.append({
                "wall_time_iso": datetime.now().isoformat(timespec="milliseconds"),
                "esp32_t_ms": esp32_t_ms,
                "trial_id": trial_id,
                "phase": phase,
                "phase_elapsed_s": phase_elapsed,
                "adc_values": adc_values,
                "resistance": resistance,
            })
    return rows


async def main() -> None:
    if not os.path.isfile(MODEL_FILE):
        raise FileNotFoundError(f"Missing model file: {MODEL_FILE}. Run train_posture_model.py first.")
    payload = joblib.load(MODEL_FILE)
    model = payload["pipeline"]
    feature_names = payload["feature_names"]
    classes = list(payload.get("classes", POSTURES))

    n_text = input("Number of blind posture trials? [10]: ").strip()
    n_trials = int(n_text) if n_text else 10

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_file = os.path.join(SCRIPT_DIR, f"posture_blind_samples_{timestamp}.csv")
    summary_file = os.path.join(SCRIPT_DIR, f"posture_blind_summary_{timestamp}.csv")

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

    print("\nBlind posture demo.")
    print("Postures:", ", ".join(classes))
    print("The software will not tell you which posture to make.")
    input("Put on glove, keep hand open, and press ENTER to begin...")

    with open(sample_file, "w", newline="") as sf, open(summary_file, "w", newline="") as tf:
        sample_writer = csv.writer(sf)
        summary_writer = csv.writer(tf)
        sample_writer.writerow(sample_header())
        summary_writer.writerow(summary_header(classes))

        async with BleakClient(device) as client:
            print("Connected.")
            await client.start_notify(TX_CHARACTERISTIC, notification_handler)

            for trial_id in range(1, n_trials + 1):
                print("\n" + "=" * 80)
                print(f"Blind trial {trial_id}/{n_trials}")
                print("=" * 80)

                # Open reference loop until clean.
                while True:
                    speak("Open hand. Hold still for reference.")
                    open_rows = await collect_phase(pending_samples, OPEN_REFERENCE_SECONDS, "open_reference", trial_id)
                    open_high = high_adc_count(open_rows)
                    usable_open = [
                        row["resistance"] for row in open_rows
                        if row["phase_elapsed_s"] >= OPEN_REFERENCE_SECONDS - OPEN_REFERENCE_USE_LAST_SECONDS
                    ]
                    if len(usable_open) == 0:
                        print("No open reference samples. Repeat reference.")
                        continue
                    if open_high > MAX_OPEN_HIGH_ADC_SAMPLES:
                        print(f"Bad open reference: {open_high} high-ADC samples. Repeat open reference.")
                        print(per_channel_high_counts(open_rows))
                        speak("Bad open reference. Repeat.")
                        continue
                    open_ref_r = np.nanmedian(np.array(usable_open, dtype=float), axis=0)
                    break

                # Fold one unknown posture.
                speak("Make one posture and hold.")
                print("Make one of these postures and hold:", ", ".join(classes))
                posture_rows = await collect_phase(pending_samples, POSTURE_SECONDS, "posture_hold", trial_id)
                posture_high = high_adc_count(posture_rows)
                if posture_high > MAX_POSTURE_HIGH_ADC_SAMPLES:
                    print(f"Warning: posture had {posture_high} high-ADC samples.")
                    print(per_channel_high_counts(posture_rows))

                local_values = []
                times = []
                for row in posture_rows:
                    if row["phase_elapsed_s"] >= POSTURE_SECONDS - POSTURE_USE_LAST_SECONDS:
                        local_values.append(local_normalized(row["resistance"], open_ref_r))
                        times.append(row["phase_elapsed_s"])
                local_arr = np.array(local_values, dtype=float)
                time_arr = np.array(times, dtype=float)
                feats = extract_features(local_arr, time_arr)
                X = pd.DataFrame([[feats.get(name, np.nan) for name in feature_names]], columns=feature_names)
                probs = model.predict_proba(X)[0]
                class_order = list(model.classes_)
                prob_map = {c: float(p) for c, p in zip(class_order, probs)}
                sorted_probs = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)
                top_label, confidence = sorted_probs[0]
                second_prob = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
                margin = confidence - second_prob
                if confidence >= MIN_CONFIDENCE and margin >= MIN_MARGIN:
                    final_prediction = top_label
                else:
                    final_prediction = "Uncertain"

                print("\nPrediction result")
                print(f"  Top label:        {top_label}")
                print(f"  Final prediction: {final_prediction}")
                print(f"  Confidence:       {confidence:.3f}")
                print(f"  Margin:           {margin:.3f}")
                speak(f"Prediction {final_prediction}")

                # Ask true label after prediction.
                true_label = None
                while true_label is None:
                    text = input("Enter true posture [Open/Fist/IndexPoint/ThumbUp/Pinch]: ").strip()
                    true_label = parse_true_label(text)
                    if true_label is None:
                        print("Could not understand. Try: Open, Fist, IndexPoint, ThumbUp, Pinch.")

                accepted_correct = int(final_prediction == true_label)
                forced_correct = int(top_label == true_label)
                local_mean = np.nanmean(local_arr, axis=0) if len(local_arr) else np.full(N_CHANNELS, np.nan)

                # Write samples now with known true/final labels.
                for row in open_rows:
                    write_sample_row(sample_writer, row, None, true_label, final_prediction)
                for row in posture_rows:
                    write_sample_row(sample_writer, row, open_ref_r, true_label, final_prediction)
                sf.flush()

                line = [
                    trial_id, true_label, top_label, final_prediction,
                    accepted_correct, forced_correct, confidence, margin,
                    open_high, posture_high,
                ]
                line += [prob_map.get(c, 0.0) for c in classes]
                line += [float(v) if np.isfinite(v) else np.nan for v in local_mean]
                summary_writer.writerow(line)
                tf.flush()

                print(f"True={true_label} | final={final_prediction} | top={top_label} | accepted_correct={accepted_correct} | forced_correct={forced_correct}")

            await client.stop_notify(TX_CHARACTERISTIC)

    print("\nBlind posture demo complete.")
    print("Saved:")
    print(" ", sample_file)
    print(" ", summary_file)
    speak("Blind posture demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
