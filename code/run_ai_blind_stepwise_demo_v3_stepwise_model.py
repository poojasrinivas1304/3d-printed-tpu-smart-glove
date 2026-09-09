import asyncio
import csv
import os
import platform
import subprocess
import time
from collections import deque
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from bleak import BleakScanner, BleakClient

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(SCRIPT_DIR, "finger_ai_stepwise_model.joblib")
OUTPUT_DIR = SCRIPT_DIR

DEVICE_NAMES = ["Glove_BLE_7_RAW", "TShirt_BLE_VP_RAW", "TShirt_BLE_10_RAW"]
TX_CHARACTERISTIC = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

CHANNELS = ["Thumb", "Index", "Middle", "Ring", "Little", "ReverseHorizontal", "ReverseThumb"]
FINGER_LABELS = ["Thumb", "Index", "Middle", "Ring", "Little"]
N_CHANNELS = len(CHANNELS)

ADC_MAX = 4095.0
R_FIXED = np.array([3400.0, 3400.0, 3400.0, 3400.0, 3400.0, 3400.0, 3400.0])
VALID_ADC_MIN = 20
VALID_ADC_MAX = 3900
BAD_ADC_THRESHOLD = 3900
MAX_OPEN_BAD_ADC_SAMPLES = 3
MAX_FOLD_BAD_ADC_SAMPLES = 5

OPEN_REFERENCE_SECONDS = 4.0
OPEN_REFERENCE_USE_LAST_SECONDS = 2.0
FOLD_SECONDS = 5.0
FOLD_USE_LAST_SECONDS = 3.0

MIN_CONFIDENCE = 0.35
MIN_MARGIN = 0.08

VOICE_ENABLED = True
VOICE_RATE = 185
_voice_process = None


def speak(text):
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


def parse_line(line):
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


def adc_to_resistance(adc_values):
    adc = np.array(adc_values, dtype=float)
    resistance = np.full(N_CHANNELS, np.nan, dtype=float)
    valid = (adc > VALID_ADC_MIN) & (adc < VALID_ADC_MAX)
    resistance[valid] = R_FIXED[valid] * adc[valid] / (ADC_MAX - adc[valid])
    return resistance


def local_normalized(resistance, open_ref):
    out = np.full(N_CHANNELS, np.nan, dtype=float)
    valid = np.isfinite(resistance) & np.isfinite(open_ref) & (open_ref != 0)
    out[valid] = (resistance[valid] - open_ref[valid]) / open_ref[valid]
    return out


def bad_adc_count(rows):
    total = 0
    by_channel = {ch: 0 for ch in CHANNELS}
    for row in rows:
        for ch, adc in zip(CHANNELS, row["adc_values"]):
            if adc >= BAD_ADC_THRESHOLD:
                total += 1
                by_channel[ch] += 1
    return total, by_channel


def print_bad(prefix, rows):
    total, by_ch = bad_adc_count(rows)
    print(f"{prefix}: high ADC >= {BAD_ADC_THRESHOLD}: {total}")
    if total:
        print("  by channel:", {k: v for k, v in by_ch.items() if v > 0})
    return total, by_ch


def compute_open_reference(rows):
    cutoff = max(0.0, OPEN_REFERENCE_SECONDS - OPEN_REFERENCE_USE_LAST_SECONDS)
    vals = [r["resistance"] for r in rows if r["phase_elapsed_s"] >= cutoff]
    if not vals:
        return None
    return np.nanmedian(np.array(vals, dtype=float), axis=0)


def slope_from_series(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    t = t[mask] - np.nanmin(t[mask])
    y = y[mask]
    try:
        return float(np.polyfit(t, y, 1)[0])
    except Exception:
        return np.nan


def extract_features(fold_rows, open_ref, feature_names):
    cutoff = max(0.0, FOLD_SECONDS - FOLD_USE_LAST_SECONDS)
    selected = [r for r in fold_rows if r["phase_elapsed_s"] >= cutoff]
    if not selected:
        selected = fold_rows
    feat = {}
    for ch_i, ch in enumerate(CHANNELS):
        y = []
        t = []
        for r in selected:
            local = local_normalized(r["resistance"], open_ref)
            y.append(local[ch_i])
            t.append(r["phase_elapsed_s"])
        y = np.asarray(y, dtype=float)
        t = np.asarray(t, dtype=float)
        feat[f"{ch}_mean"] = np.nanmean(y)
        feat[f"{ch}_median"] = np.nanmedian(y)
        feat[f"{ch}_std"] = np.nanstd(y)
        feat[f"{ch}_min"] = np.nanmin(y)
        feat[f"{ch}_max"] = np.nanmax(y)
        feat[f"{ch}_range"] = np.nanmax(y) - np.nanmin(y)
        feat[f"{ch}_last"] = y[np.where(np.isfinite(y))[0][-1]] if np.isfinite(y).any() else np.nan
        feat[f"{ch}_slope"] = slope_from_series(t, y)
    return pd.DataFrame([[feat.get(name, np.nan) for name in feature_names]], columns=feature_names)


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
        print("Could not find ESP32 BLE device.")
        return None
    print(f"Selected device: {selected.name}  {selected.address}")
    return selected


async def collect_phase(pending_samples, duration_s, session_start, trial_id, phase):
    rows = []
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
            print(f"Trial {trial_id:02d} | {phase:8s} | {duration_s - elapsed:4.1f}s left")
        while pending_samples:
            esp32_t_ms, adc_values = pending_samples.popleft()
            sample_time = time.time()
            phase_elapsed = sample_time - start
            if phase_elapsed > duration_s:
                continue
            rows.append({
                "wall_time_iso": datetime.now().isoformat(timespec="milliseconds"),
                "esp32_t_ms": esp32_t_ms,
                "trial_id": trial_id,
                "phase": phase,
                "phase_elapsed_s": phase_elapsed,
                "session_elapsed_s": sample_time - session_start,
                "adc_values": adc_values,
                "resistance": adc_to_resistance(adc_values),
            })
    return rows


def save_rows(sample_writer, rows, open_ref, pred, true_label, open_bad, fold_bad):
    for r in rows:
        local = local_normalized(r["resistance"], open_ref)
        line = [
            r["wall_time_iso"], r["esp32_t_ms"], r["trial_id"], r["phase"],
            f"{r['phase_elapsed_s']:.4f}", f"{r['session_elapsed_s']:.4f}",
            pred, true_label, open_bad, fold_bad,
        ]
        line.extend(r["adc_values"])
        for v in r["resistance"]:
            line.append(f"{v:.6f}" if np.isfinite(v) else "nan")
        for v in local:
            line.append(f"{v:.6f}" if np.isfinite(v) else "nan")
        sample_writer.writerow(line)


def sample_header():
    h = ["wall_time_iso", "esp32_t_ms", "trial_id", "phase", "phase_elapsed_s", "session_elapsed_s", "prediction", "true_label", "open_bad_adc_total", "fold_bad_adc_total"]
    h += [f"{ch}_raw_ADC" for ch in CHANNELS]
    h += [f"{ch}_R_ohm" for ch in CHANNELS]
    h += [f"{ch}_local_dR" for ch in CHANNELS]
    return h


def summary_header():
    h = ["trial_id", "prediction", "true_label", "correct", "confidence", "margin", "status", "open_bad_adc_total", "fold_bad_adc_total"]
    h += [f"prob_{c}" for c in FINGER_LABELS]
    return h


async def main():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Missing model file: {MODEL_FILE}")
    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    classes = list(bundle.get("classes", FINGER_LABELS))

    device = await find_ble_device()
    if device is None:
        return

    n_text = input("Number of accepted blind stepwise trials? [10]: ").strip()
    target_trials = 10 if n_text == "" else int(n_text)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_file = os.path.join(OUTPUT_DIR, f"ai_stepwise_model_blind_samples_{timestamp}.csv")
    summary_file = os.path.join(OUTPUT_DIR, f"ai_stepwise_model_blind_summary_{timestamp}.csv")

    line_buffer = ""
    pending_samples = deque()

    def notification_handler(sender, data):
        nonlocal line_buffer
        chunk = data.decode("utf-8", errors="ignore")
        line_buffer += chunk
        while "\n" in line_buffer:
            line, line_buffer = line_buffer.split("\n", 1)
            parsed = parse_line(line)
            if parsed is not None:
                pending_samples.append(parsed)

    with open(samples_file, "w", newline="") as sf, open(summary_file, "w", newline="") as tf:
        async with BleakClient(device) as client:
            print("Connected.")
            await client.start_notify(TX_CHARACTERISTIC, notification_handler)
            sample_writer = csv.writer(sf)
            summary_writer = csv.writer(tf)
            sample_writer.writerow(sample_header())
            summary_writer.writerow(summary_header())

            session_start = time.time()
            accepted = 0
            attempt = 0
            while accepted < target_trials:
                attempt += 1
                trial_id = accepted + 1
                print("\n" + "=" * 80)
                print(f"Blind trial {trial_id}/{target_trials}  (attempt {attempt})")
                print("The script will NOT tell you which finger to fold.")
                print("=" * 80)

                speak("Open hand")
                print("Open hand. Collecting open reference.")
                pending_samples.clear()
                open_rows = await collect_phase(pending_samples, OPEN_REFERENCE_SECONDS, session_start, trial_id, "open_ref")
                open_bad, _ = print_bad("Open reference QC", open_rows)
                open_ref = compute_open_reference(open_rows)
                if open_ref is None or open_bad > MAX_OPEN_BAD_ADC_SAMPLES:
                    print("Bad open reference. Repeat trial.")
                    speak("Bad open reference. Repeat.")
                    continue

                speak("Fold one finger and hold")
                print("Fold ONE finger and hold. The AI will predict once after the hold.")
                pending_samples.clear()
                fold_rows = await collect_phase(pending_samples, FOLD_SECONDS, session_start, trial_id, "fold")
                fold_bad, _ = print_bad("Fold QC", fold_rows)
                if fold_bad > MAX_FOLD_BAD_ADC_SAMPLES:
                    print("Bad fold window. Repeat trial.")
                    speak("Bad fold signal. Repeat.")
                    continue

                X = extract_features(fold_rows, open_ref, feature_names)
                probs = model.predict_proba(X)[0]
                class_order = list(model.classes_)
                top_idx = int(np.argmax(probs))
                pred = class_order[top_idx]
                sorted_probs = np.sort(probs)[::-1]
                conf = float(sorted_probs[0])
                margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else conf
                status = "Accepted"
                final_pred = pred
                if conf < MIN_CONFIDENCE or margin < MIN_MARGIN:
                    final_pred = "Uncertain"
                    status = "Uncertain"

                print("\nPrediction result")
                print(f"  Prediction: {final_pred}")
                print(f"  Raw top:    {pred}")
                print(f"  Confidence: {conf:.3f}")
                print(f"  Margin:     {margin:.3f}")
                print("  Probabilities:")
                for c in FINGER_LABELS:
                    p = probs[class_order.index(c)] if c in class_order else np.nan
                    print(f"    {c:7s}: {p:.3f}")
                speak(final_pred)

                true_label = input("Enter true folded finger [Thumb/Index/Middle/Ring/Little] or ENTER to skip: ").strip()
                if true_label == "":
                    true_label = "Skipped"
                elif true_label.lower() in ["thumb", "t"]:
                    true_label = "Thumb"
                elif true_label.lower() in ["index", "i"]:
                    true_label = "Index"
                elif true_label.lower() in ["middle", "m"]:
                    true_label = "Middle"
                elif true_label.lower() in ["ring", "r"]:
                    true_label = "Ring"
                elif true_label.lower() in ["little", "pinky", "p", "l"]:
                    true_label = "Little"

                correct = int(final_pred == true_label)
                accepted += 1

                save_rows(sample_writer, open_rows, open_ref, final_pred, true_label, open_bad, fold_bad)
                save_rows(sample_writer, fold_rows, open_ref, final_pred, true_label, open_bad, fold_bad)

                row = [trial_id, final_pred, true_label, correct, f"{conf:.6f}", f"{margin:.6f}", status, open_bad, fold_bad]
                for c in FINGER_LABELS:
                    p = probs[class_order.index(c)] if c in class_order else np.nan
                    row.append(f"{p:.6f}" if np.isfinite(p) else "nan")
                summary_writer.writerow(row)
                sf.flush(); tf.flush()
                print(f"Saved trial {trial_id}: true={true_label}, prediction={final_pred}, correct={correct}")

            await client.stop_notify(TX_CHARACTERISTIC)

    print("\nDemo complete.")
    print("Samples:", samples_file)
    print("Summary:", summary_file)
    speak("Demo complete")


if __name__ == "__main__":
    asyncio.run(main())
