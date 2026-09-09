import asyncio
import csv
import os
import random
import time
import platform
import subprocess
from collections import deque
from datetime import datetime

import numpy as np
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
# Arduino order: GPIO36, GPIO39, GPIO34, GPIO35, GPIO32, GPIO33, GPIO25
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
N_CHANNELS = len(CHANNELS)
FINGER_LABELS = ["Thumb", "Index", "Middle", "Ring", "Little"]

# ============================================================
# VOLTAGE DIVIDER
# 3.3V ---- fixed resistor ---- ADC ---- sensor ---- GND
# Rsensor = Rfixed * ADC / (4095 - ADC)
# ============================================================
ADC_MAX = 4095.0
R_FIXED = np.array([3400.0, 3400.0, 3400.0, 3400.0, 3400.0, 3400.0, 3400.0])
VALID_ADC_MIN = 20
VALID_ADC_MAX = 3900  # values above this are treated as open-contact artifacts

# ============================================================
# COLLECTION SETTINGS
# ============================================================
OUTPUT_DIR = "ai_stepwise_data"
OPEN_REFERENCE_SECONDS = 4.0
OPEN_REFERENCE_USE_LAST_SECONDS = 2.0
FOLD_SECONDS = 5.0
FOLD_USE_LAST_SECONDS = 3.0
DEFAULT_TRIALS_PER_FINGER = 10

# Trial QC: if too many samples are near open circuit, repeat the trial.
MAX_OPEN_BAD_ADC_SAMPLES = 3
MAX_FOLD_BAD_ADC_SAMPLES = 5
BAD_ADC_THRESHOLD = 3900

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


def bad_adc_count(sample_rows):
    count = 0
    by_channel = {ch: 0 for ch in CHANNELS}
    for row in sample_rows:
        for ch, adc in zip(CHANNELS, row["adc_values"]):
            if adc >= BAD_ADC_THRESHOLD:
                count += 1
                by_channel[ch] += 1
    return count, by_channel


def print_bad_adc_summary(prefix, rows):
    total, by_channel = bad_adc_count(rows)
    print(f"{prefix}: high ADC >= {BAD_ADC_THRESHOLD}: {total} samples")
    if total > 0:
        active = {k: v for k, v in by_channel.items() if v > 0}
        print("  by channel:", active)
    return total, by_channel


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
        print("\nCould not find expected ESP32 BLE device.")
        print("Expected one of:")
        for name in DEVICE_NAMES:
            print("  ", name)
        return None
    print(f"\nSelected device: {selected.name}  {selected.address}")
    return selected


async def collect_phase(pending_samples, duration_s, run_elapsed_start, label, phase, trial_id, true_label):
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
            print(f"Trial {trial_id:03d} | {phase:9s} | {duration_s-elapsed:4.1f}s left")
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
                "true_label": true_label,
                "phase": phase,
                "label": label,
                "phase_elapsed_s": phase_elapsed,
                "session_elapsed_s": sample_time - run_elapsed_start,
                "adc_values": adc_values,
                "resistance": resistance,
            })
    return rows


def compute_open_reference(open_rows):
    cutoff = max(0.0, OPEN_REFERENCE_SECONDS - OPEN_REFERENCE_USE_LAST_SECONDS)
    vals = [row["resistance"] for row in open_rows if row["phase_elapsed_s"] >= cutoff]
    if len(vals) == 0:
        return None
    vals = np.array(vals, dtype=float)
    return np.nanmedian(vals, axis=0)


def build_sample_header():
    header = [
        "wall_time_iso", "esp32_t_ms", "trial_id", "true_label", "phase", "label",
        "phase_elapsed_s", "session_elapsed_s", "use_for_training", "open_ref_bad_adc_total",
        "fold_bad_adc_total",
    ]
    for ch in CHANNELS:
        header.append(f"{ch}_raw_ADC")
    for ch in CHANNELS:
        header.append(f"{ch}_R_ohm")
    for ch in CHANNELS:
        header.append(f"{ch}_local_dR_over_open_ref")
    return header


def write_sample_rows(writer, rows, open_ref, use_for_training, open_bad_total, fold_bad_total):
    for row in rows:
        resistance = row["resistance"]
        local = local_normalized(resistance, open_ref)
        line = [
            row["wall_time_iso"], row["esp32_t_ms"], row["trial_id"], row["true_label"],
            row["phase"], row["label"], f"{row['phase_elapsed_s']:.4f}",
            f"{row['session_elapsed_s']:.4f}", use_for_training,
            open_bad_total, fold_bad_total,
        ]
        line.extend(row["adc_values"])
        for value in resistance:
            line.append(f"{value:.6f}" if np.isfinite(value) else "nan")
        for value in local:
            line.append(f"{value:.6f}" if np.isfinite(value) else "nan")
        writer.writerow(line)


def build_trial_header():
    header = [
        "trial_id", "true_label", "accepted", "open_bad_adc_total", "fold_bad_adc_total",
        "n_open_samples", "n_fold_samples", "n_training_samples",
    ]
    for ch in CHANNELS:
        header.append(f"open_ref_{ch}_R_ohm")
    for ch in CHANNELS:
        header.append(f"fold_train_mean_{ch}_local_dR")
    return header


def write_trial_summary(writer, trial_id, true_label, accepted, open_bad_total, fold_bad_total, open_rows, fold_rows, open_ref):
    cutoff = max(0.0, FOLD_SECONDS - FOLD_USE_LAST_SECONDS)
    train_rows = [r for r in fold_rows if r["phase_elapsed_s"] >= cutoff]
    local_values = [local_normalized(r["resistance"], open_ref) for r in train_rows]
    if len(local_values) > 0:
        fold_mean = np.nanmean(np.array(local_values, dtype=float), axis=0)
    else:
        fold_mean = np.full(N_CHANNELS, np.nan)

    line = [trial_id, true_label, int(accepted), open_bad_total, fold_bad_total,
            len(open_rows), len(fold_rows), len(train_rows)]
    for value in open_ref:
        line.append(f"{value:.6f}" if np.isfinite(value) else "nan")
    for value in fold_mean:
        line.append(f"{value:.6f}" if np.isfinite(value) else "nan")
    writer.writerow(line)


async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nStepwise finger-folding training data collector")
    print("This collects data in the SAME format as the blind stepwise demo.")
    print("For training, the script tells you which finger to fold so the dataset is balanced.\n")

    tpf = input(f"Trials per finger? [{DEFAULT_TRIALS_PER_FINGER}]: ").strip()
    trials_per_finger = DEFAULT_TRIALS_PER_FINGER if tpf == "" else int(tpf)

    shuffle_text = input("Shuffle trial order? [y]: ").strip().lower()
    shuffle_trials = (shuffle_text != "n")

    device = await find_ble_device()
    if device is None:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_file = os.path.join(OUTPUT_DIR, f"stepwise_train_samples_{timestamp}.csv")
    trial_file = os.path.join(OUTPUT_DIR, f"stepwise_train_trials_{timestamp}.csv")

    # Build balanced trial sequence.
    trial_labels = []
    for label in FINGER_LABELS:
        trial_labels.extend([label] * trials_per_finger)
    if shuffle_trials:
        random.shuffle(trial_labels)

    print("\nTrial plan:")
    print("  ", trial_labels)
    print(f"\nWill save samples to: {sample_file}")
    print(f"Will save trial summary to: {trial_file}")

    input("\nPut glove on. Keep hand open. Press ENTER to connect and start...")

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

    with open(sample_file, "w", newline="") as sf, open(trial_file, "w", newline="") as tf:
        async with BleakClient(device) as client:
            print("Connected.")
            await client.start_notify(TX_CHARACTERISTIC, notification_handler)

            sample_writer = csv.writer(sf)
            trial_writer = csv.writer(tf)
            sample_writer.writerow(build_sample_header())
            trial_writer.writerow(build_trial_header())

            session_start = time.time()
            accepted_trial_id = 0
            trial_index = 0

            while trial_index < len(trial_labels):
                true_label = trial_labels[trial_index]
                trial_number_display = trial_index + 1
                print("\n" + "=" * 80)
                print(f"Training trial {trial_number_display}/{len(trial_labels)}")
                print(f"TRUE LABEL TO PERFORM: {true_label}")
                print("=" * 80)

                speak("Open hand")
                print("Open hand. Hold still for open reference.")
                pending_samples.clear()
                open_rows = await collect_phase(
                    pending_samples, OPEN_REFERENCE_SECONDS, session_start,
                    label="OpenReference", phase="open_ref", trial_id=accepted_trial_id + 1,
                    true_label=true_label,
                )
                open_bad_total, _ = print_bad_adc_summary("Open reference QC", open_rows)
                open_ref = compute_open_reference(open_rows)

                if open_ref is None or open_bad_total > MAX_OPEN_BAD_ADC_SAMPLES:
                    print("BAD OPEN REFERENCE. Repeat this trial after checking the glove/contact.")
                    speak("Bad open reference. Repeat.")
                    continue

                print(f"Open reference OK. Now fold: {true_label}")
                speak(f"Fold {true_label}")
                pending_samples.clear()
                fold_rows = await collect_phase(
                    pending_samples, FOLD_SECONDS, session_start,
                    label=true_label, phase="fold", trial_id=accepted_trial_id + 1,
                    true_label=true_label,
                )
                fold_bad_total, _ = print_bad_adc_summary("Fold QC", fold_rows)

                if fold_bad_total > MAX_FOLD_BAD_ADC_SAMPLES:
                    print("BAD FOLD WINDOW. Repeat this trial after checking the glove/contact.")
                    speak("Bad fold signal. Repeat.")
                    continue

                accepted_trial_id += 1
                trial_index += 1
                cutoff = max(0.0, FOLD_SECONDS - FOLD_USE_LAST_SECONDS)
                for row in open_rows:
                    write_sample_rows(sample_writer, [row], open_ref, 0, open_bad_total, fold_bad_total)
                for row in fold_rows:
                    use_train = int(row["phase_elapsed_s"] >= cutoff)
                    write_sample_rows(sample_writer, [row], open_ref, use_train, open_bad_total, fold_bad_total)

                write_trial_summary(
                    trial_writer, accepted_trial_id, true_label, True,
                    open_bad_total, fold_bad_total, open_rows, fold_rows, open_ref,
                )
                sf.flush(); tf.flush()

                print(f"Accepted trial {accepted_trial_id}: {true_label}")
                speak("Accepted")

            await client.stop_notify(TX_CHARACTERISTIC)

    print("\nAll stepwise training trials complete.")
    print(f"Samples saved: {sample_file}")
    print(f"Summary saved: {trial_file}")
    speak("All training trials complete")


if __name__ == "__main__":
    asyncio.run(main())
