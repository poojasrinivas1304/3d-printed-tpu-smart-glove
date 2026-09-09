import asyncio
import csv
import time
from collections import deque
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from bleak import BleakScanner, BleakClient


# ------------------------------------------------------------
# BLE settings
# ------------------------------------------------------------

DEVICE_NAMES = [
    "Glove_BLE_7_RAW",
    "TShirt_BLE_VP_RAW",
    "TShirt_BLE_10_RAW",
]

TX_CHARACTERISTIC = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"


# ------------------------------------------------------------
# Channel order must match Arduino analogPins:
#
# analogPins = {36, 39, 34, 35, 32, 33, 25}
# ------------------------------------------------------------

CHANNELS = [
    "Thumb_GPIO36",
    "Index_GPIO39",
    "Middle_GPIO34",
    "Ring_GPIO35",
    "Pinky_GPIO32",
    "ReverseHorizontal_GPIO33",
    "ReverseThumb_GPIO25",
]

N_CHANNELS = len(CHANNELS)


# ------------------------------------------------------------
# Voltage divider
#
# Your wiring:
# 3.3V ---- fixed resistor ---- ADC pin ---- sensor ---- GND
#
# Formula:
# R_sensor = R_fixed * ADC / (4095 - ADC)
# ------------------------------------------------------------

ADC_MAX = 4095.0

# You measured your fixed resistors around 3.4 kOhm.
# Change these if you measure exact per-channel values.
R_FIXED = np.array([
    3400.0,  # Thumb / GPIO36
    3400.0,  # Index / GPIO39
    3400.0,  # Middle / GPIO34
    3400.0,  # Ring / GPIO35
    3400.0,  # Pinky / GPIO32
    3400.0,  # Reverse horizontal / GPIO33
    3400.0,  # Reverse thumb / GPIO25
])


# Treat very low/high ADC as invalid.
VALID_ADC_MIN = 20
VALID_ADC_MAX = 4075


# ------------------------------------------------------------
# Protocol
# First Open_1 is used as R0 baseline.
# ------------------------------------------------------------

STEP_DURATION_S = 5.0

PROTOCOL = [
    ("Open_1_Baseline", "OPEN HAND - keep relaxed and still"),
    ("Pinky_Bend", "BEND PINKY ONLY"),
    ("Open_2", "OPEN HAND - relax"),
    ("Ring_Bend", "BEND RING ONLY"),
    ("Open_3", "OPEN HAND - relax"),
    ("Middle_Bend", "BEND MIDDLE ONLY"),
    ("Open_4", "OPEN HAND - relax"),
    ("Index_Bend", "BEND INDEX ONLY"),
    ("Open_5", "OPEN HAND - relax"),
    ("Thumb_Bend", "BEND THUMB ONLY"),
    ("Open_6", "OPEN HAND - relax"),
    ("Fist", "MAKE A FIST"),
]

TOTAL_DURATION_S = STEP_DURATION_S * len(PROTOCOL)


# ------------------------------------------------------------
# Plot settings
# ------------------------------------------------------------

PLOT_WINDOW = 300
PLOT_UPDATE_INTERVAL_S = 0.25


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def parse_line(line):
    """
    Expected BLE/Serial line:
    t_ms,adc1,adc2,adc3,adc4,adc5,adc6,adc7
    """
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


def adc_to_resistance(adc_values):
    adc = np.array(adc_values, dtype=float)

    resistance = np.full(N_CHANNELS, np.nan, dtype=float)

    valid = (adc > VALID_ADC_MIN) & (adc < VALID_ADC_MAX)

    resistance[valid] = R_FIXED[valid] * adc[valid] / (ADC_MAX - adc[valid])

    return resistance


def adc_state(adc_value):
    if adc_value <= VALID_ADC_MIN:
        return "SHORT/GND"
    elif adc_value >= VALID_ADC_MAX:
        return "OPEN/3.3V"
    else:
        return "OK"


def get_protocol_step(elapsed_s):
    step_index = int(elapsed_s // STEP_DURATION_S)

    if step_index >= len(PROTOCOL):
        return None, None, None, None

    label, instruction = PROTOCOL[step_index]

    step_start_s = step_index * STEP_DURATION_S
    step_elapsed_s = elapsed_s - step_start_s
    step_remaining_s = STEP_DURATION_S - step_elapsed_s

    return step_index, label, instruction, step_remaining_s


def format_array(values, decimals=3):
    out = []

    for value in values:
        if np.isfinite(value):
            out.append(f"{value:.{decimals}f}")
        else:
            out.append("nan")

    return "[" + ", ".join(out) + "]"


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
        print()
        print("Could not find expected ESP32 BLE device.")
        print("Expected one of:")
        for name in DEVICE_NAMES:
            print("  ", name)
        return None

    print()
    print(f"Selected device: {selected.name}  {selected.address}")
    return selected


def build_csv_header():
    header = [
        "wall_time_iso",
        "esp32_t_ms",
        "protocol_elapsed_s",
        "step_index",
        "label",
        "instruction",
        "step_elapsed_s",
    ]

    for ch in CHANNELS:
        header.append(f"{ch}_raw_ADC")

    for ch in CHANNELS:
        header.append(f"{ch}_R_ohm")

    for ch in CHANNELS:
        header.append(f"{ch}_dR_over_R0")

    return header


def save_csv(rows, r0):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"glove_single_finger_protocol_{timestamp}.csv"

    header = build_csv_header()

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for row in rows:
            resistance = row["resistance"]
            normalized = np.full(N_CHANNELS, np.nan, dtype=float)

            valid = np.isfinite(resistance) & np.isfinite(r0) & (r0 != 0)
            normalized[valid] = (resistance[valid] - r0[valid]) / r0[valid]

            line = [
                row["wall_time_iso"],
                row["esp32_t_ms"],
                f"{row['protocol_elapsed_s']:.4f}",
                row["step_index"],
                row["label"],
                row["instruction"],
                f"{row['step_elapsed_s']:.4f}",
            ]

            line.extend(row["adc_values"])

            for value in resistance:
                line.append(f"{value:.6f}" if np.isfinite(value) else "nan")

            for value in normalized:
                line.append(f"{value:.6f}" if np.isfinite(value) else "nan")

            writer.writerow(line)

    print()
    print(f"Saved CSV file: {filename}")
    return filename


def print_r0(r0):
    print()
    print("Baseline R0 from first Open hand period:")
    for name, value in zip(CHANNELS, r0):
        if np.isfinite(value):
            print(f"  {name}: {value:.2f} ohm")
        else:
            print(f"  {name}: INVALID")


def save_summary(rows, r0):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"glove_single_finger_summary_{timestamp}.csv"

    header = ["label"]

    for ch in CHANNELS:
        header.append(f"{ch}_mean_dR_over_R0")

    for ch in CHANNELS:
        header.append(f"{ch}_std_dR_over_R0")

    labels = []
    for label, _ in PROTOCOL:
        labels.append(label)

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for label in labels:
            values = []

            for row in rows:
                if row["label"] != label:
                    continue

                resistance = row["resistance"]
                normalized = np.full(N_CHANNELS, np.nan, dtype=float)

                valid = np.isfinite(resistance) & np.isfinite(r0) & (r0 != 0)
                normalized[valid] = (resistance[valid] - r0[valid]) / r0[valid]

                values.append(normalized)

            if len(values) == 0:
                continue

            values = np.array(values, dtype=float)

            means = np.nanmean(values, axis=0)
            stds = np.nanstd(values, axis=0)

            line = [label]

            for value in means:
                line.append(f"{value:.6f}" if np.isfinite(value) else "nan")

            for value in stds:
                line.append(f"{value:.6f}" if np.isfinite(value) else "nan")

            writer.writerow(line)

    print(f"Saved summary file: {filename}")
    return filename


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------

async def main():
    device = await find_ble_device()

    if device is None:
        return

    print()
    print("Sensor order used in this script:")
    for i, name in enumerate(CHANNELS, start=1):
        print(f"  Channel {i}: {name}")

    print()
    print("Protocol:")
    for label, instruction in PROTOCOL:
        print(f"  {label:18s} | {instruction} | {STEP_DURATION_S:.0f} s")

    print()
    input("Put the glove on, keep hand open, then press ENTER to start...")

    line_buffer = ""
    pending_samples = deque()

    rows = []

    resistance_history = deque(maxlen=PLOT_WINDOW)
    normalized_history = deque(maxlen=PLOT_WINDOW)
    label_history = deque(maxlen=PLOT_WINDOW)

    r0 = None
    last_plot_time = 0.0
    last_print_step = None
    last_countdown_print = 0.0

    plt.ion()
    fig, (ax_r, ax_n) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    def notification_handler(sender, data):
        nonlocal line_buffer

        chunk = data.decode("utf-8", errors="ignore")
        line_buffer += chunk

        while "\n" in line_buffer:
            line, line_buffer = line_buffer.split("\n", 1)
            parsed = parse_line(line)

            if parsed is not None:
                pending_samples.append(parsed)

    async with BleakClient(device) as client:
        print()
        print("Connected.")
        print("Starting BLE notifications...")
        await client.start_notify(TX_CHARACTERISTIC, notification_handler)

        print()
        print("Starting protocol now.")
        print("\a")

        protocol_start_wall = time.time()

        while True:
            await asyncio.sleep(0.02)

            now = time.time()
            elapsed_s = now - protocol_start_wall

            if elapsed_s >= TOTAL_DURATION_S:
                break

            step_index, label, instruction, step_remaining_s = get_protocol_step(elapsed_s)

            if step_index is None:
                break

            if step_index != last_print_step:
                last_print_step = step_index
                print()
                print("=" * 70)
                print(f"STEP {step_index + 1}/{len(PROTOCOL)}: {instruction}")
                print(f"Label: {label}")
                print("=" * 70)
                print("\a")

            if now - last_countdown_print >= 1.0:
                last_countdown_print = now
                print(f"{instruction} | {step_remaining_s:4.1f} s remaining")

            while pending_samples:
                esp32_t_ms, adc_values = pending_samples.popleft()

                sample_wall = time.time()
                sample_elapsed_s = sample_wall - protocol_start_wall

                if sample_elapsed_s >= TOTAL_DURATION_S:
                    continue

                sample_step_index, sample_label, sample_instruction, _ = get_protocol_step(sample_elapsed_s)

                if sample_step_index is None:
                    continue

                step_elapsed_s = sample_elapsed_s - sample_step_index * STEP_DURATION_S

                resistance = adc_to_resistance(adc_values)

                row = {
                    "wall_time_iso": datetime.now().isoformat(timespec="milliseconds"),
                    "esp32_t_ms": esp32_t_ms,
                    "protocol_elapsed_s": sample_elapsed_s,
                    "step_index": sample_step_index,
                    "label": sample_label,
                    "instruction": sample_instruction,
                    "step_elapsed_s": step_elapsed_s,
                    "adc_values": adc_values,
                    "resistance": resistance,
                }

                rows.append(row)

                # Establish R0 after first Open hand step is complete.
                if r0 is None and sample_elapsed_s >= STEP_DURATION_S:
                    baseline_resistances = []

                    for saved_row in rows:
                        if saved_row["step_index"] == 0 and saved_row["step_elapsed_s"] >= 0.5:
                            baseline_resistances.append(saved_row["resistance"])

                    baseline_resistances = np.array(baseline_resistances, dtype=float)

                    r0 = np.nanmedian(baseline_resistances, axis=0)
                    print_r0(r0)
                    print()
                    print("Now live normalized resistance is active.")
                    print()

                if r0 is not None:
                    normalized = np.full(N_CHANNELS, np.nan, dtype=float)
                    valid = np.isfinite(resistance) & np.isfinite(r0) & (r0 != 0)
                    normalized[valid] = (resistance[valid] - r0[valid]) / r0[valid]
                else:
                    normalized = np.full(N_CHANNELS, np.nan, dtype=float)

                resistance_history.append(resistance)
                normalized_history.append(normalized)
                label_history.append(sample_label)

            # Live plot update
            if now - last_plot_time >= PLOT_UPDATE_INTERVAL_S and len(resistance_history) > 2:
                last_plot_time = now

                r_array = np.array(resistance_history, dtype=float)
                n_array = np.array(normalized_history, dtype=float)

                ax_r.clear()
                ax_n.clear()

                for i, ch in enumerate(CHANNELS):
                    ax_r.plot(r_array[:, i], label=ch)

                ax_r.set_title(f"Live Sensor Resistance | Current: {instruction}")
                ax_r.set_ylabel("Resistance (ohm)")
                ax_r.grid(True)
                ax_r.legend(loc="upper right", fontsize=8)

                if r0 is not None:
                    for i, ch in enumerate(CHANNELS):
                        ax_n.plot(n_array[:, i], label=ch)

                    ax_n.set_title("Live Normalized Resistance")
                    ax_n.set_ylabel("ΔR/R0")
                    ax_n.set_xlabel("Sample")
                    ax_n.grid(True)
                    ax_n.legend(loc="upper right", fontsize=8)
                else:
                    ax_n.set_title("Collecting first Open hand baseline for R0...")
                    ax_n.set_ylabel("ΔR/R0")
                    ax_n.set_xlabel("Sample")
                    ax_n.grid(True)

                plt.tight_layout()
                plt.pause(0.001)

        await client.stop_notify(TX_CHARACTERISTIC)

    print()
    print("Protocol finished.")
    print("\a")

    if len(rows) == 0:
        print("No data collected.")
        return

    if r0 is None:
        baseline_resistances = []

        for saved_row in rows:
            if saved_row["step_index"] == 0 and saved_row["step_elapsed_s"] >= 0.5:
                baseline_resistances.append(saved_row["resistance"])

        baseline_resistances = np.array(baseline_resistances, dtype=float)
        r0 = np.nanmedian(baseline_resistances, axis=0)

    print_r0(r0)

    csv_file = save_csv(rows, r0)
    summary_file = save_summary(rows, r0)

    print()
    print("Done.")
    print(f"Raw data: {csv_file}")
    print(f"Summary:  {summary_file}")


if __name__ == "__main__":
    asyncio.run(main())
    