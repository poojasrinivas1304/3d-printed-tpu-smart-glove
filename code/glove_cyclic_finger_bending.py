import asyncio
import csv
import time
import subprocess
import platform
import re
from collections import deque
from datetime import datetime

import numpy as np

# Safer Matplotlib setup for macOS.
# Backend must be selected before importing pyplot.
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
#
# Arduino order:
# GPIO36, GPIO39, GPIO34, GPIO35, GPIO32, GPIO33, GPIO25
#
# Your sensor order:
# Thumb, Index, Middle, Ring, Pinky, Reverse Horizontal, Reverse Thumb
# ============================================================

CHANNELS = [
    "Thumb",
    "Index",
    "Middle",
    "Ring",
    "Pinky",
    "ReverseHorizontal",
    "ReverseThumb",
]

N_CHANNELS = len(CHANNELS)

INTENDED_CHANNEL = {
    "Thumb": 0,
    "Index": 1,
    "Middle": 2,
    "Ring": 3,
    "Pinky": 4,
}


# ============================================================
# VOLTAGE DIVIDER SETTINGS
#
# Wiring:
# 3.3V ---- fixed resistor ---- ADC pin ---- sensor ---- GND
#
# R_sensor = R_fixed * ADC / (4095 - ADC)
# ============================================================

ADC_MAX = 4095.0

R_FIXED = np.array([
    3400.0,  # Thumb / GPIO36
    3400.0,  # Index / GPIO39
    3400.0,  # Middle / GPIO34
    3400.0,  # Ring / GPIO35
    3400.0,  # Pinky / GPIO32
    3400.0,  # Reverse Horizontal / GPIO33
    3400.0,  # Reverse Thumb / GPIO25
])

VALID_ADC_MIN = 20
VALID_ADC_MAX = 4075


# ============================================================
# CYCLIC TEST SETTINGS
# ============================================================

FINGER_ORDER = [
    "Thumb",
    "Index",
    "Middle",
    "Ring",
    "Pinky",
]

N_CYCLES = 5

# One cycle = open phase + bend phase.
OPEN_HOLD_S = 2.0
BEND_HOLD_S = 2.0

# Preparation before each finger.
PREP_REST_S = 4.0

# Initial open-hand baseline.
BASELINE_S = 10.0

# Final open-hand recovery.
FINAL_RECOVERY_S = 5.0

# For summary, use only the final part of each 2 s phase.
SUMMARY_LAST_S = 1.0


# ============================================================
# LIVE PLOT SETTINGS
# ============================================================

SHOW_LIVE_PLOT = True

# Around 24 seconds at 20 Hz.
PLOT_HISTORY_SAMPLES = 480

# 0.25 gives smoother live feedback. Use 0.5 if Mac becomes slow.
PLOT_UPDATE_INTERVAL_S = 0.25

PLOT_FIGSIZE = (6.0, 4.0)

USE_FIXED_YLIM = False
NORMALIZED_YLIM = (-0.25, 1.0)


# ============================================================
# VOICE SETTINGS
# ============================================================

VOICE_ENABLED = True
VOICE_RATE = 230

_voice_process = None


def speak(text):
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
        except Exception as e:
            print("Voice failed:", e)
    else:
        print("\a")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sanitize_filename_text(text):
    text = text.strip()
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    text = text.strip("_")
    return text


def parse_line(line):
    """
    Expected ESP32 BLE line:
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


def calculate_normalized(resistance, r0):
    normalized = np.full(N_CHANNELS, np.nan, dtype=float)

    valid = (
        np.isfinite(resistance)
        & np.isfinite(r0)
        & (r0 != 0)
    )

    normalized[valid] = (resistance[valid] - r0[valid]) / r0[valid]

    return normalized


def build_protocol_events():
    """
    Protocol:
    Initial baseline
    For each finger:
        prep open
        5 cycles of open 2 s + bend 2 s
    Final recovery
    """
    events = []

    events.append({
        "label": "Initial_Baseline",
        "target_finger": "None",
        "phase": "baseline",
        "cycle": 0,
        "duration_s": BASELINE_S,
        "instruction": "OPEN HAND - baseline",
        "voice": "Open hand. Keep relaxed and still. Baseline.",
    })

    for finger in FINGER_ORDER:
        events.append({
            "label": f"{finger}_Prep_Open",
            "target_finger": finger,
            "phase": "prep_open",
            "cycle": 0,
            "duration_s": PREP_REST_S,
            "instruction": f"OPEN HAND - prepare for {finger}",
            "voice": f"Get ready for {finger}. Keep hand open.",
        })

        for cycle in range(1, N_CYCLES + 1):
            events.append({
                "label": f"{finger}_C{cycle}_Open",
                "target_finger": finger,
                "phase": "open",
                "cycle": cycle,
                "duration_s": OPEN_HOLD_S,
                "instruction": f"{finger} cycle {cycle}: OPEN",
                "voice": "Open",
            })

            events.append({
                "label": f"{finger}_C{cycle}_Bend",
                "target_finger": finger,
                "phase": "bend",
                "cycle": cycle,
                "duration_s": BEND_HOLD_S,
                "instruction": f"{finger} cycle {cycle}: BEND {finger}",
                "voice": f"Bend {finger}",
            })

    events.append({
        "label": "Final_Open_Recovery",
        "target_finger": "None",
        "phase": "final_open",
        "cycle": 0,
        "duration_s": FINAL_RECOVERY_S,
        "instruction": "OPEN HAND - final recovery",
        "voice": "Open hand. Final recovery.",
    })

    t = 0.0
    for idx, event in enumerate(events):
        event["event_index"] = idx
        event["start_s"] = t
        event["end_s"] = t + event["duration_s"]
        t = event["end_s"]

    return events


def get_current_event(events, elapsed_s):
    for event in events:
        if event["start_s"] <= elapsed_s < event["end_s"]:
            phase_elapsed_s = elapsed_s - event["start_s"]
            phase_remaining_s = event["end_s"] - elapsed_s
            return event, phase_elapsed_s, phase_remaining_s

    return None, None, None


def get_rows_for_label(rows, label, duration_s):
    """
    Gets rows from the final SUMMARY_LAST_S seconds of an event.
    """
    selected = []
    cutoff = max(0.0, duration_s - SUMMARY_LAST_S)

    for row in rows:
        if row["label"] == label and row["phase_elapsed_s"] >= cutoff:
            selected.append(row)

    return selected


def mean_normalized_for_event(rows, label, duration_s, r0):
    selected = get_rows_for_label(rows, label, duration_s)

    if len(selected) == 0:
        return np.full(N_CHANNELS, np.nan, dtype=float)

    values = []

    for row in selected:
        values.append(calculate_normalized(row["resistance"], r0))

    values = np.array(values, dtype=float)
    return np.nanmean(values, axis=0)


def std_normalized_for_event(rows, label, duration_s, r0):
    selected = get_rows_for_label(rows, label, duration_s)

    if len(selected) == 0:
        return np.full(N_CHANNELS, np.nan, dtype=float)

    values = []

    for row in selected:
        values.append(calculate_normalized(row["resistance"], r0))

    values = np.array(values, dtype=float)
    return np.nanstd(values, axis=0)


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


def print_r0(r0):
    print()
    print("Baseline R0 from final part of initial open-hand baseline:")
    for name, value in zip(CHANNELS, r0):
        if np.isfinite(value):
            print(f"  {name}: {value:.2f} ohm")
        else:
            print(f"  {name}: INVALID")


# ============================================================
# LIVE PLOT FUNCTIONS
# ============================================================

def setup_live_plot():
    if not SHOW_LIVE_PLOT:
        return None, None

    plt.ion()
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    try:
        fig.canvas.manager.set_window_title("Cyclic Finger Bending - Live")
    except Exception:
        pass

    plt.show(block=False)
    return fig, ax


def update_live_plot(
    fig,
    ax,
    time_history,
    resistance_history,
    normalized_history,
    r0,
    current_event,
    phase_remaining_s,
):
    if fig is None or ax is None:
        return

    if len(time_history) < 2:
        return

    ax.clear()

    x = np.array(time_history, dtype=float)

    target_finger = current_event["target_finger"]
    intended_idx = INTENDED_CHANNEL.get(target_finger, None)

    if r0 is None:
        data = np.array(resistance_history, dtype=float)

        for i, ch in enumerate(CHANNELS):
            lw = 2.5 if i == intended_idx else 1.0
            ax.plot(x, data[:, i], label=ch, linewidth=lw)

        ax.set_ylabel("Resistance (ohm)")
        ax.set_title(
            f"{current_event['instruction']}\n"
            f"{phase_remaining_s:.1f} s left | collecting baseline"
        )

    else:
        data = np.array(normalized_history, dtype=float)

        for i, ch in enumerate(CHANNELS):
            lw = 2.8 if i == intended_idx else 1.0
            ax.plot(x, data[:, i], label=ch, linewidth=lw)

        ax.axhline(0, linewidth=1)
        ax.set_ylabel("Delta R / R0")
        ax.set_title(
            f"{current_event['instruction']}\n"
            f"{phase_remaining_s:.1f} s left | intended: {target_finger}"
        )

        if USE_FIXED_YLIM:
            ax.set_ylim(*NORMALIZED_YLIM)

    ax.set_xlabel("Time in current view (s)")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=7, ncol=1)

    fig.tight_layout()

    try:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)
    except Exception:
        pass


# ============================================================
# CSV SAVE FUNCTIONS
# ============================================================

def build_full_csv_header():
    header = [
        "wall_time_iso",
        "esp32_t_ms",
        "protocol_elapsed_s",
        "event_index",
        "label",
        "target_finger",
        "phase",
        "cycle",
        "phase_elapsed_s",
        "instruction",
    ]

    for ch in CHANNELS:
        header.append(f"{ch}_raw_ADC")

    for ch in CHANNELS:
        header.append(f"{ch}_R_ohm")

    for ch in CHANNELS:
        header.append(f"{ch}_dR_over_R0")

    return header


def save_full_csv(rows, r0, timestamp, run_tag):
    filename = f"glove_cyclic_finger_full_{run_tag}_{timestamp}.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(build_full_csv_header())

        for row in rows:
            resistance = row["resistance"]
            normalized = calculate_normalized(resistance, r0)

            line = [
                row["wall_time_iso"],
                row["esp32_t_ms"],
                f"{row['protocol_elapsed_s']:.4f}",
                row["event_index"],
                row["label"],
                row["target_finger"],
                row["phase"],
                row["cycle"],
                f"{row['phase_elapsed_s']:.4f}",
                row["instruction"],
            ]

            line.extend(row["adc_values"])

            for value in resistance:
                line.append(f"{value:.6f}" if np.isfinite(value) else "nan")

            for value in normalized:
                line.append(f"{value:.6f}" if np.isfinite(value) else "nan")

            writer.writerow(line)

    print(f"Saved full cyclic data CSV: {filename}")
    return filename


def save_cycle_summary_csv(rows, r0, events, timestamp, run_tag):
    """
    For each finger and cycle:
    local response = bend final 1 s - preceding open final 1 s.
    """
    filename = f"glove_cyclic_finger_cycle_summary_{run_tag}_{timestamp}.csv"

    header = [
        "target_finger",
        "cycle",
        "open_label",
        "bend_label",
        "used_last_seconds",
        "intended_sensor",
        "intended_local_response",
    ]

    for ch in CHANNELS:
        header.append(f"{ch}_local_mean_dR_over_R0")

    for ch in CHANNELS:
        header.append(f"{ch}_bend_std_dR_over_R0")

    event_by_label = {event["label"]: event for event in events}

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for finger in FINGER_ORDER:
            intended_idx = INTENDED_CHANNEL[finger]
            intended_sensor = CHANNELS[intended_idx]

            for cycle in range(1, N_CYCLES + 1):
                open_label = f"{finger}_C{cycle}_Open"
                bend_label = f"{finger}_C{cycle}_Bend"

                open_event = event_by_label[open_label]
                bend_event = event_by_label[bend_label]

                open_mean = mean_normalized_for_event(
                    rows,
                    open_label,
                    open_event["duration_s"],
                    r0,
                )

                bend_mean = mean_normalized_for_event(
                    rows,
                    bend_label,
                    bend_event["duration_s"],
                    r0,
                )

                bend_std = std_normalized_for_event(
                    rows,
                    bend_label,
                    bend_event["duration_s"],
                    r0,
                )

                local = bend_mean - open_mean
                intended_value = local[intended_idx]

                line = [
                    finger,
                    cycle,
                    open_label,
                    bend_label,
                    SUMMARY_LAST_S,
                    intended_sensor,
                    f"{intended_value:.6f}" if np.isfinite(intended_value) else "nan",
                ]

                for value in local:
                    line.append(f"{value:.6f}" if np.isfinite(value) else "nan")

                for value in bend_std:
                    line.append(f"{value:.6f}" if np.isfinite(value) else "nan")

                writer.writerow(line)

    print(f"Saved cycle summary CSV: {filename}")
    return filename


def save_finger_summary_csv(cycle_summary_file, timestamp, run_tag):
    """
    Reads cycle summary and averages local responses across 5 cycles.
    """
    filename = f"glove_cyclic_finger_summary_{run_tag}_{timestamp}.csv"

    rows = []

    with open(cycle_summary_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    header = [
        "target_finger",
        "n_cycles",
        "intended_sensor",
        "intended_mean",
        "intended_sd",
    ]

    for ch in CHANNELS:
        header.append(f"{ch}_mean_local_dR_over_R0")

    for ch in CHANNELS:
        header.append(f"{ch}_sd_local_dR_over_R0")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for finger in FINGER_ORDER:
            finger_rows = [row for row in rows if row["target_finger"] == finger]

            intended_idx = INTENDED_CHANNEL[finger]
            intended_sensor = CHANNELS[intended_idx]

            all_local = []

            for row in finger_rows:
                vals = []

                for ch in CHANNELS:
                    key = f"{ch}_local_mean_dR_over_R0"
                    vals.append(float(row[key]))

                all_local.append(vals)

            all_local = np.array(all_local, dtype=float)

            means = np.nanmean(all_local, axis=0)
            sds = np.nanstd(all_local, axis=0)

            intended_values = all_local[:, intended_idx]
            intended_mean = np.nanmean(intended_values)
            intended_sd = np.nanstd(intended_values)

            line = [
                finger,
                len(finger_rows),
                intended_sensor,
                f"{intended_mean:.6f}",
                f"{intended_sd:.6f}",
            ]

            for value in means:
                line.append(f"{value:.6f}" if np.isfinite(value) else "nan")

            for value in sds:
                line.append(f"{value:.6f}" if np.isfinite(value) else "nan")

            writer.writerow(line)

    print(f"Saved finger summary CSV: {filename}")
    return filename


# ============================================================
# MAIN PROGRAM
# ============================================================

async def main():
    events = build_protocol_events()
    total_duration_s = events[-1]["end_s"]

    device = await find_ble_device()

    if device is None:
        return

    print()
    print("Sensor order:")
    for i, name in enumerate(CHANNELS, start=1):
        print(f"  Channel {i}: {name}")

    print()
    print("Cyclic finger-bending protocol:")
    print(f"  Initial baseline: {BASELINE_S:.1f} s")
    print(f"  Fingers: {', '.join(FINGER_ORDER)}")
    print(f"  Cycles per finger: {N_CYCLES}")
    print(f"  Each cycle: open {OPEN_HOLD_S:.1f} s + bend {BEND_HOLD_S:.1f} s")
    print(f"  Summary uses final {SUMMARY_LAST_S:.1f} s of each phase")
    print(f"  Total duration: {total_duration_s:.1f} s")
    print()

    run_tag = input("Enter run label, for example cyclic_run1, then press ENTER: ")
    run_tag = sanitize_filename_text(run_tag)

    if not run_tag:
        run_tag = "cyclic"

    print()
    print("Live plot is enabled.")
    print("The intended sensor for the current finger is plotted thicker.")
    print("Follow the voice instructions.")
    print()

    input("Put the glove on, keep hand open, then press ENTER to start...")

    rows = []
    pending_samples = deque()
    line_buffer = ""

    time_history = deque(maxlen=PLOT_HISTORY_SAMPLES)
    resistance_history = deque(maxlen=PLOT_HISTORY_SAMPLES)
    normalized_history = deque(maxlen=PLOT_HISTORY_SAMPLES)

    r0 = None
    last_event_index = None
    last_countdown_print = 0.0
    last_plot_time = 0.0

    fig, ax = setup_live_plot()

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
        print("Starting cyclic protocol now.")
        print("\a")
        speak("Starting cyclic finger test. Open hand.")

        protocol_start_wall = time.time()

        while True:
            await asyncio.sleep(0.01)

            now = time.time()
            elapsed_s = now - protocol_start_wall

            if elapsed_s >= total_duration_s:
                break

            event, phase_elapsed_s, phase_remaining_s = get_current_event(events, elapsed_s)

            if event is None:
                continue

            if event["event_index"] != last_event_index:
                last_event_index = event["event_index"]

                print()
                print("=" * 80)
                print(f"EVENT {event['event_index'] + 1}/{len(events)}")
                print(f"Instruction: {event['instruction']}")
                print(f"Label:       {event['label']}")
                print(f"Finger:      {event['target_finger']}")
                print(f"Phase:       {event['phase']}")
                print(f"Cycle:       {event['cycle']}")
                print("=" * 80)
                print("\a")

                speak(event["voice"])

            if now - last_countdown_print >= 1.0:
                last_countdown_print = now
                print(
                    f"{event['instruction']} | "
                    f"{phase_remaining_s:4.1f} s left | "
                    f"total {elapsed_s:6.1f}/{total_duration_s:.1f} s"
                )

            while pending_samples:
                esp32_t_ms, adc_values = pending_samples.popleft()

                sample_wall = time.time()
                sample_elapsed_s = sample_wall - protocol_start_wall

                if sample_elapsed_s >= total_duration_s:
                    continue

                sample_event, sample_phase_elapsed_s, _ = get_current_event(events, sample_elapsed_s)

                if sample_event is None:
                    continue

                resistance = adc_to_resistance(adc_values)

                rows.append({
                    "wall_time_iso": datetime.now().isoformat(timespec="milliseconds"),
                    "esp32_t_ms": esp32_t_ms,
                    "protocol_elapsed_s": sample_elapsed_s,
                    "event_index": sample_event["event_index"],
                    "label": sample_event["label"],
                    "target_finger": sample_event["target_finger"],
                    "phase": sample_event["phase"],
                    "cycle": sample_event["cycle"],
                    "phase_elapsed_s": sample_phase_elapsed_s,
                    "instruction": sample_event["instruction"],
                    "adc_values": adc_values,
                    "resistance": resistance,
                })

                # Calculate R0 after initial baseline finishes.
                if r0 is None and sample_elapsed_s >= BASELINE_S:
                    baseline_rows = get_rows_for_label(
                        rows,
                        "Initial_Baseline",
                        BASELINE_S,
                    )

                    if len(baseline_rows) > 0:
                        baseline_resistances = np.array(
                            [row["resistance"] for row in baseline_rows],
                            dtype=float
                        )

                        r0 = np.nanmedian(baseline_resistances, axis=0)
                        print_r0(r0)
                        speak("Baseline ready.")

                if r0 is not None:
                    normalized = calculate_normalized(resistance, r0)
                else:
                    normalized = np.full(N_CHANNELS, np.nan, dtype=float)

                time_history.append(sample_elapsed_s)
                resistance_history.append(resistance)
                normalized_history.append(normalized)

            if (
                SHOW_LIVE_PLOT
                and len(time_history) > 2
                and now - last_plot_time >= PLOT_UPDATE_INTERVAL_S
            ):
                last_plot_time = now

                update_live_plot(
                    fig=fig,
                    ax=ax,
                    time_history=time_history,
                    resistance_history=resistance_history,
                    normalized_history=normalized_history,
                    r0=r0,
                    current_event=event,
                    phase_remaining_s=phase_remaining_s,
                )

        await client.stop_notify(TX_CHARACTERISTIC)

    print()
    print("Cyclic protocol finished.")
    print("\a")
    speak("Cyclic protocol finished.")

    if len(rows) == 0:
        print("No data collected.")
        return

    # Final R0 calculation using final SUMMARY_LAST_S of initial baseline.
    baseline_rows = get_rows_for_label(rows, "Initial_Baseline", BASELINE_S)

    if len(baseline_rows) == 0:
        print("Could not calculate R0: no baseline rows found.")
        return

    baseline_resistances = np.array(
        [row["resistance"] for row in baseline_rows],
        dtype=float
    )

    r0 = np.nanmedian(baseline_resistances, axis=0)

    print_r0(r0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    full_file = save_full_csv(rows, r0, timestamp, run_tag)
    cycle_summary_file = save_cycle_summary_csv(rows, r0, events, timestamp, run_tag)
    finger_summary_file = save_finger_summary_csv(cycle_summary_file, timestamp, run_tag)

    print()
    print("Done.")
    print("Upload these files for analysis:")
    print(f"  1. {full_file}")
    print(f"  2. {cycle_summary_file}")
    print(f"  3. {finger_summary_file}")


if __name__ == "__main__":
    asyncio.run(main())