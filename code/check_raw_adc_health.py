import asyncio
import time
import platform
import subprocess
from collections import deque

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
# Must match Arduino:
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

N_CHANNELS = len(CHANNELS)


# ============================================================
# ADC HEALTH LIMITS
# ============================================================

ADC_WARN = 3600       # suspicious high resistance/contact movement
ADC_BAD = 3900        # likely open-contact artifact
ADC_NEAR_OPEN = 4050  # almost open circuit

PRINT_INTERVAL_S = 0.35
VOICE_COOLDOWN_S = 3.0

VOICE_ENABLED = True


# ============================================================
# VOICE
# ============================================================

_last_voice_time = 0


def speak(text):
    global _last_voice_time

    if not VOICE_ENABLED:
        return

    now = time.time()
    if now - _last_voice_time < VOICE_COOLDOWN_S:
        return

    _last_voice_time = now

    print("VOICE:", text)

    if platform.system() == "Darwin":
        try:
            subprocess.Popen(
                ["say", text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
    else:
        print("\a")


# ============================================================
# PARSING
# ============================================================

def parse_line(line):
    """
    Expected ESP32 line:
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


def classify_adc(value):
    if value >= ADC_NEAR_OPEN:
        return "NEAR_OPEN"
    if value >= ADC_BAD:
        return "BAD_HIGH"
    if value >= ADC_WARN:
        return "WARN_HIGH"
    if value <= 20:
        return "SHORT/GND"
    return "OK"


def format_adc_line(adc_values):
    parts = []

    for name, adc in zip(CHANNELS, adc_values):
        state = classify_adc(adc)

        if state == "OK":
            parts.append(f"{name}={adc}")
        else:
            parts.append(f"{name}={adc}({state})")

    return " | ".join(parts)


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
        print()
        print("Could not find expected ESP32 BLE device.")
        print("Expected one of:")
        for name in DEVICE_NAMES:
            print("  ", name)
        return None

    print()
    print(f"Selected device: {selected.name}  {selected.address}")
    return selected


async def main():
    device = await find_ble_device()

    if device is None:
        return

    line_buffer = ""
    pending_samples = deque()

    high_counts = {name: 0 for name in CHANNELS}
    bad_counts = {name: 0 for name in CHANNELS}
    near_open_counts = {name: 0 for name in CHANNELS}

    last_print_time = 0
    start_time = time.time()

    def notification_handler(sender, data):
        nonlocal line_buffer

        chunk = data.decode("utf-8", errors="ignore")
        line_buffer += chunk

        while "\n" in line_buffer:
            line, line_buffer = line_buffer.split("\n", 1)
            parsed = parse_line(line)

            if parsed is not None:
                pending_samples.append(parsed)

    print()
    print("Starting raw ADC health check.")
    print("Move/wiggle the glove gently and watch for values above 3900.")
    print("Press Ctrl+C to stop.")
    print()

    async with BleakClient(device) as client:
        print("Connected.")
        await client.start_notify(TX_CHARACTERISTIC, notification_handler)

        try:
            while True:
                await asyncio.sleep(0.03)

                while pending_samples:
                    t_ms, adc_values = pending_samples.popleft()

                    for name, adc in zip(CHANNELS, adc_values):
                        if adc >= ADC_WARN:
                            high_counts[name] += 1
                        if adc >= ADC_BAD:
                            bad_counts[name] += 1
                        if adc >= ADC_NEAR_OPEN:
                            near_open_counts[name] += 1

                    bad_channels = [
                        name for name, adc in zip(CHANNELS, adc_values)
                        if adc >= ADC_BAD
                    ]

                    near_open_channels = [
                        name for name, adc in zip(CHANNELS, adc_values)
                        if adc >= ADC_NEAR_OPEN
                    ]

                    if near_open_channels:
                        speak("Near open contact: " + ", ".join(near_open_channels))
                    elif bad_channels:
                        speak("High ADC: " + ", ".join(bad_channels))

                    now = time.time()
                    if now - last_print_time >= PRINT_INTERVAL_S:
                        last_print_time = now

                        elapsed = now - start_time
                        print(f"t={elapsed:6.1f}s | {format_adc_line(adc_values)}")

        except KeyboardInterrupt:
            print()
            print("Stopped by user.")

        await client.stop_notify(TX_CHARACTERISTIC)

    print()
    print("=" * 80)
    print("ADC health summary")
    print("=" * 80)

    for name in CHANNELS:
        print(
            f"{name:18s} | "
            f">=3600: {high_counts[name]:5d} | "
            f">=3900: {bad_counts[name]:5d} | "
            f">=4050: {near_open_counts[name]:5d}"
        )

    print()
    print("Interpretation:")
    print("  >=3600  : suspicious / watch this channel")
    print("  >=3900  : likely loose/open-contact artifact")
    print("  >=4050  : almost open circuit")
    print()
    print("For AI/demo runs, stop and fix the connection if any important channel")
    print("repeatedly reaches 3900–4095 during open hand.")


if __name__ == "__main__":
    asyncio.run(main())