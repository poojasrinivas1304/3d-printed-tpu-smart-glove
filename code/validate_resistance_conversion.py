#!/usr/bin/env python3
"""Check stored resistance columns against the documented divider equations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
CHANNELS = ["Thumb", "Index", "Middle", "Ring", "Little", "Pinky", "ReverseHorizontal", "ReverseThumb"]


def check_esp32_directory(relative_dir: str, fixed_resistor_ohm: float) -> tuple[int, float]:
    checked = 0
    largest_error = 0.0
    for path in sorted((REPOSITORY_DIR / relative_dir).glob("*.csv")):
        frame = pd.read_csv(path)
        for channel in CHANNELS:
            adc_col = f"{channel}_raw_ADC"
            resistance_col = f"{channel}_R_ohm"
            if adc_col not in frame or resistance_col not in frame:
                continue
            adc = pd.to_numeric(frame[adc_col], errors="coerce").to_numpy(dtype=float)
            stored = pd.to_numeric(frame[resistance_col], errors="coerce").to_numpy(dtype=float)
            expected = fixed_resistor_ohm * adc / (4095.0 - adc)
            valid = np.isfinite(adc) & np.isfinite(stored) & (adc > 0) & (adc < 3900)
            if valid.any():
                checked += int(valid.sum())
                largest_error = max(largest_error, float(np.max(np.abs(stored[valid] - expected[valid]))))
    return checked, largest_error


def check_arduino_directory() -> tuple[int, float]:
    checked = 0
    largest_error = 0.0
    for path in sorted((REPOSITORY_DIR / "data" / "cyclic_compression").glob("TPU*.csv")):
        frame = pd.read_csv(path)
        adc = pd.to_numeric(frame["adc"], errors="coerce").to_numpy(dtype=float)
        stored = pd.to_numeric(frame["R_ohm"], errors="coerce").to_numpy(dtype=float)
        expected = 7500.0 * adc / (1023.0 - adc)
        valid = np.isfinite(adc) & np.isfinite(stored) & (adc > 0) & (adc < 1023)
        checked += int(valid.sum())
        largest_error = max(largest_error, float(np.max(np.abs(stored[valid] - expected[valid]))))
    return checked, largest_error


def main() -> None:
    checks = [
        ("ESP32 single-finger files", "data/single_finger", 3400.0),
        ("ESP32 cyclic-finger files", "data/cyclic_finger", 3400.0),
        ("ESP32 sequential-posture files", "data/posture_response", 3500.0),
    ]
    for label, directory, fixed_resistor in checks:
        count, error = check_esp32_directory(directory, fixed_resistor)
        print(f"{label}: {count} values checked; Rf={fixed_resistor:.0f} ohm; max absolute rounding error={error:.6g} ohm")
    count, error = check_arduino_directory()
    print(f"Arduino cyclic-compression files: {count} values checked; Rf=7500 ohm; max absolute rounding error={error:.6g} ohm")
    print("This verifies computational consistency only; it is not an independent ADC or resistor calibration.")


if __name__ == "__main__":
    main()
