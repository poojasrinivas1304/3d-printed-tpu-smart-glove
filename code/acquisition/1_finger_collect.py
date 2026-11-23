#!/usr/bin/env python3
"""Participant 1 same-sitting folded-finger calibration program.

This file contains its own BLE, quality-control, feature-extraction, and file-output logic.
It does not require _glove_core.py or any of the other glove programs.

Modes
-----
check      Hardware check only.
calibrate  Hardware check -> preconditioning -> balanced randomized calibration trials.
train      Cross-validation on one calibration features.csv, then save a frozen model bundle.
blind      Load the frozen model and run balanced randomized blind trials. The target is used
           to cue the participant, but is not supplied to the classifier; prediction is written
           before the target is scored.

Examples
--------
python finger_collect.py --mode calibrate --participant P01 --session F_CAL
python finger_ai.py --mode train --input glove_data/P01/.../features.csv \
        --model-out models/P01_finger.joblib
python finger_ai.py --mode blind --participant P01 --session F_BLIND \
        --model models/P01_finger.joblib
python posture_collect.py --mode calibrate --participant P01 --session P_CAL

Output (one folder per session)
-------------------------------
meta.json            provenance: arguments, Rf values, thresholds, versions, script hash, model hash
sequence.csv         planned trial order (written before the first trial)
samples.csv          EVERY raw ADC sample (incl. preconditioning, rejected attempts), tagged by phase
attempts.csv         one row per attempt, accepted or rejected, with rejection reasons
features.csv         one row per accepted trial (feature set v1 = the 56 manuscript features,
                     followed by the extra v2 features)
blind_predictions.csv  (blind mode) probabilities, decision, forced-choice label, model hash
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_VERSION = "3.0.5-p01-finger-cal-same-sitting"

# ============================================================
# HARDWARE / BLE (taken from your existing script)
# ============================================================
# This seven-channel collector must never auto-select the 10-channel T-shirt firmware.
DEVICE_NAMES = ["GLOVE_01", "Glove_BLE_7_RAW"]
PREFERRED_NOTIFY_CHARACTERISTIC = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

# Must match the ESP32: analogPins = {36, 39, 34, 35, 32, 33, 25}
CHANNELS = [
    "Little_VP_GPIO36",
    "Ring_VN_GPIO39",
    "Middle_D34_GPIO34",
    "Index_D35_GPIO35",
    "Thumb_D32_GPIO32",
    "FrontHorizontal_D33_GPIO33",
    "FrontThumb_D25_GPIO25",
]
N_CH = len(CHANNELS)
SHORT = ["Li", "Ri", "Mi", "In", "Th", "FH", "FT"]

FINGER_CLASSES = ["Thumb", "Index", "Middle", "Ring", "Little"]
FINGER_TO_CHANNEL = {"Thumb": 4, "Index": 3, "Middle": 2, "Ring": 1, "Little": 0}
POSTURE_CLASSES = ["Open", "Fist", "IndexPoint", "ThumbUp", "Pinch"]
TASK_CLASSES = {"finger": FINGER_CLASSES, "posture": POSTURE_CLASSES}

TARGET_CUES = {
    "Thumb": "Fold your thumb. Hold it.",
    "Index": "Fold your index finger. Hold it.",
    "Middle": "Fold your middle finger. Hold it.",
    "Ring": "Fold your ring finger. Hold it.",
    "Little": "Fold your little finger. Hold it.",
    "Open": "Keep your hand open and still.",
    "Fist": "Make a fist and hold it.",
    "IndexPoint": "Make an index-point posture and hold it.",
    "ThumbUp": "Make a thumb-up posture and hold it.",
    "Pinch": "Touch your thumb and index fingertips in a pinch and hold it.",
}

ADC_MAX = 4095.0
HIGH_ADC = 3900   # near-open-contact / saturation (same threshold as the manuscript)
LOW_ADC = 20      # shorted / floating

# Heuristic limits used only by the hardware check (adjust if you like)
CHECK_REST_LOW = 500
CHECK_REST_HIGH = 3500
CHECK_NOISE_STD = 25.0

# ============================================================
# FEATURES
# v1 = the manuscript features (8 statistics x 7 channels, channel-major)
# v2 = v1 + peak / time-to-peak / onset slope per channel + 4 neighbour differences
# ============================================================
FEATURE_STATS = ["mean", "median", "std", "min", "max", "range", "last", "slope"]
N_V1 = N_CH * len(FEATURE_STATS)


def feature_names():
    names = [f"{ch}__{s}" for ch in CHANNELS for s in FEATURE_STATS]
    names += [f"{ch}__{s}" for ch in CHANNELS for s in ("peak", "t_peak", "onset_slope")]
    names += [f"diff_{SHORT[i]}_{SHORT[i + 1]}__mean" for i in range(4)]
    return names


ALL_FEATURE_NAMES = feature_names()


def adc_to_resistance(adc, rf):
    """Voltage divider: 3.3 V - Rf - ADC node - sensor - GND  ->  Rs = Rf * ADC / (4095 - ADC)."""
    adc = np.clip(np.asarray(adc, dtype=float), 1.0, ADC_MAX - 1.0)
    return rf * adc / (ADC_MAX - adc)


def _slope(t, y):
    if len(t) < 2 or float(np.ptp(t)) == 0.0:
        return 0.0
    return float(np.polyfit(t, y, 1)[0])


def extract_features(ref_adc, fold_adc, fold_t, full_adc, full_t, rf):
    """
    ref_adc   (n,7) raw ADC of the open-hand reference window (baseline)
    fold_adc  (m,7) raw ADC of the stable folded window; fold_t its time stamps (s)
    full_adc  (k,7) raw ADC of the whole folded phase; full_t its time stamps (s)
    Trial-specific open-hand reference: x = (R - R_open) / R_open   (manuscript Eq. 3)
    Returns the full v2 vector; the first N_V1 entries are the manuscript (v1) features.
    """
    r_open = np.median(adc_to_resistance(ref_adc, rf), axis=0)
    x = (adc_to_resistance(fold_adc, rf) - r_open) / r_open
    feats = []
    for i in range(N_CH):
        v = x[:, i]
        feats += [v.mean(), np.median(v), v.std(), v.min(), v.max(),
                  float(np.ptp(v)), v[-1], _slope(fold_t, v)]
    xf = (adc_to_resistance(full_adc, rf) - r_open) / r_open
    early = full_t <= 2.0
    for i in range(N_CH):
        v = xf[:, i]
        k = int(np.argmax(np.abs(v)))
        feats += [v[k], full_t[k], _slope(full_t[early], v[early])]
    means = x.mean(axis=0)
    feats += [means[i] - means[i + 1] for i in range(4)]
    return np.array(feats, dtype=float)


# ============================================================
# SMALL HELPERS
# ============================================================
def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def parse_line(line):
    """Expected ESP32 line: t_ms,adc1,...,adc7"""
    line = line.strip()
    if not line or line.startswith("t_ms"):
        return None
    parts = line.split(",")
    if len(parts) != N_CH + 1:
        return None
    try:
        return int(parts[0]), [int(x) for x in parts[1:]]
    except ValueError:
        return None


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


async def ask(prompt):
    # Tk live plotting on macOS must stay on the same main thread as terminal input.
    # Pausing BLE callbacks at operator prompts is safe because no trial phase is recording then.
    return input(prompt)


# ============================================================
# VOICE
# ============================================================
class Speaker:
    def __init__(self, enabled, rate):
        self.enabled, self.rate, self.proc = enabled, rate, None

    def say(self, text):
        print(f"  >> {text}")
        if not self.enabled:
            return
        if platform.system() == "Darwin":
            try:
                if self.proc is not None and self.proc.poll() is None:
                    self.proc.terminate()
                self.proc = subprocess.Popen(["say", "-r", str(self.rate), text],
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:  # voice must never stop a recording
                print("   (voice failed:", e, ")")
        else:
            print("\a", end="", flush=True)


class LivePlot:
    """Live diagnostic display of all seven raw ADC channels."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.plt = self.fig = self.ax = None
        self.lines = []
        if not enabled:
            return
        try:
            import matplotlib
            try:
                matplotlib.use("TkAgg")
            except Exception:
                pass
            import matplotlib.pyplot as plt
            self.plt = plt
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(9.0, 5.2))
            for name in SHORT:
                line, = self.ax.plot([], [], linewidth=1.5, label=name)
                self.lines.append(line)
            self.ax.axhline(HIGH_ADC, color="red", linestyle="--", linewidth=1,
                            label=f"high ADC ({HIGH_ADC})")
            self.ax.axhline(LOW_ADC, color="orange", linestyle="--", linewidth=1,
                            label=f"low ADC ({LOW_ADC})")
            self.ax.set_ylim(-50, 4145)
            self.ax.set_xlabel("Recent time (s)")
            self.ax.set_ylabel("Raw ADC")
            self.ax.grid(True, alpha=0.3)
            self.ax.legend(loc="upper left", fontsize=8, ncol=3)
            self.fig.tight_layout()
            self.fig.show()
        except Exception as e:
            self.enabled = False
            print(f"Live plot unavailable ({e}). Collection can still continue.")

    def update(self, rec):
        if not self.enabled or len(rec.live_t) < 2:
            return
        try:
            t = np.asarray(rec.live_t, dtype=float)
            adc = np.asarray(rec.live_adc, dtype=float)
            recent_t = t - t[-1]
            for i, line in enumerate(self.lines):
                line.set_data(recent_t, adc[:, i])
            self.ax.set_xlim(max(-15.0, float(recent_t[0])), 0.1)
            target = f" | target: {rec.target}" if rec.target else ""
            self.ax.set_title(f"Live glove signal | phase: {rec.phase}{target}")
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.plt.pause(0.001)
        except Exception:
            pass

    def close(self):
        if self.enabled and self.plt is not None and self.fig is not None:
            try:
                self.plt.close(self.fig)
            except Exception:
                pass


# ============================================================
# RECORDER  (receives samples from BLE or from the simulator)
# ============================================================
class Recorder:
    """
    Every sample is stored with: host arrival time, the current phase, time since phase start,
    the ESP32 timestamp, raw ADC counts, and the current target. Phase is assigned from the
    host arrival time, so BLE batching can blur phase edges by a few tens of ms; the analysis
    windows use only the final seconds of each phase, away from the edges.
    """

    def __init__(self):
        self.t0 = time.perf_counter()
        self.rows = []
        self.recording = False
        self.phase = "idle"
        self.phase_t0 = self.t0
        self.target = ""
        self.disconnected = False
        self.bad_lines = 0
        self.bad_line_examples = []
        self._buf = ""
        self.live_t = deque(maxlen=300)
        self.live_adc = deque(maxlen=300)

    def set_phase(self, name):
        self.phase = name
        self.phase_t0 = time.perf_counter()

    def begin(self):
        self.rows = []
        self.recording = True

    def end(self):
        self.recording = False
        rows, self.rows = self.rows, []
        return rows

    def feed(self, t_ms, adc):
        now = time.perf_counter()
        self.live_t.append(now - self.t0)
        self.live_adc.append(tuple(adc))
        if not self.recording:
            return
        self.rows.append((now - self.t0, self.phase, now - self.phase_t0,
                          t_ms, tuple(adc), self.target))

    def on_data(self, _sender, data):
        self._buf += bytes(data).decode("utf-8", errors="ignore")
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            parsed = parse_line(line)
            if parsed is None:
                if line.strip() and not line.startswith("t_ms"):
                    self.bad_lines += 1
                    if len(self.bad_line_examples) < 3:
                        self.bad_line_examples.append(line.strip()[:180])
                continue
            self.feed(*parsed)


# ============================================================
# LINKS: real BLE and a simulator for dry runs
# ============================================================
class RealLink:
    def __init__(self, rec, cfg):
        self.rec, self.cfg = rec, cfg
        self.client = None
        self.notify_uuid = None
        self.info = {}
        self.notifications_paused = False

    def _on_disc(self, _client):
        self.rec.disconnected = True

    async def _connect(self):
        await self.client.connect()
        self.rec._buf = ""
        candidates = []
        preferred = PREFERRED_NOTIFY_CHARACTERISTIC.lower()
        for service in self.client.services:
            for char in service.characteristics:
                properties = {str(p).lower() for p in char.properties}
                if "notify" in properties or "indicate" in properties:
                    candidates.append(char.uuid)
        if not candidates:
            raise RuntimeError("The connected glove exposes no notify/indicate BLE characteristic")
        self.notify_uuid = next((u for u in candidates if u.lower() == preferred), candidates[0])
        print(f"Using BLE notify characteristic: {self.notify_uuid}")
        await self.client.start_notify(self.notify_uuid, self.rec.on_data)
        self.notifications_paused = False
        self.rec.disconnected = False

    async def open(self):
        from bleak import BleakClient, BleakScanner
        target = self.cfg.address
        if target is None:
            print("Scanning for ESP32 BLE device...")
            found = None
            for d in await BleakScanner.discover(timeout=8.0):
                name = d.name or "None"
                print(f"  Found: {name}  {d.address}")
                if name.upper() in {expected.upper() for expected in DEVICE_NAMES}:
                    found = d
                    break
            if found is None:
                raise SystemExit("Could not find the glove. Expected one of: " + ", ".join(DEVICE_NAMES))
            self.info = {"name": found.name, "address": found.address}
            target = found
        else:
            self.info = {"name": None, "address": target}
        self.client = BleakClient(target, disconnected_callback=self._on_disc)
        await self._connect()
        self.info["notify_characteristic"] = self.notify_uuid
        print(f"Connected: {self.info}")

    async def ensure(self):
        if self.client.is_connected and not self.rec.disconnected:
            return True
        for _ in range(3):
            try:
                print("BLE link lost - reconnecting...")
                await self._connect()
                return True
            except Exception as e:
                print("  reconnect failed:", e)
                await asyncio.sleep(2.0)
        return False

    async def pause_notifications(self):
        """Unsubscribe before a blocking terminal prompt so CoreBluetooth cannot build a backlog."""
        if (self.client is not None and self.client.is_connected and self.notify_uuid is not None
                and not self.notifications_paused):
            await self.client.stop_notify(self.notify_uuid)
            self.notifications_paused = True
        self.rec._buf = ""

    async def resume_notifications(self):
        """Resume with a clean line buffer; samples received while paused were intentionally dropped."""
        self.rec._buf = ""
        if (self.client is not None and self.client.is_connected and self.notify_uuid is not None
                and self.notifications_paused):
            await self.client.start_notify(self.notify_uuid, self.rec.on_data)
            self.notifications_paused = False
            self.rec.disconnected = False

    async def close(self):
        try:
            if self.client is not None and self.client.is_connected:
                if self.notify_uuid is not None:
                    await self.client.stop_notify(self.notify_uuid)
                await self.client.disconnect()
        except Exception:
            pass


SIM_GAIN = np.array([0.60, 0.50, 0.25, 0.30, 0.25])
SIM_GAIN_BY_CHANNEL = SIM_GAIN[[4, 3, 2, 1, 0]]


class SimLink:
    """Fake glove for testing the protocol, voice prompts and file outputs without hardware."""

    def __init__(self, rec, cfg):
        self.rec, self.cfg = rec, cfg
        self.task = None
        self.notifications_paused = False
        self.info = {"name": "SIMULATED", "address": None}

    async def open(self):
        print("*** SIMULATION MODE - no real glove is connected ***")
        self.task = asyncio.create_task(self._run())

    async def _run(self):
        rng = np.random.default_rng(self.cfg.seed)
        rf = self.cfg.rf
        base = rf * rng.uniform(0.7, 1.4, N_CH)
        t_ms, period = 0, 1.0 / self.cfg.rate_hz
        while True:
            await asyncio.sleep(period)
            t_ms += int(period * 1000)
            r = base * (1 + 0.02 * math.sin(t_ms / 8000.0))
            bend = np.zeros(N_CH)
            ramp = min(1.0, (time.perf_counter() - self.rec.phase_t0) / 0.6)
            if self.rec.phase == "check_fist":
                bend[:5] = SIM_GAIN_BY_CHANNEL * ramp
            elif self.rec.phase in ("target", "precond_bend"):
                if self.rec.target in FINGER_CLASSES:
                    finger_i = FINGER_CLASSES.index(self.rec.target)
                    k = FINGER_TO_CHANNEL[self.rec.target]
                    g = SIM_GAIN[finger_i] * ramp
                    bend[k] += g
                    for nb in (k - 1, k + 1):
                        if 0 <= nb < 5:
                            bend[nb] += 0.35 * g
                    bend[5] += 0.1 * g
                    bend[6] += (0.1 if self.rec.target == "Thumb" else 0.03) * g
                elif self.rec.target == "Fist":
                    bend[:5] = SIM_GAIN_BY_CHANNEL * ramp
                    bend[5:] = [0.10 * ramp, 0.08 * ramp]
                elif self.rec.target == "IndexPoint":
                    bend[:5] = SIM_GAIN_BY_CHANNEL * np.array([1.0, 1.0, 1.0, 0.05, 0.7]) * ramp
                    bend[5:] = [0.08 * ramp, 0.05 * ramp]
                elif self.rec.target == "ThumbUp":
                    bend[:5] = SIM_GAIN_BY_CHANNEL * np.array([1.0, 1.0, 1.0, 1.0, 0.05]) * ramp
                    bend[5:] = [-0.05 * ramp, 0.02 * ramp]
                elif self.rec.target == "Pinch":
                    bend[:5] = SIM_GAIN_BY_CHANNEL * np.array([0.05, 0.05, 0.05, 0.45, 0.45]) * ramp
                    bend[5:] = [-0.04 * ramp, 0.12 * ramp]
            r = r * (1 + bend)
            adc = ADC_MAX * r / (rf + r) + rng.normal(0, 6, N_CH)
            glitch = rng.random(N_CH) < self.cfg.sim_glitch_prob
            adc[glitch] = ADC_MAX
            if not self.notifications_paused:
                self.rec.feed(t_ms, np.clip(np.round(adc), 0, 4095).astype(int).tolist())

    async def ensure(self):
        return True

    async def pause_notifications(self):
        self.notifications_paused = True
        self.rec._buf = ""

    async def resume_notifications(self):
        self.rec._buf = ""
        self.notifications_paused = False

    async def close(self):
        if self.task:
            self.task.cancel()


# ============================================================
# OUTPUT FILES
# ============================================================
class Files:
    def __init__(self, folder: Path, mode: str, classes):
        folder.mkdir(parents=True, exist_ok=True)
        self.dir = folder
        self._fh = []
        self.samples = self._open("samples.csv",
                                  ["attempt_id", "trial_id", "target", "phase", "host_t_s", "phase_t_s",
                                   "esp32_t_ms"] + [f"ADC_{c}" for c in CHANNELS])
        self.attempts = self.features = self.predictions = None
        if mode in ("calibrate", "blind"):
            self.attempts = self._open("attempts.csv",
                                       ["attempt_id", "trial_id", "attempt_in_trial", "target", "accepted",
                                        "reject_reasons", "started_iso", "n_ref_window", "n_fold_window",
                                        "n_rows_total", "high_adc_total", "low_adc_total", "ref_cv_max",
                                        "mean_rate_hz", "ref_timing_ratio", "target_timing_ratio"])
            self.features = self._open("features.csv",
                                       ["trial_id", "attempt_id", "target"] + ALL_FEATURE_NAMES)
        if mode == "blind":
            self.predictions = self._open("blind_predictions.csv",
                                          ["trial_id", "attempt_id", "true_label", "forced_label", "decision",
                                           "accepted_by_rule", "correct_forced", "correct_decision",
                                           "top_prob", "margin"] + [f"p_{c}" for c in classes] +
                                          ["model_sha256", "min_prob_thr", "margin_thr", "feature_set",
                                           "predicted_at_iso"])

    def _open(self, name, header):
        fh = open(self.dir / name, "w", newline="")
        w = csv.writer(fh)
        w.writerow(header)
        self._fh.append(fh)
        return w

    def write_samples(self, attempt_id, trial_id, rows):
        for host_t, phase, phase_t, t_ms, adc, target in rows:
            self.samples.writerow([attempt_id, trial_id, target, phase, f"{host_t:.4f}",
                                   f"{phase_t:.4f}", t_ms, *adc])

    def flush(self):
        for fh in self._fh:
            fh.flush()

    def close(self):
        for fh in self._fh:
            fh.close()


def write_meta(folder, meta):
    (folder / "meta.json").write_text(json.dumps(meta, indent=2, default=json_default))


# ============================================================
# PHASES AND ANALYSIS
# ============================================================
async def run_phase(rec, name, duration, speaker=None, cue=None):
    rec.set_phase(name)
    if cue and speaker:
        speaker.say(cue)
    end = rec.phase_t0 + duration
    while True:
        remaining = end - time.perf_counter()
        if remaining <= 0:
            break
        await asyncio.sleep(min(0.05, remaining))


async def ask_with_stream_paused(link, rec, prompt, settle_s=0.35):
    """Pause BLE around synchronous input, then discard initial resubscription transients."""
    rec.recording = False
    rec.set_phase("idle")
    await link.pause_notifications()
    try:
        answer = input(prompt)
    finally:
        await link.resume_notifications()
        await asyncio.sleep(settle_s)
        rec._buf = ""
    return answer


async def update_live_plot(plotter, rec):
    while True:
        plotter.update(rec)
        await asyncio.sleep(0.10)


def phase_arrays(rows, phase):
    sel = [r for r in rows if r[1] == phase]
    if not sel:
        return np.empty((0, N_CH)), np.empty(0)
    return np.array([r[4] for r in sel], dtype=float), np.array([r[2] for r in sel], dtype=float)


def window_arrays(rows, phase, duration, win):
    adc, t = phase_arrays(rows, phase)
    m = t >= (duration - win)
    return adc[m], t[m]


def window_rows(rows, phase, duration, win):
    """Return complete records in the final analysis window of a phase."""
    return [r for r in rows if r[1] == phase and r[2] >= (duration - win)]


def timing_stats(records):
    """Compare elapsed ESP32 acquisition time with host arrival time for one analysis window."""
    if len(records) < 2:
        return float("nan"), float("nan")
    host_span = float(records[-1][0] - records[0][0])
    esp = np.asarray([r[3] for r in records], dtype=np.int64)
    delta = np.diff(esp)
    # ESP32 millis() wraps at 2**32; correct a wrap without hiding out-of-order packets.
    delta = np.where(delta < -(2 ** 31), delta + 2 ** 32, delta)
    if host_span <= 0 or np.any(delta <= 0):
        return float("nan"), float("nan")
    esp_span = float(delta.sum()) / 1000.0
    return esp_span / host_span, (len(records) - 1) / host_span


def analyse_attempt(rows, cfg):
    """Quality control + features for one attempt. Returns a dict; 'reasons' empty = accepted."""
    ref_adc, ref_t = window_arrays(rows, "ref", cfg.ref_s, cfg.ref_win_s)
    fold_adc, fold_t = window_arrays(rows, "target", cfg.fold_s, cfg.fold_win_s)
    full_adc, full_t = phase_arrays(rows, "target")
    ref_records = window_rows(rows, "ref", cfg.ref_s, cfg.ref_win_s)
    fold_records = window_rows(rows, "target", cfg.fold_s, cfg.fold_win_s)
    qc = [i for i in range(N_CH) if i not in cfg.qc_ignore_set]
    reasons = []

    for name, arr, records, win in (("ref", ref_adc, ref_records, cfg.ref_win_s),
                                    ("target", fold_adc, fold_records, cfg.fold_win_s)):
        need = cfg.min_sample_frac * cfg.rate_hz * win
        if len(arr) < need:
            reasons.append(f"too_few_samples_{name}({len(arr)}/{need:.0f})")
        maximum = cfg.max_sample_frac * cfg.rate_hz * win
        if len(arr) > maximum:
            reasons.append(f"too_many_samples_{name}({len(arr)}/{maximum:.0f}; queued BLE data?)")
        timing_ratio, _ = timing_stats(records)
        if len(records) >= 2 and (not np.isfinite(timing_ratio)
                                  or timing_ratio < cfg.min_timing_ratio
                                  or timing_ratio > cfg.max_timing_ratio):
            reasons.append(f"timing_mismatch_{name}(ESP/host={timing_ratio:.2f})")
        if len(arr):
            hi = (arr[:, qc] >= HIGH_ADC)
            if hi.sum() > cfg.high_adc_allow:
                chs = [SHORT[qc[j]] for j in np.where(hi.any(axis=0))[0]]
                reasons.append(f"high_adc_{name}({int(hi.sum())} samples; {'/'.join(chs)})")
            lo = (arr[:, qc] <= LOW_ADC)
            if lo.sum() > 0:
                chs = [SHORT[qc[j]] for j in np.where(lo.any(axis=0))[0]]
                reasons.append(f"low_adc_{name}({int(lo.sum())} samples; {'/'.join(chs)})")

    all_adc = np.array([r[4] for r in rows if r[1] in ("ref", "target")], dtype=float)
    hi_total = int((all_adc[:, qc] >= HIGH_ADC).sum()) if len(all_adc) else 0
    lo_total = int((all_adc[:, qc] <= LOW_ADC).sum()) if len(all_adc) else 0
    if len(all_adc) > 20:
        flat = [SHORT[i] for i in qc if float(np.ptp(all_adc[:, i])) == 0.0]
        if flat:
            reasons.append(f"flatline({'/'.join(flat)})")

    ref_cv = float("nan")
    if len(ref_adc) > 2:
        r = adc_to_resistance(ref_adc[:, qc], cfg.rf[qc])
        ref_cv = float(np.max(r.std(axis=0) / r.mean(axis=0)))
        if cfg.max_ref_cv and ref_cv > cfg.max_ref_cv:
            reasons.append(f"ref_unstable(cv={ref_cv:.3f})")

    ref_timing_ratio, ref_rate = timing_stats(ref_records)
    target_timing_ratio, target_rate = timing_stats(fold_records)
    valid_rates = [value for value in (ref_rate, target_rate) if np.isfinite(value)]
    rate = float(np.mean(valid_rates)) if valid_rates else float("nan")

    out = {"reasons": reasons, "n_ref": len(ref_adc), "n_fold": len(fold_adc), "n_rows": len(rows),
           "hi_total": hi_total, "lo_total": lo_total, "ref_cv": ref_cv, "rate": rate,
           "ref_timing_ratio": ref_timing_ratio, "target_timing_ratio": target_timing_ratio,
           "feats": None}
    if not reasons:
        out["feats"] = extract_features(ref_adc, fold_adc, fold_t, full_adc, full_t, cfg.rf)
    return out


# ============================================================
# HARDWARE CHECK
# ============================================================
async def hardware_check(rec, speaker, cfg, files):
    print("\n" + "=" * 72)
    print("HARDWARE CHECK  (keep the glove on)")
    print("=" * 72)
    rec.target = ""
    rec.begin()
    await run_phase(rec, "check_open", cfg.check_open_s, speaker, "Hardware check. Keep your hand open and still.")
    await run_phase(rec, "check_fist", cfg.check_fist_s, speaker, "Now make a fist and hold it.")
    rows = rec.end()
    files.write_samples(-1, "check", rows)
    files.flush()

    open_adc, open_t = phase_arrays(rows, "check_open")
    fist_adc, _ = phase_arrays(rows, "check_fist")
    open_adc = open_adc[open_t >= 0.5] if len(open_adc) else open_adc
    if len(open_adc) < 10 or len(fist_adc) < 5:
        print("Too few samples received - check the BLE link / ESP32 firmware.")
        print(f"Valid seven-channel samples: {len(rows)}; unparsed lines: {rec.bad_lines}")
        if rec.bad_line_examples:
            print("Received data did not match t_ms,adc1,...,adc7. Examples:")
            for example in rec.bad_line_examples:
                print("  ", example)
        return {"ok": False, "reason": "too_few_samples", "channels": []}

    t_ms = np.array([r[3] for r in rows], dtype=float)
    dt = np.diff(t_ms)
    dt = dt[dt > 0]
    med_dt = float(np.median(dt)) if len(dt) else float("nan")
    gaps = int((dt > 1.5 * med_dt).sum()) if len(dt) else 0
    host_rate = (len(rows) - 1) / (rows[-1][0] - rows[0][0])
    print(f"Packets: {len(rows)}  host rate {host_rate:.1f} Hz  ESP32 median period {med_dt:.0f} ms  "
          f"gaps (>1.5x period): {gaps}  unparsed lines: {rec.bad_lines}\n")

    print(f"{'Ch':<3}{'Sensor':<26}{'rest ADC':>9}{'noise':>7}{'rest R(ohm)':>12}"
          f"{'fist max':>9}{'sugg. Rf':>9}  status")
    result, n_flag = [], 0
    for i in range(N_CH):
        o, f = open_adc[:, i], fist_adc[:, i]
        med, sd = float(np.median(o)), float(o.std())
        r_rest = float(adc_to_resistance(med, cfg.rf[i]))
        fmax = float(f.max())
        issues = []
        if (o >= HIGH_ADC).any() or med > CHECK_REST_HIGH:
            issues.append("SATURATED AT REST")
        elif med < CHECK_REST_LOW:
            issues.append("LOW (short, or Rf too large)")
        if (o <= LOW_ADC).any():
            issues.append("LOW-ADC SAMPLES")
        if sd > CHECK_NOISE_STD:
            issues.append("NOISY (loose contact?)")
        if fmax >= HIGH_ADC:
            issues.append("HITS >=3900 ON FIST")
        sugg = int(round(r_rest / 100.0) * 100) if 0 < r_rest < 1e6 else 0
        n_flag += bool(issues)
        status = "OK" if not issues else "; ".join(issues)
        print(f"{i + 1:<3}{CHANNELS[i]:<26}{med:>9.0f}{sd:>7.1f}{r_rest:>12.0f}{fmax:>9.0f}{sugg:>9}  {status}")
        result.append({"channel": CHANNELS[i], "rest_adc_median": med, "rest_adc_std": sd,
                       "rest_R_ohm_nominal": r_rest, "fist_adc_max": fmax, "suggested_rf_ohm": sugg,
                       "issues": issues})

    print("\nHow to read this:")
    print(" - 'sugg. Rf' = the resting resistance, i.e. the Rf that would put the resting ADC near mid-scale.")
    print(" - SATURATED / HITS >=3900: first re-seat the contact (copper tape/wire joint) and re-run. If the")
    print("   contact is solid and the sensor genuinely rises more than ~10x, use a larger Rf for that channel.")
    print(" - NOISY: usually a loose or intermittent contact. Fix it before collecting data.")
    if gaps:
        print(f" - {gaps} packet gap(s): BLE dropouts. Move closer / reduce interference before collecting.")
    print(f"\n{n_flag} channel(s) flagged." if n_flag else "\nAll channels look healthy.")
    return {"ok": n_flag == 0 and gaps == 0, "channels": result, "host_rate_hz": host_rate,
            "esp32_median_period_ms": med_dt, "packet_gaps": gaps}


# ============================================================
# PRECONDITIONING
# ============================================================
async def preconditioning(rec, speaker, cfg, files):
    cycles = cfg.precond_cycles
    if cycles <= 0:
        return
    print("\n" + "=" * 72)
    print(f"PRECONDITIONING: {cycles} bend cycles per finger "
          f"({cycles * len(FINGER_CLASSES) * 2 * cfg.precond_phase_s / 60:.1f} min). "
          "Bend firmly on 'Bend', relax on 'Open'.")
    print("=" * 72)
    rec.begin()
    for finger in FINGER_CLASSES:
        rec.target = finger
        await run_phase(rec, "precond_prep", cfg.precond_phase_s, speaker, f"{finger}. Get ready.")
        for _ in range(cycles):
            await run_phase(rec, "precond_bend", cfg.precond_phase_s, speaker, "Bend")
            await run_phase(rec, "precond_open", cfg.precond_phase_s, speaker, "Open")
    rec.target = ""
    rows = rec.end()
    files.write_samples(0, "precond", rows)
    files.flush()
    print(f"Preconditioning done ({len(rows)} samples saved).")


# ============================================================
# TRIALS
# ============================================================
def make_sequence(classes, n_per_class, seed):
    """Balanced blocks (each class once per block, shuffled); no class twice in a row."""
    rng = random.Random(seed)
    seq = []
    for _ in range(n_per_class):
        while True:
            block = classes[:]
            rng.shuffle(block)
            if not seq or block[0] != seq[-1]:
                break
        seq += block
    return seq


def make_blind_sequence(classes, n_trials, seed):
    """Balanced randomized targets, truncated to the requested total trial count."""
    n_blocks = math.ceil(n_trials / len(classes))
    return make_sequence(classes, n_blocks, seed)[:n_trials]


async def run_attempt(rec, speaker, cfg, target):
    rec.target = target
    started = datetime.now().isoformat(timespec="seconds")
    rec.begin()
    await run_phase(rec, "ref", cfg.ref_s, speaker, "Open hand. Relax and keep still.")
    await run_phase(rec, "target", cfg.fold_s, speaker, TARGET_CUES[target])
    rows = rec.end()
    rec.set_phase("idle")
    return rows, started


def load_model(path, cfg):
    import joblib
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        fs = obj.get("feature_set", "v1")
        saved_task = obj.get("task")
        saved_channels = obj.get("channels")
        thresholds = obj.get("thresholds", {})
        if saved_task and saved_task != cfg.task:
            raise SystemExit(f"Model task {saved_task!r} does not match requested task {cfg.task!r}")
        if saved_channels and saved_channels != CHANNELS:
            raise SystemExit("Model channel order does not match this acquisition program")
    else:
        model, fs, thresholds = obj, cfg.feature_set or "v1", {}
    if cfg.feature_set and cfg.feature_set != fs:
        raise SystemExit(f"Model uses feature set {fs!r}, not requested {cfg.feature_set!r}")
    classes = [str(c) for c in model.classes_]
    if set(classes) != set(cfg.classes):
        raise SystemExit(f"Model classes {classes} do not match {cfg.classes}")
    min_prob = thresholds.get("min_prob", cfg.min_prob)
    margin = thresholds.get("margin", cfg.margin)
    if min_prob is None or margin is None:
        raise SystemExit("Legacy model needs explicit --min-prob and --margin")
    if cfg.min_prob is not None and "min_prob" in thresholds and cfg.min_prob != thresholds["min_prob"]:
        raise SystemExit("--min-prob differs from the threshold frozen in the model bundle")
    if cfg.margin is not None and "margin" in thresholds and cfg.margin != thresholds["margin"]:
        raise SystemExit("--margin differs from the threshold frozen in the model bundle")
    return model, classes, fs, sha256_file(path), float(min_prob), float(margin)


def train_model(cfg):
    """Cross-validate one calibration file and save a provenance-rich frozen model bundle."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline

    source = Path(cfg.input)
    if source.is_dir():
        source = source / "features.csv"
    if not source.is_file():
        raise SystemExit(f"Calibration feature file not found: {source}")
    selected = ALL_FEATURE_NAMES[:N_V1] if cfg.feature_set == "v1" else ALL_FEATURE_NAMES
    with source.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("Calibration feature file has no rows")
    missing = [name for name in selected if name not in rows[0]]
    if missing:
        raise SystemExit(f"Calibration file is missing feature columns: {missing[:3]}")
    y = np.array([row["target"] for row in rows])
    if set(y) != set(cfg.classes):
        raise SystemExit(f"Calibration labels {sorted(set(y))} do not match {cfg.classes}")
    x = np.array([[float(row[name]) if row[name] else np.nan for name in selected] for row in rows])
    counts = Counter(y)
    if min(counts.values()) < cfg.cv_folds:
        raise SystemExit(f"Need at least {cfg.cv_folds} trials per class; found {dict(counts)}")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=cfg.n_estimators, class_weight="balanced", max_features="sqrt",
            min_samples_leaf=1, random_state=cfg.model_seed, n_jobs=-1)),
    ])
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.model_seed)
    oof = cross_val_predict(pipeline, x, y, cv=cv, method="predict")
    pipeline.fit(x, y)

    model_out = Path(cfg.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": pipeline, "task": cfg.task, "classes": cfg.classes,
        "channels": CHANNELS, "feature_set": cfg.feature_set,
        "feature_names": selected, "script_version": SCRIPT_VERSION,
        "launcher": cfg.launcher, "launcher_sha256": cfg.launcher_sha256,
        "thresholds": {"min_prob": cfg.min_prob, "margin": cfg.margin},
        "training_source": str(source.resolve()), "training_source_sha256": sha256_file(source),
        "trained_iso": datetime.now().isoformat(timespec="seconds"),
        "cv_folds": cfg.cv_folds, "model_seed": cfg.model_seed,
        "n_estimators": cfg.n_estimators,
    }
    joblib.dump(bundle, model_out)
    labels = cfg.classes
    report = {
        "task": cfg.task, "feature_set": cfg.feature_set, "n_trials": len(y),
        "class_counts": dict(counts), "cv_folds": cfg.cv_folds,
        "oof_accuracy": float(accuracy_score(y, oof)),
        "classification_report": classification_report(y, oof, labels=labels, output_dict=True,
                                                       zero_division=0),
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(y, oof, labels=labels).tolist(),
        "thresholds_frozen_for_blind": bundle["thresholds"],
        "model_path": str(model_out.resolve()), "model_sha256": sha256_file(model_out),
        "training_source": str(source.resolve()),
        "training_source_sha256": bundle["training_source_sha256"],
    }
    report_path = model_out.with_suffix(".training_report.json")
    report_path.write_text(json.dumps(report, indent=2, default=json_default))
    print(f"OOF accuracy: {report['oof_accuracy']:.1%} ({len(y)} trials)")
    print(f"Frozen model: {model_out.resolve()}")
    print(f"Training report: {report_path.resolve()}")


def fmt_resp(feats):
    means = feats[0:N_V1:len(FEATURE_STATS)]
    return " ".join(f"{SHORT[i]}{means[i]:+.2f}" for i in range(N_CH))


async def main_async(cfg):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = cfg.participant or "NA"
    sess = f"{pid}_{cfg.session or 'NA'}_{cfg.mode}_{stamp}"
    folder = Path(cfg.out_dir) / pid / sess
    model = classes = fs = model_sha = None
    if cfg.mode == "blind":
        model, classes, fs, model_sha, cfg.min_prob, cfg.margin = load_model(cfg.model, cfg)
    files = Files(folder, cfg.mode, cfg.classes)
    rec = Recorder()
    speaker = Speaker(not cfg.no_voice, cfg.voice_rate)
    link = SimLink(rec, cfg) if cfg.simulate else RealLink(rec, cfg)
    plotter = LivePlot(not cfg.no_live_plot)
    plot_task = None

    meta = {
        "status": "running", "session_id": sess, "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(__file__), "started_iso": datetime.now().isoformat(timespec="seconds"),
        "launcher": {"path": cfg.launcher, "sha256": cfg.launcher_sha256},
        "args": vars(cfg), "task": cfg.task, "rf_ohm": cfg.rf.tolist(),
        "channels": CHANNELS, "classes": cfg.classes,
        "thresholds": {"high_adc": HIGH_ADC, "low_adc": LOW_ADC, "min_prob": cfg.min_prob,
                       "margin": cfg.margin, "min_sample_frac": cfg.min_sample_frac,
                       "max_sample_frac": cfg.max_sample_frac,
                       "timing_ratio": [cfg.min_timing_ratio, cfg.max_timing_ratio],
                       "high_adc_allow": cfg.high_adc_allow, "max_ref_cv": cfg.max_ref_cv},
        "model": {"path": cfg.model, "sha256": model_sha, "feature_set": fs}
        if model is not None else None,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "bleak": pkg_version("bleak"), "scikit-learn": pkg_version("scikit-learn"),
                     "joblib": pkg_version("joblib")},
        "platform": platform.platform(),
        "timing_note": "BLE notifications are paused during terminal prompts to prevent queued data. "
                       "Each analysis window is rejected if ESP32 elapsed time is inconsistent with host time.",
        "feature_names": ALL_FEATURE_NAMES, "n_v1_features": N_V1,
        "hardware_check": None, "device": None,
    }
    write_meta(folder, meta)
    print(f"\nSession folder: {folder.resolve()}")

    accepted_log, reject_counter, n_accepted, n_attempts = [], Counter(), 0, 0
    try:
        await link.open()
        meta["device"] = link.info
        plot_task = asyncio.create_task(update_live_plot(plotter, rec))

        if cfg.mode == "check" or not cfg.skip_check:
            meta["hardware_check"] = await hardware_check(rec, speaker, cfg, files)
            write_meta(folder, meta)
            if cfg.mode == "check":
                meta["status"] = "completed"
                return
            if meta["hardware_check"].get("reason") == "too_few_samples":
                print("\nSTOPPED: no valid seven-channel glove stream was received.")
                print("Do not continue to trials. Confirm the ESP32 advertises as Glove_BLE_7_RAW.")
                meta["status"] = "stopped_no_valid_glove_stream"
                return
            if not meta["hardware_check"].get("ok", False):
                print("\nSTOPPED: one or more channels failed the hardware check.")
                print("Correct the flagged contact or glove fit, then restart this program.")
                meta["status"] = "stopped_failed_hardware_check"
                return
            ans = await ask_with_stream_paused(link, rec, "\nContinue with data collection? [y/N]: ")
            if ans.strip().lower() not in ("y", "yes"):
                meta["status"] = "stopped_after_hardware_check"
                return

        if cfg.precond_cycles > 0:
            await ask_with_stream_paused(
                link, rec, "\nGlove on, hand relaxed. Press ENTER to start preconditioning... ")
            await preconditioning(rec, speaker, cfg, files)
        else:
            await ask_with_stream_paused(
                link, rec, "\nGlove on, hand relaxed. Press ENTER to prepare the trials... ")

        if cfg.mode == "calibrate":
            seq = make_sequence(cfg.classes, cfg.trials_per_class, cfg.seed)
            design = f"{cfg.trials_per_class} per class"
        else:
            seq = make_blind_sequence(cfg.classes, cfg.blind_trials, cfg.seed)
            design = f"{cfg.blind_trials} total, balanced as closely as possible"
        with open(folder / "sequence.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["trial_id", "target"])
            w.writerows((i + 1, t) for i, t in enumerate(seq))
        n_total = len(seq)
        est_min = n_total * (cfg.ref_s + cfg.fold_s) * 1.15 / 60
        print(f"\n{n_total} trials ({design}), about {est_min:.0f} min plus retries and breaks. "
              f"Seed {cfg.seed}.")
        await ask_with_stream_paused(link, rec, "Press ENTER to begin the trials... ")

        attempt_id, stop = 0, False
        for trial_id, target in enumerate(seq, 1):
            in_trial = 0
            while True:
                attempt_id += 1
                in_trial += 1
                if not await link.ensure():
                    print("Could not restore the BLE link. Stopping; data so far is saved.")
                    stop = True
                    break
                rows, started = await run_attempt(rec, speaker, cfg, target)
                res = analyse_attempt(rows, cfg)
                if rec.disconnected:
                    res["reasons"].insert(0, "ble_disconnect")
                    res["feats"] = None
                accepted = not res["reasons"]
                files.write_samples(attempt_id, trial_id, rows)
                files.attempts.writerow([
                    attempt_id, trial_id, in_trial, target, int(accepted), " | ".join(res["reasons"]),
                    started, res["n_ref"], res["n_fold"], res["n_rows"], res["hi_total"], res["lo_total"],
                    f"{res['ref_cv']:.4f}", f"{res['rate']:.2f}",
                    f"{res['ref_timing_ratio']:.3f}", f"{res['target_timing_ratio']:.3f}"])
                n_attempts += 1
                head = f"[{trial_id}/{n_total}] {target:<6} attempt {in_trial}"

                if accepted:
                    n_accepted += 1
                    feats = res["feats"]
                    files.features.writerow([trial_id, attempt_id, target, *[f"{v:.6g}" for v in feats]])
                    line = f"{head}: ACCEPTED  response {fmt_resp(feats)}"
                    if cfg.mode == "blind":
                        n_feat = N_V1 if fs == "v1" else len(ALL_FEATURE_NAMES)
                        proba = model.predict_proba(feats[:n_feat].reshape(1, -1))[0]
                        order = np.argsort(proba)[::-1]
                        top_p, second_p = float(proba[order[0]]), float(proba[order[1]])
                        forced = classes[order[0]]
                        ok_rule = top_p >= cfg.min_prob and (top_p - second_p) >= cfg.margin
                        decision = forced if ok_rule else "Uncertain"
                        pmap = {c: float(p) for c, p in zip(classes, proba)}
                        files.predictions.writerow([
                            trial_id, attempt_id, target, forced, decision, int(ok_rule),
                            int(forced == target), int(ok_rule and forced == target),
                            f"{top_p:.4f}", f"{top_p - second_p:.4f}",
                            *[f"{pmap[c]:.4f}" for c in cfg.classes],
                            model_sha, cfg.min_prob, cfg.margin, fs,
                            datetime.now().isoformat(timespec="milliseconds")])
                        accepted_log.append((target, forced, decision))
                        line += f"  ->  {decision} (p={top_p:.2f}, margin={top_p - second_p:.2f})"
                    else:
                        accepted_log.append((target, None, None))
                    print(line)
                    files.flush()
                    break

                reject_counter.update(r.split("(")[0] for r in res["reasons"])
                print(f"{head}: REJECTED  {' | '.join(res['reasons'])}")
                files.flush()
                speaker.say("Rejected. Repeating this trial.")
                if in_trial >= cfg.max_attempts:
                    ans = await ask_with_stream_paused(
                        link, rec, f"{in_trial} rejected attempts in a row. Check the contacts / fit. "
                                   f"ENTER to keep trying, q to stop the session: ")
                    if ans.strip().lower() == "q":
                        stop = True
                        break
                    in_trial = 0
            if stop:
                break
            if cfg.rest_every and trial_id % cfg.rest_every == 0 and trial_id < n_total:
                speaker.say("Rest break.")
                await ask_with_stream_paused(
                    link, rec,
                    f"\n--- Rest ({trial_id}/{n_total} done). Relax your hand, then press ENTER ---")

        meta["status"] = "stopped_early" if stop else "completed"
    except BaseException as e:
        meta["status"] = f"interrupted: {type(e).__name__}"
        raise
    finally:
        if plot_task is not None:
            plot_task.cancel()
        await link.close()
        plotter.close()
        meta["ended_iso"] = datetime.now().isoformat(timespec="seconds")
        meta["counts"] = {"attempts": n_attempts, "accepted_trials": n_accepted,
                          "rejected_attempts": n_attempts - n_accepted,
                          "rejection_reasons": dict(reject_counter)}
        write_meta(folder, meta)
        files.flush()
        files.close()

    print("\n" + "=" * 72)
    print(f"Session {meta['status']}: {n_accepted} accepted trials, {n_attempts - n_accepted} rejected attempts")
    if reject_counter:
        print("Rejection reasons:", dict(reject_counter))
    per_class = Counter(t for t, _, _ in accepted_log)
    print("Accepted per class:", dict(per_class))
    if cfg.mode == "blind" and accepted_log:
        n = len(accepted_log)
        acc = [(t, f, d) for t, f, d in accepted_log if d != "Uncertain"]
        corr = sum(t == d for t, _, d in acc)
        forced = sum(t == f for t, f, _ in accepted_log)
        print(f"Blind trials: {n}")
        if acc:
            lo, hi = wilson(corr, len(acc))
            print(f"  accepted accuracy {corr}/{len(acc)} = {corr / len(acc):.1%} (95% Wilson {lo:.1%}-{hi:.1%}), "
                  f"coverage {len(acc) / n:.1%}")
        lo, hi = wilson(forced, n)
        print(f"  forced-choice accuracy {forced}/{n} = {forced / n:.1%} (95% Wilson {lo:.1%}-{hi:.1%})")
    print(f"Files in: {folder.resolve()}")


# ============================================================
# COMMAND LINE
# ============================================================
def parse_rf(s):
    vals = [float(v) for v in str(s).split(",")]
    if len(vals) == 1:
        vals *= N_CH
    if len(vals) != N_CH:
        raise argparse.ArgumentTypeError(f"--rf needs 1 or {N_CH} comma-separated values")
    return np.array(vals, dtype=float)


def build_parser(fixed_task=None, allowed_modes=None):
    allowed_modes = allowed_modes or ["check", "calibrate", "train", "blind"]
    p = argparse.ArgumentParser(description="Unified glove acquisition, training, and blind evaluation",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=allowed_modes, default=allowed_modes[0])
    if fixed_task is None:
        p.add_argument("--task", choices=sorted(TASK_CLASSES), default="finger")
    else:
        p.set_defaults(task=fixed_task)
    g = p.add_argument_group("session metadata")
    g.add_argument("--participant")
    g.add_argument("--session")
    g.add_argument("--glove", default="G1", help="glove ID (if you build more than one)")
    g.add_argument("--fit", choices=["snug", "normal", "loose", "unspecified"], default="unspecified")
    g.add_argument("--donning", type=int, default=1, help="how many times the glove has been re-donned for this participant")
    g.add_argument("--notes", default="")
    g.add_argument("--out-dir", default="glove_data")
    g = p.add_argument_group("hardware")
    g.add_argument("--rf", type=parse_rf, default=parse_rf("3400"),
                   help="fixed resistor in ohm: one value, or 7 comma-separated values (use MEASURED values)")
    g.add_argument("--address", help="BLE address/UUID (skips scanning)")
    g.add_argument("--rate-hz", type=float, default=20.0, help="nominal sample rate")
    g = p.add_argument_group("protocol timing")
    g.add_argument("--ref-s", type=float, default=4.0)
    g.add_argument("--ref-win-s", type=float, default=2.0, help="final seconds of the reference used as baseline")
    g.add_argument("--fold-s", type=float, default=5.0, help="target hold duration")
    g.add_argument("--fold-win-s", type=float, default=3.0,
                   help="final seconds of the target hold used for features")
    g.add_argument("--trials-per-class", type=int, default=10, help="calibration trials per class")
    g.add_argument("--blind-trials", type=int, default=10, help="total balanced randomized blind trials")
    g.add_argument("--rest-every", type=int, default=25, help="rest break every N trials (0 = none)")
    g.add_argument("--precond-cycles", type=int, default=5, help="bend cycles per finger before trials (0 = skip)")
    g.add_argument("--precond-phase-s", type=float, default=2.0)
    g.add_argument("--check-open-s", type=float, default=5.0)
    g.add_argument("--check-fist-s", type=float, default=4.0)
    g.add_argument("--skip-check", action="store_true")
    g.add_argument("--seed", type=int, default=None)
    g = p.add_argument_group("quality control")
    g.add_argument("--high-adc-allow", type=int, default=0,
                   help="max samples >= 3900 allowed in the analysis windows")
    g.add_argument("--min-sample-frac", type=float, default=0.7)
    g.add_argument("--max-sample-frac", type=float, default=1.5,
                   help="reject implausibly dense analysis windows (usually queued BLE data)")
    g.add_argument("--min-timing-ratio", type=float, default=0.8,
                   help="minimum ESP32-time / host-time ratio allowed in an analysis window")
    g.add_argument("--max-timing-ratio", type=float, default=1.25,
                   help="maximum ESP32-time / host-time ratio allowed in an analysis window")
    g.add_argument("--max-ref-cv", type=float, default=0.20,
                   help="reject if the open reference is less stable than this")
    g.add_argument("--qc-ignore", default="", help="1-based channel numbers excluded from QC, e.g. 6,7")
    g.add_argument("--max-attempts", type=int, default=5, help="pause after this many rejected attempts in a row")
    g = p.add_argument_group("training and blind modes")
    g.add_argument("--input", help="calibration features.csv or its session folder (train mode)")
    g.add_argument("--model-out", help="destination .joblib model bundle (train mode)")
    g.add_argument("--model")
    g.add_argument("--min-prob", type=float, default=None,
                   help="confidence threshold; defaults to 0.35 when training and is frozen in the bundle")
    g.add_argument("--margin", type=float, default=None,
                   help="probability-margin threshold; defaults to 0.08 when training and is frozen in the bundle")
    g.add_argument("--feature-set", choices=["v1", "v2"], default=None)
    g.add_argument("--cv-folds", type=int, default=5)
    g.add_argument("--n-estimators", type=int, default=700)
    g.add_argument("--model-seed", type=int, default=42)
    g = p.add_argument_group("misc")
    g.add_argument("--no-voice", action="store_true")
    g.add_argument("--no-live-plot", action="store_true", help="disable the live seven-channel ADC plot")
    g.add_argument("--voice-rate", type=int, default=185)
    g.add_argument("--simulate", action="store_true", help="fake glove for a dry run")
    g.add_argument("--sim-glitch-prob", type=float, default=0.0002)
    return p


def main(argv=None, fixed_task=None, allowed_modes=None):
    p = build_parser(fixed_task=fixed_task, allowed_modes=allowed_modes)
    cfg = p.parse_args(argv)
    launcher = Path(sys.argv[0]).resolve()
    cfg.launcher = str(launcher)
    cfg.launcher_sha256 = sha256_file(launcher)
    out_dir = Path(cfg.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = launcher.parent / out_dir
    cfg.out_dir = str(out_dir.resolve())
    cfg.classes = TASK_CLASSES[cfg.task]

    def prompt_missing(value, label, default=None):
        if value:
            return value
        suffix = f" [{default}]" if default else ""
        try:
            entered = input(f"{label}{suffix}: ").strip()
        except EOFError:
            entered = ""
        value = entered or default
        if not value:
            p.error(f"{label} is required")
        return value

    if cfg.mode in ("calibrate", "blind"):
        cfg.participant = prompt_missing(cfg.participant, "Participant ID (for example P01)")
        default_session = f"{cfg.task.upper()}_{'CAL' if cfg.mode == 'calibrate' else 'BLIND'}"
        cfg.session = prompt_missing(cfg.session, "Session name", default_session)
    if cfg.mode == "train":
        cfg.input = prompt_missing(cfg.input, "Calibration session folder or features.csv")
        cfg.model_out = prompt_missing(
            cfg.model_out, "Model output filename", f"models/{cfg.task}_model.joblib")
        cfg.feature_set = cfg.feature_set or "v1"
        cfg.min_prob = 0.35 if cfg.min_prob is None else cfg.min_prob
        cfg.margin = 0.08 if cfg.margin is None else cfg.margin
    if cfg.mode == "blind":
        cfg.model = prompt_missing(cfg.model, "Frozen model filename", f"models/{cfg.task}_model.joblib")
    if cfg.trials_per_class < 1 or cfg.blind_trials < 1:
        p.error("trial counts must be positive")
    if not (0 <= (cfg.min_prob if cfg.min_prob is not None else 0.35) <= 1):
        p.error("--min-prob must be between 0 and 1")
    if not (0 <= (cfg.margin if cfg.margin is not None else 0.08) <= 1):
        p.error("--margin must be between 0 and 1")
    if cfg.ref_win_s > cfg.ref_s or cfg.fold_win_s > cfg.fold_s:
        p.error("analysis windows cannot be longer than their recording phases")
    if not (0 < cfg.min_sample_frac <= cfg.max_sample_frac):
        p.error("sample fractions must satisfy 0 < --min-sample-frac <= --max-sample-frac")
    if not (0 < cfg.min_timing_ratio <= 1.0 <= cfg.max_timing_ratio):
        p.error("timing ratios must satisfy 0 < minimum <= 1 <= maximum")
    if cfg.max_ref_cv is not None and cfg.max_ref_cv < 0:
        p.error("--max-ref-cv cannot be negative")
    cfg.seed = cfg.seed if cfg.seed is not None else int(time.time()) % (2 ** 31)
    try:
        cfg.qc_ignore_set = {int(x) - 1 for x in cfg.qc_ignore.split(",") if x.strip()}
    except ValueError:
        p.error("--qc-ignore must be comma-separated channel numbers")
    if any(i < 0 or i >= N_CH for i in cfg.qc_ignore_set):
        p.error(f"--qc-ignore channel numbers must be between 1 and {N_CH}")
    try:
        if cfg.mode == "train":
            train_model(cfg)
        else:
            asyncio.run(main_async(cfg))
    except KeyboardInterrupt:
        print("\nInterrupted. Everything recorded so far was saved.")


def run_participant1_finger_calibration():
    """Collect the new P01 finger calibration for the paired same-sitting run."""
    base_dir = Path(__file__).resolve().parent

    while True:
        fit = input("Enter glove fit (snug, normal, or loose): ").strip().lower()
        if fit in {"snug", "normal", "loose"}:
            break
        print("Please enter snug, normal, or loose.")

    while True:
        entered = input(
            "Enter donning number for Participant 1 today (1, 2, 3, ...): "
        ).strip()
        if entered.isdigit() and int(entered) >= 1:
            donning = entered
            break
        print("Please enter a whole number: 1, 2, 3, ...")

    print("\nParticipant 1 new finger calibration")
    print("This starts the four-recording same-sitting protocol.")
    print("Left hand only; keep every sensor fixed in the back-of-hand arrangement.")
    print("The program will stop if any channel fails the hardware check.\n")

    args = [
        "--mode", "calibrate",
        "--participant", "P01",
        "--session", "FINGER_CAL_SAME_SITTING",
        "--fit", fit,
        "--donning", donning,
        "--notes", (
            "Participant 1 same-sitting retake; finger calibration; left hand; "
            "sensors fixed on back/dorsum; right hand excluded"
        ),
        "--out-dir", str(base_dir / "glove_data"),
        "--trials-per-class", "10",
        "--rest-every", "25",
        "--precond-cycles", "5",
    ]
    main(argv=args, fixed_task="finger", allowed_modes=["calibrate"])


if __name__ == "__main__":
    run_participant1_finger_calibration()
