# 3D-printed TPU smart glove

Acquisition code, de-identified derived data, analysis scripts, and result files supporting the manuscript **“Additively manufactured TPU textile sensor glove for participant-independent hand-motion recognition.”**

## Study snapshot

- Five participants, identified only as `P01`–`P05`.
- All participants were right-hand dominant; testing was performed on the left hand.
- Two five-class tasks: individual-finger bending and static hand posture.
- Four sessions per participant: finger calibration, finger evaluation, posture calibration, and posture evaluation.
- 600 accepted labelled trials in total (300 per task).
- Participant-independent performance was evaluated using leave-one-participant-out (LOSO) cross-validation. Every model and imputer was fitted using four participants and tested on the entirely unseen fifth participant.

The word `blind` remains in a few acquisition filenames because those are the exact scripts used during collection. In the public data and analysis outputs, these held-out same-sitting sessions are called **evaluation sessions** to distinguish them from participant-independent LOSO testing.

## Sensor mapping

| Channel | Sensor location | ESP32 input |
|---:|---|---|
| 1 | Little finger (`VP`) | GPIO36 |
| 2 | Ring finger (`VN`) | GPIO39 |
| 3 | Middle finger (`D34`) | GPIO34 |
| 4 | Index finger (`D35`) | GPIO35 |
| 5 | Thumb (`D32`) | GPIO32 |
| 6 | Front horizontal (`D33`) | GPIO33 |
| 7 | Front thumb (`D25`) | GPIO25 |

The glove is discovered over BLE as `GLOVE_01` or the legacy name `Glove_BLE_7_RAW`. Data are received from Nordic UART Service TX characteristic `6E400003-B5A3-F393-E0A9-E50E24DCCA9E`.

## Repository layout

```text
code/acquisition/       Four study scripts plus the combined menu program
code/analysis/          Portable LOSO and sensor-ablation analyses
data/derived/           De-identified trial features and audit records
data/cyclic_compression Material-characterization data used in the manuscript
results/tables/         Reported predictions, metrics, and confusion matrices
results/figures/        Manuscript-ready figures in PNG and PDF formats
```

Raw participant time series are not placed in this public repository because they are human-participant records. The de-identified trial-level feature table needed to reproduce the reported machine-learning analysis is included.

## Installation

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the analyses

From the repository root:

```bash
python code/analysis/run_loso.py
python code/analysis/run_sensor_ablation.py
```

New files are written to `reproduced_results/`, leaving the archived reported results unchanged. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for protocol definitions and verification targets.

## Run data collection

For a guided menu that asks for a de-identified participant ID and study section:

```bash
python code/acquisition/glove_study.py
```

The four individual programs are also retained exactly as used operationally:

```bash
python code/acquisition/1_finger_collect.py
python code/acquisition/2_finger_blind.py
python code/acquisition/3_posture_collect.py
python code/acquisition/4_posture_blind.py
```

Calibration sections 1 and 3 use five preconditioning cycles. Evaluation sections 2 and 4 skip preconditioning. The scripts perform a hardware check before accepting trials and stop when a channel fails the configured quality criteria.

## Data availability

De-identified derived data and analysis code are available in this repository. Raw participant recordings are available from the corresponding author on reasonable request, subject to ethics and institutional data-sharing requirements.

## Citation

Please cite the associated manuscript when it becomes available. Repository citation metadata are provided in [CITATION.cff](CITATION.cff).
