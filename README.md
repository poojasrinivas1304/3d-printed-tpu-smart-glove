# 3D-printed TPU smart glove

Host-side acquisition and analysis code supporting the manuscript **“Additively manufactured TPU textile sensor glove for wireless multi-channel hand-motion sensing and AI-assisted recognition.”**

## System overview

The prototype uses seven 3D-printed TPU sensing channels connected to an ESP32 and streamed over Bluetooth Low Energy (BLE). The channel order is:

| Channel | Sensor | ESP32 input |
|---:|---|---|
| 1 | Thumb | GPIO36 |
| 2 | Index | GPIO39 |
| 3 | Middle | GPIO34 |
| 4 | Ring | GPIO35 |
| 5 | Little finger | GPIO32 |
| 6 | Reverse horizontal | GPIO33 |
| 7 | Reverse thumb | GPIO25 |

The voltage-divider circuits used nominal 3.3 kΩ fixed resistors. A measured value of 3.4 kΩ is used by the resistance-conversion scripts. BLE notifications use the Nordic UART Service TX characteristic `6E400003-B5A3-F393-E0A9-E50E24DCCA9E`.

## Repository contents

The `code/` directory contains scripts for:

- sequential and cyclic finger-bending acquisition;
- posture and folded-finger calibration-data collection;
- raw-ADC quality checking;
- resistance conversion and local open-hand baseline correction;
- Random Forest training and cross-validation;
- blind folded-finger and posture prediction; and
- MATLAB figure generation.

This repository contains the host-side software package. ESP32 firmware and participant-level datasets are not included in this release.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Typical workflow

From the repository root:

```bash
cd code
python check_raw_adc_health.py
python collect_stepwise_finger_training_data.py
python train_stepwise_finger_model.py
python run_ai_blind_stepwise_demo_v3_stepwise_model.py
```

For posture recognition:

```bash
cd code
python collect_posture_training_data.py
python train_posture_model.py
python run_posture_blind_demo.py
python plot_posture_blind_demo.py
```

The acquisition scripts create timestamped CSV files and the training scripts create model, report, feature, and confusion-matrix outputs in or below `code/`. Generated files are excluded by `.gitignore`.

## Data availability

Participant-level datasets are not included in this initial code release. Aggregate results are reported in the manuscript and its Supplementary Information.

## Citation

Citation metadata are provided in `CITATION.cff`. The manuscript citation will be updated after publication.

