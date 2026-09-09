# 3D-printed TPU smart glove

Host-side acquisition, analysis code, and verified de-identified data supporting the manuscript **“Additively manufactured TPU textile sensor glove for wireless multi-channel hand-motion sensing and machine-learning-assisted recognition.”**

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

The `data/` directory contains verified material-characterization data, sequential and cyclic glove measurements, posture-response measurements, and representative folded-finger and posture-classification sessions. Its `README.md`, `data_dictionary.md`, and `exclusions.csv` files describe the release and analysis scope.

The exact ESP32 BLE firmware and Arduino cyclic-compression firmware are not included because verified study versions have not yet been located. The unrelated six-channel serial Arduino sketch found with the working files was deliberately excluded.

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

The training scripts automatically use the released `data/classification/` files when they are run from a repository checkout. Set `GLOVE_STEPWISE_DATA_DIR` or `GLOVE_POSTURE_DATA_DIR` to train from another compatible directory.

The acquisition scripts create timestamped CSV files and the training scripts create model, report, feature, and confusion-matrix outputs in or below `code/`. Generated files are excluded by `.gitignore`.

To regenerate the cyclic-compression summaries and plots from the released data:

```bash
cd data/cyclic_compression
python ../../code/material_characterization/analyze_cyclic_resistance.py
```

## Data availability

This release includes the verified single-session and exploratory datasets described in `data/README.md`. Candidate files for the three-participant validation have not yet been reconciled with the participant-level aggregate table and are therefore not included. Aggregate three-participant results remain reported in the manuscript and Supplementary Information.

## Citation

Citation metadata are provided in `CITATION.cff`. The manuscript citation will be updated after publication.
