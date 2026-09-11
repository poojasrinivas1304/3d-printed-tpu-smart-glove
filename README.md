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

The voltage-divider circuits used nominal 3.3 kΩ fixed resistors. Archived finger and classification files used a 3.4 kΩ conversion constant; the sequential-posture files correspond to 3.5 kΩ. No traceable channel-level resistor or ESP32 ADC calibration record was retained, so absolute resistance is treated as nominal. The normalized within-channel results are invariant to a constant fixed-resistor scale. BLE notifications use the Nordic UART Service TX characteristic `6E400003-B5A3-F393-E0A9-E50E24DCCA9E`.

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

The `outputs/` directory contains regenerated feature tables, training reports, confusion matrices, and fitted model bundles for the released calibration sessions. See `REPRODUCIBILITY.md` for exact commands.

The exact ESP32 BLE firmware and Arduino cyclic-compression firmware are not included because verified study versions have not yet been located. The unrelated six-channel serial Arduino sketch found with the working files was deliberately excluded.

## Installation

Python 3.10 or later is recommended. `requirements-tested.txt` records the exact environment used to verify this release on 11 September 2026.

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

The acquisition scripts create timestamped CSV files. The training scripts write model bundles, reports, feature tables, and confusion matrices to `outputs/`; the regenerated outputs supplied with this release are tracked.

## Model-evaluation details

- `train_stepwise_finger_model.py` evaluates the released 50-trial folded-finger dataset using out-of-fold predictions from stratified five-fold cross-validation. With multiple calibration files, it uses leave-one-file-out validation.
- `train_posture_model.py` follows the same rule: stratified five-fold validation for one calibration file and leave-one-file-out validation when multiple files are supplied.
- The training scripts place median imputation and the Random Forest classifier in one scikit-learn `Pipeline`. The imputer and classifier are therefore fitted only on the training portion of each fold before predictions are generated for its held-out trials.
- Both blind-test scripts use the same uncertainty rule: the top-class probability must be at least 0.35 and the probability margin between the two leading classes must be at least 0.08; otherwise, the final decision is `Uncertain`.

To regenerate the cyclic-compression summaries and plots from the released data:

```bash
cd data/cyclic_compression
python ../../code/material_characterization/analyze_cyclic_resistance.py
```

To verify that archived raw ADC and resistance columns match the documented conversion constants:

```bash
python code/validate_resistance_conversion.py
```

This is a computational consistency check, not an independent physical calibration. See `REPRODUCIBILITY.md` for the complete workflow and known limitations.

## Data availability

This release includes the verified single-session and exploratory datasets described in `data/README.md`, plus a de-identified table of the complete archived blind-session counts used in the revised multi-session figure. Candidate participant-level raw files are not released until their source sessions are fully reconciled and institutional requirements permit public sharing.

## Citation

Citation metadata are provided in `CITATION.cff`. The manuscript citation will be updated after publication.
