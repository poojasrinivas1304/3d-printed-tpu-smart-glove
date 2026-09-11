# Reproducing the released glove analyses

## Environment

Python 3.10 or later is recommended. Install the declared dependencies with:

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

The original acquisition environment was not frozen. The lower bounds in `requirements.txt` describe the maintained environment, while `requirements-tested.txt` records the exact versions used to verify this release on 11 September 2026; neither file is a claim about the historical acquisition environment.

## Resistance conversion audit

    python code/validate_resistance_conversion.py

This checks that the stored resistance columns reproduce the divider equations and constants recorded in the archived files. It does not replace a physical ESP32 ADC calibration.

## Folded-finger model

    python code/train_stepwise_finger_model.py

With the released session_01 data, the script constructs one feature vector per complete trial and reproduces the 82% within-session accuracy using out-of-fold predictions from stratified five-fold cross-validation. Median imputation and Random Forest fitting occur inside each training fold.

Outputs are written to outputs/folded_finger/session_01/.

## Posture model

    python code/train_posture_model.py

With one calibration file, the script uses stratified five-fold cross-validation. If multiple calibration files are supplied through GLOVE_POSTURE_DATA_DIR, it uses leave-one-file-out validation. Outputs are written to outputs/posture/session_01/.

## Cyclic-compression summaries

    cd data/cyclic_compression
    python ../../code/material_characterization/analyze_cyclic_resistance.py

The released cyclic-compression files contain electrical readings and relative time only. They do not contain platen gap, preload, force, or synchronized crosshead position; mechanical stress, strain, and force–displacement curves cannot be reconstructed from this release.

## Acquisition

Acquisition scripts require the glove hardware and BLE notifications formatted as one device-time value followed by seven raw ADC values. The exact ESP32 study firmware has not been located; see firmware/README.md.
