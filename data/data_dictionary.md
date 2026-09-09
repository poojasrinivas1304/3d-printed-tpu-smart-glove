# Data dictionary

The files are machine-readable CSV tables. Each row is one recorded sample, trial, cycle, or aggregate result, depending on the file type. Units are included in field names where applicable.

## Common timing and identity fields

- `esp32_t_ms`, `arduino_t_ms`: device time in milliseconds.
- `elapsed_s`, `run_elapsed_s`, `session_elapsed_s`, `phase_elapsed_s`: elapsed time in seconds from the indicated reference point.
- `run_id`, `trial_id`, `cycle`, `event_index`, `segment_id`: experiment-local identifiers.
- `label`, `event_label`, `true_label`, `target_label`, `target_finger`, `posture_label`: instructed or recorded movement class.
- `phase`: open-reference, held-state, bending, or other protocol phase.

Absolute wall-clock timestamp fields were removed from the public sample tables. Timestamp-like text retained in filenames functions only as a unique session identifier.

## Electrical measurements

- Fields ending in `_raw_ADC`: raw 12-bit ESP32 analog-to-digital converter counts.
- `adc`: raw Arduino analog-to-digital converter count in cyclic-compression files.
- `R_ohm`, `R_kOhm`, and channel fields ending in `_R_kOhm`: resistance in ohms or kilohms.
- Fields containing `dR_over_R0` or `dR_R0`: normalized resistance change, `(R - R0) / R0`.
- Fields containing `local`: response after subtracting the immediately preceding open-hand reference.
- `open_bad_adc_total`, `fold_bad_adc_total`, `posture_high_adc_total`, and similar fields: counts of readings meeting the high-ADC quality-control rule.

## Cyclic-compression files

The three raw files correspond to conductive TPU alone at 2.5 mm and 5 mm/s, backed conductive TPU at 2.5 mm and 5 mm/s, and backed conductive TPU at 5 mm and 10 mm/s. `cyclic_characterization_summary.csv` reports duration, sampling rate, baseline resistance, extrema, peak-to-peak response, mean, and standard deviation.

## Sequential and cyclic glove files

Protocol files contain time-resolved measurements and event labels. Files containing `local` or `cycle_summary` contain locally referenced responses used to construct response matrices and cycle-level summaries. Files containing `summary_run` aggregate the cycle-level values within a run.

Channel order is thumb, index, middle, ring, little finger (`Pinky` in some source fields), reverse horizontal, and reverse thumb. These correspond to GPIO36, GPIO39, GPIO34, GPIO35, GPIO32, GPIO33, and GPIO25.

## Classification files

- Files containing `train_samples`: time-resolved open-reference and held-state calibration samples.
- Files containing `train_trials`: one record per calibration trial with quality-control counts and trial summaries.
- Files containing `blind_samples`: time-resolved samples from blind evaluations.
- Files containing `blind_summary`: one record per blind trial, including the true class, predicted class, probability, uncertainty status, and quality-control fields.
- Probability fields beginning with `prob_`: Random Forest class probabilities.
- `confidence`: highest class probability.
- `margin`: difference between the two highest class probabilities.
- `status` or `final_prediction`: whether the prediction passed the uncertainty criteria.
- `correct` or `is_correct_accepted`: correctness after uncertainty rejection.
- `is_correct_forced`: correctness when the highest-probability class is used for every trial.

The public participant/session identifier `session_01` does not encode a participant name.
