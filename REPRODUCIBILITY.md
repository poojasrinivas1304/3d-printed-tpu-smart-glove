# Reproducibility guide

## Analysis input

`data/derived/combined_selected_features.csv` contains one row per accepted trial. It includes 600 rows: 5 participants × 2 tasks × (50 calibration + 10 evaluation trials). Participant names have been replaced with `P01`–`P05`.

Each trial has 56 model inputs: eight summary statistics for each of seven sensing channels. Preconditioning samples, rejected attempts, and raw time-series samples are not included in this table.

## Primary LOSO analysis

Run:

```bash
python code/analysis/run_loso.py
```

The analysis uses a median imputer and a Random Forest classifier with 700 trees, balanced class weights, square-root feature sampling, and random seed 42. Both imputation and classification are fitted separately inside every LOSO training fold.

Three protocols are reported:

- `all_trials`: calibration and evaluation trials from four participants are used to predict both session types for the unseen fifth participant. This is the manuscript's primary participant-independent analysis.
- `calibration_only`: calibration trials from four participants predict calibration trials from the unseen fifth participant.
- `calibration_to_evaluation`: calibration trials from four participants predict the evaluation session of the unseen fifth participant.

Expected primary pooled accuracies in `loso_summary.csv` are approximately:

| Task | Protocol | Accuracy |
|---|---|---:|
| Finger | all trials | 78.7% |
| Posture | all trials | 67.3% |

Small floating-point differences can occur across library or platform versions; predicted classes should otherwise reproduce.

## Sensor ablation

Run:

```bash
python code/analysis/run_sensor_ablation.py
```

This repeats LOSO using all seven channels (56 features) and the five dorsal finger channels only (40 features). The output is `reproduced_results/sensor_ablation_loso.csv`.

## Archived outputs

`results/tables/` contains the outputs used for manuscript reporting, including fold-level metrics, trial-level predictions, class metrics, and confusion matrices. `results/figures/` contains the final PNG and vector-PDF figures. Reproduction scripts write to a separate directory so these archived files are never overwritten.

## Audit trail

- `data/derived/data_audit.csv` documents the 20 selected sessions, accepted-trial counts, hardware-check state, packet gaps, class balance, and source hashes.
- `data/derived/analysis_manifest.json` records the model configuration and the selected session hashes.
- Failed, partial, superseded, and duplicate sessions were excluded before the derived table was created.

The raw records represented by the source hashes are retained by the research team and are not part of the public repository.
