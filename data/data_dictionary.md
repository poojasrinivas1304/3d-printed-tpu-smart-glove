# Data dictionary

## `combined_selected_features.csv`

| Field | Meaning |
|---|---|
| `participant` | De-identified participant (`P01`–`P05`) |
| `task` | `finger` or `posture` |
| `session_type` | `cal` (calibration) or `evaluation` |
| `source_row` | Row number in the retained source feature table |
| `trial_id` | Accepted trial sequence number within the session |
| `attempt_id` | Attempt sequence number, including rejected attempts in the original session |
| `target` | Ground-truth class |
| `source_session` | De-identified timestamped acquisition-session identifier |

The primary 56 features follow `<sensor>__<statistic>`. Seven sensors each contribute `mean`, `median`, `std`, `min`, `max`, `range`, `last`, and `slope`. The table also retains 25 exploratory dynamic and adjacent-channel features (`peak`, `t_peak`, `onset_slope`, and four adjacent-channel differences); these are not used by the reported v1 LOSO models.

## `data_audit.csv`

Each row represents one selected session. Audit fields describe completion status, hardware-check result, packet gaps, accepted and rejected trial counts, preconditioning setting, class balance, and SHA-256 hashes of the retained private source files.

## `results/tables/`

Prediction tables contain the held-out participant, true and predicted class, confidence, class probabilities, correctness, and de-identified source session. Fold tables report accuracy, balanced accuracy, and macro-F1. Confusion-matrix rows are true classes and columns are predicted classes.
