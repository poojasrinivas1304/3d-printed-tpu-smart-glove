# Glove study data

This directory contains de-identified data supporting the material-characterization, glove-response, and representative classification results reported in the manuscript. Absolute wall-clock timestamps were removed from the public copies. Relative experiment time, ESP32 or Arduino time, trial labels, ADC readings, resistance values, normalized responses, and quality-control fields were retained.

## Directory structure

- `cyclic_compression/`: three cyclic-compression conditions and their aggregate characterization summary.
- `single_finger/`: three sequential finger-bending runs and their locally corrected response tables.
- `cyclic_finger/`: four repeated finger-bending acquisitions, cycle-level tables, and run-level summaries.
- `posture_response/`: three sequential posture runs and their locally corrected response tables.
- `classification/folded_finger/session_01/`: the 50-trial calibration dataset and representative 10-trial blind folded-finger evaluation.
- `classification/posture/session_01/`: the 50-trial posture calibration dataset and two representative 10-trial blind posture evaluations.
- `classification/archived_session_summary.csv`: de-identified counts for the three complete folded-finger and three complete posture blind files used in the revised multi-session figure.
- `exclusions.csv`: analysis exclusions and reasons.

The complete cyclic-finger acquisition for run 2 is retained for transparency but was excluded from the cycle-level repeatability analysis for the reason recorded in `exclusions.csv`.

## Scope

The public trial-level classification data support the primary within-session and representative blind results reported in the main text: 82% out-of-fold five-fold cross-validated folded-finger accuracy; seven correct accepted predictions among seven accepted blind trials; nine correct forced-choice folded-finger predictions among 10 trials; and seven correct posture decisions among eight accepted trials in the representative posture session.

The de-identified session-level count table records all six complete archived blind files. Participant-level raw files beyond the representative sessions should not be released until the source-session mapping is fully reconciled and institutional requirements permit public sharing.

## Reuse

See `data_dictionary.md` for field definitions. The Python and MATLAB scripts in `../code/` read these CSV formats. Some acquisition scripts create timestamped filenames; public copies retain those filenames as session identifiers even though their absolute wall-clock timestamp columns were removed.
