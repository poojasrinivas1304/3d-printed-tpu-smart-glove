# Glove study data

This directory contains de-identified data supporting the material-characterization, glove-response, and representative classification results reported in the manuscript. Absolute wall-clock timestamps were removed from the public copies. Relative experiment time, ESP32 or Arduino time, trial labels, ADC readings, resistance values, normalized responses, and quality-control fields were retained.

## Directory structure

- `cyclic_compression/`: three cyclic-compression conditions and their aggregate characterization summary.
- `single_finger/`: three sequential finger-bending runs and their locally corrected response tables.
- `cyclic_finger/`: four repeated finger-bending acquisitions, cycle-level tables, and run-level summaries.
- `posture_response/`: three sequential posture runs and their locally corrected response tables.
- `classification/folded_finger/session_01/`: the 50-trial calibration dataset and representative 10-trial blind folded-finger evaluation.
- `classification/posture/session_01/`: the 50-trial posture calibration dataset and two representative 10-trial blind posture evaluations.
- `exclusions.csv`: analysis exclusions and reasons.

The complete cyclic-finger acquisition for run 2 is retained for transparency but was excluded from the cycle-level repeatability analysis for the reason recorded in `exclusions.csv`.

## Scope

The public classification data in this release support the single-session and exploratory results reported in the main text: 82% five-fold cross-validated folded-finger accuracy, seven correct accepted predictions among seven accepted blind trials, nine correct forced-choice predictions among 10 blind trials, and 10 correct predictions among 14 accepted posture trials across two sessions.

The three-participant validation files are not included in this release because the candidate local files have not yet been reconciled with the participant-level aggregate table in the manuscript. They should not be added until the exact source sessions are identified and verified.

## Reuse

See `data_dictionary.md` for field definitions. The Python and MATLAB scripts in `../code/` read these CSV formats. Some acquisition scripts create timestamped filenames; public copies retain those filenames as session identifiers even though their absolute wall-clock timestamp columns were removed.
