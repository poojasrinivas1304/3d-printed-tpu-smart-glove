# Data release

## `derived/`

- `combined_selected_features.csv`: de-identified trial-level modelling table (600 rows; 56 primary model features plus 25 retained exploratory features).
- `data_audit.csv`: one record for each of the 20 included acquisition sessions.
- `analysis_manifest.json`: participant IDs, analysis configuration, session identifiers, and integrity hashes.

## `cyclic_compression/`

Material-characterization data retained from the sensor study, including the three specimen CSV files and their summary table.

Raw human-participant time series are intentionally excluded. The public derived table is sufficient to reproduce the reported LOSO and sensor-ablation results.

See [data_dictionary.md](data_dictionary.md) for field definitions.
