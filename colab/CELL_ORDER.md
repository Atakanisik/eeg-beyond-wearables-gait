# Order of the Colab cells

Paste each cell into its own Colab cell and run them top to bottom. Every cell prints progress and ends with ✅ (pass), ⚠️ (look, but you may continue) or ❌ (the cell stops; fix before continuing). The files keep the cells exactly as run, including development-phase cells whose results were superseded (noted below), so that the analysis history is transparent.

| Step | File | Cells | What it does |
|---|---|---|---|
| 1 | `N0_colab_cells.py` | CELL 1-4 (5 optional) | Setup and Drive; download from the PhysioNet S3 mirror; checksum verification; IMU extraction, manifest, gate G0 (51 complete participants) |
| 2 | `N1_colab_cells.py` | CELL 6, 7, 8, 8b | One-trial inspection: sampling rates, force-plate events, stream alignment, EEG-IMU clock (8b superseded by CELL 10) |
| 3 | `N1_colab_cells.py` | CELL 9, 9b, 9c, 10 | All trials: events, force-plate alignment by heel-strike sequence matching, alignment tiers, stride table (`strides_v3.csv`); pooled EEG-IMU clock test |
| 4 | `N2_colab_cells.py` | CELL 11, 11b, 12, 13 | EEG preprocessing; head-motion regressors limited to 1-25 Hz; gate G1 (gait-cycle modulation); artifact diagnostics |
| 5 | `N3_colab_cells.py` | CELL 14, 15, 15b, 15c | Stride table with the initial [-600, -100] ms window; development split and joint-ridge model (kept as a record; superseded by the two-stage model); two-stage model and stride-to-stride check; power analysis |
| 6 | `N3w_colab_cells.py` | CELL P, 14w | Stride table with the primary [-400, -100] ms window in the Drive sub-folder `w400` (raw data needed) |
| 7 | `N3_colab_cells.py` | CELL 15, 15b, 15c | Development analyses repeated on the [-400, -100] ms features (run after CELL P, which switches the output folder) |
| 8 | `N4_confirm_cells.py` | CELL 16a | Writes and hashes the frozen analysis plan (never rewritten) |
| 9 | `N4_confirm_cells.py` | CELL 16b, 16c | Confirmatory test on the 33 held-out participants |
| 10 | `N4_confirm_cells.py` | CELL 16d, 16e, 16f | Secondary and sensitivity analyses; circular-shift null; descriptive analyses and export of all result files |

After a lost runtime (raw data gone, Drive intact): CELL 1 -> `R_resume_cell.py` -> 16a -> 16b -> 16c -> the cell you need. Do not use "Run all": it starts the download.
