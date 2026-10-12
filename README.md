# Does pre-landing dry EEG add information about landing mechanics beyond wearable sensors?

Analysis code and result files for:

**Result in one sentence.** In 33 held-out participants, adding EEG relative band power (400-100 ms before heel strike) to IMU and EMG features did not improve prediction of peak vertical ground reaction force, loading rate or early-stance knee flexion (median ΔR² 0.0000 / 0.0000 / -0.0002); gains of ΔR² ≥ 0.01 were excluded by equivalence tests (Holm-corrected p < 0.001).

## Data

The raw data are **not** included. They are publicly available from PhysioNet under the Creative Commons Attribution 4.0 licence:

> Katmah R, AlShehhi A, Kosaji D, Al-Rahmani N, Abdullah M, Hulleck AAV, Khalaf K. *A multimodal gait dataset of brain activity, muscle activity, kinematics and ground forces in young adults* (version 1.0.0). PhysioNet; 2026. https://doi.org/10.13026/r0ea-7161

CELL 2 in `colab/N0_colab_cells.py` downloads the dataset from the PhysioNet open-data mirror on AWS S3 and CELL 3 verifies the SHA-256 checksums.

## Repository layout

| Path | Content |
|---|---|
| `colab/` | The Google Colab cells exactly as run, in order (see `colab/CELL_ORDER.md`) |
| `results_w400/w400/` | Result files of the primary analysis ([-400, -100] ms window): frozen analysis plan and its SHA-256, development and confirmation tables, circular-shift null distributions, Colab figures |
| `results_w400/window600_and_methods/` | Development results with the initial [-600, -100] ms window, gait-cycle EEG modulation (G1) outputs, trial QC and manifest |
| `analysis/` | Local scripts that recompute the confirmatory statistics from the per-participant results and draw the paper figures (`figures/`) |
| `docs/protocol_and_amendments.md` | Original study design and the dated amendments made before the confirmatory analysis |

## Reproducing the analysis

1. **Colab (full pipeline, raw data).** Open a new Colab notebook and paste the cells one by one in the order given in `colab/CELL_ORDER.md`. Each cell prints progress and ends with ✅/❌ checks; a ❌ stops the cell. Outputs are written to Google Drive (`MyDrive/gait_eeg_paper/n0/`). After a lost runtime, run CELL 1 and then `colab/R_resume_cell.py` instead of re-running everything.
2. **Local (statistics and figures from the exported results).**

   ```bash
   pip install -r requirements.txt
   python analysis/make_figures.py   # recomputes the confirmatory statistics, writes analysis/stats_recomputed.csv and figures
   python analysis/make_fig1.py      # design schematic
   ```

3. **Checking the frozen plan.** The confirmatory analysis plan was fixed on 2026-10-09 before the confirmation participants were analysed:

   ```bash
   sha256sum results_w400/w400/frozen_plan.json   # must equal results_w400/w400/frozen_plan.sha256
   ```

   Every confirmation result file stores the first 16 characters of this hash (`plan_sha256` column). The plan was not deposited in an external registry.

## Analysis summary

- **Stride clock.** Force-plate heel strikes (5%/3% body-weight hysteresis), aligned to the IMU clock per trial by heel-strike sequence matching; EEG on the IMU clock with a 25 ms correction.
- **Targets.** Peak vertical GRF, loading rate and peak knee flexion in the first 25% of stance.
- **Features.** IMU and EMG (37 features, M0); EEG log relative band power (19 channels x theta/alpha/beta/gamma) after notch and band-pass filtering, common average reference and head-acceleration regression.
- **Model.** Two-stage ridge regression: stage 1 on M0, stage 2 on EEG made orthogonal to M0, fitted to the stage-1 residuals.
- **Validation.** Each of 33 confirmation participants held out once; ΔR² per participant.
- **Statistics.** One-sided Wilcoxon (gain > 0) and equivalence (gain < 0.01), Holm across three targets; secondary analyses; circular-shift null (100 rotations).

## Software

Google Colab, Python 3.13.16, NumPy 2.1.3, SciPy 1.16.3, pandas 2.2.3, scikit-learn 1.6.1, Matplotlib 3.10.0.

## Licence

Code: MIT (see `LICENSE`). Result files in `results_w400/` are derived from the CC BY 4.0 dataset above and are shared under CC BY 4.0; please cite the dataset and this work.

## Contact

Atakan Işık, Department of Biomedical Engineering, Başkent University — atakani@baskent.edu.tr
