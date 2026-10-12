# Study protocol and dated amendments

Original design written on 2026-10-08, before any EEG-target analysis, followed by the dated amendments made on development participants and synthetic data, and the confirmatory results. The original design mentions an OSF preregistration; instead, the confirmatory analysis plan was fixed by its SHA-256 hash (results_w400/w400/frozen_plan.json and frozen_plan.sha256) on 2026-10-09 and was not deposited in an external registry. Sections on idea selection, timeline, journal plan and internal questions are omitted.

## 2. Research Question Card

```
Question: Do EEG features recorded just before foot landing predict stride-to-stride variability
  in landing mechanics (loading peak vGRF, loading rate, knee flexion) in held-out subjects,
  beyond what IMU and EMG-envelope features from the same window already predict?
Type: confirmatory (pre-specified contrasts, dev/confirm split)
Hypothesis: EEG adds a small, reliable out-of-sample gain (delta R2 > 0) that (a) exceeds a
  stride-permuted EEG surrogate, (b) survives head-motion regression and ICA cleaning,
  (c) holds when the test speed is excluded from training.
Why it matters: EEG-based gait decoding papers rarely separate neural information from
  stride-locked artifact. A surrogate-controlled test is publishable whichever way it goes.
Current evidence: pre-heel-strike sensorimotor coupling and gait-cycle-locked alpha/beta modulation
  are reported in the literature (observed). EEG2GAIT, NeuroDyGait and similar decoders report gains
  without this control (observed, other datasets).
Missing evidence: whether any increment exists in this dataset (dry EEG, slow speeds 0.5-1.0 m/s,
  healthy young adults); minimum detectable effect at N about 51.
Supported by: paired subject-level delta R2 > 0 with CI excluding 0, above the S1 and S2 nulls,
  after the strongest cleaning, on the confirmatory subjects, Holm-corrected.
Falsified by: no gain over S1, or a gain that disappears after head-acceleration regression / ICA.
Minimal next action: Step 0 audit notebook, then the sanity and power gates (section 8).
Decision: run experiment
```

## 3. Verified dataset facts and what they force

Verified from file listings on 2026-10-08 (not yet from local files):

| Fact | Source | Consequence |
|---|---|---|
| EMG column headers read `... EMG n->MAV [Volts]`; timestamps step about 0.0623 s (about 16 Hz) | `EMG_S1_0.5.csv` head | EMG is a **smoothed MAV envelope, not raw**. No spectral or coherence analysis. Only stride-level amplitude features, about 6 samples in a 400 ms window. |
| Each trial is about 1 min in every modality (EMG about 271 KB per file; force plate about 12.4 MB at 1000 Hz) | Folder listings | About 3 h of EEG in total. About 35-50 strides per leg per trial (estimate), about 10^4 strides overall. |
| EEG: 300 Hz, raw, 19 scalp channels, Pz reference, `EEG_referencing.m` provided | Page, README, folder | Port the Pz reconstruction to Python. Filter the continuous signal, then window. |
| IMU: 60 Hz, sensor-level files (free acceleration, magnetic field, sensor orientation) and segment/joint-level MVN outputs | Page | Use the head sensor's free acceleration for EEG artifact regression. Exact file names are not published on the page; read them from `IMU.tar.gz`. |
| Force plate not hardware-synced; EEG, EMG, IMU got trigger pulses | Page | Use **IMU-defined heel strike** as the shared stride clock. Use the force plate only for target values after per-trial lag estimation. |
| Missing files: EMG (S3, S8, S11, S43, S50), IMU (S2, S3, S24), force plate (S3, S8, S45) | Page | Complete-case subjects: 59 minus {2, 3, 8, 11, 24, 43, 45, 50} = **51** (verified in Colab: G0 PASS, 162 trials with all 4 modalities) |
| **Verified in Colab 2026-10-08:** all streams exported as the same ~1-min window; FP offset near 0 in most trials (median +37 ms), one exception (S1_0.5, +3.455 s) | N1 CELL 9/9b | FP aligned per trial by heel-strike sequence matching, +-0.5 s window first |
| **Verified:** EEG shares the IMU clock: pooled over 157 trials, EEG-artifact vs head-acceleration cross-correlation peaks at -33 ms (P2, r = 0.37) and -17 ms (P3), p = 0.001 vs a per-trial lag-shuffle null | N1 CELL 10 | Global EEG correction -0.025 s; well inside the 100 ms guard. Methods-section result. |
| **Verified:** EEG carries a strong head-motion-locked artifact on average (pooled r = 0.37 at zero lag, sidelobes at +-1 step) | N1 CELL 10 | Head-acceleration regression (S2) and the cleaning ladder are mandatory; add per-trial artifact index as a covariate / dose-response control |

## 4. Design

**Stride clock.** Heel strike (HS) and toe-off from the foot IMU segment. Per trial, estimate the lag between IMU-HS and Fz-HS (median offset of matched events) and apply it to force-plate targets. Drop strides with unmatched events.

**Pre-landing window.** EEG, EMG and IMU features come from [HS-600, HS-100] ms (500 ms; primary). The 100 ms guard keeps the impact transient and event-detection error out of the window. Sensitivity analysis: windows ending at -50 and -200 ms, and 300 / 700 ms lengths.

**Targets (primary, per leg)**
- T1: peak vertical GRF in the first 25% of stance.
- T2: vertical loading rate (peak dFz/dt in early stance).
- T3: peak knee flexion in the first 25% of stance (Xsens joint angle).
- Secondary: tibialis anterior post-HS MAV peak; mediolateral COP excursion.

**Within-trial centering.** For scoring, subtract each subject x speed x leg mean from targets and features, so only stride-to-stride covariation counts (no body-mass or speed offsets).

**Feature sets**
- M0 = IMU features (shank/foot angular velocity, foot orientation, clearance, pelvis and trunk state, last-sample joint angles and velocities) + EMG envelope features (mean, slope for the 12 channels).
- M1 = M0 + EEG features. EEG: continuous filtering (0.5-4, 4-8, 8-13, 13-30 Hz), Hilbert envelope mean per window, plus slow-potential mean and slope; 19 channels. Gamma (above 30 Hz) is reported as a labeled artifact-prone exploratory band only.

**Model families (conclusion must hold in both)**
- Linear: ridge / PLS on handcrafted features.
- Deep: EEGNet-style CNN on the filtered pre-window EEG, late-fused with the M0 feature vector.

**EEG cleaning ladder:** (a) minimal (filter, Pz reconstruction, bad-channel interpolation); (b) ICA with ICLabel; (c) ICA + head free-acceleration regression (lagged). Report the increment at every rung.

**Controls**
- S1: permute EEG windows across strides within each subject x speed x leg.
- S2: artifact-only EEG: the part of EEG linearly predicted from the head accelerometer.
- Sanity gate: grand-average gait-cycle-locked alpha/beta modulation at C3/Cz/C4 must be visible (reported in the literature, e.g. PMC7134343). If it is not, EEG quality is insufficient and the study stops.

## 5. Evaluation and statistics

- **Splits.** Pick 12-15 development subjects (stratified by sex and BMI category) for all design decisions. Freeze the pipeline and the analysis plan (OSF preregistration), then run once on the remaining about 36 confirmatory subjects.
- **Validation.** LOSO for linear models; subject-grouped 10-fold for deep models.
- **Cross-speed.** Train on two speeds from training subjects, test on the third speed from held-out subjects (subject- and speed-disjoint).
- **Primary metric.** delta R2 = R2(M1) - R2(M0) on within-trial-centered targets, computed per held-out subject.
- **Inference.** One-sided Wilcoxon signed-rank on subject-level delta R2 (and bootstrap CI over subjects); compare against the S1 permutation null (1000 permutations). Holm correction across 3 targets x 2 legs. Secondary: stride-level mixed model with subject random effects.
- **Power.** Simulate planted EEG effects on real M0 features to get the minimum detectable delta R2 at N about 51. Provisional rule: if MDE > 0.03, reconsider before spending the confirmatory subjects.

## 6. Pipeline (Colab notebooks, features cached to Drive as parquet)

| Notebook | Output |
|---|---|
| N0_audit (`notebooks/N0_audit.ipynb`, written and tested on synthetic data; not yet run on the real dataset) | Download + SHA256 check; complete-case manifest; durations; EMG rate; IMU file map |
| N1_events | IMU-HS/TO detection; Fz-HS; per-trial lag; stride table with QC flags |
| N2_eeg | Pz reconstruction; cleaning ladder rungs a/b/c; continuous filtered bands |
| N3_features | Stride-level M0 and EEG feature tables; targets |
| N4_models | Linear + CNN, controls S1/S2, cross-speed, statistics |
| N5_figures | Figures and tables below |

## 7. Figures and tables

1. Design schematic (stride clock, pre-landing window, guard gap).
2. Grand-average gait-cycle-locked alpha/beta modulation (sanity gate).
3. Forest plot of delta R2 per target and leg versus the S1 and S2 nulls.
4. Cleaning ladder: delta R2 at rungs a/b/c.
5. Cross-speed results (held-out 0.5 / 0.75 / 1.0).
6. Sensitivity to window end and length.
7. Table: M0 baseline performance (this is the idea-4 arm).

## 8. Gates

| Gate | Pass condition | If it fails |
|---|---|---|
| G0 audit | at least 45 complete-case subjects; sync lags estimable | Re-scope subject set |
| G1 sanity | alpha/beta gait-cycle modulation visible | Stop; EEG unusable |
| G2 power | MDE <= 0.03 (provisional) | Reconsider targets or pooled analysis |
| G3 dev result | pipeline frozen and preregistered | Iterate only on dev subjects |

## 9. Outcomes and allowed claims

| Outcome | Paper | Allowed wording | Forbidden wording |
|---|---|---|---|
| Gain survives all controls and cross-speed | "EEG carries small incremental pre-landing information" | "EEG-derived features improve prediction of ... by delta R2 = x (95% CI ...)" | "decodes intent", "brain controls landing", clinical claims |
| Gain appears only before cleaning or in S2 | Methodological: apparent EEG gains are explained by head motion | "the apparent gain is attributable to motion-related components" | "EEG is useless for gait" |
| No gain anywhere | Null result + M0 mechanics baseline | "no detectable incremental information under these conditions" | generalization beyond healthy young adults at 0.5-1.0 m/s on a treadmill |

## 10. Risks

1. **Weak EMG baseline.** A 16 Hz envelope is a weak M0 component, so EEG may stand in for unobserved muscle state. State this limit; claim "beyond IMU + 16 Hz EMG envelope", not "beyond muscle activity".
2. **Dry-EEG gamma** is contaminated by muscle and motion. Keep it exploratory.
3. **Forking paths.** Many windows, bands and cleaning rungs. The dev/confirm split and preregistration are the defence.
4. **Small N and small effects.** The power gate exists for this.
5. **Generalization.** Healthy young adults, slow treadmill speeds. Say so.

## 13b. Protocol amendment, 2026-10-08 (made before any EEG-vs-target analysis)

Evidence that triggered it (N2 CELL 12-13, 110 strict trials, 47 subjects):
- G1 passed as pre-specified (Cz beta after head-motion regression: 6.20 dB, p = 0.001; central 5.17 > temporal 2.43 dB), and power peaks exactly at both heel strikes, which validates the alignment chain.
- But the gait-cycle modulation is dominated by one broadband heel-strike-locked profile (mean between-band r = 0.93), the known signature of mechanical artifact. Beta-specific modulation beyond that profile: p = 0.029 (not significant at the pre-set 0.01). Alpha-specific: p = 0.002 (exploratory only). Dose-response with the artifact index: rho = +0.26.
- Head-motion regressors are now limited to 1-25 Hz: with 1-45 Hz, interpolating the 60 Hz IMU injected 30-45 Hz content (Cz gamma depth 5.41 -> 7.77 dB; back to 5.45 dB after the fix).

Changes:
1. Primary EEG features = log relative band power (band / 1-45 Hz broadband) per channel in the pre-landing window, from motion-regressed EEG. A multiplicative broadband artifact cancels in the ratio. Absolute broadband power is kept as an artifact covariate.
2. M0 (mechanics baseline) includes head kinematics (head acceleration mean/RMS per axis, |acc| RMS) in the same window, so EEG must add beyond head motion.
3. Primary trial set: strict tier minus the 10 trials whose left/right IMU match was ambiguous; relaxed tier and those 10 trials only in sensitivity analyses. S32_1 excluded from EEG (corrupted channel labels).
4. Report the G1 diagnostics in the paper as a methods result (dominant broadband artifact profile in dry-electrode treadmill EEG).

Correction, 2026-10-08 (still before any confirmatory analysis): the first dev/confirm split (v1) stratified on gender x numeric BMI, which made every subject its own stratum and yielded only 3 dev subjects; its dev results were also numerically unstable (per-subject R2 down to -55) because features were not scaled within trial. Split v2 bins BMI at 25/30 kg/m2, targets 30% dev subjects, and keeps the 3 v1 dev subjects in dev so that no subject whose results were seen enters the confirmation set. Features and targets are z-scored within trial, features clipped at +-5 SD, strides with |target z| > 4 dropped. Report this in the methods.

## 13c. Methods amendment, 2026-10-08 (dev subjects only; confirm subjects untouched)

Dev result with the CELL 15 v2 joint ridge (14 dev subjects, 3,033 strides): M0 median held-out R2 0.044 / 0.224 / 0.126 (T1 / T2 / T3); EEG relative power median delta R2 +0.0016 / -0.0152 / -0.0054, Wilcoxon p 0.52 / 0.92 / 0.64, within-trial shuffle p 0.039 / 0.65 / 0.10; broadband delta R2 about 0; dose-response rho -0.45 / -0.39 / -0.14.

Three problems found on synthetic feature-level data (no-link, autocorrelated no-link, shared slow drift, planted stride-level link), not on the real dev results:
1. **Joint ridge bias.** With one shared penalty, adding uninformative EEG features raises alpha and can improve the mechanics part on held-out subjects, giving delta R2 > 0 without EEG information (false-positive rates up to 0.80 on stride-to-stride data). Replaced by a **two-stage model**: ridge M0; EEG made orthogonal to the M0 features on the training subjects; ridge (alphas 1 to 1e6) of the M0 residuals on that EEG. Without EEG information this stage only adds noise, so E[delta R2] <= 0 and a one-sided Wilcoxon over subjects against 0 is valid.
2. **Within-trial shuffle null (S1) is anti-conservative** when EEG features and targets are both autocorrelated over strides (3 of 6 false positives on one no-link dataset). S1 is replaced by a **circular-shift surrogate**: each trial's EEG sequence rotated by >= 3 strides.
3. **Shared slow drift** (e.g. electrode settling + gait adaptation) is real dependence but not stride-level information. Added a **stride-to-stride (first-difference) analysis**; on synthetic drift it reduced the spurious gain from delta R2 0.19 to 0.005 (Wilcoxon p 0.08).

Validation of the new cells (CELL 15b/15c) on synthetic data: no-link datasets (3, two with AR(1) targets) gave 0/18 Wilcoxon p < 0.05 and 1/18 circular p < 0.05; a planted stride-level link was detected in both variants (p 0.0001); power simulation false-positive rate 0.017-0.019 at nominal 0.0167.

Power gate method (G2): planted effect on circularly shifted real EEG, paired gain over q = 0, resampled to the 33 confirm subjects, one-sided Wilcoxon at alpha 0.05/3. Conservative because dev training uses 13 subjects (confirm: 46).

Real dev results, 2026-10-09 (two-stage model, 14 dev subjects):
- CELL 15b, trial-centered: median delta R2 +0.0033 / +0.0002 / +0.0026 (T1/T2/T3); Wilcoxon p 0.38 / 0.27 / 0.31; circular p 0.039 / 0.22 / 0.078. Stride-to-stride: +0.0049 / +0.0062 / +0.0003; Wilcoxon p 0.43 / 0.31 / 0.57; circular p 0.020 / 0.020 / 0.16 (0.020 = smallest possible with 50 rotations). 5-6 of 14 subjects get worse with EEG. Read: no gain that generalizes to held-out subjects; weak pooled dependence for T1 and T2 that does not transfer consistently.
- Design flaw found while interpreting this (independent of the direction of the result): the contralateral heel strike falls inside the [-600, -100] ms window at faster speeds (step time about 0.73 / 0.63 / 0.56 s at 0.5 / 0.75 / 1.0 m/s), so the window can contain an impact artifact that is mechanically linked to the next landing. Proposed fix: primary window [-400, -100] ms.
- CELL 15c (G2 power, N = 33 confirm subjects, one-sided Wilcoxon alpha 0.05/3): false-positive rate 0.015-0.020; MDE (planted incremental R2, power >= 0.8) trial-centered 0.01 / 0.01 / 0.01, stride-to-stride 0.03 / 0.01 / 0.02. G2 passes for both variants; trial-centered is clearly more powerful.

## 13d. Freeze decisions, 2026-10-09 (Atakan; before any confirm-subject analysis)

1. Primary pre-landing window **[-400, -100] ms** (other foot's heel strike outside the window at every speed); [-600, -100] ms becomes a sensitivity analysis. Features rebuilt by N3w CELL P + CELL 14w into the Drive sub-folder `w400`; CELL 15/15b/15c re-run there on the same frozen dev subjects before the plan file is written.
2. Primary variant **trial-centered**, two-stage model; stride-to-stride is the drift-robust secondary analysis.
3. **Legs pooled**; Holm correction across the 3 targets.
4. Primary test: one-sided Wilcoxon over held-out confirm subjects on two-stage delta R2; secondary: circular-shift null of the median. Equivalence bound (SESOI) **delta R2 = 0.01**.
5. **CNN arm = preregistered secondary analysis**, run after the primary confirm test, same two-stage logic (CNN predicts M0 residuals from the pre-landing EEG) and the same circular-shift control.

Dev re-check with the [-400, -100] ms features (same 14 dev subjects, 2026-10-09):
- M0 held-out R2 0.068 / 0.219 / 0.148 (T1 / T2 / T3; T1 up from 0.044 with [-600, -100]).
- Two-stage, trial-centered: median delta R2 -0.0003 / +0.0015 / +0.0031; Wilcoxon p 0.62 / 0.18 / 0.36; circular p 0.90 / 0.078 / 0.059. Stride-to-stride: +0.0016 / +0.0006 / +0.0011; Wilcoxon p 0.27 / 0.25 / 0.79; circular p 0.039 / 0.078 / 0.078.
- The strongest [-600, -100] dependence (trial-centered T1, circular p 0.039) disappeared (p 0.90) once the other foot's impact was outside the window; the remaining circular p values are borderline (0.04-0.08) in both windows. No gain that generalizes to held-out subjects in any comparison.
- Power (N = 33, alpha 0.05/3): MDE trial-centered 0.01 / 0.01 / 0.005 (G2 passes); stride-to-stride 0.05 / 0.02 / 0.05 (secondary analysis, lower power; its nulls are less informative). False-positive rate 0.014-0.016.
- Decision: freeze with CELL 16a (N4_confirm_cells.py) and run the primary confirm test (CELL 16b/16c).

## 13e. CONFIRMATORY RESULT (primary, 2026-10-09; frozen plan, 33 held-out confirm subjects, window [-400, -100] ms)

| Target | M0 R2 median (IQR) | delta R2 median (95% CI) | subjects with gain | p (gain > 0), Holm | p (gain < 0.01), Holm | Decision |
|---|---|---|---|---|---|---|
| T1 peak vGRF | 0.028 (-0.012..0.063) | 0.0000 (-0.0015..0.0013) | 52% | 1.0 | 3.5e-10 | no gain >= 0.01 |
| T2 loading rate | 0.194 (0.086..0.276) | 0.0000 (-0.0015..0.0024) | 52% | 1.0 | 4.7e-10 | no gain >= 0.01 |
| T3 early knee flexion | 0.082 (-0.025..0.207) | -0.0002 (-0.0027..0.0015) | 48% | 1.0 | 6.4e-9 | no gain >= 0.01 |

Uncorrected Wilcoxon p (gain > 0): 0.63 / 0.48 / 0.61. All CHECKs passed (plan hash unchanged, no dev subject held out, first run saved to confirm_primary.csv).
Secondary/sensitivity (CELL 16d, same pipeline, 33 confirm subjects): all 12 analysis x target rows "no gain >= 0.01". Stride-to-stride -0.0016 / -0.0007 / +0.0004 (equivalence Holm p 2.5e-7 / 3.2e-6 / 3.5e-9; wider CIs, lower power as expected); window [-600, -100] ms -0.0002 / +0.0002 / -0.0003; + relaxed tier and odd-mapping trials +0.0003 / +0.0003 / -0.0003; per leg and per speed all |median delta R2| <= 0.0022. Re-derived primary reproduced CELL 16c.
Circular-shift null (CELL 16e, 100 within-trial rotations, confirm subjects): primary window [-400, -100] ms p = 0.386 / 0.248 / 0.921 (Holm 0.77 / 0.74 / 0.92); exploratory window [-600, -100] ms p = 0.881 / 0.129 / 0.950. No EEG-landing dependence of any kind in either window. Correction to the dev-phase interpretation: the weak dependence seen on the 14 dev subjects with [-600, -100] ms (circular p 0.02-0.04) did not replicate in the confirm subjects, so the paper must NOT claim that the other foot's impact produced an artifactual dependence; the window change stays a design decision (an impact transient was excluded a priori), reported with this exploratory null.
Descriptive (CELL 16f, confirm subjects, no claims): broadband-power two-stage model median delta R2 0.0000 / +0.0018 / -0.0005 (95% CI -0.0009..0.0005 / -0.0006..0.0037 / -0.0015..0.0020; uncorrected Wilcoxon p 0.63 / 0.09 / 0.60), i.e. even signal + artifact power adds nothing beyond M0 with head kinematics; dose-response Spearman rho between subject artifact index and primary delta R2 -0.19 (p 0.28) / -0.27 (p 0.13) / +0.14 (p 0.43). All results exported to paper_results_w400.zip (42 files, now in the project folder).
Deviations from the frozen plan (decided 2026-10-12, after the primary, secondary and circular results; reported as deviations in the paper):
1. Cross-speed analysis not performed: the primary result and every per-speed result showed no gain, so there was no gain whose transfer across speeds could be tested.
2. CNN arm not performed in this paper (moved to future work / limitations): all linear analyses were null with equivalence and the circular-shift test showed no dependence of any kind; claims are restricted to linear band-power features.
Outcome = section 9 row 3 ("no gain anywhere") with an equivalence result: pre-landing dry-EEG relative band power did not improve held-out prediction of stride-to-stride landing mechanics by delta R2 >= 0.01 beyond IMU + EMG; the 95% CIs exclude gains above 0.0024.

Superseded open items (kept for the record) -- were to be decided at the freeze after CELL 15b/15c on the real dev subjects: primary variant (trial-centered vs stride-to-stride), legs pooled vs per leg (CELL 15 pooled them; section 5 said per leg), SESOI for an equivalence statement, and whether the CNN arm stays in scope.
