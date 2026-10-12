# ===== CELL P · Prepare the [-400, -100] ms rebuild: what CELL 14w needs + a new output folder (~10 s) =====
# Freeze decision 2026-10-09: primary pre-landing window [-400, -100] ms, so the other foot's heel strike (one step,
# 0.56-0.73 s earlier) is outside the window at every speed; [-600, -100] stays as a sensitivity analysis.
# Definitions below are verbatim copies from CELL 6, 11, 11b and 12, so CELL 6-13 need not be re-run.
# OUT_ROOT is switched to the sub-folder "w400": nothing from the [-600, -100] run is overwritten.
import json
import shutil
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, hilbert

if "inv" not in globals():
    raise RuntimeError("run CELL 1, 2, 3 and 4 first (raw data + IMU extraction)")
BASE_OUT = OUT_ROOT.parent if OUT_ROOT.name == "w400" else OUT_ROOT     # safe to re-run this cell
W_OUT = BASE_OUT / "w400"
IMU_FS = 60.0
EEG_SHIFT = 0.025                                  # CELL 10: EEG time = IMU time + 0.025 s
BAND = (1.0, 45.0)
REG_BAND = (1.0, 25.0)                             # CELL 11b
MOTION_LAGS_S = (-0.06, -0.03, 0.0, 0.03, 0.06)
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
EEG_EXCLUDE = {"S32_1": "channel labels corrupted in CSV and EDF"}
qc3 = pd.read_csv(BASE_OUT / "trials_qc_v3.csv")
strides_v3 = pd.read_csv(BASE_OUT / "strides_v3.csv")
st_ok = qc3[(qc3.tier == "strict") & np.isfinite(qc3.near_mismatch_ms)]
odd = st_ok.loc[st_ok.near_mismatch_ms >= st_ok.near_swapped_mismatch_ms, "tag"].tolist()


def one(pattern: str, root: Path) -> Path:
    hits = sorted(root.rglob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected 1 match for {pattern} under {root}, found {len(hits)}")
    return hits[0]


hdr = pd.read_csv(one("S1_0.5_raw.csv", DATA), comment="#", nrows=1).columns
eeg_ch = list(hdr[1:hdr.get_loc("Trigger")])
scalp = [c for c in eeg_ch if c not in ("CM", "A1", "A2", "X1", "X2", "X3")]


def load_eeg(tg: str) -> tuple:
    """Scalp EEG (channels x samples, uV) and its rate; time rebuilt from the sample count (file Time is rounded)."""
    df = pd.read_csv(one(f"{tg}_raw.csv", DATA), comment="#", usecols=["Time"] + scalp, low_memory=False)
    t_file = df["Time"].to_numpy(float)
    fs = (len(t_file) - 1) / (t_file[-1] - t_file[0])
    return df[scalp].to_numpy(float).T, fs


def preprocess(x: np.ndarray, fs: float) -> tuple:
    """Add Pz (recording reference) as a zero row, notch 50/100 Hz, band-pass, flag bad channels, common average."""
    x = np.vstack([x - np.median(x, axis=1, keepdims=True), np.zeros((1, x.shape[1]))])
    for f0 in (50.0, 100.0):
        bn_, an_ = iirnotch(f0, 30.0, fs)
        x = filtfilt(bn_, an_, x, axis=1)
    b_, a_ = butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)], btype="band")
    x = filtfilt(b_, a_, x, axis=1)
    sd = x[:-1].std(axis=1)
    med = float(np.median(sd))
    bad = np.flatnonzero((sd < 0.5) | (sd > 3 * med) | (sd < med / 5))
    good = np.setdiff1d(np.arange(x.shape[0]), bad)
    return x - x[good].mean(axis=0, keepdims=True), bad


def head_regressors(tg: str, n: int, fs: float) -> np.ndarray:
    """Head acceleration (Segment_Acceleration) on the EEG clock, band-passed 1-25 Hz, with lags."""
    acc = pd.read_csv(one(f"IMU_{tg}", IMU_ROOT) / "Segment_Acceleration.csv", usecols=["Head x", "Head y", "Head z"]).to_numpy(float)
    t_imu = np.arange(len(acc)) / IMU_FS
    t_eeg = np.arange(n) / fs
    a = np.column_stack([np.interp(t_eeg - EEG_SHIFT, t_imu, acc[:, k]) for k in range(3)])
    b_, a_ = butter(4, [REG_BAND[0] / (fs / 2), REG_BAND[1] / (fs / 2)], btype="band")
    a = filtfilt(b_, a_, a, axis=0)
    lagged = [np.roll(a, int(round(l_ * fs)), axis=0) for l_ in MOTION_LAGS_S]
    return np.column_stack(lagged + [np.ones(n)])


def regress_out(x: np.ndarray, R: np.ndarray) -> tuple:
    """Least-squares removal of the head-motion regressors from every channel; returns residual and R^2 per channel."""
    beta_, *_ = np.linalg.lstsq(R, x.T, rcond=None)
    fitted = (R @ beta_).T
    return x - fitted, 1 - (x - fitted).var(axis=1) / x.var(axis=1)


W_OUT.mkdir(exist_ok=True)
shared = ["dev_confirm_split_v2.json", "dev_confirm_split.json", "g1_trials.csv"]   # same subjects, same artifact index
for f in shared:
    if (BASE_OUT / f).exists() and not (W_OUT / f).exists():
        shutil.copy2(BASE_OUT / f, W_OUT / f)
old_cols = pd.read_parquet(BASE_OUT / "stride_features.parquet").columns
OUT_ROOT = W_OUT
log(f"OUT_ROOT is now {OUT_ROOT} (the [-600, -100] results stay in {BASE_OUT})")

check(len(scalp) == 18, f"{len(scalp)} scalp channels from the EEG header (expected 18)")
check([c for c in old_cols if c.startswith("eeg_bb_")] == [f"eeg_bb_{c}" for c in scalp + ["Pz"]],
      "channel order matches the [-600, -100] stride table")
check(all((W_OUT / f).exists() for f in shared[:1] + shared[2:]), "frozen split v2 and g1_trials.csv copied to the new folder")
check(len(qc3) > 0 and len(strides_v3) > 0, f"trial QC ({len(qc3)} trials) and stride clock ({len(strides_v3)} strides) loaded")
end_cell("CELL P")

# ===== CELL 14w · Stride table with the primary window [-400, -100] ms -> folder w400 (~5-8 min) =====
# Same code as CELL 14 except the window, a folder guard, and two checks against the [-600, -100] table.
import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)  # bad channels are NaN by design
need =["load_eeg", "preprocess", "head_regressors", "regress_out", "strides_v3", "qc3", "EEG_EXCLUDE", "BANDS", "scalp", "one"]
missing_defs = [n for n in need if n not in globals()]
if missing_defs:
    raise RuntimeError(f"run CELL P first (missing: {missing_defs})")
if OUT_ROOT.name != "w400":
    raise RuntimeError("run CELL P first: it switches OUT_ROOT to the w400 folder, so the [-600, -100] table is kept")
WIN = (-0.4, -0.1)        # s relative to heel strike; the other foot's heel strike (0.56-0.73 s earlier) stays outside
ch_all = scalp + ["Pz"]
FEAT_PATH = OUT_ROOT / "stride_features.parquet"


def find_col(cols: pd.Index, *parts: str) -> str:
    """First column whose name contains every part (case-insensitive); '' if none."""
    hits = [c for c in cols if all(p.lower() in c.lower() for p in parts)]
    return hits[0] if hits else ""


def frames(t0: float, t1: float, fs: float, n: int) -> slice:
    i0, i1 = int(np.floor(t0 * fs)), int(np.ceil(t1 * fs))
    return slice(i0, i1) if 0 <= i0 < i1 <= n else slice(0, 0)


def trial_features(tg: str, st: pd.DataFrame) -> list:
    """One dict per stride of this trial: identifiers, targets, EEG/IMU/EMG pre-landing features."""
    x0, fs = load_eeg(tg)
    xa, bad = preprocess(x0, fs)
    xc, _ = regress_out(xa, head_regressors(tg, xa.shape[1], fs))          # motion-regressed (1-25 Hz regressors)
    xc[bad] = np.nan
    n_e = xc.shape[1]
    band_pw = {}
    for b, (lo, hi) in BANDS.items():
        b_, a_ = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
        band_pw[b] = np.abs(hilbert(filtfilt(b_, a_, np.nan_to_num(xc), axis=1), axis=1)) ** 2
    broad = xc ** 2

    d = one(f"IMU_{tg}", IMU_ROOT)
    acc = pd.read_csv(d / "Segment_Acceleration.csv")
    angv = pd.read_csv(d / "Segment_Angular_Velocity.csv")
    ja = pd.read_csv(d / "Joint_Angles_ZXY.csv")
    emg = pd.read_csv(one(f"EMG_{tg}.csv", DATA))
    t_emg = emg.iloc[:, 0].to_numpy(float)                                # EMG shares the IMU clock (CELL 9b)
    emg_names = [c.split(":")[0].replace(" ", "_") for c in emg.columns[1:]]
    n_i = len(acc)

    rows = []
    for r in st.itertuples():
        side = "Left" if r.leg == "L" else "Right"
        f = {"subject": r.subject, "speed": r.speed, "leg": r.leg, "tier": r.tier, "hs_eeg": r.hs_eeg, "stride_s": r.stride_s,
             "T1_peak_fz": r.peak_fz_25, "T2_loading_rate": r.loading_rate}
        se = frames(r.hs_eeg + WIN[0], r.hs_eeg + WIN[1], fs, n_e)
        f["eeg_ok"] = se.stop > se.start
        if f["eeg_ok"]:
            bb = np.nanmean(broad[:, se], axis=1)
            for ci, ch in enumerate(ch_all):
                f[f"eeg_bb_{ch}"] = np.log10(bb[ci])
            for b in BANDS:
                p_ = np.nanmean(band_pw[b][:, se], axis=1)
                p_[bad] = np.nan
                for ci, ch in enumerate(ch_all):
                    f[f"eeg_abs_{b}_{ch}"] = np.log10(p_[ci])
                    f[f"eeg_rel_{b}_{ch}"] = np.log10(p_[ci] / bb[ci])
        si = frames(r.hs_imu + WIN[0], r.hs_imu + WIN[1], IMU_FS, n_i)
        f["imu_ok"] = si.stop > si.start
        if f["imu_ok"]:
            for ax_ in "xyz":
                h_ = acc[f"Head {ax_}"].to_numpy(float)[si]
                f[f"imu_head_acc_{ax_}_mean"], f[f"imu_head_acc_{ax_}_rms"] = h_.mean(), np.sqrt(np.mean(h_ ** 2))
                f[f"imu_pelvis_acc_{ax_}_mean"] = acc[f"Pelvis {ax_}"].to_numpy(float)[si].mean()
                fw = angv[f"{side} Foot {ax_}"].to_numpy(float)[si]
                f[f"imu_foot_angv_{ax_}_mean"], f[f"imu_foot_angv_{ax_}_last"] = fw.mean(), fw[-1]
                f[f"imu_shank_angv_{ax_}_last"] = angv[f"{side} Lower Leg {ax_}"].to_numpy(float)[si][-1]
            hv = acc[[f"Head {a}" for a in "xyz"]].to_numpy(float)[si]
            f["imu_head_acc_norm_rms"] = np.sqrt(np.mean(np.sum(hv ** 2, axis=1)))
            for jn, parts in (("hip", ("Hip", "Flexion")), ("knee", ("Knee", "Flexion")), ("ankle", ("Ankle", "flexion"))):
                col = find_col(ja.columns, side, *parts)
                if col:
                    v = ja[col].to_numpy(float)[si]
                    f[f"imu_{jn}_angle_mean"], f[f"imu_{jn}_angle_last"] = v.mean(), v[-1]
            knee = find_col(ja.columns, side, "Knee", "Flexion")
            ss = frames(r.hs_imu, r.hs_imu + 0.25 * (r.to_imu - r.hs_imu), IMU_FS, n_i)
            f["T3_knee_flex_early"] = ja[knee].to_numpy(float)[ss].max() if knee and ss.stop > ss.start else np.nan
        m_ = (t_emg >= r.hs_imu + WIN[0]) & (t_emg < r.hs_imu + WIN[1])
        if m_.sum() >= 2:
            vals = emg.iloc[:, 1:].to_numpy(float)[m_].mean(axis=0)
            for nm, v in zip(emg_names, vals):
                f[f"emg_{nm}"] = v
        rows.append(f)
    return rows


trial_list = qc3[(qc3.tier != "excluded") & ~qc3.tag.isin(list(EEG_EXCLUDE))].reset_index(drop=True)
log(f"building features for {len(trial_list)} trials ({trial_list.tier.value_counts().to_dict()})")
all_rows, failed_f = [], {}
t0_f = time.time()
for k_, r in enumerate(trial_list.itertuples(), 1):
    st = strides_v3[(strides_v3.subject == r.subject) & (strides_v3.speed == r.speed)].dropna(subset=["hs_eeg"])
    try:
        all_rows += trial_features(r.tag, st)
    except (OSError, ValueError, KeyError, IndexError, np.linalg.LinAlgError) as exc:
        failed_f[r.tag] = f"{type(exc).__name__}: {str(exc)[:120]}"
    if k_ % 10 == 0 or k_ == len(trial_list):
        log(f"{k_}/{len(trial_list)} trials | {len(all_rows)} strides | failed {len(failed_f)} | {time.time() - t0_f:.0f}s")

feat = pd.DataFrame(all_rows)
feat["odd_mapping"] = [f"S{s}_{sp:g}" in set(odd) for s, sp in zip(feat.subject, feat.speed)] if "odd" in globals() else False
feat.to_parquet(FEAT_PATH, index=False)

groups = {g: [c for c in feat.columns if c.startswith(g)] for g in ("eeg_bb_", "eeg_abs_", "eeg_rel_", "imu_", "emg_", "T")}
print(f"\nstride table: {feat.shape[0]} strides x {feat.shape[1]} columns -> {FEAT_PATH}")
print("feature counts:", {g: len(c) for g, c in groups.items()})
print("joint-angle columns used (first trial, left):",
      {jn: find_col(pd.read_csv(one(f"IMU_{trial_list.tag[0]}", IMU_ROOT) / "Joint_Angles_ZXY.csv", nrows=1).columns, "Left", *pt)
       for jn, pt in (("hip", ("Hip", "Flexion")), ("knee", ("Knee", "Flexion")), ("ankle", ("Ankle", "flexion")))})
print("share of strides with a complete EEG / IMU window:", round(float(feat.eeg_ok.mean()), 3), "/", round(float(feat.imu_ok.mean()), 3))
print("\ntargets by speed (median):")
print(feat.groupby("speed")[["T1_peak_fz", "T2_loading_rate", "T3_knee_flex_early", "stride_s"]].median().round(2).to_string())
if failed_f:
    print("failed:", failed_f)

check(len(failed_f) <= 0.1 * len(trial_list), f"{len(trial_list) - len(failed_f)}/{len(trial_list)} trials -> features")
check(feat.eeg_ok.mean() > 0.9, f"{100 * feat.eeg_ok.mean():.0f}% of strides have a complete EEG window", critical=False)
check("T3_knee_flex_early" in feat and feat.T3_knee_flex_early.notna().mean() > 0.9,
      "knee-flexion target found for > 90% of strides", critical=False)
check(len(groups["eeg_rel_"]) == len(BANDS) * len(ch_all), f"{len(groups['eeg_rel_'])} relative-power EEG features")
old = pd.read_parquet(BASE_OUT / "stride_features.parquet")
key = ["subject", "speed", "leg", "hs_eeg"]
same_rows = len(old) == len(feat) and bool(np.allclose(old[key[:2] + key[3:]].to_numpy(float), feat[key[:2] + key[3:]].to_numpy(float), equal_nan=True)
                                           and (old.leg.to_numpy() == feat.leg.to_numpy()).all())
check(same_rows, f"same {len(feat)} strides in the same order as the [-600, -100] table")
t3_both = old.T3_knee_flex_early.notna() & feat.T3_knee_flex_early.notna()
check(same_rows and np.allclose(old[["T1_peak_fz", "T2_loading_rate"]].to_numpy(float), feat[["T1_peak_fz", "T2_loading_rate"]].to_numpy(float), equal_nan=True)
      and np.allclose(old.T3_knee_flex_early[t3_both], feat.T3_knee_flex_early[t3_both]),
      "targets identical to the [-600, -100] table (they do not depend on the window)")
print(f"complete EEG windows: {old.eeg_ok.mean():.3f} ([-600, -100]) -> {feat.eeg_ok.mean():.3f} ([-400, -100])")
end_cell("CELL 14w")
