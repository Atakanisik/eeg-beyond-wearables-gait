# ===== CELL 14 · Stride table: pre-landing EEG / IMU / EMG features + landing targets (~5-8 min) =====
import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)  # bad channels are NaN by design
need =["load_eeg", "preprocess", "head_regressors", "regress_out", "strides_v3", "qc3", "EEG_EXCLUDE", "BANDS", "scalp", "one"]
missing_defs = [n for n in need if n not in globals()]
if missing_defs:
    raise RuntimeError(f"run CELL 11, 11b and 12 first (missing: {missing_defs})")
WIN = (-0.6, -0.1)        # s relative to heel strike: pre-landing window, 100 ms guard before impact
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
end_cell("CELL 14")

# ===== CELL 15 (v2) · Dev/confirm split + main test on the DEV subjects only (~3-5 min) =====
# v2 fixes: (1) numeric BMI made every subject its own stratum -> only 3 dev subjects; now BMI is binned and the
# 3 subjects already seen stay in dev; (2) features/targets z-scored within trial and features clipped at +-5 SD,
# because scale differences between subjects made the mechanics model explode on held-out subjects (R2 down to -55).
import json
from scipy.stats import wilcoxon, spearmanr
from sklearn.linear_model import RidgeCV

feat = pd.read_parquet(OUT_ROOT / "stride_features.parquet")
TARGETS = ["T1_peak_fz", "T2_loading_rate", "T3_knee_flex_early"]
M0_COLS = [c for c in feat.columns if c.startswith(("imu_", "emg_"))]       # mechanics incl. head kinematics + EMG
EEG_REL = [c for c in feat.columns if c.startswith("eeg_rel_")]             # primary EEG features (amendment 13b)
EEG_BB = [c for c in feat.columns if c.startswith("eeg_bb_")]               # broadband power (signal + artifact)
ALPHAS = np.logspace(-2, 4, 25)
N_PERM_S1, DEV_FRACTION, SEED = 50, 0.3, 20261008
Z_CLIP, Y_OUTLIER = 5.0, 4.0
SPLIT_V1 = OUT_ROOT / "dev_confirm_split.json"         # faulty v1 (kept as a record)
SPLIT_PATH = OUT_ROOT / "dev_confirm_split_v2.json"
rng_m = np.random.default_rng(SEED)

prim = feat[(feat.tier == "strict") & ~feat.odd_mapping.astype(bool) & feat.eeg_ok & feat.imu_ok].copy()
prim["trial"] = prim.subject.astype(str) + "_" + prim.speed.astype(str) + "_" + prim.leg
num = TARGETS + M0_COLS + EEG_REL + EEG_BB
grp = prim.groupby("trial")[num]
prim[num] = (prim[num] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)   # within-trial z-scores
xcols = M0_COLS + EEG_REL + EEG_BB
clipped_share = float((prim[xcols].abs() > Z_CLIP).to_numpy().mean())
prim[xcols] = prim[xcols].clip(-Z_CLIP, Z_CLIP)
subjects_all = sorted(int(s) for s in prim.subject.unique())


def strata_from_info(subjects: list) -> tuple:
    """gender | BMI category per subject; numeric BMI is cut at the WHO limits 25 and 30."""
    strata = {s: "all" for s in subjects}
    info_path = next(iter(DATA.rglob("subject-info.csv")), None)
    if info_path is None:
        return strata, "subject-info.csv not found -> random split"
    info = pd.read_csv(info_path)
    print("subject-info columns:", list(info.columns)[:12])
    gcol = next((c for c in info.columns if any(k in c.lower() for k in ("gender", "sex"))), None)
    bcols = [c for c in info.columns if "bmi" in c.lower()]
    bcol = next((c for c in bcols if "cat" in c.lower()), bcols[0] if bcols else None)
    bmi = info[bcol] if bcol else pd.Series([""] * len(info))
    if bcol and pd.api.types.is_numeric_dtype(bmi):
        bmi = pd.cut(bmi, [0, 25, 30, np.inf], labels=["normal", "overweight", "obese"]).astype(str)
    ids = info[info.columns[0]].astype(str).str.extract(r"(\d+)")[0].astype(float)
    for i, sid in enumerate(ids):
        if np.isfinite(sid) and int(sid) in strata:
            strata[int(sid)] = f"{info[gcol].iloc[i] if gcol else ''}|{bmi.iloc[i]}"
    return strata, f"gender={gcol!r}, bmi={bcol!r}"


if SPLIT_PATH.exists():
    split = json.loads(SPLIT_PATH.read_text())
    log(f"loaded frozen split v2 (created {split['created']}) - never re-drawn")
else:
    keep = [int(s) for s in json.loads(SPLIT_V1.read_text())["dev"]] if SPLIT_V1.exists() else []
    strata, used_cols = strata_from_info(subjects_all)
    dev = [s for s in keep if s in subjects_all]
    for stratum in sorted(set(strata.values())):
        members = [s for s in subjects_all if strata[s] == stratum]
        quota = int(round(DEV_FRACTION * len(members))) - sum(strata[s] == stratum for s in dev)
        pool = [s for s in members if s not in dev]
        if quota > 0 and pool:
            dev += [int(x) for x in rng_m.choice(pool, size=min(quota, len(pool)), replace=False)]
    rest = [s for s in subjects_all if s not in dev]
    while len(dev) < round(DEV_FRACTION * len(subjects_all)) and rest:
        dev.append(rest.pop(int(rng_m.integers(len(rest)))))
    split = {"dev": sorted(dev), "confirm": [s for s in subjects_all if s not in dev], "seed": SEED, "fraction": DEV_FRACTION,
             "strata_columns": used_cols, "kept_from_v1": keep, "created": time.strftime("%Y-%m-%d %H:%M"),
             "note": "v1 had too few dev subjects (numeric BMI -> one stratum per subject); v1 dev subjects kept in dev"}
    SPLIT_PATH.write_text(json.dumps(split, indent=1))
    log(f"split v2 saved ({used_cols}); kept {len(keep)} v1 dev subjects")
dev_subj = [int(s) for s in split["dev"]]
dev_d = prim[prim.subject.isin(dev_subj)]
print(f"dev subjects ({len(dev_subj)}): {dev_subj} | {len(dev_d)} strides | confirm subjects: {len(split['confirm'])} (not touched)")
print(f"feature values clipped at +-{Z_CLIP} SD: {100 * clipped_share:.2f}%")


def ridge_fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    return RidgeCV(alphas=ALPHAS).fit(Xtr, ytr).predict(Xte)       # inputs are already within-trial z-scores


def r2_vs_trial_mean(y: np.ndarray, yhat: np.ndarray) -> float:
    """y is a within-trial z-score, so the trial mean (= 0) is the baseline prediction."""
    return float(1 - np.sum((y - yhat) ** 2) / np.sum(y ** 2))


def within_trial_perm(trial_ids: np.ndarray) -> np.ndarray:
    out = np.arange(len(trial_ids))
    for t in np.unique(trial_ids):
        pos = np.flatnonzero(trial_ids == t)
        out[pos] = pos[rng_m.permutation(len(pos))]
    return out


def loso_r2(d: pd.DataFrame, target: str, cols: list, perm_cols: tuple = (), perm_idx=None) -> dict:
    """Leave-one-subject-out R2 per held-out subject; perm_cols rows reordered by perm_idx (within-trial shuffle)."""
    y = d[target].to_numpy(float)
    X = d[cols].fillna(0.0).to_numpy(float)
    if perm_idx is not None:
        pc = [cols.index(c) for c in perm_cols]
        X[:, pc] = X[perm_idx][:, pc]
    out = {}
    for s in sorted(d.subject.unique()):
        te = (d.subject == s).to_numpy()
        out[int(s)] = r2_vs_trial_mean(y[te], ridge_fit_predict(X[~te], y[~te], X[te]))
    return out


art = pd.read_csv(OUT_ROOT / "g1_trials.csv").assign(a=lambda t: t.artifact_a.abs()).groupby("subject").a.mean()
res_rows, subj_rows = [], []
fig, ax = plt.subplots(1, len(TARGETS), figsize=(15, 4))
for ti, t in enumerate(TARGETS):
    d = dev_d[dev_d[t].abs() <= Y_OUTLIER].reset_index(drop=True)     # also drops NaN targets
    n_out = int((dev_d[t].abs() > Y_OUTLIER).sum())
    r0, r1, rbb = loso_r2(d, t, M0_COLS), loso_r2(d, t, M0_COLS + EEG_REL), loso_r2(d, t, M0_COLS + EEG_BB)
    subs = sorted(r0)
    dR = np.array([r1[s] - r0[s] for s in subs])
    dB = np.array([rbb[s] - r0[s] for s in subs])
    tid = d.trial.to_numpy()
    perm_med = np.array([np.median([v - r0[s] for s, v in loso_r2(d, t, M0_COLS + EEG_REL, tuple(EEG_REL),
                                                                    within_trial_perm(tid)).items()])
                         for _ in range(N_PERM_S1)])
    p_perm = (1 + int((perm_med >= np.median(dR)).sum())) / (N_PERM_S1 + 1)
    p_wil = wilcoxon(dR, alternative="greater").pvalue if len(dR) >= 5 else float("nan")
    a_sub = np.array([art.get(s, np.nan) for s in subs])
    ok = np.isfinite(a_sub)
    rho = spearmanr(a_sub[ok], dR[ok]).correlation if ok.sum() >= 5 else float("nan")
    r0v = np.array([r0[s] for s in subs])
    res_rows.append({"target": t, "n_strides": len(d), "outliers_dropped": n_out,
                     "M0_R2_median": round(float(np.median(r0v)), 4), "M0_R2_min": round(float(r0v.min()), 3),
                     "dR2_EEGrel_median": round(float(np.median(dR)), 4), "dR2_EEGrel_IQR": f"{np.percentile(dR, 25):.4f}..{np.percentile(dR, 75):.4f}",
                     "p_wilcoxon": round(float(p_wil), 4), "p_shuffled_EEG": round(p_perm, 4),
                     "dR2_broadband_median": round(float(np.median(dB)), 4), "rho_dR2_vs_artifact": round(float(rho), 3)})
    subj_rows += [{"target": t, "subject": s, "R2_M0": r0[s], "dR2_EEGrel": r1[s] - r0[s], "dR2_broadband": rbb[s] - r0[s]} for s in subs]
    ax[ti].axhline(0, color="grey", lw=0.6)
    ax[ti].axhline(np.percentile(perm_med, 95), color="tab:red", ls="--", lw=0.8, label="shuffled-EEG 95th pct (median)")
    ax[ti].scatter(np.zeros(len(dR)) + rng_m.normal(0, 0.03, len(dR)), dR, s=18, label="EEG relative power")
    ax[ti].scatter(np.ones(len(dB)) + rng_m.normal(0, 0.03, len(dB)), dB, s=18, label="broadband (signal + artifact)")
    ax[ti].set(xticks=[0, 1], xticklabels=["M1 - M0", "M0+bb - M0"], ylabel="delta R2 per dev subject", title=t)
ax[0].legend(fontsize=7)
fig.suptitle("DEV subjects only: what EEG adds beyond mechanics (stride-to-stride, leave-one-subject-out)")
fig.tight_layout()
fig.savefig(OUT_ROOT / "cell15_dev.png", dpi=110)
plt.show()

dev_res = pd.DataFrame(res_rows)
dev_res.to_csv(OUT_ROOT / "dev_results.csv", index=False)
pd.DataFrame(subj_rows).to_csv(OUT_ROOT / "dev_subject_deltas.csv", index=False)
print("\nDEV results (guide the frozen plan; NOT the confirmatory test):")
print(dev_res.to_string(index=False))

check(SPLIT_PATH.exists(), f"split frozen in {SPLIT_PATH.name}")
check(len(dev_subj) >= 0.2 * len(subjects_all), f"{len(dev_subj)} dev subjects out of {len(subjects_all)} (>= 20%)")
check(not set(dev_d.subject) & set(split["confirm"]), "no confirm subject used in this cell")
check(bool((dev_res.M0_R2_min > -1).all()), f"models are numerically stable (lowest per-subject M0 R2 {dev_res.M0_R2_min.min():.2f} > -1)", critical=False)
check(bool((dev_res.M0_R2_median > 0.01).any()),
      "mechanics (M0) predicts some stride-to-stride landing variability (median R2 > 0.01 for at least one target)", critical=False)
print("note: broadband power carries both signal and artifact, so its delta R2 is descriptive; "
      "the artifact controls are the shuffled-EEG null and the dose-response rho.")
end_cell("CELL 15")

# ===== CELL 15b · Two-stage model + slow-drift check, DEV subjects only (~6-9 min) =====
# Three problems found on synthetic data before freezing the plan (CELL 15 results stay as a record):
# 1) One joint ridge (CELL 15): 76 uninformative EEG features raise the shared penalty, which can help the mechanics
#    part generalize -> delta R2 > 0 without EEG information. Now two stages: fit M0; make EEG orthogonal to the M0
#    features (training subjects); ridge of the M0 residuals on that EEG. Useless EEG only adds noise here, so
#    delta R2 <= 0 on average and the one-sided Wilcoxon over subjects against 0 is valid (primary test).
# 2) The within-trial shuffle null is too optimistic when EEG and target are both autocorrelated over strides. Null is
#    now a circular shift: each trial's EEG sequence is rotated by >= 3 strides (keeps its own autocorrelation).
# 3) Slow drifts shared by EEG and target (electrode settling, gait adaptation) are real dependence but not stride-level
#    information. Re-test on stride-to-stride changes (stride n minus stride n-1 of the same leg): slow drifts cancel.
if "loso_r2" not in globals() or "dev_d" not in globals():
    raise RuntimeError("CELL 15 variables are missing (runtime restarted?) -> run CELL 1 and CELL 15 first")
ALPHAS_E = np.logspace(0, 6, 25)                  # EEG stage: allows strong shrinkage, so useless EEG predicts ~0
MIN_SHIFT = 3                                     # circular-shift null: rotate by at least 3 strides
NUM_COLS = TARGETS + M0_COLS + EEG_REL + EEG_BB
X_COLS = M0_COLS + EEG_REL + EEG_BB
rng_m = np.random.default_rng(SEED + 1)


def time_ordered(d: pd.DataFrame) -> pd.DataFrame:
    return d.sort_values(["trial", "hs_eeg"]).reset_index(drop=True)


def differenced(d: pd.DataFrame) -> pd.DataFrame:
    """Stride n minus stride n-1 of the same trial (consecutive strides only), re-z-scored within trial, features clipped."""
    d = time_ordered(d)
    gap = d.groupby("trial", sort=False).hs_eeg.diff()
    consecutive = (gap <= 1.5 * gap.groupby(d.trial).transform("median")).to_numpy()   # False for first stride / gaps
    out = d.copy()
    out[NUM_COLS] = d.groupby("trial", sort=False)[NUM_COLS].diff()
    out = out[consecutive].reset_index(drop=True)
    g = out.groupby("trial")[NUM_COLS]
    out[NUM_COLS] = (out[NUM_COLS] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    out[X_COLS] = out[X_COLS].clip(-Z_CLIP, Z_CLIP)
    return out


def within_trial_roll(trial_ids: np.ndarray) -> np.ndarray:
    """Row index that rotates each trial's (time-ordered) rows by a random offset >= MIN_SHIFT."""
    out = np.arange(len(trial_ids))
    for t in np.unique(trial_ids):
        pos = np.flatnonzero(trial_ids == t)
        if len(pos) > 2 * MIN_SHIFT:
            out[pos] = np.roll(pos, int(rng_m.integers(MIN_SHIFT, len(pos) - MIN_SHIFT + 1)))
    return out


def m0_folds(d: pd.DataFrame, target: str) -> dict:
    """Mechanics model fitted once per held-out subject; keeps what the EEG stage needs."""
    y, X0 = d[target].to_numpy(float), d[M0_COLS].fillna(0.0).to_numpy(float)
    folds = []
    for s in sorted(d.subject.unique()):
        te = (d.subject == s).to_numpy()
        m0 = RidgeCV(alphas=ALPHAS).fit(X0[~te], y[~te])
        folds.append({"s": int(s), "te": te, "p0": m0.predict(X0[te]), "res": y[~te] - m0.predict(X0[~te]),
                      "proj": np.linalg.pinv(X0[~te]), "x0tr": X0[~te], "x0te": X0[te]})
    return {"y": y, "folds": folds}


def stage_r2(c: dict, E: np.ndarray) -> tuple:
    """Per held-out subject: R2 of M0, and of M0 + EEG stage (EEG orthogonal to M0, ridge on the M0 residuals)."""
    r0, r1 = {}, {}
    for f in c["folds"]:
        te = f["te"]
        B = f["proj"] @ E[~te]                                     # EEG part that mechanics explains (training subjects)
        m1 = RidgeCV(alphas=ALPHAS_E).fit(E[~te] - f["x0tr"] @ B, f["res"])
        r0[f["s"]] = r2_vs_trial_mean(c["y"][te], f["p0"])
        r1[f["s"]] = r2_vs_trial_mean(c["y"][te], f["p0"] + m1.predict(E[te] - f["x0te"] @ B))
    return r0, r1


def stage_test(d: pd.DataFrame, t: str, n_perm: int) -> tuple:
    """Two-stage delta R2 per held-out subject, Wilcoxon against 0, and circular-shift null of the median."""
    c = m0_folds(d, t)
    E = d[EEG_REL].fillna(0.0).to_numpy(float)
    r0, r1 = stage_r2(c, E)
    subs = sorted(r0)
    dR = np.array([r1[s] - r0[s] for s in subs])
    null = []
    for _ in range(n_perm):
        q0, q1 = stage_r2(c, E[within_trial_roll(d.trial.to_numpy())])
        null.append(np.median([q1[s] - q0[s] for s in subs]))
    return {"M0_R2_median": round(float(np.median([r0[s] for s in subs])), 4), "dR2_median": round(float(np.median(dR)), 4),
            "dR2_IQR": f"{np.percentile(dR, 25):.4f}..{np.percentile(dR, 75):.4f}",
            "p_wilcoxon": round(float(wilcoxon(dR, alternative="greater").pvalue), 4),
            "p_circular": round((1 + int((np.array(null) >= np.median(dR)).sum())) / (n_perm + 1), 4),
            "null_median": round(float(np.median(null)), 4)}, dR


dev_raw, dev_df = time_ordered(dev_d), differenced(dev_d)
ac = dev_raw[NUM_COLS].corrwith(dev_raw.groupby("trial", sort=False)[NUM_COLS].shift(1))
print(f"stride-to-stride changes: {len(dev_df)} of {len(dev_raw)} strides have a consecutive previous stride")
print("lag-1 autocorrelation over strides (~0 = none):", {t: round(float(ac[t]), 2) for t in TARGETS},
      f"| median M0 feature {ac[M0_COLS].median():.2f} | median EEG rel feature {ac[EEG_REL].median():.2f}")

rows, subj_b = [], []
t0_b = time.time()
for vname, vdata in (("trial-centered", dev_raw), ("stride-to-stride", dev_df)):
    for t in TARGETS:
        d = vdata[vdata[t].abs() <= Y_OUTLIER].reset_index(drop=True)      # keeps time order within trial
        res, dR = stage_test(d, t, N_PERM_S1)
        rows.append({"variant": vname, "target": t, "n_strides": len(d), **res})
        subj_b += [{"variant": vname, "target": t, "dR2": v} for v in dR]
        log(f"{vname} / {t}: delta R2 median {res['dR2_median']:+.4f}, p_wilcoxon {res['p_wilcoxon']}, "
            f"p_circular {res['p_circular']} ({time.time() - t0_b:.0f}s)")

b_res = pd.DataFrame(rows)
b_res.to_csv(OUT_ROOT / "dev_two_stage_results.csv", index=False)
sb = pd.DataFrame(subj_b)
fig, ax = plt.subplots(1, len(TARGETS), figsize=(15, 3.8))
for ti, t in enumerate(TARGETS):
    for xi, vname in enumerate(("trial-centered", "stride-to-stride")):
        v = sb[(sb.variant == vname) & (sb.target == t)].dR2.to_numpy()
        ax[ti].scatter(xi + rng_m.normal(0, 0.03, len(v)), v, s=18)
        ax[ti].plot([xi - 0.2, xi + 0.2], [np.median(v)] * 2, color="k")
    ax[ti].axhline(0, color="grey", lw=0.6)
    ax[ti].set(xticks=[0, 1], xticklabels=["trial-centered", "stride-to-stride"], ylabel="delta R2 per dev subject", title=t)
fig.suptitle("DEV only, two-stage model: EEG gain beyond mechanics (line = median; <= 0 expected without EEG information)")
fig.tight_layout()
fig.savefig(OUT_ROOT / "cell15b_two_stage.png", dpi=110)
plt.show()
print("\nDEV, two-stage model (CELL 15 joint-model medians for comparison: "
      + ", ".join(f"{r.target} {r.dR2_EEGrel_median:+.4f}" for r in dev_res.itertuples()) + "):")
print(b_res.to_string(index=False))

check(len(dev_df) > 0.8 * len(dev_raw), f"{100 * len(dev_df) / len(dev_raw):.0f}% of strides have a consecutive previous stride (> 80%)")
check(bool((b_res.null_median <= 0.001).all()), f"EEG without alignment does not raise delta R2 (circular-null medians "
      f"{b_res.null_median.min():+.4f}..{b_res.null_median.max():+.4f} <= 0.001)")
for r_ in b_res.itertuples():
    check(r_.p_wilcoxon < 0.05 and r_.p_circular < 0.05, f"{r_.variant} / {r_.target}: EEG gain "
          f"(median {r_.dR2_median:+.4f}, p_wilcoxon {r_.p_wilcoxon}, p_circular {r_.p_circular})", critical=False)
print("⚠️ lines above are results, not errors: they say where the dev subjects show an EEG gain.")
end_cell("CELL 15b")

# ===== CELL 15c · Power gate G2: the smallest EEG gain the confirm run can detect (MDE), DEV subjects only (~5 min) =====
# Each run: (1) rotate each trial's EEG sequence (circular shift), so no real EEG-target link is left; (2) plant a known
# effect y' = y + b*s, s = mean of 6 of those EEG features (C3/Cz/C4 x alpha/beta) made orthogonal to M0 and y, so it
# adds q = b^2/(1+b^2) of variance beyond mechanics; (3) two-stage LOSO delta R2 per dev subject. Same rotations for
# every q, so the gain over q = 0 is paired. Power: resample subjects to the number of confirm subjects, value = no-effect
# value with random sign + paired gain, one-sided Wilcoxon at alpha 0.05/3 (Holm, worst case).
# Training here uses 13 subjects, the confirm run 46 -> the MDE is conservative.
if "stage_r2" not in globals():
    raise RuntimeError("run CELL 15b first")
from scipy.stats import norm, rankdata

Q_GRID = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
N_RUNS, N_BOOT, POWER_GOAL, G2_LIMIT = 3, 4000, 0.8, 0.03
N_CONFIRM, ALPHA_H = len(split["confirm"]), 0.05 / 3
PLANT_COLS = [f"eeg_rel_{b}_{c}" for b in ("alpha", "beta") for c in ("C3", "Cz", "C4")]
rng_m = np.random.default_rng(SEED + 2)
missing_plant = [c for c in PLANT_COLS if c not in EEG_REL]
if missing_plant:
    raise RuntimeError(f"planted-signal channels not found: {missing_plant}")


def planted(d: pd.DataFrame, t: str, q: float, roll: np.ndarray) -> tuple:
    """Copy of d with EEG rotated within trial (roll) and a planted EEG component worth q of target variance beyond M0."""
    out = d.copy()
    out[EEG_REL] = d[EEG_REL].to_numpy()[roll]
    Z = np.column_stack([d[M0_COLS].fillna(0.0).to_numpy(), d[t].to_numpy(), np.ones(len(d))])
    s = out[PLANT_COLS].fillna(0.0).mean(axis=1).to_numpy()
    s = s - Z @ np.linalg.lstsq(Z, s, rcond=None)[0]           # orthogonal to M0 and to the real target
    s = s / s.std()
    y = pd.Series(d[t].to_numpy() / d[t].std() + np.sqrt(q / (1 - q)) * s, index=d.index)
    g = y.groupby(d.trial)
    out[t] = (y - g.transform("mean")) / g.transform("std")
    return out, float(np.abs(Z[:, :-1].T @ s).max() / len(s))


def wilcoxon_power(x0: np.ndarray, gain: np.ndarray, n: int) -> float:
    """Share of size-n resamples (no-effect value with random sign + paired gain) where the one-sided Wilcoxon is significant."""
    i = rng_m.integers(0, len(x0), size=(N_BOOT, n))
    b = x0[i] * rng_m.choice([-1.0, 1.0], size=(N_BOOT, n)) + gain[i]
    w = np.where(b > 0, rankdata(np.abs(b), axis=1), 0).sum(axis=1)
    z = (w - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    return float((norm.sf(z) < ALPHA_H).mean())


pw_rows, max_cov = [], 0.0
t0_c = time.time()
for vname, vdata in (("trial-centered", dev_raw), ("stride-to-stride", dev_df)):
    for t in TARGETS:
        d = vdata[vdata[t].abs() <= Y_OUTLIER].reset_index(drop=True)
        rolls = [within_trial_roll(d.trial.to_numpy()) for _ in range(N_RUNS)]
        x0 = None
        for q in Q_GRID:
            dR = []
            for roll in rolls:
                dq, cov = planted(d, t, q, roll)
                max_cov = max(max_cov, cov)
                r0, r1 = stage_r2(m0_folds(dq, t), dq[EEG_REL].fillna(0.0).to_numpy(float))
                dR += [r1[s] - r0[s] for s in sorted(r0)]
            dR = np.array(dR)
            x0 = dR if q == 0 else x0
            pw_rows.append({"variant": vname, "target": t, "q_planted": q, "dR2_median": round(float(np.median(dR)), 4),
                            "gain_median": round(float(np.median(dR - x0)), 4),
                            "power_N_confirm": wilcoxon_power(x0, dR - x0, N_CONFIRM)})
        log(f"{vname} / {t} done ({time.time() - t0_c:.0f}s)")

pw = pd.DataFrame(pw_rows)
pw.to_csv(OUT_ROOT / "dev_power.csv", index=False)
idx = pd.MultiIndex.from_product([["trial-centered", "stride-to-stride"], TARGETS], names=["variant", "target"])
mde = pw[(pw.q_planted > 0) & (pw.power_N_confirm >= POWER_GOAL)].groupby(["variant", "target"]).q_planted.min().reindex(idx)
fig, ax = plt.subplots(1, len(TARGETS), figsize=(15, 3.8))
for ti, t in enumerate(TARGETS):
    for vname, ls in (("trial-centered", "-"), ("stride-to-stride", "--")):
        s_ = pw[(pw.variant == vname) & (pw.target == t)]
        ax[ti].plot(s_.q_planted, s_.power_N_confirm, "o" + ls, label=vname)
    ax[ti].axhline(POWER_GOAL, color="grey", lw=0.6)
    ax[ti].axvline(G2_LIMIT, color="tab:red", ls=":", lw=0.8, label="G2 limit 0.03")
    ax[ti].set(xlabel="planted EEG gain q (variance beyond M0)", ylabel=f"power at N = {N_CONFIRM}", title=t, ylim=(0, 1.02))
ax[0].legend(fontsize=7)
fig.suptitle("DEV only: power to detect a planted EEG gain in the confirm subjects (one-sided Wilcoxon, alpha 0.05/3)")
fig.tight_layout()
fig.savefig(OUT_ROOT / "cell15c_power.png", dpi=110)
plt.show()
print(pw.to_string(index=False))
print(f"\nMDE = smallest planted q with power >= {POWER_GOAL} at N = {N_CONFIRM} (NaN = above {max(Q_GRID)}):")
print(mde.unstack("target").to_string())

fp = pw[pw.q_planted == 0].power_N_confirm
top = pw[pw.q_planted == max(Q_GRID)].gain_median
check(max_cov < 1e-6, f"planted signal is orthogonal to M0 and to the real target (max covariance {max_cov:.1e})")
check(bool((fp <= 0.03).all()), f"false-positive rate with no planted effect ~ alpha {ALPHA_H:.3f} (max {fp.max():.3f})")
check(bool((top > 0.25 * max(Q_GRID)).all()), f"the model recovers the largest planted gain (median gain "
      f"{top.min():.3f}..{top.max():.3f} for q = {max(Q_GRID)})", critical=False)
for vname in ("trial-centered", "stride-to-stride"):
    check(bool((mde.loc[vname] <= G2_LIMIT).all()), f"G2 power gate, {vname}: MDE <= {G2_LIMIT} for all 3 targets "
          f"({', '.join(f'{v:.3f}' if v == v else '>' + str(max(Q_GRID)) for v in mde.loc[vname])})", critical=False)
end_cell("CELL 15c")
