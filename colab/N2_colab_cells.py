# ===== CELL 11 · EEG preprocessing, one trial: CAR, filters, bad channels, head-motion regression (~30 s) =====
from scipy.signal import welch, hilbert

if "scalp" not in globals():
    raise RuntimeError("variables missing (runtime reset?) -> run CELL 1, 4, 6, 7, 8 first")
qc3 = pd.read_csv(OUT_ROOT / "trials_qc_v3.csv")
strides_v3 = pd.read_csv(OUT_ROOT / "strides_v3.csv")
EEG_EXCLUDE = {"S32_1": "channel labels corrupted in CSV and EDF"}
EEG_SHIFT = 0.025          # CELL 10: EEG time = IMU time + 0.025 s
BAND = (1.0, 45.0)
REG_BAND = (1.0, 25.0)     # head-motion regressors: IMU is 60 Hz, above ~25 Hz interpolation only adds spurious images
MOTION_LAGS_S = (-0.06, -0.03, 0.0, 0.03, 0.06)
TRIAL = "S1_0.5"           # change to inspect another strict trial

st_ok = qc3[(qc3.tier == "strict") & np.isfinite(qc3.near_mismatch_ms)]
left_strict = float((st_ok.near_mismatch_ms < st_ok.near_swapped_mismatch_ms).mean())
odd = st_ok.loc[st_ok.near_mismatch_ms >= st_ok.near_swapped_mismatch_ms, "tag"].tolist()
print(f"force plate '1:' = left in {100 * left_strict:.0f}% of strict near-window trials; exceptions: {odd}")


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
    # ratio rule (a z-score on similar channels flags channels that merely carry extra signal, e.g. motor beta)
    bad = np.flatnonzero((sd < 0.5) | (sd > 3 * med) | (sd < med / 5))
    good = np.setdiff1d(np.arange(x.shape[0]), bad)
    return x - x[good].mean(axis=0, keepdims=True), bad


def head_regressors(tg: str, n: int, fs: float) -> np.ndarray:
    """Head acceleration (Segment_Acceleration, present for every subject) on the EEG clock, band-passed, with lags."""
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


def artifact_index(x: np.ndarray, fs: float, R: np.ndarray, good: np.ndarray) -> float:
    """corr(EEG 1-30 Hz envelope, head |acc| envelope) at lag 0 -- the P2 measure of CELL 10, one trial."""
    env = np.convolve(np.abs(x[good]).mean(axis=0), np.ones(int(0.1 * fs)) / int(0.1 * fs), mode="same")
    head_mag = np.convolve(np.linalg.norm(R[:, 6:9], axis=1), np.ones(int(0.1 * fs)) / int(0.1 * fs), mode="same")
    cut = int(3 * fs)
    return float(np.corrcoef(env[cut:-cut], head_mag[cut:-cut])[0, 1])


def stride_locked(sig: np.ndarray, fs: float, hs_times: np.ndarray, pre: float = 0.6, post: float = 0.8) -> np.ndarray:
    idx = [int(round(h * fs)) for h in hs_times if pre + 1 < h < len(sig) / fs - post - 1]
    w0, w1 = int(pre * fs), int(post * fs)
    return np.mean([sig[i - w0:i + w1] for i in idx], axis=0)


if TRIAL in EEG_EXCLUDE:
    raise RuntimeError(f"{TRIAL} is excluded from EEG: {EEG_EXCLUDE[TRIAL]}")
x_raw, fs_e = load_eeg(TRIAL)
x_a, bad = preprocess(x_raw, fs_e)                         # rung a: minimal
R = head_regressors(TRIAL, x_a.shape[1], fs_e)
x_c, r2 = regress_out(x_a, R)                              # rung c: minus head motion
ch_all = scalp + ["Pz"]
good = np.setdiff1d(np.arange(len(ch_all)), bad)
print(f"{TRIAL}: {x_a.shape[1]} samples at {fs_e:.1f} Hz | bad channels: {[ch_all[i] for i in bad] or 'none'}")
print("motion R^2 per channel (share of variance explained by head acceleration):",
      {ch_all[i]: round(float(r2[i]), 3) for i in good if ch_all[i] != "Pz"})
ai_a, ai_c = artifact_index(x_a, fs_e, R, good), artifact_index(x_c, fs_e, R, good)
print(f"artifact index (EEG envelope vs head |acc|, r at lag 0): before {ai_a:+.3f} -> after {ai_c:+.3f}")

hs_t = strides_v3.loc[(strides_v3.subject == int(TRIAL[1:].split("_")[0])) &
                      (strides_v3.speed == float(TRIAL.split("_")[1])), "hs_eeg"].dropna().to_numpy()
cz = ch_all.index("Cz")
bb, ab = butter(4, [13 / (fs_e / 2), 30 / (fs_e / 2)], btype="band")
beta_env = {k: np.abs(hilbert(filtfilt(bb, ab, v[cz]))) for k, v in (("a", x_a), ("c", x_c))}
bb_env = {k: np.convolve(np.abs(v[good]).mean(axis=0), np.ones(30) / 30, mode="same") for k, v in (("a", x_a), ("c", x_c))}
t_lock = np.arange(-int(0.6 * fs_e), int(0.8 * fs_e)) / fs_e
depth = {k: float(np.ptp(stride_locked(beta_env[k], fs_e, hs_t)) / beta_env[k].mean()) for k in ("a", "c")}
print(f"{len(hs_t)} heel strikes | Cz beta (13-30 Hz) stride-locked modulation depth: before {depth['a']:.2f} -> after {depth['c']:.2f}")

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for k, lab in (("a", "minimal (rung a)"), ("c", "minus head motion (rung c)")):
    f_, p_ = welch(x_a[cz] if k == "a" else x_c[cz], fs_e, nperseg=int(2 * fs_e))
    ax[0].semilogy(f_, p_, label=lab)
    ax[1].plot(t_lock, stride_locked(bb_env[k], fs_e, hs_t), label=lab)
    ax[2].plot(t_lock, stride_locked(beta_env[k], fs_e, hs_t), label=lab)
ax[0].set(xlim=(0, 50), xlabel="Hz", ylabel="Cz PSD", title="spectrum")
ax[1].set(xlabel="s from heel strike", ylabel="mean |EEG| (uV)", title="stride-locked broadband envelope (artifact)")
ax[2].set(xlabel="s from heel strike", ylabel="Cz beta amplitude (uV)", title="stride-locked Cz beta (G1 preview)")
for a_ in ax:
    a_.legend(fontsize=8)
fig.suptitle(f"{TRIAL}: EEG preprocessing check")
fig.tight_layout()
fig.savefig(OUT_ROOT / f"cell11_{TRIAL}.png", dpi=110)
plt.show()

check(abs(fs_e - 300) < 1, f"EEG rate {fs_e:.2f} Hz")
check(len(hs_t) >= 20, f"{len(hs_t)} heel strikes on the EEG clock for this trial")
check(len(bad) <= 3, f"bad channels: {len(bad)} (<= 3)", critical=False)
check(left_strict >= 0.95, f"force plate '1:' = left in {100 * left_strict:.0f}% of strict trials", critical=False)
check(abs(ai_c) < 0.5 * abs(ai_a) or abs(ai_a) < 0.05, f"head-motion regression removes the motion artifact ({ai_a:+.3f} -> {ai_c:+.3f})", critical=False)
end_cell("CELL 11")

# ===== CELL 11b · Patch: head-motion regressors limited to 1-25 Hz (~1 s) =====
# IMU is sampled at 60 Hz: above ~25 Hz the interpolated regressors only contain interpolation images, which let the
# regression inject 30-45 Hz content (CELL 12: Cz gamma depth rose from 5.41 to 7.77 dB after "cleaning").
REG_BAND = (1.0, 25.0)


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


x_test, fs_test = load_eeg(TRIAL)
f_r, p_r = welch(head_regressors(TRIAL, x_test.shape[1], fs_test)[:, 8], fs_test, nperseg=int(2 * fs_test))
share_hi = float(p_r[f_r > 30].sum() / p_r.sum())
print(f"share of regressor power above 30 Hz: {100 * share_hi:.3f}%")
check(share_hi < 0.001, "head-motion regressors carry no 30+ Hz content")
end_cell("CELL 11b -> now re-run CELL 12 (unchanged), then CELL 13")

# ===== CELL 12 · All trials: EEG preprocessing + gate G1 (gait-cycle EEG modulation, permutation test) (~5 min) =====
need = ["load_eeg", "preprocess", "head_regressors", "regress_out", "artifact_index", "qc3", "strides_v3", "odd", "EEG_EXCLUDE"]
missing_defs = [n for n in need if n not in globals()]
if missing_defs:
    raise RuntimeError(f"run CELL 11 first (missing: {missing_defs})")
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
N_BINS, N_PERM = 101, 1000
CENTRAL, TEMPORAL = ["C3", "Cz", "C4"], ["T3", "T4", "T5", "T6"]
rng_g = np.random.default_rng(1)
ch_all = scalp + ["Pz"]

# primary set: strict trials, minus the 10 whose left/right IMU match looked odd in CELL 11, minus EEG exclusions
primary = qc3[(qc3.tier == "strict") & ~qc3.tag.isin(odd) & ~qc3.tag.isin(list(EEG_EXCLUDE))].reset_index(drop=True)
log(f"G1 on {len(primary)} strict trials from {primary.subject.nunique()} subjects ({len(odd)} odd-mapping trials kept for sensitivity)")


def cycle_profiles(x: np.ndarray, fs: float, hs: np.ndarray, stride: np.ndarray) -> tuple:
    """Power over the left gait cycle (heel strike -> next heel strike, 101 bins) per band and channel,
    in dB relative to the trial's cycle mean."""
    ok = [(h, s) for h, s in zip(hs, stride) if np.isfinite(s) and 0.7 < s < 2.6 and h > 1 and h + s < x.shape[1] / fs - 1]
    out = np.full((len(BANDS), x.shape[0], N_BINS), np.nan)
    for bi, (lo, hi) in enumerate(BANDS.values()):
        b_, a_ = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
        pw = np.abs(hilbert(filtfilt(b_, a_, x, axis=1), axis=1)) ** 2
        acc_ = np.zeros((x.shape[0], N_BINS))
        for h, s in ok:
            idx = np.round(np.linspace(h * fs, (h + s) * fs, N_BINS)).astype(int)
            acc_ += pw[:, idx]
        P = acc_ / max(len(ok), 1)
        out[bi] = 10 * np.log10(P / P.mean(axis=1, keepdims=True))
    return out, len(ok)


rows_g, prof = [], {"a": [], "c": []}
failed_g1 = {}
t0_g = time.time()
for k_, r in enumerate(primary.itertuples(), 1):
    try:
        x0, fs_ = load_eeg(r.tag)
        xa, bad_ = preprocess(x0, fs_)
        R_ = head_regressors(r.tag, xa.shape[1], fs_)
        xc, r2_ = regress_out(xa, R_)
        good_ = np.setdiff1d(np.arange(xa.shape[0]), bad_)
        st = strides_v3[(strides_v3.subject == r.subject) & (strides_v3.speed == r.speed) & (strides_v3.leg == "L")]
        pa_, n_ok = cycle_profiles(xa, fs_, st.hs_eeg.to_numpy(), st.stride_s.to_numpy())
        pc_, _ = cycle_profiles(xc, fs_, st.hs_eeg.to_numpy(), st.stride_s.to_numpy())
        pa_[:, bad_, :], pc_[:, bad_, :] = np.nan, np.nan
        prof["a"].append(pa_)
        prof["c"].append(pc_)
        rows_g.append({"tag": r.tag, "subject": r.subject, "speed": r.speed, "n_strides": n_ok,
                       "bad": ",".join(ch_all[i] for i in bad_), "motion_r2_median": round(float(np.median(r2_[good_])), 4),
                       "artifact_a": round(artifact_index(xa, fs_, R_, good_), 3),
                       "artifact_c": round(artifact_index(xc, fs_, R_, good_), 3)})
    except (OSError, ValueError, KeyError, IndexError, np.linalg.LinAlgError) as exc:
        failed_g1[r.tag] = f"{type(exc).__name__}: {str(exc)[:100]}"
    if k_ % 10 == 0 or k_ == len(primary):
        log(f"{k_}/{len(primary)} trials | failed {len(failed_g1)} | {time.time() - t0_g:.0f}s")

g1_trials = pd.DataFrame(rows_g)
PA, PC = np.array(prof["a"]), np.array(prof["c"])      # trials x bands x channels x bins
subj = g1_trials.subject.to_numpy()
subjects = np.unique(subj)


def grand(P: np.ndarray) -> np.ndarray:
    """Average within subject first, then across subjects (each subject counts once)."""
    return np.nanmean(np.stack([np.nanmean(P[subj == s], axis=0) for s in subjects]), axis=0)


def depth(G: np.ndarray) -> np.ndarray:
    return np.nanmax(G, axis=-1) - np.nanmin(G, axis=-1)   # bands x channels, dB


obs = {"a": grand(PA), "c": grand(PC)}
dep = {k: depth(v) for k, v in obs.items()}
null = {k: np.empty((N_PERM,) + dep[k].shape) for k in obs}
for j in range(N_PERM):  # rotate each trial's profile by a random phase: keeps within-trial modulation, breaks cross-trial consistency
    sh = rng_g.integers(0, N_BINS, len(PA))
    for k, P in (("a", PA), ("c", PC)):
        null[k][j] = depth(grand(np.stack([np.roll(P[i], sh[i], axis=-1) for i in range(len(P))])))
pval = {k: (1 + (null[k] >= dep[k][None]).sum(axis=0)) / (N_PERM + 1) for k in obs}

bn = list(BANDS)
res_g1 = pd.DataFrame([{"rung": k, "band": b, "channel": ch, "depth_dB": round(float(dep[k][bi, ci]), 3),
                        "p": round(float(pval[k][bi, ci]), 4)}
                       for k in ("a", "c") for bi, b in enumerate(bn) for ci, ch in enumerate(ch_all)])
g1_trials.to_csv(OUT_ROOT / "g1_trials.csv", index=False)
res_g1.to_csv(OUT_ROOT / "g1_results.csv", index=False)
np.savez_compressed(OUT_ROOT / "g1_profiles.npz", tags=g1_trials.tag.to_numpy(), PA=PA.astype(np.float32), PC=PC.astype(np.float32))

show = res_g1[res_g1.channel.isin(CENTRAL + ["Fz", "Pz"] + TEMPORAL)].pivot_table(index=["rung", "band"], columns="channel", values="depth_dB")
print("\nmodulation depth over the gait cycle (dB, grand average):")
print(show.round(2).to_string())
ci_ = {ch: ch_all.index(ch) for ch in ch_all}
b_beta, b_gam = bn.index("beta"), bn.index("gamma")
cz_p, cz_d = float(pval["c"][b_beta, ci_["Cz"]]), float(dep["c"][b_beta, ci_["Cz"]])
cen = float(np.nanmean([dep["c"][b_beta, ci_[c]] for c in CENTRAL]))
tem = float(np.nanmean([dep["c"][b_beta, ci_[c]] for c in TEMPORAL]))
print(f"\nGATE G1 (pre-specified): Cz beta, motion-regressed: depth {cz_d:.2f} dB, p = {cz_p:.4f} -> {'PASS' if cz_p < 0.01 else 'FAIL'}")
print(f"origin diagnostics (beta, rung c): central {cen:.2f} dB vs temporal {tem:.2f} dB | Cz beta {cz_d:.2f} vs Cz gamma {dep['c'][b_gam, ci_['Cz']]:.2f} dB")
print(f"artifact index |r| median: before {g1_trials.artifact_a.abs().median():.3f} -> after {g1_trials.artifact_c.abs().median():.3f} | "
      f"trials with |r| > 0.2: {int((g1_trials.artifact_a.abs() > 0.2).sum())} -> {int((g1_trials.artifact_c.abs() > 0.2).sum())}")
if failed_g1:
    print("failed:", failed_g1)

fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
xc_ = np.linspace(0, 100, N_BINS)
for bi, b in enumerate(bn):
    ax[0].plot(xc_, obs["c"][bi, ci_["Cz"]], label=f"{b} (p={pval['c'][bi, ci_['Cz']]:.3f})")
ax[0].set(xlabel="% left gait cycle (0 = left heel strike)", ylabel="dB vs cycle mean", title="Cz, motion-regressed (grand average)")
ax[0].axhline(0, color="grey", lw=0.5)
order = np.argsort(dep["c"][b_beta])[::-1]
ax[1].bar([ch_all[i] for i in order], dep["c"][b_beta, order],
          color=["tab:red" if pval["c"][b_beta, i] < 0.01 else "tab:grey" for i in order])
ax[1].set(ylabel="beta depth (dB)", title="beta modulation by channel (red: p < 0.01)")
ax[1].tick_params(axis="x", rotation=90)
ax[2].scatter(g1_trials.artifact_a, g1_trials.artifact_c, s=12)
lim = max(0.05, float(np.nanmax(np.abs(g1_trials[["artifact_a", "artifact_c"]].to_numpy()))))
ax[2].plot([-lim, lim], [-lim, lim], "k:", lw=0.8)
ax[2].set(xlabel="artifact index, minimal", ylabel="artifact index, motion-regressed", title="per-trial motion artifact")
for a_ in ax[:1]:
    a_.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_ROOT / "cell12_G1.png", dpi=110)
plt.show()

check(len(g1_trials) >= 0.8 * len(primary), f"{len(g1_trials)}/{len(primary)} trials preprocessed")
check(cz_p < 0.01, f"GATE G1: gait-cycle Cz beta modulation after motion regression ({cz_d:.2f} dB, p = {cz_p:.4f})", critical=False)
check(cen > 1.5 * tem, f"beta modulation clearly central (central {cen:.2f} > 1.5 x temporal {tem:.2f} dB) -> less likely neck muscle", critical=False)
med_a, med_c = g1_trials.artifact_a.abs().median(), g1_trials.artifact_c.abs().median()
check(med_c <= med_a or med_a < 0.05, f"motion regression lowers the per-trial artifact index (median |r| {med_a:.3f} -> {med_c:.3f})", critical=False)
end_cell("CELL 12")

# ===== CELL 13 · Brain or artifact? Diagnostics of the gait-cycle modulation (~1 min) =====
from scipy.stats import spearmanr, mannwhitneyu

need = ["PC", "PA", "g1_trials", "grand", "depth", "BANDS", "ch_all", "subj", "N_BINS"]
missing_defs = [n for n in need if n not in globals()]
if missing_defs:
    raise RuntimeError(f"run CELL 12 first (missing: {missing_defs})")
bn = list(BANDS)
cz_i, b_be, b_al = ch_all.index("Cz"), bn.index("beta"), bn.index("alpha")
rng_d = np.random.default_rng(2)


def rotation_p(P: np.ndarray, n_perm: int = 1000) -> tuple:
    """Grand-average depth and its p-value under random per-trial phase rotation (as in CELL 12)."""
    d_obs = depth(grand(P))
    null_ = np.empty((n_perm,) + d_obs.shape)
    for j in range(n_perm):
        sh = rng_d.integers(0, N_BINS, len(P))
        null_[j] = depth(grand(np.stack([np.roll(P[i], sh[i], axis=-1) for i in range(len(P))])))
    return d_obs, (1 + (null_ >= d_obs[None]).sum(axis=0)) / (n_perm + 1)


# 1. are all bands the same shape? (broadband = typical of mechanical artifact)
G = grand(PC)[:, cz_i]                      # bands x bins at Cz
shape_r = np.corrcoef(G)
print("Cz profile correlation between bands (motion-regressed):")
print(pd.DataFrame(shape_r, index=bn, columns=bn).round(2).to_string())
broadband = float(np.mean(shape_r[np.triu_indices(len(bn), 1)]))

# 2. beta-specific modulation: per trial and channel, remove the best-scaled copy of the other bands' shared profile
#    (scale-free: a broadband artifact present in beta at any size is removed; a beta-only shape survives)
def specific(P: np.ndarray, b: int) -> np.ndarray:
    others = np.nanmean(np.delete(P, b, axis=1), axis=1)          # trials x channels x bins
    tgt = P[:, b]
    oc = others - np.nanmean(others, axis=-1, keepdims=True)
    tc = tgt - np.nanmean(tgt, axis=-1, keepdims=True)
    k = np.nansum(oc * tc, axis=-1, keepdims=True) / np.nansum(oc * oc, axis=-1, keepdims=True)
    return (tc - k * oc)[:, None]                                   # trials x 1 x channels x bins


PS = np.concatenate([specific(PC, b_be), specific(PC, b_al)], axis=1)   # [beta-specific, alpha-specific]
d_spec, p_spec = rotation_p(PS)
print(f"\nband-specific modulation at Cz (scaled shared profile of the other bands removed): "
      f"beta {d_spec[0, cz_i]:.2f} dB p={p_spec[0, cz_i]:.3f} | alpha {d_spec[1, cz_i]:.2f} dB p={p_spec[1, cz_i]:.3f}")

# 3. dose-response: does modulation grow with the trial's motion artifact?
trial_depth = np.array([np.nanmax(PC[i, b_be, cz_i]) - np.nanmin(PC[i, b_be, cz_i]) for i in range(len(PC))])
art = g1_trials.artifact_a.abs().to_numpy()
rho, p_rho = spearmanr(art, trial_depth, nan_policy="omit")
lo_, hi_ = trial_depth[art < 0.1], trial_depth[art > 0.3]
p_lohi = mannwhitneyu(lo_, hi_).pvalue if len(lo_) > 2 and len(hi_) > 2 else float("nan")
print(f"\ndose-response, Cz beta depth vs motion-artifact index: Spearman rho = {rho:+.2f} (p = {p_rho:.3f})")
print(f"cleanest trials (|artifact| < 0.1, n={len(lo_)}): median depth {np.median(lo_) if len(lo_) else float('nan'):.2f} dB | "
      f"artifact-heavy (|artifact| > 0.3, n={len(hi_)}): {np.median(hi_) if len(hi_) else float('nan'):.2f} dB | p = {p_lohi:.3f}")
by_speed = g1_trials.assign(depth=trial_depth).groupby("speed")[["depth", "artifact_a"]].median().round(3)
print("\nmedian per speed:\n" + by_speed.to_string())

fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
xb = np.linspace(0, 100, N_BINS)
for i, b in enumerate(bn):
    ax[0].plot(xb, (G[i] - G[i].mean()) / G[i].std(), label=b)
ax[0].set(xlabel="% left gait cycle", ylabel="z-scored dB", title=f"Cz profile shapes (mean between-band r = {broadband:.2f})")
GS = grand(PS)[:, cz_i]
for i, b in enumerate(["beta", "alpha"]):
    ax[1].plot(xb, GS[i], label=f"{b}-specific p={p_spec[i, cz_i]:.3f}")
ax[1].axhline(0, color="grey", lw=0.5)
ax[1].set(xlabel="% left gait cycle", ylabel="dB beyond the scaled shared profile", title="band-specific modulation at Cz")
sc = ax[2].scatter(art, trial_depth, c=g1_trials.speed, cmap="viridis", s=14)
ax[2].set(xlabel="|motion-artifact index| (minimal)", ylabel="Cz beta depth per trial (dB)", title=f"dose-response: rho = {rho:+.2f}")
plt.colorbar(sc, ax=ax[2], label="speed (m/s)")
for a_ in ax[:2]:
    a_.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_ROOT / "cell13_origin.png", dpi=110)
plt.show()

spec_ok = p_spec[0, cz_i] < 0.01
dose_ok = not (rho > 0.3 and p_rho < 0.05)
verdict = ("neural component likely: beta-specific modulation, not driven by motion-artifact strength" if spec_ok and dose_ok else
           "mixed: beta-specific modulation exists but scales with motion artifact" if spec_ok else
           "no beta-specific modulation beyond the broadband (artifact-like) profile")
print(f"\n=> VERDICT: {verdict}")
check(spec_ok, f"beta modulates beyond the scaled shared profile at Cz (p={p_spec[0, cz_i]:.3f}; alpha p={p_spec[1, cz_i]:.3f} reported only)", critical=False)
check(dose_ok, f"Cz beta modulation is not explained by motion-artifact strength (rho = {rho:+.2f})", critical=False)
check(broadband < 0.8, f"bands do not all share one profile (mean between-band r = {broadband:.2f})", critical=False)
end_cell("CELL 13")
