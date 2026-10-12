# ===== CELL 6/9 · One trial, all 4 modalities: rates, durations, sync channels (~30 s) =====
import matplotlib.pyplot as plt

if "inv" not in globals():
    raise RuntimeError("variables missing (runtime was reset?) -> run CELL 1 and CELL 4 first")
SUBJ, SPEED = 1, 0.5  # trial to inspect; change and re-run to look at another one
tag = f"S{SUBJ}_{SPEED:g}"


def one(pattern: str, root: Path) -> Path:
    hits = sorted(root.rglob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected 1 match for {pattern} under {root}, found {len(hits)}")
    return hits[0]


def rate_and_duration(t: np.ndarray) -> tuple:
    """Rate from sample count / time span (the EEG Time column is rounded to 4 decimals, so median steps mislead)."""
    fs = (len(t) - 1) / float(t[-1] - t[0])
    return fs, len(t) / fs


eeg_df = pd.read_csv(one(f"{tag}_raw.csv", DATA), comment="#", low_memory=False)
emg_df = pd.read_csv(one(f"EMG_{tag}.csv", DATA))
fp_df = pd.read_csv(one(f"FP_{tag}.csv", DATA))
imu_dir = one(f"IMU_{tag}", IMU_ROOT)
seg_acc = pd.read_csv(imu_dir / "Segment_Acceleration.csv")
info = pd.read_csv(imu_dir / "General_Information.csv", header=None)
markers = pd.read_csv(imu_dir / "Markers.csv")
sensor_cols = pd.read_csv(imu_dir / "Sensor_Free_Acceleration.csv", nrows=1).columns

eeg_t = eeg_df["Time"].to_numpy(float)
eeg_fs, eeg_dur = rate_and_duration(eeg_t)
eeg_ch = list(eeg_df.columns[1:eeg_df.columns.get_loc("Trigger")])
trig = eeg_df["Trigger"].to_numpy(float)
trig_changes = np.flatnonzero(np.diff(trig) != 0) + 1
print(f"--- EEG: {len(eeg_df)} rows | fs {eeg_fs:.2f} Hz | {eeg_dur:.2f} s | {len(eeg_ch)} channels: {eeg_ch}")
print(f"    Trigger values: {sorted(pd.unique(trig))[:10]} | {len(trig_changes)} changes, first at t = "
      f"{[round(float(eeg_t[i]), 3) for i in trig_changes[:6]]}")

emg_t = emg_df.iloc[:, 0].to_numpy(float)
emg_fs, emg_dur = rate_and_duration(emg_t)
print(f"--- EMG: {len(emg_df)} rows | fs {emg_fs:.2f} Hz | {emg_dur:.2f} s | first t = {emg_t[0]:.3f} s")

print(f"--- FP: {len(fp_df)} rows = {len(fp_df) / 1000:.2f} s at 1000 Hz")
for leg in ("1", "2"):
    for col in ("Timestamp", "Sync", "Aux"):
        v = fp_df[f"{leg}:{col}"].to_numpy(float)
        steps = np.diff(v)
        nonzero = steps[steps != 0]
        print(f"    {leg}:{col:9s} min {v.min():.6g} max {v.max():.6g} | unique {len(np.unique(v))} | "
              f"changes {len(nonzero)} | median non-zero step {np.median(nonzero) if len(nonzero) else 0:.6g}")

rate_rows = info[info[0].astype(str).str.contains("rate", case=False)]
try:
    imu_fs = float(rate_rows.iloc[0, 1])
except (IndexError, ValueError):
    imu_fs = 60.0  # not stated in the export -> dataset page value
print("--- IMU General_Information.csv:")
print(info.to_string(index=False, header=False))
frames = seg_acc["Frame"].to_numpy()
imu_dur = len(frames) / imu_fs
segments = sorted({c.rsplit(" ", 1)[0] for c in seg_acc.columns[1:]})
sensors = sorted({c.rsplit(" ", 1)[0] for c in sensor_cols[1:]})
print(f"--- IMU: {len(frames)} frames | frame steps {sorted(set(np.diff(frames).tolist()))[:5]} | {imu_dur:.2f} s at {imu_fs:g} Hz")
print(f"    segments ({len(segments)}): {segments}")
print(f"    Sensor_Free_Acceleration entries ({len(sensors)}): {sensors}")
print(f"    Markers.csv: {len(markers)} rows; first: {markers.head(5).to_dict('records')}")

durations = {"EEG": eeg_dur, "EMG": emg_dur, "FP @1000Hz": len(fp_df) / 1000, f"IMU @{imu_fs:g}Hz": imu_dur}
spread = max(durations.values()) - min(durations.values())
print("durations (s):", {k: round(v, 2) for k, v in durations.items()})

fig, ax = plt.subplots(5, 1, figsize=(14, 12))
win = 15  # seconds shown; each stream on its own clock
t_fp = np.arange(len(fp_df)) / 1000
m = t_fp < win
ax[0].plot(t_fp[m], fp_df["1:Fz"].to_numpy()[m], label="1:Fz (left?)")
ax[0].plot(t_fp[m], fp_df["2:Fz"].to_numpy()[m], label="2:Fz (right?)")
ax[0].set_ylabel("Fz [N]")
t_imu = np.arange(len(seg_acc)) / imu_fs
mi = t_imu < win
foot = [c for c in seg_acc.columns if "Foot" in c and c.endswith(" z")]
for c in foot:
    ax[1].plot(t_imu[mi], seg_acc[c].to_numpy()[mi], label=c)
ax[1].set_ylabel("foot acc z [m/s²]")
me = emg_t < win
for c in emg_df.columns[1:3]:
    ax[2].plot(emg_t[me], emg_df[c].to_numpy()[me], label=c.split(":")[0])
ax[2].set_ylabel("EMG MAV [V]")
mg = eeg_t < win
cz = eeg_df["Cz"].to_numpy(float)[mg]
ax[3].plot(eeg_t[mg], cz - cz.mean(), lw=0.5, label="Cz")
ax[3].set_ylabel("EEG raw [µV]")
ax[4].plot(eeg_t[mg], trig[mg], label="EEG Trigger")
ax[4].plot(t_fp[m], fp_df["1:Sync"].to_numpy()[m], label="FP 1:Sync")
ax[4].set_ylabel("sync channels")
ax[4].set_xlabel("seconds from each file's own start (streams NOT aligned yet)")
for a in ax:
    a.legend(loc="upper right", fontsize=8)
fig.suptitle(f"{tag}: first {win} s of each stream")
fig.tight_layout()
fig.savefig(OUT_ROOT / f"cell6_{tag}.png", dpi=110)
plt.show()

check(min(len(eeg_df), len(emg_df), len(fp_df), len(seg_acc)) > 0, "all 4 files loaded")
check(abs(eeg_fs - 300) < 1, f"EEG rate {eeg_fs:.2f} Hz (expected 300)", critical=False)
check(abs(emg_fs - 16.05) < 0.5, f"EMG rate {emg_fs:.2f} Hz (expected ~16)", critical=False)
check(abs(imu_fs - 60) < 1, f"IMU rate {imu_fs:g} Hz (expected 60)", critical=False)
check(len(foot) == 2, f"both foot acceleration columns found: {foot}", critical=False)
check("Head" in sensors, "IMU has a Head sensor (needed later for EEG motion-artifact regression)", critical=False)
check(spread < 2, f"durations agree within 2 s (spread {spread:.1f} s -> if not, the streams start/stop at "
      f"different times and must be aligned)", critical=False)
end_cell("CELL 6")

# ===== CELL 7/9 · Heel strike / toe-off from the force plates, one trial (~10 s) =====
from scipy.signal import butter, filtfilt

FS_FP = 1000.0
LEGS = {"L": "1:Fz", "R": "2:Fz"}  # dataset page: '1:' = left belt, '2:' = right belt (re-checked in CELL 8 with the IMU)


def lowpass(x: np.ndarray, fs: float, fc: float = 20.0) -> np.ndarray:
    b, a = butter(4, fc / (fs / 2))
    return filtfilt(b, a, x)


def gait_events(f: np.ndarray, fs: float, on: float, off: float) -> pd.DataFrame:
    """One row per stance: heel strike (Fz rises above `on`) to toe-off (Fz falls below `off`)."""
    up = np.flatnonzero((f[1:] > on) & (f[:-1] <= on)) + 1
    down = np.flatnonzero((f[1:] < off) & (f[:-1] >= off)) + 1
    seq = []
    for idx, kind in sorted([(i, "HS") for i in up] + [(i, "TO") for i in down]):
        if not seq or seq[-1][1] != kind:  # hysteresis: keep the first crossing of each kind
            seq.append((idx, kind))
    pairs = [(seq[k][0], seq[k + 1][0]) for k in range(len(seq) - 1) if seq[k][1] == "HS" and seq[k + 1][1] == "TO"]
    ev = pd.DataFrame(pairs, columns=["hs", "to"], dtype=float) / fs
    ev = ev[(ev.to - ev.hs) > 0.15].reset_index(drop=True)  # drop contacts shorter than 150 ms
    ev["stride_s"] = ev.hs.shift(-1) - ev.hs
    ev["stance_pct"] = 100 * (ev.to - ev.hs) / ev.stride_s
    return ev


fz_f = {leg: lowpass(fp_df[col].to_numpy(float), FS_FP) for leg, col in LEGS.items()}
bw = float(np.median(fz_f["L"] + fz_f["R"]))
events, thresholds = {}, {}
for leg, f in fz_f.items():
    base = float(np.percentile(f, 2))
    thresholds[leg] = (base + 0.05 * bw, base + 0.03 * bw)
    events[leg] = gait_events(f, FS_FP, *thresholds[leg])
    ev = events[leg].dropna()
    print(f"{leg}: baseline {base:6.1f} N | thresholds {thresholds[leg][0]:.0f}/{thresholds[leg][1]:.0f} N | "
          f"{len(ev)} strides | stride {ev.stride_s.median():.3f} s (CV {100 * ev.stride_s.std() / ev.stride_s.mean():.1f}%) | "
          f"stance {ev.stance_pct.median():.1f}% | cadence {120 / ev.stride_s.median():.1f} steps/min")

L, R = events["L"].dropna(), events["R"].dropna()
phase = []
for hs in R.hs:
    k = int(np.searchsorted(L.hs.to_numpy(), hs)) - 1
    if 0 <= k < len(L):
        phase.append((hs - L.hs.iloc[k]) / L.stride_s.iloc[k])
phase_med = float(np.median(phase)) if phase else float("nan")

fig, ax = plt.subplots(figsize=(14, 4))
t = np.arange(len(fp_df)) / FS_FP
w = (t > 10) & (t < 18)
for leg, color in (("L", "tab:blue"), ("R", "tab:orange")):
    ax.plot(t[w], fz_f[leg][w], color=color, label=f"{leg} Fz (filtered)")
    ev = events[leg]
    hs = ev.hs[(ev.hs > 10) & (ev.hs < 18)].to_numpy()
    to = ev.to[(ev.to > 10) & (ev.to < 18)].to_numpy()
    ax.plot(hs, np.interp(hs, t, fz_f[leg]), "^", color=color, ms=9, label=f"{leg} heel strike")
    ax.plot(to, np.interp(to, t, fz_f[leg]), "v", color=color, ms=9, label=f"{leg} toe-off")
ax.set_xlabel("s (force-plate clock)")
ax.set_ylabel("N")
ax.legend(ncol=3, fontsize=8)
ax.set_title(f"{tag}: detected gait events, 10-18 s")
fig.tight_layout()
fig.savefig(OUT_ROOT / f"cell7_{tag}.png", dpi=110)
plt.show()

check(350 < bw < 1500, f"body weight from total Fz = {bw:.0f} N ({bw / 9.81:.0f} kg) is plausible", critical=False)
for leg, ev in (("L", L), ("R", R)):
    med = ev.stride_s.median() if len(ev) else float("nan")
    cv = 100 * ev.stride_s.std() / ev.stride_s.mean() if len(ev) > 1 else float("nan")
    check(len(ev) >= 20, f"{leg}: at least 20 strides detected ({len(ev)})")
    check(0.8 < med < 2.5, f"{leg}: median stride time plausible ({med:.2f} s)")
    check(cv < 15, f"{leg}: stride-time CV {cv:.1f}% < 15% (higher -> missed or extra events)", critical=False)
    check(50 < ev.stance_pct.median() < 75, f"{leg}: stance {ev.stance_pct.median():.0f}% of stride (normal 55-70%)", critical=False)
check(abs(len(L) - len(R)) <= 2, f"left/right stride counts close ({len(L)} vs {len(R)})", critical=False)
check(0.4 < phase_med < 0.6, f"right heel strike at {100 * phase_med:.0f}% of the left stride (expected ~50%)", critical=False)
end_cell("CELL 7")

# ===== CELL 8/9 · Align the 4 streams for this trial (~20 s) =====
from scipy.signal import find_peaks

if "events" not in globals():
    raise RuntimeError("run CELL 6 and CELL 7 first (same trial)")


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    b, a = butter(2, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, np.nan_to_num(x))


def nearest_residuals(t_ref: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """For each query time: (nearest reference time) - (query time)."""
    idx = np.clip(np.searchsorted(t_ref, t_query), 1, len(t_ref) - 1)
    left, right = t_ref[idx - 1], t_ref[idx]
    return np.where(np.abs(right - t_query) < np.abs(left - t_query), right, left) - t_query


def match_offset(pairs: list, max_shift: float = 15.0) -> dict:
    """Offset d with FP time + d = IMU time. Every FP/IMU heel-strike pair is a candidate d; the best d has the
    smallest median |mismatch|. Stride-to-stride variability makes the right d fit clearly better than d +- 1 stride."""
    cands = np.unique(np.round(np.concatenate([(imu[None, :] - fp[:, None]).ravel() for fp, imu in pairs]), 3))
    cands = cands[np.abs(cands) <= max_shift]

    def residuals(d: float) -> np.ndarray:
        parts = []
        for fp, imu in pairs:
            q = fp + d
            q = q[(q > imu[0]) & (q < imu[-1])]
            parts.append(nearest_residuals(imu, q))
        return np.concatenate(parts)

    scores = np.array([np.median(np.abs(r)) if len(r := residuals(d)) >= 20 else np.inf for d in cands])
    best = int(np.argmin(scores))
    d = float(cands[best] + np.median(residuals(cands[best])))
    far = np.abs(cands - cands[best]) > 0.3
    return {"offset_s": round(d, 4), "mismatch_ms": round(1000 * float(np.median(np.abs(residuals(d)))), 1),
            "runner_up_ms": round(1000 * float(scores[far].min()), 1) if far.any() else float("inf"),
            "n_matched": int(len(residuals(d)))}


def xcorr_lag(a: np.ndarray, b: np.ndarray, fs: float, max_lag_s: float) -> tuple:
    """Lag (s) maximising |corr(a[t], b[t + lag])|; positive lag = b happens later than a."""
    n = min(len(a), len(b))
    a = (a[:n] - a[:n].mean()) / a[:n].std()
    b = (b[:n] - b[:n].mean()) / b[:n].std()
    lags = np.arange(-int(max_lag_s * fs), int(max_lag_s * fs) + 1)
    r = np.array([np.mean(a[max(0, -k):n - max(0, k)] * b[max(0, k):n - max(0, -k)]) for k in lags])
    i = int(np.argmax(np.abs(r)))
    return lags[i] / fs, float(r[i])


def imu_heel_strikes(acc_z: np.ndarray, fs: float, stride_s: float) -> np.ndarray:
    """Foot-impact times (s): the largest 2-20 Hz acceleration spike per ~stride."""
    x = np.abs(bandpass(acc_z, fs, 2, 20))
    mad = np.median(np.abs(x - np.median(x)))
    peaks, _ = find_peaks(x, distance=int(0.6 * stride_s * fs), height=np.median(x) + 3 * mad)
    return peaks / fs


# --- 1. what sync signals exist? ---
rise = np.flatnonzero((trig[1:] > 0) & (trig[:-1] == 0)) + 1
fall = np.flatnonzero((trig[1:] == 0) & (trig[:-1] > 0)) + 1
high = [eeg_t[fall[fall > r][0]] - eeg_t[r] for r in rise if (fall > r).any()]
fp_sync_flat = all(fp_df[f"{leg}:{c}"].nunique() == 1 for leg in "12" for c in ("Timestamp", "Sync", "Aux"))
print(f"EEG Trigger: {len(rise)} pulses | period {np.median(np.diff(eeg_t[rise])):.3f} s | "
      f"high time min/median/max {min(high):.2f}/{np.median(high):.2f}/{max(high):.2f} s")
print(f"FP Timestamp/Sync/Aux all constant (no sync signal): {fp_sync_flat}")
print(f"IMU Markers.csv: {len(markers)} rows {markers.head(3).to_dict('records')}")

# --- 2. force plate -> IMU clock, from heel-strike sequences ---
stride_fp = float(pd.concat([events["L"].stride_s, events["R"].stride_s]).median())
imu_hs = {side: imu_heel_strikes(seg_acc[f"{side} Foot z"].to_numpy(float), imu_fs, stride_fp) for side in ("Left", "Right")}
fp_hs = {leg: events[leg].hs.to_numpy() for leg in ("L", "R")}
map_a = match_offset([(fp_hs["L"], imu_hs["Left"]), (fp_hs["R"], imu_hs["Right"])])
map_b = match_offset([(fp_hs["L"], imu_hs["Right"]), (fp_hs["R"], imu_hs["Left"])])
left_is_1 = map_a["mismatch_ms"] <= map_b["mismatch_ms"]
fp_align = map_a if left_is_1 else map_b
print(f"IMU heel strikes: Left {len(imu_hs['Left'])}, Right {len(imu_hs['Right'])} (FP median stride {stride_fp:.2f} s)")
print(f"FP '1:'=Left : {map_a}")
print(f"FP '1:'=Right: {map_b}")

# --- 3. EEG vs IMU: gait artifact in EEG vs head vertical acceleration ---
head = bandpass(pd.read_csv(imu_dir / "Sensor_Free_Acceleration.csv", usecols=["Head z"])["Head z"].to_numpy(float), imu_fs, 1, 8)
scalp = [c for c in eeg_ch if c not in ("CM", "A1", "A2", "X1", "X2", "X3")]
eeg_motion = bandpass(eeg_df[scalp].to_numpy(float).mean(axis=1), eeg_fs, 1, 8)[:: int(round(eeg_fs / imu_fs))]
eeg_lag, eeg_r = xcorr_lag(head, eeg_motion, imu_fs, 0.8)
eeg_lag_wide, eeg_r_wide = xcorr_lag(head, eeg_motion, imu_fs, 10.0)

# --- 4. EMG vs IMU: R tibialis anterior vs right-foot impacts; channel quality ---
mav = emg_df.iloc[:, 1:]
mod_index = ((mav.quantile(0.95) - mav.quantile(0.05)) / mav.median()).rename(lambda c: c.split(":")[0])
flat = mod_index[mod_index < 0.3]
t_grid = np.arange(len(seg_acc)) / imu_fs
ta_col = [c for c in mav.columns if c.startswith("R TIBIALIS ANTERIOR")][0]
ta_r = np.interp(t_grid, emg_t, mav[ta_col].to_numpy(float))
foot_env = np.convolve(np.abs(bandpass(seg_acc["Right Foot z"].to_numpy(float), imu_fs, 2, 20)), np.ones(12) / 12, mode="same")
emg_lag, emg_r = xcorr_lag(foot_env, ta_r, imu_fs, 0.8)
print("EMG modulation index (p95-p5)/median:", mod_index.round(2).to_dict())

d = fp_align["offset_s"]
print(f"\n=> FP time + {d:.3f} s = IMU time | EEG vs head: lag {1000 * eeg_lag:.0f} ms, r {eeg_r:.2f} "
      f"(wide +-10 s search: {eeg_lag_wide:.2f} s) | R TA vs right foot: lag {1000 * emg_lag:.0f} ms, r {emg_r:.2f}")

fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
w0, w1 = 10, 20
t_fp_imu = np.arange(len(fp_df)) / FS_FP + d
mf = (t_fp_imu > w0) & (t_fp_imu < w1)
for leg, color in (("L", "tab:blue"), ("R", "tab:orange")):
    ax[0].plot(t_fp_imu[mf], fz_f[leg][mf], color=color, label=f"FP {leg} Fz, shifted {d:+.3f} s")
for side, color in (("Left", "tab:blue" if left_is_1 else "tab:orange"), ("Right", "tab:orange" if left_is_1 else "tab:blue")):
    for x in imu_hs[side][(imu_hs[side] > w0) & (imu_hs[side] < w1)]:
        ax[0].axvline(x, color=color, ls="--", lw=0.8)
ax[0].set_ylabel("Fz [N] (dashed = IMU impacts)")
mi = (t_grid > w0) & (t_grid < w1)
ax[1].plot(t_grid[mi], (head / head.std())[mi], label="IMU head acc z (1-8 Hz)")
ax[1].plot(t_grid[mi], (eeg_motion[: len(t_grid)] / eeg_motion.std())[mi[: len(eeg_motion)]], lw=0.8, label="EEG scalp mean (1-8 Hz)")
ax[1].set_ylabel("z-score")
ax[2].plot(t_grid[mi], ta_r[mi], label="R tibialis anterior MAV")
for x in imu_hs["Right"][(imu_hs["Right"] > w0) & (imu_hs["Right"] < w1)]:
    ax[2].axvline(x, color="tab:orange", ls="--", lw=0.8)
ax[2].set_ylabel("EMG [V]")
ax[2].set_xlabel("s (IMU clock)")
for a in ax:
    a.legend(loc="upper right", fontsize=8)
fig.suptitle(f"{tag}: streams on the IMU clock")
fig.tight_layout()
fig.savefig(OUT_ROOT / f"cell8_{tag}.png", dpi=110)
plt.show()

align = {"tag": tag, "fp_offset_s": d, "fp_1_is_left": bool(left_is_1), **{f"fp_{k}": v for k, v in fp_align.items() if k != "offset_s"},
         "eeg_lag_ms": round(1000 * eeg_lag), "eeg_r": round(eeg_r, 3), "emg_lag_ms": round(1000 * emg_lag), "emg_r": round(emg_r, 3),
         "emg_flat_channels": list(flat.index)}
check(min(len(imu_hs["Left"]), len(imu_hs["Right"])) >= 20, "IMU heel strikes detected on both feet (>= 20 each)")
check(fp_align["mismatch_ms"] < 50, f"force plate -> IMU: median heel-strike mismatch {fp_align['mismatch_ms']:.0f} ms (< 50 ms)")
check(fp_align["runner_up_ms"] > 1.8 * fp_align["mismatch_ms"],
      f"offset is unambiguous (next-best offset fits {fp_align['runner_up_ms']:.0f} ms vs {fp_align['mismatch_ms']:.0f} ms)", critical=False)
check(left_is_1, "force plate '1:' is the LEFT foot (as the dataset page says)", critical=False)
check(abs(eeg_lag) < 0.1 and abs(eeg_r) > 0.15,
      f"rough EEG check (weak proxy, superseded by CELL 8b): lag {1000 * eeg_lag:.0f} ms, |r| {abs(eeg_r):.2f}", critical=False)
check(abs(emg_lag) < 0.4, f"R tibialis anterior vs right-foot impacts: lag {1000 * emg_lag:.0f} ms (r {emg_r:.2f})", critical=False)
check(flat.empty, f"every EMG channel is gait-modulated (flat: {list(flat.index)})", critical=False)
print("align =", align)
end_cell("CELL 8")

# ===== CELL 8b · EEG <-> IMU clock: independent estimates (~30 s) =====
# P2 and P3 decide (they agree on synthetic tests with planted offsets); P1 is shown for reference only.
if "imu_hs" not in globals():
    raise RuntimeError("run CELL 8 first (same trial)")


def bandpass_cols(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    b, a = butter(2, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, np.nan_to_num(x), axis=0)


def smooth(x: np.ndarray, fs: float, win_s: float) -> np.ndarray:
    w = max(int(win_s * fs), 1)
    return np.convolve(x, np.ones(w) / w, mode="same")


def xcorr_curve(a: np.ndarray, b: np.ndarray, fs: float, max_lag_s: float) -> tuple:
    """r(lag) for lags in +-max_lag_s; positive lag = b later than a."""
    n = min(len(a), len(b))
    a = (a[:n] - a[:n].mean()) / a[:n].std()
    b = (b[:n] - b[:n].mean()) / b[:n].std()
    lags = np.arange(-int(max_lag_s * fs), int(max_lag_s * fs) + 1)
    return lags / fs, np.array([np.mean(a[max(0, -k):n - max(0, k)] * b[max(0, k):n - max(0, -k)]) for k in lags])


# EEG proxies of head motion, brought to the 60 Hz IMU rate
step = int(round(eeg_fs / imu_fs))
X = eeg_df[scalp].to_numpy(float)
X = np.nan_to_num(X - np.nanmedian(X, axis=0))
P1, P2, P3 = "P1 scalp mean 1-8 Hz vs head z", "P2 EEG 1-30 Hz envelope vs head |acc|", "P3 EEG 55-95 Hz bursts vs foot impacts"
eeg_sig = {
    P1: bandpass_cols(X.mean(axis=1), eeg_fs, 1, 8)[::step],
    P2: smooth(np.abs(bandpass_cols(X, eeg_fs, 1, 30)).mean(axis=1), eeg_fs, 0.1)[::step],
    P3: smooth((bandpass_cols(X, eeg_fs, 55, 95) ** 2).mean(axis=1), eeg_fs, 0.05)[::step],
}
sfa = pd.read_csv(imu_dir / "Sensor_Free_Acceleration.csv", usecols=["Head x", "Head y", "Head z"]).to_numpy(float)
impacts = smooth(sum(np.abs(bandpass_cols(seg_acc[f"{s} Foot z"].to_numpy(float), imu_fs, 2, 20)) for s in ("Left", "Right")), imu_fs, 0.05)
imu_sig = {
    P1: bandpass_cols(sfa[:, 2], imu_fs, 1, 8),
    P2: smooth(np.linalg.norm(bandpass_cols(sfa, imu_fs, 1, 20), axis=1), imu_fs, 0.1),
    P3: impacts,
}

# convention for every estimate: EEG time + d = IMU time  (d = 0 means EEG and IMU share a clock)
rows, curves = [], {}
for name in eeg_sig:
    lags, r = xcorr_curve(imu_sig[name], eeg_sig[name], imu_fs, 10.0)
    ar = np.abs(r)
    i, i0 = int(np.argmax(ar)), int(np.argmin(np.abs(lags)))
    curves[name] = (-lags, ar)
    rows.append({"estimate": name, "d_s": round(float(-lags[i]), 3), "|r|": round(float(ar[i]), 3),
                 "z": round(float((ar[i] - np.median(ar)) / ar.std()), 1), "|r| at d=0": round(float(ar[i0]), 3)})
est = pd.DataFrame(rows).set_index("estimate")
print(est.to_string())

p3 = eeg_sig[P3]
d2, d3 = est.loc[P2, "d_s"], est.loc[P3, "d_s"]
agree = abs(d2 - d3) < 0.05
strong = min(est.loc[P2, "z"], est.loc[P3, "z"]) > 4
eeg_d = float((d2 + d3) / 2) if agree and strong else float("nan")
long_t = float(eeg_t[rise[int(np.argmax(high))]]) if high else float("nan")
print(f"\nEEG {eeg_dur:.2f} s vs IMU {imu_dur:.2f} s | long trigger pulse starts at EEG t = {long_t:.2f} s | "
      f"FP offset (CELL 8) = {d:+.3f} s")
print(f"=> EEG time + {eeg_d:.3f} s = IMU time  (P2 {d2:+.3f} s, P3 {d3:+.3f} s, agree: {agree}, both z > 4: {strong})")

fig, ax = plt.subplots(2, 1, figsize=(14, 8))
for name, (dd, ar) in curves.items():
    ax[0].plot(dd, ar, lw=0.8, label=name)
if not np.isnan(eeg_d):
    ax[0].axvline(eeg_d, color="k", ls="--", label=f"consensus d = {eeg_d:+.3f} s")
ax[0].axvline(0, color="grey", ls=":", label="d = 0 (shared clock)")
ax[0].set_xlabel("d (s): EEG time + d = IMU time")
ax[0].set_ylabel("|r|")
shift = 0.0 if np.isnan(eeg_d) else eeg_d
t_e = np.arange(len(p3)) / imu_fs + shift
we, wi = (t_e > 10) & (t_e < 20), (t_grid > 10) & (t_grid < 20)
ax[1].plot(t_grid[wi], impacts[wi] / impacts.std(), label="IMU foot impacts (both feet)")
ax[1].plot(t_e[we], p3[we] / p3.std(), lw=0.8, label=f"EEG 55-95 Hz burst power, shifted {shift:+.3f} s")
ax[1].set_xlabel("s (IMU clock)")
for a in ax:
    a.legend(loc="upper right", fontsize=8)
fig.suptitle(f"{tag}: where does the EEG sit on the IMU clock?")
fig.tight_layout()
fig.savefig(OUT_ROOT / f"cell8b_{tag}.png", dpi=110)
plt.show()

align.update({"eeg_offset_s": eeg_d, "eeg_p2_s": float(d2), "eeg_p3_s": float(d3)})
check(agree, f"P2 and P3 agree on the EEG offset within 50 ms ({d2:+.3f} vs {d3:+.3f} s)", critical=False)
check(strong, f"both have a clear peak (z = {est.loc[P2, 'z']:.1f}, {est.loc[P3, 'z']:.1f}; need > 4)", critical=False)
check(not np.isnan(eeg_d) and abs(eeg_d) < 0.05, f"EEG and IMU share a clock (d = {eeg_d:+.3f} s, close to 0)", critical=False)
end_cell("CELL 8b")

# ===== CELL 9/10 · All complete trials: events, alignment, EMG QC, EEG sync signals (~5-10 min) =====
from scipy.signal import iirnotch

need = ["one", "rate_and_duration", "lowpass", "gait_events", "match_offset", "nearest_residuals", "imu_heel_strikes",
        "bandpass", "xcorr_lag", "bandpass_cols", "smooth", "scalp", "manifest"]
missing_defs = [n for n in need if n not in globals()]
if missing_defs:
    raise RuntimeError(f"run CELL 1, 4, 6, 7, 8 and 8b first (missing: {missing_defs})")
IMU_FS = 60.0


def trial_events(fz: dict, fs: float) -> tuple:
    """Force-plate stances per leg (force-plate clock) and body weight, exactly as in CELL 7."""
    bw_ = float(np.median(fz["L"] + fz["R"]))
    ev = {}
    for leg, f in fz.items():
        base = float(np.percentile(f, 2))
        ev[leg] = gait_events(f, fs, base + 0.05 * bw_, base + 0.03 * bw_)
    return ev, bw_


def window_max(x: np.ndarray, t0: float, t1: float, fs: float) -> float:
    seg = x[int(t0 * fs):max(int(t1 * fs), int(t0 * fs) + 1)]
    return float(seg.max()) if len(seg) else float("nan")


def process_trial(subj: int, speed: float) -> tuple:
    """One trial -> (summary row, stride table on the IMU clock, EEG/IMU sync signals at 60 Hz)."""
    tg = f"S{subj}_{speed:g}"
    fp_ = pd.read_csv(one(f"FP_{tg}.csv", DATA), usecols=list(LEGS.values()))
    fz = {leg: lowpass(fp_[col].to_numpy(float), FS_FP) for leg, col in LEGS.items()}
    ev, bw_ = trial_events(fz, FS_FP)
    stride = float(pd.concat([ev["L"].stride_s, ev["R"].stride_s]).median())

    idir = one(f"IMU_{tg}", IMU_ROOT)
    acc = pd.read_csv(idir / "Segment_Acceleration.csv", usecols=["Left Foot z", "Right Foot z"])
    ihs = {s: imu_heel_strikes(acc[f"{s} Foot z"].to_numpy(float), IMU_FS, stride) for s in ("Left", "Right")}
    fit = match_offset([(ev["L"].hs.to_numpy(), ihs["Left"]), (ev["R"].hs.to_numpy(), ihs["Right"])])
    swap = match_offset([(ev["L"].hs.to_numpy(), ihs["Right"]), (ev["R"].hs.to_numpy(), ihs["Left"])])
    off = fit["offset_s"]

    tabs = []
    for leg, side in (("L", "Left"), ("R", "Right")):
        e = ev[leg].copy()
        res = nearest_residuals(ihs[side], e.hs.to_numpy() + off)
        dfz = np.gradient(fz[leg]) * FS_FP
        e["hs_imu"] = e.hs + off + res          # matched IMU heel strike = stride clock (IMU time)
        e["hs_res_ms"] = 1000 * res
        e["to_imu"] = e.to + off
        e["peak_fz_25"] = [window_max(fz[leg], h, h + 0.25 * (t - h), FS_FP) for h, t in zip(e.hs, e.to)]
        e["loading_rate"] = [window_max(dfz, h, h + 0.25 * (t - h), FS_FP) for h, t in zip(e.hs, e.to)]
        e["subject"], e["speed"], e["leg"] = subj, speed, leg
        tabs.append(e)

    emg_ = pd.read_csv(one(f"EMG_{tg}.csv", DATA))
    et = emg_.iloc[:, 0].to_numpy(float)
    mav_ = emg_.iloc[:, 1:]
    mi = ((mav_.quantile(0.95) - mav_.quantile(0.05)) / mav_.median()).rename(lambda c: "MI " + c.split(":")[0])
    tgrid = np.arange(len(acc)) / IMU_FS
    ta = np.interp(tgrid, et, mav_[[c for c in mav_.columns if c.startswith("R TIBIALIS ANTERIOR")][0]].to_numpy(float))
    fenv = np.convolve(np.abs(bandpass(acc["Right Foot z"].to_numpy(float), IMU_FS, 2, 20)), np.ones(12) / 12, mode="same")
    ta_lag, ta_r = xcorr_lag(fenv, ta, IMU_FS, 0.8)

    eeg_ = pd.read_csv(one(f"{tg}_raw.csv", DATA), comment="#", usecols=["Time"] + scalp, low_memory=False)
    efs, edur = rate_and_duration(eeg_["Time"].to_numpy(float))
    xe = eeg_[scalp].to_numpy(float)
    xe = np.nan_to_num(xe - np.nanmedian(xe, axis=0))
    stp = int(round(efs / IMU_FS))
    bn, an = iirnotch(50.0, 30.0, efs)  # mains 50 Hz (EEG header)
    hd = pd.read_csv(idir / "Sensor_Free_Acceleration.csv", usecols=["Head x", "Head y", "Head z"]).to_numpy(float)
    sig = {"p2e": smooth(np.abs(bandpass_cols(xe, efs, 1, 30)).mean(axis=1), efs, 0.1)[::stp],
           "p3e": smooth((bandpass_cols(filtfilt(bn, an, xe, axis=0), efs, 55, 95) ** 2).mean(axis=1), efs, 0.05)[::stp],
           "p2i": smooth(np.linalg.norm(bandpass_cols(hd, IMU_FS, 1, 20), axis=1), IMU_FS, 0.1),
           "p3i": smooth(sum(np.abs(bandpass_cols(acc[f"{s} Foot z"].to_numpy(float), IMU_FS, 2, 20))
                             for s in ("Left", "Right")), IMU_FS, 0.05)}

    row = {"subject": subj, "speed": speed, "bw_N": round(bw_, 1), "stride_s": round(stride, 3),
           "n_strides_L": int(ev["L"].stride_s.notna().sum()), "n_strides_R": int(ev["R"].stride_s.notna().sum()),
           "imu_hs_L": len(ihs["Left"]), "imu_hs_R": len(ihs["Right"]),
           "fp_offset_s": off, "fp_mismatch_ms": fit["mismatch_ms"], "fp_runner_up_ms": fit["runner_up_ms"],
           "fp_swapped_mismatch_ms": swap["mismatch_ms"],
           "dur_fp_s": round(len(fp_) / FS_FP, 2), "dur_imu_s": round(len(acc) / IMU_FS, 2),
           "dur_emg_s": round(rate_and_duration(et)[1], 2), "dur_eeg_s": round(edur, 2),
           "ta_lag_ms": round(1000 * ta_lag), "ta_r": round(ta_r, 3), **mi.round(3).to_dict()}
    return row, pd.concat(tabs, ignore_index=True), sig


trials = [(int(s), float(sp)) for s, sp in manifest.index[manifest.all4.to_numpy()]]
log(f"{len(trials)} complete trials to process")
rows, stride_tabs, sigs, failed = [], [], {}, {}
t_start = time.time()
for i, (s, sp) in enumerate(trials, 1):
    try:
        row, st, sg = process_trial(s, sp)
        rows.append(row)
        stride_tabs.append(st)
        sigs[f"S{s}_{sp:g}"] = sg
    except (OSError, ValueError, KeyError, IndexError, ZeroDivisionError) as exc:
        failed[f"S{s}_{sp:g}"] = f"{type(exc).__name__}: {exc}"
    if i % 10 == 0 or i == len(trials):
        log(f"{i}/{len(trials)} trials | failed {len(failed)} | {time.time() - t_start:.0f}s")

trial_tab = pd.DataFrame(rows)
strides_all = pd.concat(stride_tabs, ignore_index=True)
trial_tab.to_csv(OUT_ROOT / "trials_alignment.csv", index=False)
strides_all.to_csv(OUT_ROOT / "strides.csv", index=False)
n_common = min(len(v) for sg in sigs.values() for v in sg.values())
np.savez_compressed(OUT_ROOT / "eeg_sync_signals.npz", tags=np.array(list(sigs)), fs=IMU_FS,
                    **{k: np.array([sg[k][:n_common] for sg in sigs.values()], dtype=np.float32) for k in ("p2e", "p2i", "p3e", "p3i")})

ok_fp = (trial_tab.fp_mismatch_ms < 50) & (trial_tab.fp_runner_up_ms > 1.8 * trial_tab.fp_mismatch_ms)
print(trial_tab[["fp_offset_s", "fp_mismatch_ms", "stride_s", "bw_N", "ta_lag_ms",
                 "dur_fp_s", "dur_imu_s", "dur_emg_s", "dur_eeg_s"]].describe().round(3).to_string())
if (~ok_fp).any():
    print("trials with weak force-plate alignment:", trial_tab.loc[~ok_fp, ["subject", "speed", "fp_mismatch_ms", "fp_runner_up_ms"]].to_dict("records"))
mi_cols = [c for c in trial_tab.columns if c.startswith("MI ")]
print("\nshare of trials where each EMG channel is flat (MI < 0.3):")
print((trial_tab[mi_cols] < 0.3).mean().sort_values(ascending=False).round(2).to_string())
if failed:
    print("failed trials:", failed)

check(len(failed) <= 0.05 * len(trials), f"{len(trials) - len(failed)}/{len(trials)} trials processed")
check(ok_fp.mean() >= 0.9, f"force plate aligned cleanly in {int(ok_fp.sum())}/{len(trial_tab)} trials (< 50 ms, unambiguous)", critical=False)
check(bool((trial_tab.fp_mismatch_ms < trial_tab.fp_swapped_mismatch_ms).all()), "force plate '1:' = left foot in every trial", critical=False)
check(trial_tab.ta_lag_ms.std() < 100, f"R tibialis anterior delay is stable across trials (SD {trial_tab.ta_lag_ms.std():.0f} ms)", critical=False)
check(len(strides_all) > 0, f"stride table saved: {len(strides_all)} strides -> {OUT_ROOT / 'strides.csv'}")
end_cell("CELL 9")

# ===== CELL 9b · Fix-ups: failed trials, force-plate offset with a near-zero prior, QC flags (~2-3 min) =====
# All streams were exported as the same 1-min window (EEG 59.87 s in every trial, FP mostly 60.0 s) and CELL 9's
# offsets cluster near 0, so first search +-0.5 s; use the wide search only when that fails (e.g. S1_0.5, 55.5 s).
if "trial_tab" not in globals():
    raise RuntimeError("run CELL 9 first")
PRIOR_WIN = 0.5
MAX_MISMATCH_MS = 40.0


def match_window(pairs: list, max_shift: float) -> dict:
    """Like match_offset (CELL 8), restricted to |d| <= max_shift and safe when nothing matches."""
    cands = np.unique(np.round(np.concatenate([(imu[None, :] - fp[:, None]).ravel() for fp, imu in pairs]), 3))
    cands = cands[np.abs(cands) <= max_shift]

    def residuals(d: float) -> np.ndarray:
        parts = []
        for fp, imu in pairs:
            q = fp + d
            q = q[(q > imu[0]) & (q < imu[-1])]
            parts.append(nearest_residuals(imu, q))
        return np.concatenate(parts)

    scores = np.array([np.median(np.abs(r)) if len(r := residuals(d)) >= 20 else np.inf for d in cands])
    if not len(scores) or not np.isfinite(scores.min()):
        return {"offset_s": float("nan"), "mismatch_ms": float("inf")}
    best = cands[int(np.argmin(scores))]
    d = float(best + np.median(residuals(best)))
    return {"offset_s": round(d, 4), "mismatch_ms": round(1000 * float(np.median(np.abs(residuals(d)))), 1)}


# --- 1. why did trials fail? ---
for tg_, err in failed.items():
    s_, sp_ = tg_[1:].split("_")
    s_, sp_ = int(s_), float(sp_)
    print(f"\n--- {tg_}: {err[:140]}")
    for f in list(inv[(inv.subject == s_) & (inv.speed == sp_)].path) + [str(p) for p in DATA.rglob(f"{tg_}_raw.csv")]:
        p_ = Path(f)
        if p_.is_dir():
            csvs = list(p_.rglob("*.csv"))
            print(f"    {p_.name}/ : {len(csvs)} csv | empty: {[c.name for c in csvs if c.stat().st_size == 0]}")
            continue
        try:
            cols = list(pd.read_csv(p_, nrows=2, comment="#").columns) if p_.suffix == ".csv" else ["(edf, not read)"]
        except pd.errors.EmptyDataError:
            cols = ["<EMPTY FILE>"]
        print(f"    {p_.name}: {p_.stat().st_size / 1e6:.2f} MB | {len(cols)} columns: {cols[:26]}")

# --- 2. force-plate offset with the near-zero prior ---
qc_rows, new_tabs = [], []
for k_, r in enumerate(trial_tab.itertuples(), 1):
    s_, sp_ = int(r.subject), float(r.speed)
    tg_ = f"S{s_}_{sp_:g}"
    st = strides_all[(strides_all.subject == s_) & (strides_all.speed == sp_)]
    acc = pd.read_csv(one(f"IMU_{tg_}", IMU_ROOT) / "Segment_Acceleration.csv", usecols=["Left Foot z", "Right Foot z"])
    ihs = {sd: imu_heel_strikes(acc[f"{sd} Foot z"].to_numpy(float), IMU_FS, r.stride_s) for sd in ("Left", "Right")}
    fp_hs = {leg: st[st.leg == leg].hs.to_numpy() for leg in ("L", "R")}
    near = match_window([(fp_hs["L"], ihs["Left"]), (fp_hs["R"], ihs["Right"])], PRIOR_WIN)
    near_sw = match_window([(fp_hs["L"], ihs["Right"]), (fp_hs["R"], ihs["Left"])], PRIOR_WIN)
    wide_ok = r.fp_mismatch_ms < MAX_MISMATCH_MS and r.fp_runner_up_ms > 1.8 * r.fp_mismatch_ms
    near_ok = near["mismatch_ms"] < MAX_MISMATCH_MS
    if near_ok and wide_ok and abs(r.fp_offset_s - near["offset_s"]) > 0.3 and r.fp_mismatch_ms < near["mismatch_ms"]:
        d_fin, mm, how = r.fp_offset_s, r.fp_mismatch_ms, "wide (beats near-zero)"
    elif near_ok:
        d_fin, mm, how = near["offset_s"], near["mismatch_ms"], "near-zero window"
    elif wide_ok:
        d_fin, mm, how = r.fp_offset_s, r.fp_mismatch_ms, "wide search (unambiguous)"
    else:
        d_fin, mm, how = float("nan"), float("nan"), "UNALIGNED"
    bw_ok = 300 < r.bw_N < 1500
    qc_rows.append({"subject": s_, "speed": sp_, "fp_offset_final_s": d_fin, "fp_mismatch_final_ms": mm, "method": how,
                    "near_offset_s": near["offset_s"], "near_mismatch_ms": near["mismatch_ms"],
                    "near_swapped_mismatch_ms": near_sw["mismatch_ms"], "wide_offset_s": r.fp_offset_s,
                    "wide_mismatch_ms": r.fp_mismatch_ms, "bw_N": r.bw_N, "bw_ok": bw_ok, "dur_fp_s": r.dur_fp_s,
                    "fp_usable": bool(bw_ok and how != "UNALIGNED")})
    if bw_ok and how != "UNALIGNED":
        for leg, side in (("L", "Left"), ("R", "Right")):
            e = st[st.leg == leg].copy()
            res = nearest_residuals(ihs[side], e.hs.to_numpy() + d_fin)
            e["hs_imu"] = e.hs + d_fin + res
            e["hs_res_ms"] = 1000 * res
            e["to_imu"] = e.to + d_fin
            new_tabs.append(e)
    if k_ % 20 == 0 or k_ == len(trial_tab):
        log(f"{k_}/{len(trial_tab)} trials re-aligned")

qc = pd.DataFrame(qc_rows)
strides_v2 = pd.concat(new_tabs, ignore_index=True)
qc.to_csv(OUT_ROOT / "trials_qc.csv", index=False)
strides_v2.to_csv(OUT_ROOT / "strides_v2.csv", index=False)

print("\nmethod used:")
print(qc.method.value_counts().to_string())
print("\nfinal offset / mismatch (usable trials):")
print(qc.loc[qc.fp_usable, ["fp_offset_final_s", "fp_mismatch_final_ms"]].describe().round(3).to_string())
bad = qc[~qc.fp_usable]
if len(bad):
    print("\nNOT usable for force-plate targets:", bad[["subject", "speed", "method", "bw_N", "near_mismatch_ms", "wide_mismatch_ms"]].to_dict("records"))
conflict = qc[qc.method.str.startswith("wide (beats")]
if len(conflict):
    print("\nwide search beat the near-zero window (check these):", conflict[["subject", "speed", "wide_offset_s", "near_offset_s", "dur_fp_s"]].to_dict("records"))
ta_q = trial_tab.ta_lag_ms.quantile([0.25, 0.5, 0.75])
ta_near = float((trial_tab.ta_lag_ms.abs() <= 100).mean())
print(f"\nR tibialis anterior vs right-foot impact lag: median {ta_q[0.5]:.0f} ms, IQR {ta_q[0.25]:.0f} to {ta_q[0.75]:.0f} ms, "
      f"{100 * ta_near:.0f}% of trials within +-100 ms")

usable = qc.fp_usable.mean()
left_ok = (qc.near_mismatch_ms <= qc.near_swapped_mismatch_ms)[qc.method == "near-zero window"].mean()
check(usable >= 0.9, f"force plate usable in {int(qc.fp_usable.sum())}/{len(qc)} trials ({100 * usable:.0f}%)", critical=False)
check(qc.loc[qc.fp_usable, "fp_mismatch_final_ms"].median() < 20,
      f"median heel-strike mismatch {qc.loc[qc.fp_usable, 'fp_mismatch_final_ms'].median():.1f} ms (< 20 ms)", critical=False)
check(left_ok >= 0.95, f"force plate '1:' = left foot in {100 * left_ok:.0f}% of near-zero trials", critical=False)
check(ta_near >= 0.7, f"EMG on the IMU clock: R TA lag within +-100 ms in {100 * ta_near:.0f}% of trials", critical=False)
check(len(strides_v2) > 0, f"corrected stride table saved: {len(strides_v2)} strides -> {OUT_ROOT / 'strides_v2.csv'}")
end_cell("CELL 9b")

# ===== CELL 9c · Fix-ups 2: robust readers, alignment tiers, stride clock on IMU + EEG time (~2-3 min) =====
need = ["one", "lowpass", "gait_events", "trial_events", "window_max", "match_offset", "match_window",
        "nearest_residuals", "imu_heel_strikes", "trials", "failed"]
missing_defs = [n for n in need if n not in globals()]
if missing_defs:
    raise RuntimeError(f"run CELL 7, 8, 9 and 9b first (missing: {missing_defs})")
FP_COLS = [f"{leg}:{c}" for leg in (1, 2) for c in ["Timestamp", "Sync", "Aux", "Fx", "Fy", "Fz", "Mx", "My", "Mz", "COPx", "COPy"]]
NEAR_WIN = 0.3      # below half the shortest stride (~0.48 s): a swapped-leg match cannot fake a fit
EEG_SHIFT = 0.025   # CELL 10: EEG time = IMU time + 0.025 s
MAX_MM, RELAXED_MM = 40.0, 80.0


def read_fp(tg: str) -> pd.DataFrame:
    """Force-plate Fz columns; a few files lack the header row (S2_0.5) -> use the standard 22 names."""
    path = one(f"FP_{tg}.csv", DATA)
    if "1:Fz" in pd.read_csv(path, nrows=0).columns:
        return pd.read_csv(path, usecols=list(LEGS.values()))
    return pd.read_csv(path, header=None, names=FP_COLS, usecols=list(LEGS.values()))


def diagnose(tg: str) -> None:
    """Try every file of a trial separately and show which one fails, with its size and first bytes."""
    files = [one(f"FP_{tg}.csv", DATA), one(f"EMG_{tg}.csv", DATA), one(f"{tg}_raw.csv", DATA)]
    files += sorted(one(f"IMU_{tg}", IMU_ROOT).rglob("*.csv"))
    for p_ in files:
        try:
            pd.read_csv(p_, nrows=3, comment="#" if p_.name.endswith("_raw.csv") else None)
        except (ValueError, OSError) as exc:  # EmptyDataError is a ValueError
            print(f"    FAILS {p_.name}: {type(exc).__name__} | {p_.stat().st_size} bytes | first bytes {p_.read_bytes()[:60]!r}")
    print(f"    (all other {tg} files read fine)")


# --- 1. diagnose the trials that failed in CELL 9 ---
for tg_ in failed:
    print(f"--- {tg_}")
    try:
        diagnose(tg_)
    except (OSError, ValueError) as exc:
        print(f"    could not list files: {exc}")
try:
    import mne
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mne"], check=True)
    import mne
for tg_ in ("S32_1", "S32_0.75"):
    try:
        edf_ch = mne.io.read_raw_edf(one(f"{tg_}_raw.edf", DATA), preload=False, verbose="ERROR").ch_names
        print(f"--- {tg_} EDF channel labels: {edf_ch}")
    except (OSError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"--- {tg_} EDF not readable: {exc}")

# --- 2. pass 1: per-trial force-plate events, targets, IMU heel strikes, offset estimates ---
est, errors = {}, {}
for k_, (s_, sp_) in enumerate(trials, 1):
    tg_ = f"S{s_}_{sp_:g}"
    try:
        fp_ = read_fp(tg_)
        fz = {leg: lowpass(fp_[col].to_numpy(float), FS_FP) for leg, col in LEGS.items()}
        ev, bw_ = trial_events(fz, FS_FP)
        stride = float(pd.concat([ev["L"].stride_s, ev["R"].stride_s]).median())
        acc = pd.read_csv(one(f"IMU_{tg_}", IMU_ROOT) / "Segment_Acceleration.csv", usecols=["Left Foot z", "Right Foot z"])
        ihs = {sd: imu_heel_strikes(acc[f"{sd} Foot z"].to_numpy(float), IMU_FS, stride) for sd in ("Left", "Right")}
        pa = [(ev["L"].hs.to_numpy(), ihs["Left"]), (ev["R"].hs.to_numpy(), ihs["Right"])]
        pb = [(ev["L"].hs.to_numpy(), ihs["Right"]), (ev["R"].hs.to_numpy(), ihs["Left"])]
        tabs = []
        for leg in ("L", "R"):
            e = ev[leg].copy()
            dfz = np.gradient(fz[leg]) * FS_FP
            e["peak_fz_25"] = [window_max(fz[leg], h, h + 0.25 * (t - h), FS_FP) for h, t in zip(e.hs, e.to)]
            e["loading_rate"] = [window_max(dfz, h, h + 0.25 * (t - h), FS_FP) for h, t in zip(e.hs, e.to)]
            e["leg"] = leg
            tabs.append(e)
        est[tg_] = {"subject": s_, "speed": sp_, "bw": bw_, "dur_fp": len(fp_) / FS_FP, "ihs": ihs,
                    "strides": pd.concat(tabs, ignore_index=True), "near": match_window(pa, NEAR_WIN),
                    "near_sw": match_window(pb, NEAR_WIN), "wide": match_offset(pa)}
    except (OSError, ValueError, KeyError, IndexError, ZeroDivisionError) as exc:
        errors[tg_] = f"{type(exc).__name__}: {str(exc)[:100]}"
    if k_ % 20 == 0 or k_ == len(trials):
        log(f"{k_}/{len(trials)} trials | errors {len(errors)}")

# --- 3. pass 2: alignment tiers, stride clock on IMU and EEG time ---
prior = float(np.median([e["near"]["offset_s"] for e in est.values() if e["near"]["mismatch_ms"] < MAX_MM]))
qc_rows, out_tabs = [], []
for tg_, e in est.items():
    near, wide = e["near"], e["wide"]
    near_ok = near["mismatch_ms"] < MAX_MM
    wide_ok = wide["mismatch_ms"] < MAX_MM and wide["runner_up_ms"] > 1.8 * wide["mismatch_ms"]
    if not 300 < e["bw"] < 1500:
        tier, why, d_ = "excluded", "force plate empty/dead (body weight)", float("nan")
    elif near_ok and wide_ok and abs(wide["offset_s"] - near["offset_s"]) > 0.3 and wide["mismatch_ms"] < near["mismatch_ms"]:
        tier, why, d_ = "strict", "wide search beats near window", wide["offset_s"]
    elif near_ok:
        tier, why, d_ = "strict", "near window", near["offset_s"]
    elif wide_ok:
        tier, why, d_ = "strict", "wide search (unambiguous)", wide["offset_s"]
    elif near["mismatch_ms"] < RELAXED_MM and abs(near["offset_s"] - prior) < 0.1 and e["dur_fp"] >= 59.9:
        tier, why, d_ = "relaxed", "near window, noisy IMU events", near["offset_s"]
    else:
        tier, why, d_ = "excluded", "no reliable alignment", float("nan")
    qc_rows.append({"tag": tg_, "subject": e["subject"], "speed": e["speed"], "tier": tier, "reason": why, "fp_offset_s": d_,
                    "near_offset_s": near["offset_s"], "near_mismatch_ms": near["mismatch_ms"],
                    "near_swapped_mismatch_ms": e["near_sw"]["mismatch_ms"], "wide_offset_s": wide["offset_s"],
                    "wide_mismatch_ms": wide["mismatch_ms"], "bw_N": round(e["bw"], 1), "dur_fp_s": round(e["dur_fp"], 2)})
    if tier != "excluded":
        st = e["strides"].copy()
        st["hs_imu"], st["to_imu"] = st.hs + d_, st.to + d_
        st["hs_eeg"], st["to_eeg"] = st.hs_imu + EEG_SHIFT, st.to_imu + EEG_SHIFT
        st["imu_res_ms"] = np.concatenate([1000 * nearest_residuals(e["ihs"][side], st.loc[st.leg == leg, "hs_imu"].to_numpy())
                                           for leg, side in (("L", "Left"), ("R", "Right"))])
        st["subject"], st["speed"], st["tier"] = e["subject"], e["speed"], tier
        out_tabs.append(st)

qc3 = pd.DataFrame(qc_rows)
strides_v3 = pd.concat(out_tabs, ignore_index=True)
qc3.to_csv(OUT_ROOT / "trials_qc_v3.csv", index=False)
strides_v3.to_csv(OUT_ROOT / "strides_v3.csv", index=False)

print(f"\nprior offset (median of near-window fits < {MAX_MM:.0f} ms): {prior:+.3f} s")
print(qc3.groupby(["tier", "reason"]).size().to_string())
strict = qc3[qc3.tier == "strict"]
print("\nstrict trials per speed:", strict.groupby("speed").size().to_dict(),
      "| subjects with >= 1 strict trial:", strict.subject.nunique(),
      "| with all 3 speeds strict:", int((strict.groupby("subject").size() == 3).sum()))
print("stride counts by tier:", strides_v3.groupby("tier").size().to_dict())
excl = qc3[qc3.tier == "excluded"]
if len(excl):
    print("\nexcluded:", excl[["tag", "reason", "near_mismatch_ms", "wide_mismatch_ms"]].to_dict("records"))
if errors:
    print("\nstill failing:", errors)
fin = qc3[np.isfinite(qc3.near_mismatch_ms)]  # swapped mapping = inf means it found no fit at all (left mapping wins)
left_ok = float((fin.near_mismatch_ms < fin.near_swapped_mismatch_ms).mean()) if len(fin) else float("nan")

check(len(strides_v3) > 0, f"stride table saved: {len(strides_v3)} strides -> {OUT_ROOT / 'strides_v3.csv'}")
check(len(strict) >= 0.7 * len(qc3), f"strict tier: {len(strict)}/{len(qc3)} trials ({100 * len(strict) / len(qc3):.0f}%)", critical=False)
check(left_ok >= 0.95, f"force plate '1:' = left foot in {100 * left_ok:.0f}% of trials (+-{NEAR_WIN} s window)", critical=False)
check("S2_0.5" in est, "S2_0.5 (no header row) is now read", critical=False)
end_cell("CELL 9c")

# ===== CELL 10/10 · Pooled EEG <-> IMU clock test with a permutation null (~30 s) =====
# If all trials share ONE EEG offset (0 = shared clock), the trial-averaged cross-correlation peaks there. Null: keep
# each trial's own EEG-IMU pair but shift its lag randomly (+-20 s), i.e. "no common offset across trials".
z = np.load(OUT_ROOT / "eeg_sync_signals.npz")
TRIM = 3.0  # s cut from both ends: filter/smoothing edge transients line up across trials and fake a common lag
fs_s = float(z["fs"])
cut = int(TRIM * fs_s)
z = {k: z[k][:, cut:-cut] for k in ("p2e", "p2i", "p3e", "p3i")}
n_tr, n_s = z["p2e"].shape
MAX_LAG, N_PERM, NULL_SHIFT = 10.0, 1000, 20.0
nfft = 1 << (2 * n_s - 1).bit_length()
lag_k = np.arange(-int(MAX_LAG * fs_s), int(MAX_LAG * fs_s) + 1)
d_ax = -lag_k / fs_s  # convention: EEG time + d = IMU time
rng_p = np.random.default_rng(0)
rows_idx = np.arange(n_tr)[:, None]


def spectra(x: np.ndarray) -> np.ndarray:
    x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    return np.fft.rfft(x, nfft, axis=1)


fig, ax = plt.subplots(figsize=(14, 5))
results = {}
for name, ek, ik in (("P2 EEG 1-30 Hz envelope vs head |acc|", "p2e", "p2i"), ("P3 EEG 55-95 Hz bursts vs foot impacts", "p3e", "p3i")):
    A, B = spectra(z[ik].astype(float)), spectra(z[ek].astype(float))
    full = np.fft.irfft(np.conj(A) * B, nfft, axis=1) / n_s  # per-trial r(lag) = corr(imu[t], eeg[t + lag]), all lags
    obs = full[:, lag_k % nfft].mean(axis=0)
    i = int(np.argmax(obs))
    null_max = np.empty(N_PERM)
    for j in range(N_PERM):
        s = rng_p.integers(-int(NULL_SHIFT * fs_s), int(NULL_SHIFT * fs_s) + 1, n_tr)
        null_max[j] = full[rows_idx, (lag_k[None, :] - s[:, None]) % nfft].mean(axis=0).max()
    p = (1 + int((null_max >= obs[i]).sum())) / (N_PERM + 1)
    results[name] = {"peak_d_s": round(float(d_ax[i]), 3), "peak_r": round(float(obs[i]), 4),
                     "null_99pct_r": round(float(np.quantile(null_max, 0.99)), 4), "p": round(p, 4)}
    ax.plot(d_ax, obs, lw=1, label=f"{name}: peak d = {d_ax[i]:+.3f} s, p = {p:.3f}")
    ax.axhline(np.quantile(null_max, 0.99), ls="--", lw=0.8, color=ax.lines[-1].get_color())
ax.axvline(0, color="grey", ls=":", label="d = 0 (shared clock)")
ax.set_xlabel("d (s): EEG time + d = IMU time")
ax.set_ylabel(f"mean r over {n_tr} trials (dashed = null 99th pct)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_ROOT / "cell10_pooled_eeg_sync.png", dpi=110)
plt.show()

res_df = pd.DataFrame(results).T
print(res_df.to_string())
sig = res_df[res_df.p < 0.01]
eeg_offset_global = float(sig.peak_d_s.median()) if len(sig) and (sig.peak_d_s.max() - sig.peak_d_s.min()) < 0.05 else float("nan")
print(f"\n=> {len(sig)} of 2 proxies significant (p < 0.01); pooled EEG offset: EEG time + {eeg_offset_global:+.3f} s = IMU time")
pd.Series({"eeg_offset_global_s": eeg_offset_global, "n_trials": n_tr, "n_perm": N_PERM,
           **{f"{k.split()[0]}_{c}": v for k, r in results.items() for c, v in r.items()}}).to_json(OUT_ROOT / "eeg_sync_decision.json")
check(len(sig) >= 1, f"at least one EEG motion proxy has a significant pooled peak ({len(sig)} of 2, p < 0.01)", critical=False)
check(len(sig) >= 1 and not np.isnan(eeg_offset_global), "the significant proxies agree on one offset (within 50 ms)", critical=False)
check(not np.isnan(eeg_offset_global) and abs(eeg_offset_global) < 0.05,
      f"pooled EEG offset is ~0 ({eeg_offset_global:+.3f} s) -> EEG and IMU share a clock", critical=False)
end_cell("CELL 10")
