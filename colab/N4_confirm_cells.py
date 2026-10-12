# ===== CELL 16a · Freeze the confirmatory analysis plan (reads no confirm-subject data) (~5 s) =====
# Writes frozen_plan.json and its SHA-256 to the w400 folder. CELL 16b refuses to run without them, and every confirm
# result file carries the hash. Recommended before CELL 16b: upload frozen_plan.json to OSF (or e-mail it to a
# co-author) so the freeze has an outside time stamp. Never edit the file afterwards; a change = new plan + new hash.
import hashlib
import json

if OUT_ROOT.name != "w400":
    raise RuntimeError("run CELL 1 and CELL R (WINDOW_DIR = 'w400') first")
PLAN_PATH, HASH_PATH = OUT_ROOT / "frozen_plan.json", OUT_ROOT / "frozen_plan.sha256"
ALREADY_FROZEN = PLAN_PATH.exists()
split_ = json.loads((OUT_ROOT / "dev_confirm_split_v2.json").read_text())
cols_ = pd.read_parquet(OUT_ROOT / "stride_features.parquet").columns
m0_cols = [c for c in cols_ if c.startswith(("imu_", "emg_")) and not c.endswith("_ok")]   # imu_ok = flag, not a feature
eeg_cols = [c for c in cols_ if c.startswith("eeg_rel_")]
dev_two = pd.read_csv(OUT_ROOT / "dev_two_stage_results.csv")
dev_pw = pd.read_csv(OUT_ROOT / "dev_power.csv")
mde_ = (dev_pw[(dev_pw.q_planted > 0) & (dev_pw.power_N_confirm >= 0.8)].groupby(["variant", "target"]).q_planted.min())

plan = {
    "title": "Does pre-landing dry EEG add stride-level information about landing mechanics beyond IMU + EMG?",
    "dataset": "PhysioNet multimodal-gait-dataset 1.0.0 (doi 10.13026/r0ea-7161)",
    "frozen": time.strftime("%Y-%m-%d %H:%M"),
    "split": {"file": "dev_confirm_split_v2.json", "dev": split_["dev"], "confirm": split_["confirm"]},
    "window_s": [-0.4, -0.1],
    "trial_set": "tier == strict, not odd_mapping, eeg_ok, imu_ok (S32_1 excluded upstream)",
    "targets": ["T1_peak_fz", "T2_loading_rate", "T3_knee_flex_early"],
    "m0_columns": m0_cols,
    "m0_note": "the imu_ok flag is excluded; in the dev cells it was a constant (all-zero after z-scoring) column, so predictions are unchanged",
    "eeg_columns": eeg_cols,
    "preprocessing": "targets and features z-scored within subject x speed x leg; features clipped at +-5 SD; "
                     "per target, strides with |target z| > 4 dropped; missing EEG features = 0 after z-scoring",
    "model": {"m0": "RidgeCV, alphas logspace(-2, 4, 25), efficient leave-one-out (GCV)",
              "eeg_stage": "EEG made orthogonal to M0 by least squares on the training subjects; RidgeCV, alphas "
                           "logspace(0, 6, 25), on the M0 training residuals; prediction = M0 + EEG stage"},
    "cv": "each confirm subject held out once; training = all other primary-set subjects (dev and confirm)",
    "metric": "per held-out subject: delta R2 = R2(M0 + EEG stage) - R2(M0), R2 relative to the trial mean (= 0)",
    "primary": {"variant": "trial-centered", "test": "one-sided Wilcoxon signed-rank, delta R2 > 0, over the 33 confirm subjects",
                "alpha": 0.05, "multiplicity": "Holm over the 3 targets"},
    "equivalence": {"sesoi_delta_r2": 0.01, "test": "one-sided Wilcoxon, delta R2 - 0.01 < 0, Holm over the 3 targets",
                    "claim_if_significant": "EEG did not improve held-out prediction by delta R2 >= 0.01"},
    "estimates": "median delta R2 with bootstrap 95% CI over subjects (4000 resamples, seed 20261009); M0 R2 reported as the mechanics baseline",
    "secondary": ["stride-to-stride variant (first differences within trial), same model and tests",
                  "circular-shift null of the median delta R2 (each trial's EEG rotated by >= 3 strides; 100 rotations)",
                  "window [-600, -100] ms (sensitivity)", "relaxed tier + odd-mapping trials added (sensitivity)",
                  "cross-speed: train on two speeds of the training subjects, test on the third speed of the held-out subject",
                  "per leg and per speed (descriptive)", "broadband-power model and dose-response vs artifact index (descriptive)",
                  "CNN on the pre-landing EEG predicting the M0 residuals, same two-stage logic and circular control (after the primary test)"],
    "dev_evidence": {"two_stage_results": dev_two.to_dict("records"),
                     "mde_planted_q_power80_N33": {f"{v}/{t}": float(q) for (v, t), q in mde_.items()}},
    "deviations": "any deviation from this plan is reported as such in the paper",
}
if ALREADY_FROZEN:                                 # never re-written: re-running this cell only verifies the frozen file
    log(f"plan already frozen on {json.loads(PLAN_PATH.read_text())['frozen']} -> not re-written")
else:
    PLAN_PATH.write_text(json.dumps(plan, indent=1))
    HASH_PATH.write_text(hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest() + "\n")
plan_hash = hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
print(f"frozen plan: {PLAN_PATH}\nSHA-256: {plan_hash}")

check(not set(split_["dev"]) & set(split_["confirm"]), f"dev ({len(split_['dev'])}) and confirm ({len(split_['confirm'])}) subjects are disjoint")
check(len(eeg_cols) == 76 and len(m0_cols) >= 30 and not any(c.endswith("_ok") for c in m0_cols + eeg_cols),
      f"plan lists {len(m0_cols)} M0 features (IMU + EMG, no flags) and {len(eeg_cols)} EEG features")
check(len(dev_two) == 6 and len(mde_) == 6, "dev evidence (two-stage results, MDE) embedded in the plan")
check(HASH_PATH.exists() and HASH_PATH.read_text().strip() == plan_hash, "plan file matches its stored SHA-256 (unchanged since the freeze)")
end_cell("CELL 16a")

# ===== CELL 16b · Confirm-run setup: frozen-plan check, primary data set, model functions (~20 s) =====
# Self-contained: needs only CELL 1 + CELL R + CELL 16a. Functions are verbatim copies of CELL 15 / 15b, except that
# the folds hold out only the given test subjects and the circular shift takes its own random generator.
import hashlib
import json
from scipy.stats import wilcoxon
from sklearn.linear_model import RidgeCV

if OUT_ROOT.name != "w400":
    raise RuntimeError("run CELL 1 and CELL R (WINDOW_DIR = 'w400') first")
PLAN_PATH, HASH_PATH = OUT_ROOT / "frozen_plan.json", OUT_ROOT / "frozen_plan.sha256"
if not (PLAN_PATH.exists() and HASH_PATH.exists()):
    raise RuntimeError("no frozen plan -> run CELL 16a first")
PLAN = json.loads(PLAN_PATH.read_text())
PLAN_HASH = hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
TARGETS, M0_COLS, EEG_REL = PLAN["targets"], PLAN["m0_columns"], PLAN["eeg_columns"]
CONFIRM, DEV = [int(s) for s in PLAN["split"]["confirm"]], [int(s) for s in PLAN["split"]["dev"]]
ALPHAS, ALPHAS_E = np.logspace(-2, 4, 25), np.logspace(0, 6, 25)
Z_CLIP, Y_OUTLIER, MIN_SHIFT, N_BOOT_C, SEED_C = 5.0, 4.0, 3, 4000, 20261009
SESOI = float(PLAN["equivalence"]["sesoi_delta_r2"])

feat = pd.read_parquet(OUT_ROOT / "stride_features.parquet")
EEG_BB = [c for c in feat.columns if c.startswith("eeg_bb_")]
NUM_COLS = TARGETS + M0_COLS + EEG_REL + EEG_BB
X_COLS = M0_COLS + EEG_REL + EEG_BB
prim = feat[(feat.tier == "strict") & ~feat.odd_mapping.astype(bool) & feat.eeg_ok & feat.imu_ok].copy()
prim["trial"] = prim.subject.astype(str) + "_" + prim.speed.astype(str) + "_" + prim.leg
grp = prim.groupby("trial")[NUM_COLS]
prim[NUM_COLS] = (prim[NUM_COLS] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)   # within-trial z-scores
prim[X_COLS] = prim[X_COLS].clip(-Z_CLIP, Z_CLIP)
prim = prim.sort_values(["trial", "hs_eeg"]).reset_index(drop=True)                                    # time order within trial


def r2_vs_trial_mean(y: np.ndarray, yhat: np.ndarray) -> float:
    """y is a within-trial z-score, so the trial mean (= 0) is the baseline prediction."""
    return float(1 - np.sum((y - yhat) ** 2) / np.sum(y ** 2))


def m0_folds(d: pd.DataFrame, target: str, test_subjects: list) -> dict:
    """Mechanics model fitted once per held-out test subject (training = every other subject in d)."""
    y, X0 = d[target].to_numpy(float), d[M0_COLS].fillna(0.0).to_numpy(float)
    folds = []
    for s in sorted(set(test_subjects) & set(d.subject.unique())):
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
        B = f["proj"] @ E[~te]
        m1 = RidgeCV(alphas=ALPHAS_E).fit(E[~te] - f["x0tr"] @ B, f["res"])
        r0[f["s"]] = r2_vs_trial_mean(c["y"][te], f["p0"])
        r1[f["s"]] = r2_vs_trial_mean(c["y"][te], f["p0"] + m1.predict(E[te] - f["x0te"] @ B))
    return r0, r1


def within_trial_roll(trial_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row index that rotates each trial's (time-ordered) rows by a random offset >= MIN_SHIFT."""
    out = np.arange(len(trial_ids))
    for t in np.unique(trial_ids):
        pos = np.flatnonzero(trial_ids == t)
        if len(pos) > 2 * MIN_SHIFT:
            out[pos] = np.roll(pos, int(rng.integers(MIN_SHIFT, len(pos) - MIN_SHIFT + 1)))
    return out


def differenced(d: pd.DataFrame) -> pd.DataFrame:
    """Stride n minus stride n-1 of the same trial (consecutive strides only), re-z-scored within trial, features clipped."""
    d = d.sort_values(["trial", "hs_eeg"]).reset_index(drop=True)
    gap = d.groupby("trial", sort=False).hs_eeg.diff()
    consecutive = (gap <= 1.5 * gap.groupby(d.trial).transform("median")).to_numpy()
    out = d.copy()
    out[NUM_COLS] = d.groupby("trial", sort=False)[NUM_COLS].diff()
    out = out[consecutive].reset_index(drop=True)
    g = out.groupby("trial")[NUM_COLS]
    out[NUM_COLS] = (out[NUM_COLS] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    out[X_COLS] = out[X_COLS].clip(-Z_CLIP, Z_CLIP)
    return out


print(f"frozen plan of {PLAN['frozen']} | SHA-256 {PLAN_HASH[:16]}... | window {PLAN['window_s']} s | SESOI {SESOI}")
print(f"primary set: {len(prim)} strides, {prim.subject.nunique()} subjects ({len(CONFIRM)} confirm held out one by one, "
      f"trained on all other subjects)")
check(HASH_PATH.read_text().strip() == PLAN_HASH, "frozen plan unchanged since CELL 16a (SHA-256 matches)")
check(PLAN["window_s"] == [-0.4, -0.1] and SESOI == 0.01, "plan window [-0.4, -0.1] s and SESOI 0.01 as decided")
check(all(c in feat.columns for c in M0_COLS + EEG_REL), f"all {len(M0_COLS)} M0 and {len(EEG_REL)} EEG plan columns present")
check(set(CONFIRM) <= set(prim.subject.unique()), f"all {len(CONFIRM)} confirm subjects in the primary set")
check(not set(CONFIRM) & set(DEV), "confirm and dev subjects are disjoint")
end_cell("CELL 16b")

# ===== CELL 16c · CONFIRMATORY TEST (primary, trial-centered): 33 held-out confirm subjects (~1-3 min) =====
# Exactly the frozen plan: two-stage model, each confirm subject held out once, one-sided Wilcoxon (gain > 0) and
# equivalence (gain < 0.01), both Holm-corrected over the 3 targets. Re-running gives the same numbers; if a saved
# result differs, the cell stops (something changed after the first run).
if "PLAN_HASH" not in globals():
    raise RuntimeError("run CELL 16b first")
RES_PATH, SUBJ_PATH = OUT_ROOT / "confirm_primary.csv", OUT_ROOT / "confirm_subject_deltas.csv"


def holm(p: list) -> list:
    """Holm step-down adjusted p-values."""
    order, m, run, adj = np.argsort(p), len(p), 0.0, np.empty(len(p))
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj.tolist()


rng_c = np.random.default_rng(SEED_C)
rows, subj_c = [], []
t0_16 = time.time()
for t in TARGETS:
    d = prim[prim[t].abs() <= Y_OUTLIER].reset_index(drop=True)
    r0, r1 = stage_r2(m0_folds(d, t, CONFIRM), d[EEG_REL].fillna(0.0).to_numpy(float))
    subs = sorted(r0)
    dR, r0v = np.array([r1[s] - r0[s] for s in subs]), np.array([r0[s] for s in subs])
    boot = np.median(rng_c.choice(dR, size=(N_BOOT_C, len(dR))), axis=1)
    rows.append({"target": t, "n_subjects": len(subs), "n_strides": int(d.subject.isin(subs).sum()),
                 "M0_R2_median": round(float(np.median(r0v)), 4), "M0_R2_IQR": f"{np.percentile(r0v, 25):.3f}..{np.percentile(r0v, 75):.3f}",
                 "dR2_median": round(float(np.median(dR)), 4),
                 "dR2_CI95": f"{np.percentile(boot, 2.5):.4f}..{np.percentile(boot, 97.5):.4f}",
                 "share_subjects_gain": round(float((dR > 0).mean()), 2),
                 "p_wilcoxon": float(wilcoxon(dR, alternative="greater").pvalue),
                 "p_equiv": float(wilcoxon(dR - SESOI, alternative="less").pvalue)})
    subj_c += [{"target": t, "subject": s, "R2_M0": r0[s], "dR2": r1[s] - r0[s]} for s in subs]
    log(f"{t}: delta R2 median {np.median(dR):+.4f} over {len(subs)} confirm subjects ({time.time() - t0_16:.0f}s)")

res_c = pd.DataFrame(rows)
res_c["p_holm"], res_c["p_equiv_holm"] = holm(res_c.p_wilcoxon.tolist()), holm(res_c.p_equiv.tolist())
res_c["decision"] = np.where(res_c.p_holm < 0.05, "EEG gain",
                             np.where(res_c.p_equiv_holm < 0.05, f"no gain >= {SESOI}", "inconclusive"))
res_c["plan_sha256"] = PLAN_HASH[:16]
same_as_saved, first_run = True, not RES_PATH.exists()
if not first_run:
    saved = pd.read_csv(RES_PATH)
    same_as_saved = bool(np.allclose(saved[["dR2_median", "p_wilcoxon", "p_equiv"]].to_numpy(float),
                                     res_c[["dR2_median", "p_wilcoxon", "p_equiv"]].to_numpy(float), atol=1e-6))
else:
    res_c.to_csv(RES_PATH, index=False)
    pd.DataFrame(subj_c).to_csv(SUBJ_PATH, index=False)

sc = pd.DataFrame(subj_c)
fig, ax = plt.subplots(1, len(TARGETS), figsize=(15, 4))
for ti, t in enumerate(TARGETS):
    v = sc[sc.target == t].dR2.to_numpy()
    r_ = res_c.set_index("target").loc[t]
    lo, hi = (float(x) for x in r_.dR2_CI95.split(".."))
    ax[ti].scatter(rng_c.normal(0, 0.04, len(v)), v, s=18, alpha=0.8)
    ax[ti].errorbar([0.3], [r_.dR2_median], yerr=[[r_.dR2_median - lo], [hi - r_.dR2_median]], fmt="o", color="k", capsize=4,
                    label="median, 95% CI")
    ax[ti].axhline(0, color="grey", lw=0.6)
    ax[ti].axhline(SESOI, color="tab:red", ls="--", lw=0.8, label=f"equivalence bound {SESOI}")
    ax[ti].set(xlim=(-0.4, 0.6), xticks=[], ylabel="delta R2 per confirm subject",
               title=f"{t}\n{r_.decision} (p_holm {r_.p_holm:.3g}, p_equiv_holm {r_.p_equiv_holm:.3g})")
ax[0].legend(fontsize=7)
fig.suptitle(f"CONFIRMATORY: EEG gain beyond mechanics, {len(CONFIRM)} held-out subjects, window [-400, -100] ms")
fig.tight_layout()
fig.savefig(OUT_ROOT / "cell16c_confirm.png", dpi=130)
plt.show()
print(res_c.drop(columns="plan_sha256").to_string(index=False, float_format=lambda v: f"{v:.3g}"))

check(bool((res_c.n_subjects == len(CONFIRM)).all()), f"every target scored on all {len(CONFIRM)} confirm subjects")
check(not set(sc.subject) & set(DEV), "no dev subject among the held-out subjects")
check(HASH_PATH.read_text().strip() == PLAN_HASH, "frozen plan still unchanged (SHA-256)")
check(same_as_saved, f"first run saved to {RES_PATH.name}" if first_run else "result identical to the first saved run")
end_cell("CELL 16c")

# ===== CELL 16d · Secondary + sensitivity analyses on the confirm subjects (frozen-plan list) (~4-6 min) =====
# Same frozen pipeline (two-stage model, 33 confirm subjects held out one by one, Wilcoxon gain > 0 and equivalence
# gain < 0.01, Holm over the 3 targets within each analysis). These are SECONDARY: they qualify, never replace, CELL 16c.
#  A primary, re-derived (must reproduce CELL 16c)    B stride-to-stride changes
#  C window [-600, -100] ms (other foot's impact inside) D relaxed tier + odd-mapping trials added
#  + per leg and per speed for the primary analysis (descriptive, no correction)
if "holm" not in globals() or not (OUT_ROOT / "confirm_primary.csv").exists():
    raise RuntimeError("run CELL 16b and CELL 16c first")


def prepare(feat_df: pd.DataFrame, tiers: tuple = ("strict",), include_odd: bool = False) -> pd.DataFrame:
    """Frozen preprocessing (as CELL 16b) for any stride table and trial set."""
    keep = feat_df.tier.isin(tiers) & feat_df.eeg_ok & feat_df.imu_ok
    if not include_odd:
        keep &= ~feat_df.odd_mapping.astype(bool)
    p = feat_df[keep].copy()
    p["trial"] = p.subject.astype(str) + "_" + p.speed.astype(str) + "_" + p.leg
    g = p.groupby("trial")[NUM_COLS]
    p[NUM_COLS] = (p[NUM_COLS] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    p[X_COLS] = p[X_COLS].clip(-Z_CLIP, Z_CLIP)
    return p.sort_values(["trial", "hs_eeg"]).reset_index(drop=True)


def stage_pred(c: dict, E: np.ndarray) -> tuple:
    """Held-out predictions of M0 and of M0 + EEG stage for every test row (NaN for training-only rows)."""
    p0, p1 = np.full(len(c["y"]), np.nan), np.full(len(c["y"]), np.nan)
    for f in c["folds"]:
        te = f["te"]
        B = f["proj"] @ E[~te]
        m1 = RidgeCV(alphas=ALPHAS_E).fit(E[~te] - f["x0tr"] @ B, f["res"])
        p0[te], p1[te] = f["p0"], f["p0"] + m1.predict(E[te] - f["x0te"] @ B)
    return p0, p1


def summary(dR: np.ndarray, rng: np.random.Generator) -> dict:
    boot = np.median(rng.choice(dR, size=(N_BOOT_C, len(dR))), axis=1)
    return {"n_subjects": len(dR), "dR2_median": round(float(np.median(dR)), 4),
            "ci_lo": round(float(np.percentile(boot, 2.5)), 4), "ci_hi": round(float(np.percentile(boot, 97.5)), 4),
            "share_gain": round(float((dR > 0).mean()), 2), "p_wilcoxon": float(wilcoxon(dR, alternative="greater").pvalue),
            "p_equiv": float(wilcoxon(dR - SESOI, alternative="less").pvalue)}


def run_analysis(d_all: pd.DataFrame, label: str, rng: np.random.Generator) -> tuple:
    rows_, held_ = [], {}
    for t in TARGETS:
        d = d_all[d_all[t].abs() <= Y_OUTLIER].reset_index(drop=True)
        c = m0_folds(d, t, CONFIRM)
        p0, p1 = stage_pred(c, d[EEG_REL].fillna(0.0).to_numpy(float))
        m_ = ~np.isnan(p0)
        h = pd.DataFrame({"subject": d.subject.to_numpy()[m_], "leg": d.leg.to_numpy()[m_], "speed": d.speed.to_numpy()[m_],
                          "y": c["y"][m_], "p0": p0[m_], "p1": p1[m_]})
        dR = np.array([r2_vs_trial_mean(g.y.to_numpy(), g.p1.to_numpy()) - r2_vs_trial_mean(g.y.to_numpy(), g.p0.to_numpy())
                       for _, g in h.groupby("subject")])
        rows_.append({"analysis": label, "target": t, **summary(dR, rng)})
        held_[t] = h
    out = pd.DataFrame(rows_)
    out["p_holm"], out["p_equiv_holm"] = holm(out.p_wilcoxon.tolist()), holm(out.p_equiv.tolist())
    out["decision"] = np.where(out.p_holm < 0.05, "EEG gain", np.where(out.p_equiv_holm < 0.05, f"no gain >= {SESOI}", "inconclusive"))
    log(f"{label}: " + ", ".join(f"{r.target} {r.dR2_median:+.4f} ({r.decision})" for r in out.itertuples()))
    return out, held_


rng_d = np.random.default_rng(SEED_C + 1)
t0_d = time.time()
res_A, held_A = run_analysis(prim, "A primary (re-derived)", rng_d)
res_B, _ = run_analysis(differenced(prim), "B stride-to-stride", rng_d)
res_C, _ = run_analysis(prepare(pd.read_parquet(OUT_ROOT.parent / "stride_features.parquet")), "C window [-600, -100] ms", rng_d)
res_D, _ = run_analysis(prepare(feat, tiers=("strict", "relaxed"), include_odd=True), "D + relaxed tier + odd-mapping trials", rng_d)
sec = pd.concat([res_A, res_B, res_C, res_D], ignore_index=True)
sec["plan_sha256"] = PLAN_HASH[:16]
sec.to_csv(OUT_ROOT / "confirm_secondary.csv", index=False)

desc = []
for t, h in held_A.items():
    for key in ("leg", "speed"):
        for lev, hk in h.groupby(key):
            dR = np.array([r2_vs_trial_mean(g.y.to_numpy(), g.p1.to_numpy()) - r2_vs_trial_mean(g.y.to_numpy(), g.p0.to_numpy())
                           for _, g in hk.groupby("subject")])
            desc.append({"target": t, "by": key, "level": lev, "n_subjects": len(dR), "dR2_median": round(float(np.median(dR)), 4),
                         "share_gain": round(float((dR > 0).mean()), 2),
                         "p_wilcoxon_uncorrected": round(float(wilcoxon(dR, alternative="greater").pvalue), 4)})
desc = pd.DataFrame(desc)
desc.to_csv(OUT_ROOT / "confirm_descriptive_leg_speed.csv", index=False)
log(f"all secondary analyses done ({time.time() - t0_d:.0f}s)")

fig, ax = plt.subplots(1, len(TARGETS), figsize=(16, 3.8), sharey=True)
labels = sec.analysis.unique().tolist()
for ti, t in enumerate(TARGETS):
    s_ = sec[sec.target == t].set_index("analysis").loc[labels]
    yy = np.arange(len(labels))[::-1]
    ax[ti].errorbar(s_.dR2_median, yy, xerr=[s_.dR2_median - s_.ci_lo, s_.ci_hi - s_.dR2_median], fmt="o", color="k", capsize=3)
    ax[ti].axvline(0, color="grey", lw=0.6)
    ax[ti].axvline(SESOI, color="tab:red", ls="--", lw=0.8)
    ax[ti].set(yticks=yy, yticklabels=labels, xlabel="delta R2, median and 95% CI over confirm subjects", title=t)
fig.suptitle("Confirm subjects: EEG gain beyond mechanics across secondary and sensitivity analyses (red = equivalence bound 0.01)")
fig.tight_layout()
fig.savefig(OUT_ROOT / "cell16d_forest.png", dpi=130)
plt.show()
print(sec.drop(columns="plan_sha256").to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print("\nper leg / per speed (primary analysis, descriptive):")
print(desc.pivot_table(index=["by", "level"], columns="target", values="dR2_median").to_string(float_format=lambda v: f"{v:+.4f}"))

prim_saved = pd.read_csv(OUT_ROOT / "confirm_primary.csv").set_index("target").dR2_median
check(np.allclose(res_A.set_index("target").dR2_median.loc[prim_saved.index], prim_saved, atol=1e-4),
      "analysis A reproduces the CELL 16c medians (same pipeline)")
check(bool((sec.n_subjects == len(CONFIRM)).all()), f"every analysis scored on all {len(CONFIRM)} confirm subjects")
check(HASH_PATH.read_text().strip() == PLAN_HASH, "frozen plan still unchanged (SHA-256)")
end_cell("CELL 16d")

# ===== CELL 16e · Circular-shift null on the confirm subjects: is there ANY EEG-landing dependence? (~30-40 min) =====
# Frozen-plan secondary: observed median delta R2 vs 100 rotations (each trial's EEG rotated by >= 3 strides, which keeps
# its own autocorrelation and breaks only the stride alignment). Run for the primary window [-400, -100] ms
# (preregistered) and, EXPLORATORY, for [-600, -100] ms, where the other foot's heel strike falls inside the window.
# Speed: ridge_loo() is the same estimator as RidgeCV (tested equal to 1e-15), about 8x faster; the cell checks that it
# reproduces CELL 16c. Results are saved after every target: after a lost runtime run CELL 1 -> R -> 16a -> 16b -> 16c
# -> 16e again and it continues where it stopped (the primary window is done first).
if "holm" not in globals() or not (OUT_ROOT / "confirm_primary.csv").exists():
    raise RuntimeError("run CELL 16b and CELL 16c first")
N_ROT, CIRC_PATH = 100, OUT_ROOT / "confirm_circular.csv"


def ridge_loo(X: np.ndarray, y: np.ndarray, alphas: np.ndarray) -> tuple:
    """Same estimator as RidgeCV(alphas) with its default efficient leave-one-out choice of alpha (intercept unpenalised)."""
    xm, ym = X.mean(axis=0), y.mean()
    Xc, yc = X - xm, y - ym
    lam, V = np.linalg.eigh(Xc.T @ Xc)
    Z = Xc @ V
    inv = 1.0 / (np.clip(lam, 0, None)[:, None] + alphas[None, :])
    c = Z.T @ yc
    loo = ((yc[:, None] - Z @ (c[:, None] * inv)) / (1 - (Z ** 2) @ inv - 1.0 / len(y))) ** 2
    k = int(np.argmin(loo.mean(axis=0)))
    coef = V @ (c * inv[:, k])
    return coef, ym - xm @ coef


def median_gain(c: dict, E: np.ndarray) -> float:
    """Median over held-out subjects of delta R2 for the EEG stage (two-stage model, fast ridge)."""
    dR = []
    for f in c["folds"]:
        te = f["te"]
        B = f["proj"] @ E[~te]
        coef, b0 = ridge_loo(E[~te] - f["x0tr"] @ B, f["res"], ALPHAS_E)
        y = c["y"][te]
        dR.append(r2_vs_trial_mean(y, f["p0"] + (E[te] - f["x0te"] @ B) @ coef + b0) - r2_vs_trial_mean(y, f["p0"]))
    return float(np.median(dR))


def prepare_w(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Frozen preprocessing (as CELL 16b) for another stride table."""
    p = feat_df[(feat_df.tier == "strict") & ~feat_df.odd_mapping.astype(bool) & feat_df.eeg_ok & feat_df.imu_ok].copy()
    p["trial"] = p.subject.astype(str) + "_" + p.speed.astype(str) + "_" + p.leg
    g = p.groupby("trial")[NUM_COLS]
    p[NUM_COLS] = (p[NUM_COLS] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    p[X_COLS] = p[X_COLS].clip(-Z_CLIP, Z_CLIP)
    return p.sort_values(["trial", "hs_eeg"]).reset_index(drop=True)


done = pd.read_csv(CIRC_PATH) if CIRC_PATH.exists() else pd.DataFrame(columns=["window", "target"])
windows = [("[-400, -100] ms (primary)", prim),
           ("[-600, -100] ms (exploratory)", prepare_w(pd.read_parquet(OUT_ROOT.parent / "stride_features.parquet")))]
t0_e = time.time()
for wi, (wname, data) in enumerate(windows):
    for ti, t in enumerate(TARGETS):
        if ((done.window == wname) & (done.target == t)).any():
            log(f"{wname} / {t}: already done -> skipped")
            continue
        rng_e = np.random.default_rng([SEED_C + 2, wi, ti])      # own seed per window x target: a resumed run is identical
        d = data[data[t].abs() <= Y_OUTLIER].reset_index(drop=True)
        c = m0_folds(d, t, CONFIRM)
        E, tid = d[EEG_REL].fillna(0.0).to_numpy(float), d.trial.to_numpy()
        obs = median_gain(c, E)
        null = np.empty(N_ROT)
        for k in range(N_ROT):
            null[k] = median_gain(c, E[within_trial_roll(tid, rng_e)])
            if (k + 1) % 25 == 0:
                log(f"{wname} / {t}: {k + 1}/{N_ROT} rotations ({time.time() - t0_e:.0f}s)")
        row = {"window": wname, "target": t, "dR2_median_obs": round(obs, 5), "null_median": round(float(np.median(null)), 5),
               "null_p95": round(float(np.percentile(null, 95)), 5),
               "p_circular": (1 + int((null >= obs).sum())) / (N_ROT + 1), "plan_sha256": PLAN_HASH[:16]}
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        done.to_csv(CIRC_PATH, index=False)
        np.save(OUT_ROOT / f"circular_null_{'w400' if 'primary' in wname else 'w600'}_{t}.npy", null)

res_e = done.copy()
res_e["p_holm_within_window"] = res_e.groupby("window").p_circular.transform(lambda p: pd.Series(holm(p.tolist()), index=p.index))
res_e.to_csv(CIRC_PATH, index=False)
fig, ax = plt.subplots(2, len(TARGETS), figsize=(15, 6.5))
for wi, (wname, _) in enumerate(windows):
    for ti, t in enumerate(TARGETS):
        f_ = OUT_ROOT / f"circular_null_{'w400' if 'primary' in wname else 'w600'}_{t}.npy"
        r_ = res_e[(res_e.window == wname) & (res_e.target == t)].iloc[0]
        ax[wi, ti].hist(np.load(f_), bins=20, color="lightgrey", edgecolor="grey")
        ax[wi, ti].axvline(r_.dR2_median_obs, color="tab:red", lw=1.5, label="observed")
        ax[wi, ti].set(title=f"{t} | {wname}\np_circular {r_.p_circular:.3f}", xlabel="median delta R2 (rotated EEG)")
ax[0, 0].legend(fontsize=7)
fig.suptitle(f"Confirm subjects: observed EEG gain vs {N_ROT} circular shifts of the EEG within each trial")
fig.tight_layout()
fig.savefig(OUT_ROOT / "cell16e_circular.png", dpi=130)
plt.show()
print(res_e.drop(columns="plan_sha256").to_string(index=False, float_format=lambda v: f"{v:.4g}"))

prim_saved = pd.read_csv(OUT_ROOT / "confirm_primary.csv").set_index("target").dR2_median
obs_w400 = res_e[res_e.window.str.contains("primary")].set_index("target").dR2_median_obs
check(np.allclose(obs_w400.loc[prim_saved.index], prim_saved, atol=1e-4), "fast ridge reproduces the CELL 16c medians")
check(len(res_e) == 2 * len(TARGETS), f"{len(res_e)}/6 window x target null distributions finished")
check(bool((res_e.null_median <= 0.001).all()), "EEG without stride alignment does not raise delta R2 (null medians <= 0.001)", critical=False)
check(HASH_PATH.read_text().strip() == PLAN_HASH, "frozen plan still unchanged (SHA-256)")
end_cell("CELL 16e")

# ===== CELL 16f · Descriptive (broadband model, artifact dose-response) + one zip with every result file (~3-5 min) =====
# Frozen-plan descriptive items, no claims attached: (a) the two-stage model with broadband EEG power (signal + artifact)
# instead of relative band power; (b) Spearman correlation between each confirm subject's EEG artifact index and its
# primary delta R2. Then all result files (frozen plan, dev + confirm tables, figures, null distributions, G1 outputs)
# are zipped to Drive and offered for download, for the analysis bundle and the manuscript.
import zipfile
from scipy.stats import spearmanr

if "holm" not in globals() or not (OUT_ROOT / "confirm_subject_deltas.csv").exists():
    raise RuntimeError("run CELL 16b and CELL 16c first")
rng_f = np.random.default_rng(SEED_C + 3)
desc_rows = []
for t in TARGETS:
    d = prim[prim[t].abs() <= Y_OUTLIER].reset_index(drop=True)
    r0, r1 = stage_r2(m0_folds(d, t, CONFIRM), d[EEG_BB].fillna(0.0).to_numpy(float))
    dR = np.array([r1[s] - r0[s] for s in sorted(r0)])
    boot = np.median(rng_f.choice(dR, size=(N_BOOT_C, len(dR))), axis=1)
    desc_rows.append({"item": "broadband-power model", "target": t, "n_subjects": len(dR), "dR2_median": round(float(np.median(dR)), 4),
                      "ci_lo": round(float(np.percentile(boot, 2.5)), 4), "ci_hi": round(float(np.percentile(boot, 97.5)), 4),
                      "p_wilcoxon_uncorrected": float(wilcoxon(dR, alternative="greater").pvalue), "rho": np.nan, "p_rho": np.nan})
    log(f"broadband model / {t}: median delta R2 {np.median(dR):+.4f}")

art_idx = pd.read_csv(OUT_ROOT / "g1_trials.csv").assign(a=lambda x: x.artifact_a.abs()).groupby("subject").a.mean()
sd_ = pd.read_csv(OUT_ROOT / "confirm_subject_deltas.csv")
sd_["artifact_index"] = sd_.subject.map(art_idx)
fig, ax = plt.subplots(1, len(TARGETS), figsize=(15, 3.8))
for ti, t in enumerate(TARGETS):
    s_ = sd_[(sd_.target == t) & sd_.artifact_index.notna()]
    rho, p_rho = spearmanr(s_.artifact_index, s_.dR2)
    desc_rows.append({"item": "dose-response: artifact index vs primary delta R2", "target": t, "n_subjects": len(s_),
                      "dR2_median": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p_wilcoxon_uncorrected": np.nan,
                      "rho": round(float(rho), 3), "p_rho": float(p_rho)})
    ax[ti].scatter(s_.artifact_index, s_.dR2, s=18)
    ax[ti].axhline(0, color="grey", lw=0.6)
    ax[ti].set(xlabel="EEG artifact index (|r| EEG envelope vs head acceleration)", ylabel="primary delta R2", title=f"{t}: Spearman rho {rho:+.2f} (p {p_rho:.2f})")
fig.suptitle("Confirm subjects: does the EEG gain grow with motion artifact? (descriptive)")
fig.tight_layout()
fig.savefig(OUT_ROOT / "cell16f_dose_response.png", dpi=130)
plt.show()
desc_c = pd.DataFrame(desc_rows)
desc_c["plan_sha256"] = PLAN_HASH[:16]
desc_c.to_csv(OUT_ROOT / "confirm_descriptive.csv", index=False)
print(desc_c.drop(columns="plan_sha256").to_string(index=False, float_format=lambda v: f"{v:.3g}"))

BASE = OUT_ROOT.parent
ZIP_PATH = OUT_ROOT / "paper_results_w400.zip"
picks = [(OUT_ROOT, p) for pat in ("frozen_plan.*", "confirm_*.csv", "dev_*.csv", "dev_confirm_split*.json", "cell1[456]*.png",
                                   "circular_null_*.npy") for p in sorted(OUT_ROOT.glob(pat))]
picks += [(BASE, p) for pat in ("dev_*.csv", "g1_*.csv", "g1_*.npz", "cell1[1-5]*.png", "trials_qc_v3.csv", "manifest.csv")
          for p in sorted(BASE.glob(pat))]
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, p in picks:
        zf.write(p, arcname=("w400/" if root == OUT_ROOT else "window600_and_methods/") + p.name)
log(f"{len(picks)} files -> {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1e6:.1f} MB)")
try:
    from google.colab import files
    files.download(str(ZIP_PATH))
except ImportError:
    log("not in Colab -> download skipped (the zip is in the folder above)")

with zipfile.ZipFile(ZIP_PATH) as zf:
    names = zf.namelist()
check(len(desc_c) == 2 * len(TARGETS), "broadband model and dose-response computed for all 3 targets")
check(all(f"w400/{f}" in names for f in ("frozen_plan.json", "confirm_primary.csv", "confirm_secondary.csv", "confirm_circular.csv")),
      "zip holds the frozen plan and the primary, secondary and circular results")
check(HASH_PATH.read_text().strip() == PLAN_HASH, "frozen plan still unchanged (SHA-256)")
end_cell("CELL 16f")
