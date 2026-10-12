"""Analysis bundle: recompute the confirmatory statistics from the per-subject files and draw the paper figures.

Input : ../results_w400/ (unzipped paper_results_w400.zip exported by Colab CELL 16f)
Output: figures/*.png|pdf and stats_recomputed.csv next to this script.
Run   : python make_figures.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp, wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results_w400" / "w400"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
TARGETS = ["T1_peak_fz", "T2_loading_rate", "T3_knee_flex_early"]
LABEL = {"T1_peak_fz": "Peak vGRF (T1)", "T2_loading_rate": "Loading rate (T2)", "T3_knee_flex_early": "Early-stance knee flexion (T3)"}
SHORT = {"T1_peak_fz": "Peak vGRF", "T2_loading_rate": "Loading rate", "T3_knee_flex_early": "Knee flexion"}
SESOI, SEED, N_BOOT = 0.01, 20261009, 4000
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
W2 = 6.85                                           # double-column width (in)
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlecolor": INK,
                     "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})


def holm(p: list) -> list:
    order, m, run, adj = np.argsort(p), len(p), 0.0, np.empty(len(p))
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj.tolist()


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)


# ---------- 1. recompute the confirmatory statistics ----------
sd = pd.read_csv(RES / "confirm_subject_deltas.csv")
saved = pd.read_csv(RES / "confirm_primary.csv").set_index("target")
rng = np.random.default_rng(SEED)
rows = []
for t in TARGETS:
    d = sd[sd.target == t].sort_values("subject").dR2.to_numpy()
    boot = np.median(rng.choice(d, size=(N_BOOT, len(d))), axis=1)          # same draw order as CELL 16c
    rows.append({"target": t, "n": len(d), "median": np.median(d), "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
                 "mean": d.mean(), "sd": d.std(ddof=1), "share_gain": (d > 0).mean(),
                 "p_wilcoxon": wilcoxon(d, alternative="greater").pvalue, "p_equiv": wilcoxon(d - SESOI, alternative="less").pvalue,
                 "p_sign_gain": binomtest(int((d > 0).sum()), int((d != 0).sum()), alternative="greater").pvalue,
                 "p_t_gain": ttest_1samp(d, 0, alternative="greater").pvalue,
                 "p_t_equiv": ttest_1samp(d, SESOI, alternative="less").pvalue,
                 "m0_median": np.median(sd[sd.target == t].R2_M0), "max_subject_gain": d.max()})
st = pd.DataFrame(rows)
st["p_holm"], st["p_equiv_holm"] = holm(st.p_wilcoxon.tolist()), holm(st.p_equiv.tolist())
st.to_csv(HERE / "stats_recomputed.csv", index=False)
cmp_ = pd.DataFrame({"median_colab": saved.dR2_median, "median_local": st.set_index("target")["median"].round(4),
                     "p_wil_colab": saved.p_wilcoxon, "p_wil_local": st.set_index("target").p_wilcoxon,
                     "p_eq_colab": saved.p_equiv, "p_eq_local": st.set_index("target").p_equiv,
                     "ci_colab": saved.dR2_CI95,
                     "ci_local": [f"{a:.4f}..{b:.4f}" for a, b in zip(st.ci_lo, st.ci_hi)]})
print(cmp_.to_string())

# ---------- 2. Figure: primary result, per-subject delta R2 ----------
fig, ax = plt.subplots(1, 3, figsize=(W2, 2.5), sharey=True)
jit = np.random.default_rng(1)
for i, t in enumerate(TARGETS):
    d = sd[sd.target == t].dR2.to_numpy()
    r = st.set_index("target").loc[t]
    ax[i].axhline(0, color=BASE, lw=0.8, zorder=1)
    ax[i].axhline(SESOI, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax[i].scatter(jit.uniform(-0.18, 0.18, len(d)), d, s=12, color=BLUE, alpha=0.75, lw=0, zorder=2)
    ax[i].errorbar([0.45], [r["median"]], yerr=[[r["median"] - r.ci_lo], [r.ci_hi - r["median"]]], fmt="o", ms=4,
                   color=INK, ecolor=INK, elinewidth=1, capsize=3, zorder=3)
    ax[i].set(xlim=(-0.4, 0.75), xticks=[])
    ax[i].set_title(f"{LABEL[t]}\nmedian {r['median']:+.4f}\n95% CI {r.ci_lo:+.4f} to {r.ci_hi:+.4f}", fontsize=7.5)
    ax[i].text(0.74, SESOI + 0.0004, f"equivalence p = {r.p_equiv_holm:.1e}", ha="right", va="bottom", fontsize=6.5, color=INK2)
ax[0].set_ylabel("ΔR² (EEG stage − mechanics only)\nper held-out confirm subject")
fig.text(0.5, -0.02, "Dots: confirm subjects (N = 33). Black: median with bootstrap 95% CI. Dashed line: equivalence bound ΔR² = 0.01 "
         "(p values Holm-corrected over the three targets).", ha="center", fontsize=6.5, color=INK2)
fig.tight_layout(w_pad=1.0)
save(fig, "fig2_primary_confirm")

# ---------- 3. Figure: forest plot of primary, secondary, sensitivity and descriptive analyses ----------
sec = pd.read_csv(RES / "confirm_secondary.csv")
desc = pd.read_csv(RES / "confirm_descriptive.csv")
bb = desc[desc["item"] == "broadband-power model"].assign(analysis="E broadband")
order = [("A primary (re-derived)", "Primary: trial-centred, −400 to −100 ms"), ("B stride-to-stride", "Stride-to-stride changes"),
         ("C window [-600, -100] ms", "Window −600 to −100 ms"), ("D + relaxed tier + odd-mapping trials", "+ relaxed / odd-mapping trials"),
         ("E broadband", "Broadband power (descriptive)")]
prim_row = st.assign(analysis="A primary (re-derived)").rename(columns={"median": "dR2_median"})   # CELL 16c values (same as Fig. 2)
allr = pd.concat([prim_row[["analysis", "target", "dR2_median", "ci_lo", "ci_hi"]],
                  sec[sec.analysis != "A primary (re-derived)"][["analysis", "target", "dR2_median", "ci_lo", "ci_hi"]],
                  bb[["analysis", "target", "dR2_median", "ci_lo", "ci_hi"]]])
fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3), sharey=True)
yy = np.arange(len(order))[::-1]
for i, t in enumerate(TARGETS):
    s = allr[allr.target == t].set_index("analysis").loc[[a for a, _ in order]]
    ax[i].axvline(0, color=BASE, lw=0.8)
    ax[i].axvline(SESOI, color=INK2, lw=0.8, ls=(0, (4, 3)))
    cols = [INK] * 4 + [MUTED]
    for k, (a, _) in enumerate(order):
        ax[i].errorbar(s.loc[a, "dR2_median"], yy[k], xerr=[[s.loc[a, "dR2_median"] - s.loc[a, "ci_lo"]], [s.loc[a, "ci_hi"] - s.loc[a, "dR2_median"]]],
                       fmt="o", ms=3.5, color=cols[k], ecolor=cols[k], elinewidth=1, capsize=2.5)
    ax[i].set(title=SHORT[t], xlim=(-0.006, 0.012))
    ax[i].xaxis.grid(True, color=GRID, lw=0.5)
ax[1].set_xlabel("ΔR², median and 95% CI over the 33 confirm subjects")
ax[0].set_yticks(yy, [lab for _, lab in order])
fig.text(0.5, -0.04, "Dashed line: equivalence bound ΔR² = 0.01. Grey: descriptive analysis (no test planned).",
         ha="center", fontsize=6.5, color=INK2)
fig.tight_layout(w_pad=0.8)
save(fig, "fig3_forest_secondary")

# ---------- 4. Figure: circular-shift null ----------
circ = pd.read_csv(RES / "confirm_circular.csv")
fig, ax = plt.subplots(2, 3, figsize=(W2, 3.6))
for wi, (tag, wname, lab) in enumerate((("w400", "[-400, -100] ms (primary)", "−400 to −100 ms (primary)"),
                                        ("w600", "[-600, -100] ms (exploratory)", "−600 to −100 ms (exploratory)"))):
    for ti, t in enumerate(TARGETS):
        null = np.load(RES / f"circular_null_{tag}_{t}.npy")
        r = circ[(circ.window == wname) & (circ.target == t)].iloc[0]
        a = ax[wi, ti]
        a.hist(1e3 * null, bins=20, color=BASE, edgecolor="white", lw=0.5)
        a.axvline(1e3 * r.dR2_median_obs, color=BLUE, lw=1.6)
        a.set_title(f"{SHORT[t]}: p = {r.p_circular:.2f}", fontsize=7.5)
        if ti == 0:
            a.set_ylabel(f"{lab}\nrotations")
        if wi == 1:
            a.set_xlabel("median ΔR² (×10⁻³), rotated EEG")
ax[0, 0].text(0.98, 0.95, "observed", transform=ax[0, 0].transAxes, ha="right", va="top", color=BLUE, fontsize=6.5)
fig.tight_layout(h_pad=1.0)
save(fig, "fig4_circular_null")

# ---------- 5. Figure: power (dev subjects, planted effects) ----------
pw = pd.read_csv(RES / "dev_power.csv")
fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3), sharey=True)
for i, t in enumerate(TARGETS):
    for var, col, lab in (("trial-centered", BLUE, "trial-centred (primary)"), ("stride-to-stride", ORANGE, "stride-to-stride")):
        s = pw[(pw.variant == var) & (pw.target == t)].sort_values("q_planted")
        ax[i].plot(s.q_planted, s.power_N_confirm, "-o", color=col, lw=2, ms=4, label=lab)
    ax[i].axhline(0.8, color=BASE, lw=0.8)
    ax[i].set(title=SHORT[t], xlabel="planted incremental R² (q)", ylim=(0, 1.03))
    ax[i].yaxis.grid(True, color=GRID, lw=0.5)
ax[0].set_ylabel(f"power, N = 33 confirm subjects\n(one-sided Wilcoxon, α = 0.05/3)")
ax[0].legend(loc="lower right", fontsize=6.5)
fig.tight_layout(w_pad=0.8)
save(fig, "figS1_power_dev")

# ---------- 6. Figure: mechanics-only baseline (M0) ----------
fig, ax = plt.subplots(figsize=(3.35, 2.4))
for i, t in enumerate(TARGETS):
    v = sd[sd.target == t].R2_M0.to_numpy()
    ax.scatter(i + jit.uniform(-0.15, 0.15, len(v)), v, s=10, color=BLUE, alpha=0.7, lw=0)
    ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=INK, lw=1.5)
    ax.text(i + 0.28, np.median(v), f"{np.median(v):.3f}", va="center", fontsize=6.5, color=INK2)
ax.axhline(0, color=BASE, lw=0.8)
ax.set_xticks(range(3), ["Peak vGRF", "Loading rate", "Knee flexion"])
ax.set_ylabel("held-out R², mechanics only (M0)")
fig.tight_layout()
save(fig, "figS2_mechanics_baseline")
print("figures:", sorted(p.name for p in FIG.glob("*.png")))
