"""Fig 1: design schematic (timing of the pre-landing window, two-stage model, development/confirmation design)."""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

FIG = Path(__file__).resolve().parent / "figures"
INK, INK2, MUTED, BASE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 7.5, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "savefig.dpi": 300, "savefig.bbox": "tight"})


def grf(t: np.ndarray, t0: float, stance: float) -> np.ndarray:
    """Schematic double-hump vertical force of one stance phase starting at t0."""
    u = (t - t0) / stance
    g = lambda c, w: np.exp(-((u - c) / w) ** 2)  # noqa: E731
    f = 1.1 * g(0.22, 0.13) + 1.1 * g(0.78, 0.13) + 0.55 * g(0.5, 0.2)
    return np.where((u > 0) & (u < 1), f * np.sin(np.pi * np.clip(u, 0, 1)) ** 0.35, 0.0)


def box(ax: plt.Axes, xy: tuple, w: float, h: float, text: str, fc: str = "white", ec: str = INK2, bold: bool = False, fs: float = 6.5) -> None:
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec, lw=0.8))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal")


def arrow(ax: plt.Axes, a: tuple, b: tuple) -> None:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, color=INK2, lw=0.8))


fig = plt.figure(figsize=(6.85, 5.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], width_ratios=[1.35, 1.0], hspace=0.35, wspace=0.08)

# A: timing
axA = fig.add_subplot(gs[0, :])
t = np.linspace(-1.0, 0.75, 2000)
stride, stance = 1.12, 0.69                                   # 1.0 m/s medians
axA.add_patch(Rectangle((-0.4, 0), 0.3, 1.45, fc=BLUE, alpha=0.15, lw=0))
axA.add_patch(Rectangle((-0.1, 0), 0.1, 1.45, fc="none", ec=MUTED, hatch="////", lw=0))
axA.add_patch(Rectangle((-0.6, 0), 0.5, 1.45, fc="none", ec=MUTED, ls=(0, (3, 2)), lw=0.8))
axA.plot(t, grf(t, -stride / 2, stance), color=MUTED, lw=1.5)
axA.plot(t, grf(t, 0.0, stance), color=INK, lw=2)
axA.axvline(0, color=INK, lw=0.8)
axA.axvline(-stride / 2, color=MUTED, lw=0.8, ls=":")
axA.add_patch(Rectangle((0, 0), 0.25 * stance, 1.45, fc=ORANGE, alpha=0.12, lw=0))
axA.text(-0.25, 1.52, "feature window\n−400 to −100 ms\n(EEG, IMU, EMG)", ha="center", va="bottom", fontsize=7, color=BLUE)
axA.text(-0.05, 1.52, "guard", ha="center", va="bottom", fontsize=7, color=MUTED)
axA.text(-0.62, 1.52, "sensitivity window\nstarts at −600 ms", ha="right", va="bottom", fontsize=7, color=MUTED)
axA.text(0.25 * stance / 2, 1.52, "targets\n(first 25%\nof stance)", ha="center", va="bottom", fontsize=7, color=ORANGE)
axA.text(0.02, 1.25, "heel strike\n(landing leg)", ha="left", va="center", fontsize=6.5, color=INK)
axA.text(-stride / 2 + 0.015, 1.25, "heel strike,\nother leg", ha="left", va="center", fontsize=6.5, color=MUTED)
axA.text(0.62, 1.1, "landing leg", fontsize=7, color=INK)
axA.text(-0.95, 1.1, "other leg", fontsize=7, color=MUTED)
axA.set(xlim=(-1.0, 0.75), ylim=(0, 2.15), yticks=[], xlabel="time relative to heel strike of the landing leg (s)")
axA.set_ylabel("vertical force\n(schematic)")
for s in ("top", "right", "left"):
    axA.spines[s].set_visible(False)
axA.set_title("A  Pre-landing window and landing targets", loc="left", fontsize=8, fontweight="bold")

# B: two-stage model
axB = fig.add_subplot(gs[1, 0])
axB.set(xlim=(-0.02, 1.04), ylim=(0, 1))
axB.axis("off")
axB.set_title("B  Two-stage incremental model", loc="left", fontsize=8, fontweight="bold")
box(axB, (0.0, 0.70), 0.30, 0.16, "IMU + EMG\n37 features (M0)")
box(axB, (0.40, 0.70), 0.24, 0.16, "Stage 1\nridge")
box(axB, (0.74, 0.70), 0.26, 0.16, "mechanics\nprediction ŷ₀")
box(axB, (0.0, 0.22), 0.30, 0.18, "EEG relative\nband power\n76 features")
box(axB, (0.40, 0.22), 0.24, 0.18, "orthogonalize\nto M0, then\nStage 2 ridge")
box(axB, (0.74, 0.22), 0.26, 0.18, "EEG part ŷ_EEG\n(fit to stage-1\nresiduals)")
arrow(axB, (0.30, 0.78), (0.40, 0.78))
arrow(axB, (0.64, 0.78), (0.74, 0.78))
arrow(axB, (0.30, 0.31), (0.40, 0.31))
arrow(axB, (0.64, 0.31), (0.74, 0.31))
arrow(axB, (0.52, 0.70), (0.52, 0.40))
axB.text(0.54, 0.55, "residuals", fontsize=6.5, color=INK2, va="center")
axB.text(0.5, 0.06, "ΔR² = R²(ŷ₀ + ŷ_EEG) − R²(ŷ₀); no EEG information ⇒ ΔR² ≤ 0 on average",
         ha="center", fontsize=7, color=INK)

# C: design
axC = fig.add_subplot(gs[1, 1])
axC.set(xlim=(-0.02, 1.04), ylim=(0, 1))
axC.axis("off")
axC.set_title("C  Development and confirmation", loc="left", fontsize=8, fontweight="bold")
box(axC, (0.10, 0.80), 0.80, 0.12, "47 participants, 110 strict trials")
box(axC, (0.0, 0.52), 0.44, 0.18, "14 development\ndesign decisions,\npower analysis")
box(axC, (0.56, 0.52), 0.44, 0.18, "33 confirmation\nuntouched until\nplan fixed")
box(axC, (0.0, 0.24), 0.44, 0.14, "analysis plan\nfixed (SHA-256)", fc="#eef4fc", ec=BLUE)
box(axC, (0.53, 0.18), 0.50, 0.26, "one-shot test:\nWilcoxon (gain > 0)\nequivalence (< 0.01)\nHolm, 3 targets", fc="#eef4fc", ec=BLUE)
arrow(axC, (0.32, 0.80), (0.22, 0.70))
arrow(axC, (0.68, 0.80), (0.78, 0.70))
arrow(axC, (0.22, 0.52), (0.22, 0.38))
arrow(axC, (0.44, 0.31), (0.56, 0.31))
arrow(axC, (0.78, 0.52), (0.78, 0.44))

for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig1_design.{ext}")
print("saved", FIG / "fig1_design.png")
