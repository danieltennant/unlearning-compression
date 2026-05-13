"""
Generate figures for the compression-reverses-unlearning paper.
8B results only: GradDiff, SimNPO, RMU across quantization and pruning.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────────

ORACLE_FORGET  = 0.104
ORACLE_UTILITY = 0.648
FULL_FORGET    = 0.992

METHODS = ["GradDiff", "SimNPO", "RMU"]
COLORS  = ["#2166ac", "#d6604d", "#4dac26"]  # blue, red, green

# forget_Q_A_Prob / model_utility
QUANT = {
    "GradDiff": {"Unlearned": (0.028, 0.465), "8-bit": (0.033, 0.467), "4-bit": (0.672, 0.589)},
    "SimNPO":   {"Unlearned": (0.088, 0.653), "8-bit": (0.096, 0.657), "4-bit": (0.210, 0.636)},
    "RMU":      {"Unlearned": (0.067, 0.637), "8-bit": (0.083, 0.638), "4-bit": (0.649, 0.614)},
}

PRUNE = {
    "GradDiff": {0: (0.028, 0.465), 10: (0.187, 0.543), 20: (0.979, 0.637), 30: (0.938, 0.630)},
    "SimNPO":   {0: (0.088, 0.653), 10: (0.112, 0.660), 20: (0.333, 0.649), 30: (0.935, 0.630)},
    "RMU":      {0: (0.067, 0.637), 10: (0.124, 0.630), 20: (0.963, 0.622), 30: (0.940, 0.639)},
}

PRUNE_X = [0, 10, 20, 30]
QUANT_LABELS = ["Unlearned", "8-bit", "4-bit"]

# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

out_dir = Path(__file__).parent.parent / "results" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

bar_width = 0.22
x_base    = np.arange(len(QUANT_LABELS))

# ── Figure 1: Quantization (2 panels stacked) ─────────────────────────────────

fig1, (ax_top, ax_bot) = plt.subplots(1, 2, figsize=(9, 4))
fig1.subplots_adjust(wspace=0.32)

# Top: forget_Q_A_Prob
for i, (method, color) in enumerate(zip(METHODS, COLORS)):
    vals = [QUANT[method][lbl][0] for lbl in QUANT_LABELS]
    ax_top.bar(x_base + (i - 1) * bar_width, vals, bar_width,
               label=method, color=color, alpha=0.85, zorder=3)

ax_top.axhline(ORACLE_FORGET, color="black", linestyle=":", linewidth=1.2,
               label=f"Oracle ({ORACLE_FORGET})", zorder=4)
ax_top.axhline(FULL_FORGET, color="gray", linestyle="--", linewidth=1.0,
               label="Full model", zorder=4)
ax_top.set_ylabel("Forget-set probability")
ax_top.set_ylim(0, 1.05)
ax_top.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
ax_top.set_xticks(x_base)
ax_top.set_xticklabels(QUANT_LABELS)
ax_top.legend(fontsize=8, loc="upper left")
ax_top.set_title("Knowledge recovery", fontsize=10, pad=6)

# Right: model_utility
for i, (method, color) in enumerate(zip(METHODS, COLORS)):
    vals = [QUANT[method][lbl][1] for lbl in QUANT_LABELS]
    ax_bot.bar(x_base + (i - 1) * bar_width, vals, bar_width,
               label=method, color=color, alpha=0.85, zorder=3)

ax_bot.axhline(ORACLE_UTILITY, color="black", linestyle=":", linewidth=1.2,
               label=f"Oracle ({ORACLE_UTILITY})", zorder=4)
ax_bot.set_xticks(x_base)
ax_bot.set_xticklabels(QUANT_LABELS)
ax_bot.set_ylabel("Model utility")
ax_bot.set_ylim(0, 0.75)
ax_bot.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
ax_bot.legend(fontsize=8, loc="lower right")
ax_bot.set_title("Model utility", fontsize=10, pad=6)

fig1.suptitle("Quantization", fontsize=12, fontweight="bold", y=1.02)

for ext in ("pdf", "png"):
    p = out_dir / f"quantization.{ext}"
    fig1.savefig(p, bbox_inches="tight", dpi=150)
    print(f"Saved {p}")

# ── Figure 2: Pruning (2 panels stacked) ──────────────────────────────────────

fig2, (ax_top, ax_bot) = plt.subplots(1, 2, figsize=(9, 4))
fig2.subplots_adjust(wspace=0.32)

# Top: forget_Q_A_Prob
for method, color in zip(METHODS, COLORS):
    vals = [PRUNE[method][x][0] for x in PRUNE_X]
    ax_top.plot(PRUNE_X, vals, marker="o", color=color, label=method,
                linewidth=1.8, markersize=5, zorder=3)

ax_top.axhline(ORACLE_FORGET, color="black", linestyle=":", linewidth=1.2,
               label=f"Oracle ({ORACLE_FORGET})", zorder=4)
ax_top.axhline(FULL_FORGET, color="gray", linestyle="--", linewidth=1.0,
               label="Full model", zorder=4)
ax_top.set_ylabel("Forget-set probability")
ax_top.set_ylim(0, 1.05)
ax_top.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
ax_top.set_xticks(PRUNE_X)
ax_top.set_xticklabels([f"{x}%" for x in PRUNE_X])
ax_top.set_xlabel("Sparsity (%)")
ax_top.legend(fontsize=8, loc="upper left")
ax_top.set_title("Knowledge recovery", fontsize=10, pad=6)

# Right: model_utility
for method, color in zip(METHODS, COLORS):
    vals = [PRUNE[method][x][1] for x in PRUNE_X]
    ax_bot.plot(PRUNE_X, vals, marker="o", color=color, label=method,
                linewidth=1.8, markersize=5, zorder=3)

ax_bot.axhline(ORACLE_UTILITY, color="black", linestyle=":", linewidth=1.2,
               label=f"Oracle ({ORACLE_UTILITY})", zorder=4)
ax_bot.set_xlabel("Sparsity (%)")
ax_bot.set_ylabel("Model utility")
ax_bot.set_xticks(PRUNE_X)
ax_bot.set_xticklabels([f"{x}%" for x in PRUNE_X])
ax_bot.set_ylim(0, 0.75)
ax_bot.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
ax_bot.legend(fontsize=8, loc="lower right")
ax_bot.set_title("Model utility", fontsize=10, pad=6)

fig2.suptitle("Magnitude pruning", fontsize=12, fontweight="bold", y=1.02)

for ext in ("pdf", "png"):
    p = out_dir / f"pruning.{ext}"
    fig2.savefig(p, bbox_inches="tight", dpi=150)
    print(f"Saved {p}")

plt.show()
