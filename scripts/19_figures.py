"""19_figures.py -- the paper's two figures, regenerated from data/ and results/.

Figure 1: shooting-efficiency break-even by possession share (every qualified player-season, 1997-98 .. 2025-26).
Figure 2: measured replacement gap by share bin (results/gamelevel.json `nonparametric`) against PAC's 0.25*L.

Writes results/figures/figure1_breakeven.png and results/figures/figure2_price.png. The published versions in
paper/figures/ were drawn by this same code; fonts can differ by machine (Cambria where installed).
"""
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pac import paths  # noqa: E402

S = 0.25
plt.rcParams["font.family"] = ["Cambria", "DejaVu Serif"]


def qualified_seasons():
    rows = []
    for y in paths.YEARS:
        with open(paths.pac_season(paths.code_of(y)), encoding="utf-8") as f:
            for r in json.load(f)["rows"]:
                rows.append(dict(r, season=y))
    s = pd.DataFrame(rows)
    q = s[(s.gp >= 40) & (s.poss >= 2500)].copy()
    q["tsag"] = q.tsa / q.gp
    return q


def figure1(q, path):
    fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=220)
    ok = q.pac >= 0
    ax.scatter(q.l[ok], q.rts[ok], s=(q.tsag[ok] ** 1.5) * 0.3, c="#2b6cb0", alpha=0.22, linewidths=0, label="PAC ≥ 0")
    ax.scatter(q.l[~ok], q.rts[~ok], s=(q.tsag[~ok] ** 1.5) * 0.3, c="#c53030", alpha=0.30, linewidths=0, label="PAC < 0")
    xx = np.linspace(0.05, 0.42, 50)
    ax.plot(xx, -(S / 2) * 100 * xx, color="#1a202c", lw=2, label="PAC break-even, rTS = −(0.25/2)·L")
    ax.axhline(0, color="#718096", lw=1.5, ls="--", label="TS Points Added break-even, rTS = 0")
    for nm, yr, dx, dy in [("Allen Iverson", 2001, -0.035, -9.0), ("Russell Westbrook", 2016, -0.075, 5.5),
                           ("Stephen Curry", 2015, -0.07, 3.0), ("Kobe Bryant", 2015, -0.075, -6.0)]:
        r = q[(q.name == nm) & (q.season == yr)].iloc[0]
        ax.scatter([r.l], [r.rts], s=36, facecolors="none", edgecolors="black", lw=1.1, zorder=5)
        ax.annotate(f"{nm.split()[-1]} {yr % 100:02d}-{(yr + 1) % 100:02d}", (r.l, r.rts), xytext=(r.l + dx, r.rts + dy),
                    fontsize=7.5, arrowprops=dict(arrowstyle="-", lw=0.6, color="black"), zorder=6)
    ax.set_xlim(0.05, 0.42)
    ax.set_ylim(-16, 19)
    ax.set_xlabel("possession share, L = TSA/POSS", fontsize=8.5)
    ax.set_ylabel("relative true shooting, rTS (points)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure2(res, path):
    b = res["nonparametric"]
    L = np.array([x["mean_L"] for x in b])
    g = np.array([x["gap"] for x in b])
    se = np.array([x["se"] for x in b])
    n = [x["absences"] for x in b]
    fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=220)
    x = np.linspace(0.08, 0.36, 50)
    ax.axhline(0, color="#718096", lw=1.5, ls="--", label="league-average replacement (TS Points Added, s = 0)")
    ax.plot(x, S * x, color="#1a202c", lw=2, label="PAC price, 0.25·L")
    ax.errorbar(L, g, yerr=1.96 * se, fmt="o", color="#2b6cb0", ms=5, capsize=3, lw=1.2,
                label="measured gap by share bin (95% CI)")
    for xi, ni in zip(L, n):
        ax.annotate(f"{ni:,}", (xi, -0.112), ha="center", fontsize=6.5, color="#555")
    ax.set_xlim(0.08, 0.36)
    ax.set_ylim(-0.125, 0.14)
    ax.set_ylabel("replacement gap\n(points per shooting possession)", fontsize=8.5)
    ax.set_xlabel("absent player's possession share, L   (numbers: absences per bin)", fontsize=8.5)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.tick_params(labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    out = os.path.join(paths.RESULTS, "figures")
    os.makedirs(out, exist_ok=True)
    q = qualified_seasons()
    assert len(q) == 6315, len(q)
    figure1(q, os.path.join(out, "figure1_breakeven.png"))
    with open(paths.result("gamelevel.json"), encoding="utf-8") as f:
        res = json.load(f)
    figure2(res, os.path.join(out, "figure2_price.png"))
    print("figures ->", out)


if __name__ == "__main__":
    main()
