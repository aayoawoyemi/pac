"""check_abstract_numbers.py -- every number in the working paper v1, checked against the regenerated results.

Each check formats the regenerated value at the precision printed in the paper and compares strings, so
"matches" means "prints the same". Exit code 1 on any mismatch. Run after scripts/run_all.py.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pac import paths  # noqa: E402

Z = 1.959963984540054
fails = []


def check(label, got, want):
    ok = got == want
    print(f"{'ok ' if ok else 'BAD'} {label}: {got}" + ("" if ok else f"   (paper: {want})"))
    if not ok:
        fails.append(label)


def f3(x):
    return f"{x:.3f}"


def md(name):
    with open(paths.result(name), encoding="utf-8") as f:
        return f.read()


def row(text, start):
    """Cells of the first markdown table row whose first cell starts with `start`."""
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0].startswith(start):
            return cells
    raise KeyError(start)


def num(cell):
    return re.sub(r"[*\s]", "", cell)


with open(paths.result("gamelevel.json"), encoding="utf-8") as f:
    G = json.load(f)

# --- Methods / Introduction: primary design
m = G["main"]
check("s (primary design)", f3(m["s_origin"]), "0.257")
check("s 95% CI", f"[{f3(m['s_origin'] - Z * m['s_origin_se'])}, {f3(m['s_origin'] + Z * m['s_origin_se'])}]", "[0.223, 0.291]")
check("team-games", f"{G['descriptives']['team_games']:,}", "68,708")
check("rotation-player absences", f"{G['descriptives']['absences']:,}", "108,895")

# --- held-out design
h = row(md("heldout.md"), "pooled")
check("held-out carriers / games missed", f"{h[1]} / {h[2]}", "739 / 10676")
check("held-out s and CI", f"{h[3]} {h[5]}", "0.245 [0.204, 0.284]")

# --- break-even at s = 0.25
check("break-even at 30% / 40% share", f"{-0.25 / 2 * 0.30 * 100:.1f} / {-0.25 / 2 * 0.40 * 100:.1f}", "-3.8 / -5.0")

# --- placebo
p = G["permutation"]
check("placebo draws", str(p["n"]), "500")
g = p["gap30"]
check("placebo gap at 30%: real vs 95% range, p", f"{f3(g['real'])} vs [{f3(g['lo'])}, {f3(g['hi'])}], p = {f3(g['p'])}",
      "0.098 vs [-0.011, 0.012], p = 0.002")

# --- 11 alternative specifications (robustness[0] is the main specification)
alt = [r["s_origin"] for r in G["robustness"][1:]]
check("alternative specifications", str(len(alt)), "11")
check("their range of s", f"[{f3(min(alt))}, {f3(max(alt))}]", "[0.217, 0.276]")

# --- scarcity vs playmaking at 30% share
sc = md("scarcity.md")
check("low-assist scorers, gap at 30%", num(row(sc, "pure scorers")[-1]).split("[")[0], "0.072")
check("high-assist playmakers, gap at 30%", num(row(sc, "playmakers")[-1]).split("[")[0], "0.106")
check("PAC's implied gap at 30%", f3(0.25 * 0.30), "0.075")

# --- Table 1
t = md("pac_vs_tsadd.md")
want = {"0%-15%": ("20,669", "-0.39[-0.59,-0.19]", "-0.08", "0.10"),
        "15%-20%": ("46,627", "-0.12[-0.28,0.03]", "-0.21", "0.10"),
        "20%-25%": ("29,319", "0.32[0.13,0.51]", "-0.25", "0.33"),
        "25%-30%": ("9,543", "2.00[1.68,2.32]", "0.07", "1.14"),
        "30%-100%": ("2,737", "3.52[2.92,4.12]", "0.52", "2.30")}
for k, w in want.items():
    c = row(t, k)
    check(f"Table 1 row {k}", (c[1], num(c[3]), c[4], c[5]), w)

# --- Iverson 2001-02 and the Figure 1 pool
cs = md("cases.md")
iv = [ln for ln in cs.splitlines() if ln.startswith("| 01-02 | Allen Iverson")][0]
cells = [c.strip() for c in iv.strip("|").split("|")]
check("Iverson 01-02: TS Added/g (rank) vs PAC/g (rank) of n", f"{cells[4]} ({cells[5]}) vs {cells[6]} ({cells[7]}) of {cells[8]}",
      "-1.78 (214) vs +1.23 (60) of 216")
check("qualified player-seasons (Figure 1)", re.search(r"Pool: ([\d,]+) qualified", cs).group(1), "6,315")

print()
if fails:
    print(f"{len(fails)} MISMATCH(ES):", ", ".join(fails))
    sys.exit(1)
print("all paper numbers reproduce")
