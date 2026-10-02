"""run_all.py -- regenerate every result in the paper from data/, then check them against the paper.

    python scripts/run_all.py           full run (about 20 minutes; the primary design's 500-draw placebo dominates)
    python scripts/run_all.py --fast    smoke test: 20 placebo draws and 20 bootstrap draws, written to
                                        results/gamelevel_fast.*; the paper check is skipped

The build scripts (00-03) are not run: they need raw play-by-play and lineup stints that are not in the
repository (see data/README.md). Their outputs are what data/ contains.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FAST = "--fast" in sys.argv

STEPS = [
    ["10_estimate_price.py"] + (["--perms", "20", "--boot", "20", "--tag", "fast"] if FAST else []),
    ["11_heldout_design.py"],
    ["12_pac_vs_tsadd.py"],
    ["13_cases.py"],
    ["14_scarcity.py"],
    ["15_known_answer.py"],
    ["16_team_price.py"],
    ["17_mechanism.py"],
    ["19_figures.py"],
]
if not FAST:
    STEPS.append(["check_abstract_numbers.py"])

t0 = time.time()
for step in STEPS:
    t = time.time()
    print(f"== {' '.join(step)}", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, step[0])] + step[1:], stdout=subprocess.DEVNULL if step[0] != "check_abstract_numbers.py" else None)
    if r.returncode:
        sys.exit(f"{step[0]} failed (exit {r.returncode})")
    print(f"   {time.time() - t:.0f}s", flush=True)
print(f"done in {(time.time() - t0) / 60:.1f} min")
