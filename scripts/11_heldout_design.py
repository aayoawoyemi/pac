"""11_heldout_design.py -- Second, held-out design: s = 0.245 [0.204, 0.284] from data/heldout/carriers.json. Writes results/heldout.md."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pac import heldout  # noqa: E402

if __name__ == "__main__":
    heldout.estimate()
