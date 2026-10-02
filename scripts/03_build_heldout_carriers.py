"""03_build_heldout_carriers.py -- Build the held-out design's 739 carrier-seasons from raw lineup stints and per-game rows (needs raw/). Writes data/heldout/carriers.json."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pac import heldout  # noqa: E402

if __name__ == "__main__":
    heldout.build()
