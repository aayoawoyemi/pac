"""10_estimate_price.py -- Primary design: s = 0.257 [0.223, 0.291], placebo, bootstrap, robustness, leave-one-season-out, event study, mechanism. About 17 minutes; --perms 20 --boot 20 --tag fast for a smoke test. Writes results/gamelevel.{json,md}, results/price_schedule.png."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pac import gamelevel  # noqa: E402

if __name__ == "__main__":
    gamelevel.main()
