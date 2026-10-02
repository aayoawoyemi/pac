"""01_build_player_values.py -- Build per-game and per-season PAC inputs from raw play-by-play (needs raw/). Writes raw/_sv_pergame_{code}.json and data/player_seasons/pac_season_{code}.json. Also the way to compute PAC for any season: python scripts/01_build_player_values.py --years 2025"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pac import player_values  # noqa: E402

if __name__ == "__main__":
    player_values.main()
