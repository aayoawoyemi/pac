"""Where everything lives. Every script resolves its inputs and outputs through here.

    data/       inputs shipped with the repository (see data/README.md)
    results/    everything the analysis scripts regenerate
    raw/        optional, not in the repository: raw play-by-play, box scores and lineup stints, only
                needed by the build scripts (01-03). See data/README.md for how to obtain them.

Environment overrides: PAC_DATA_DIR, PAC_RESULTS_DIR, PAC_RAW_DIR.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("PAC_DATA_DIR", os.path.join(ROOT, "data"))
RESULTS = os.environ.get("PAC_RESULTS_DIR", os.path.join(ROOT, "results"))
RAW = os.environ.get("PAC_RAW_DIR", os.path.join(ROOT, "raw"))
PAPER = os.path.join(ROOT, "paper")

YEARS = list(range(1997, 2026))          # season start years, 1997-98 .. 2025-26


def code_of(y: int) -> str:
    """1997 -> '9798'."""
    return f"{y % 100:02d}{(y + 1) % 100:02d}"


def team_games(code: str) -> str:
    return os.path.join(DATA, "team_games", f"{code}_team_games.parquet")


def player_games(code: str) -> str:
    return os.path.join(DATA, "team_games", f"{code}_player_games.parquet")


def team_games_meta(code: str) -> str:
    return os.path.join(DATA, "team_games", f"{code}_meta.json")


def pac_season(code: str) -> str:
    """Per player-season PAC inputs and values at s = 0.25 (built by 01_build_player_values.py)."""
    return os.path.join(DATA, "player_seasons", f"pac_season_{code}.json")


AST_PCT = os.path.join(DATA, "player_seasons", "ast_pct.csv")
SCHEDULE = os.path.join(DATA, "schedules", "espn_schedule.json")
HELDOUT_CARRIERS = os.path.join(DATA, "heldout", "carriers.json")


def raw(*parts: str) -> str:
    return os.path.join(RAW, *parts)


def result(name: str) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    return os.path.join(RESULTS, name)
