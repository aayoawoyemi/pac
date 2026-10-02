# Archive: retired work, kept for the record

**Nothing in this folder supports a number in the working paper.** It is the earlier record, kept public so the
history of the estimate can be checked. The scripts still reference their old file layout and are not maintained.

| file | what it was | why it was retired |
|---|---|---|
| `scripts/_ingame_slope.py`, `results/_ingame_*.json` | in-game rest design, s = 0.324 | contaminated by bench-unit lineups |
| `scripts/_sitout_slope.py`, `results/_sitout_*.json` | pooled sit-out WLS and its placebo (0.004) | replaced by the game-level design and the 500-draw permutation placebo |
| `scripts/_blind_sitout.py`, `docs/_blind_sitout_results.md` | independent reimplementation of the sit-out design, s = 0.28 | superseded by the game-level design |
| `scripts/_calib_events.py`, `results/_calib_events_data.json`, `docs/_calib_results_events.md` | trades and season-ending injuries | superseded |
| `scripts/_bench_scoring.py`, `docs/_bench_scoring_results.md` | implied prices of other statistics (OBPM 0.77, RAPM 0.36-0.62) | retired |
| `scripts/_pac_rapm.py` | PAC-RAPM at s = 0.42 | **0.42 is retracted** (see the main README) |
| `scripts/_pac_boards.py`, `_pac_examples.py`, `_pac_predict_games.py`, `_pac_replacement_examples.py`, `_pac_cases.py` and their docs | leaderboards, examples and game prediction | computed at the hinge price, not the paper's 0.25 |
| `scripts/_pac_scarcity_vs_creation.py`, `docs/_pac_scarcity_vs_creation.md` | first version of the scorer / playmaker split | superseded by `scripts/14_scarcity.py` |
| `docs/COEFFICIENT_RECORD.md` | the coefficient record before the game-level design | superseded by `docs/PAC_VALIDATED_NUMBERS.md` |
| `docs/DATA_v0.md` | data notes for the old layout | superseded by `data/README.md` |

Also retired and not to be cited: the teammate-absence instrument (-0.30), the next-season design (0.27), the
"per-carrier correlation 0.346" argument, and the U-shaped held-out MSE offered as proof.
