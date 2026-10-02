# Data

Everything the analysis scripts (10-19) read is in this folder, about 17 MB. Season codes are two-year: `9798` is
1997-98, `2526` is 2025-26. Regular seasons only.

| folder / file | rows | what it is | built by |
|---|---|---|---|
| `team_games/{code}_team_games.parquet` | 68,708 team-games in total | one row per team per game | `02_build_team_games.py` |
| `team_games/{code}_player_games.parquet` | every appearance | one row per player per game he appeared in | `02_build_team_games.py` |
| `team_games/{code}_meta.json` | 29 | match diagnostics for the build (games, dated games, ESPN matches) | `02_build_team_games.py` |
| `player_seasons/pac_season_{code}.json` | one per player-season | PAC inputs and values at s = 0.25 | `01_build_player_values.py` |
| `player_seasons/ast_pct.csv` | 14,128 | NBA.com assist percentage by player-season (used to split scorers from playmakers) | NBA.com, see below |
| `schedules/espn_schedule.json` | 29 seasons | game dates, home and away, used only for rest controls | `00_fetch_schedules.py` |
| `heldout/carriers.json` | 739 | the held-out design's carrier-seasons | `03_build_heldout_carriers.py` |
| `heldout/build_stats.json` | 1 | candidate and drop counts from that build | `03_build_heldout_carriers.py` |

## Schemas

**`*_team_games.parquet`**: `gid` NBA game id · `team`, `opp` NBA team ids · `home` 1/0 · `date` ISO date (ESPN; null
if unmatched) · `pts`, `fga`, `fta`, `tov` team totals · `rest` days since the team's previous game minus one ·
`game_no` game number within the team-season · `opp_rest` the opponent's `rest`.

**`*_player_games.parquet`**: `gid`, `pid` NBA player id, `team`, `name` · `pts`, `fga`, `fta`, `tov` from the box
score · `l_season` the player's season possession share L = TSA / on-court possessions (null where on-court
possessions are unavailable). An appearance is any play-by-play event, so zero-stat appearances are appearances,
not absences.

**`pac_season_{code}.json`**: `summary` (league TS%, slope = 0.25, on-court coverage, a build check) and `rows`, one
per player-season: `pid`, `name`, `gp`, `pts`, `tsa` (FGA + 0.44 FTA), `poss` on-court possessions, `l` = tsa/poss,
`ts`, `rts` (TS% minus league, points), `tsadd` (TS Points Added), `credit` = 0.25·TSA·L, `pac` = tsadd + credit,
`pac_g` per game, `pac100` per 100 TSA. Totals are season totals.

**`ast_pct.csv`**: `season` (start year), `player_id`, `ast_pct`. Rows whose source value did not parse as a number
were dropped, which is the same rule the analysis applied to the original API files.

**`carriers.json`**: one record per (player, team, season) with L ≥ 0.24 and 40+ games for that team: `code`, `year`,
`pid`, `name`, `team`, `L` season share, `L_on` share on his on-court stints, `t_on` teammates' points per TSA while
he is on the floor, `t_off` team points per TSA in games he missed, `price` league points per TSA, `own_ts` his
points per TSA, `n_on` usable on-court games, `n_on_bad` games dropped for unreconciled stints, `n_off` games missed.

## Freeze note: the held-out carriers

`heldout/carriers.json` is the build of 2026-09-21, the one the paper's held-out numbers (0.245 [0.204, 0.284]) come
from. The lineup stint files were rebuilt on 2026-09-26, and the new build reconciles more games with the box score
(45,977 usable on-court games instead of 44,006). Rerunning `03_build_heldout_carriers.py` on the rebuilt stints gives
the same 739 player-seasons and the same 10,676 missed games; only the on-court measurements change, and the estimate
becomes s = 0.244 [0.203, 0.282]. The paper reports the frozen build; the next version will use the rebuilt one.

## Not in this repository

| what | size | why | how to get it |
|---|---|---|---|
| raw play-by-play, `raw/nba_data_raw/nbastats_{year}.csv` (stats.nba.com, 1997-2024) and `cdnnba_{year}.csv` (NBA CDN feed, 2021-2025; PAC uses it from 2021) | 3.7 GB | size, and it belongs to its source | public NBA.com stats and CDN play-by-play endpoints |
| NBA.com player dumps, `raw/_nba_pl_adv_{code}.json`, `raw/_nba_pl_base_{code}.json` | 21 MB | source data | NBA.com `leaguedashplayerstats` (Advanced, Base), regular season |
| lineup stints, `raw/_leaf_stints_{code}.json` (2000-01 onward) | 352 MB | size | built from the play-by-play |
| per-game PAC rows, `raw/_sv_pergame_{code}.json` | 188 MB | size; an intermediate | `01_build_player_values.py` |

Only the build scripts (00-03) need these. The scripts that download the play-by-play and segment it into lineup
stints are part of the LUMA data pipeline and are not in this repository. Put them under `raw/` (or set `PAC_RAW_DIR`) to rebuild `data/` from
scratch.

## Sources and terms

Play-by-play, box scores and assist percentage come from NBA.com; game dates come from ESPN's public schedule API.
Those underlying records remain subject to their providers' terms. What this repository adds (the team-game panel,
the PAC values, the carrier records) is released under CC BY 4.0, see `LICENSE-DATA`.
