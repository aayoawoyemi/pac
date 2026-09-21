# Data

## What the scripts read

| input | what it is | how it is built |
|---|---|---|
| `nba_data_raw/nbastats_{year}.csv` | NBA play-by-play, one file per season start year. 1997 through 2024. | scraped from public NBA endpoints |
| `cdnnba_2025.csv` | 2025-26 play-by-play, richer CDN schema | same |
| `_leaf_stints_{code}.json` | game -> lineup -> possession stints, per season | `_leaf_poss.load_leaf` |
| `_nba_pl_adv_{code}.json` | player-season on-court possessions | derived from stints |

Season codes are two-year: `0001` is 1999-2000, `2526` is 2025-26. File names keyed by
**start year**: `nbastats_2023` is the 2023-24 season.

## What this repository does not redistribute

Raw play-by-play is not included. It is large, and it belongs to its source. The scripts expect it
at `nba_data_raw/` relative to the script directory.

Everything derived **is** included, in `results/`:

| file | contents |
|---|---|
| `_sitout_2000_2025.json` | 739 carrier-seasons: share, on-court rate, absent-side rate, price, absence count |
| `_sitout_2000_2014.json`, `_sitout_2015_2025.json` | the era splits |
| `_sitout_2015_2025_placebo.json` | fake absences drawn from games played |
| `_ingame_2000_2025.json` | 1,194 carrier-seasons, in-game rest design |
| `_ingame_2000_2025_placebo.json` | random split of the player's own stints |
| `_calib_sitout_carriers.json` | the held-out calibration frame |
| `_calib_events_data.json` | trades and season-ending injuries, 278 events |

Those files are enough to reproduce every coefficient in the README without touching raw
play-by-play. The row schema is one record per carrier-season:

```
code    season, two-year code
team    NBA team id
pid     player id
L       season share of on-court possessions
L_on    share measured on his on-court stints
lam     league price index for the season
t_on    team points per attempt with him on the floor
t_off   team points per attempt in the games or stints he missed
qbar    observed return on the redistributed attempts
price   the priced return at the candidate coefficient
n_off   absences (sit-out file) / w: possession weight (in-game file)
n_on    games or stints with him available
```

The implied per-carrier slope is `(price - qbar) / L`. Pooled estimates weight by opportunity
count, never by carrier, because a player with five absences and one with sixty should not count
equally.

## Verifying a rebuild

`_sv_pergame.py` prints a per-season check. Two invariants:

- `credit / (TSA * L)` equals the slope on every row. At `SLOPE = 0.25` the median is exactly
  0.250.
- The team with the higher summed PAC wins 83-87% of games in every season, correlation with
  final margin .82-.87. This holds for any slope, because additivity does not depend on the
  coefficient. It is a build check, not evidence for the number.
