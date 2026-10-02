# Blind sit-out replication: return on redistributed attempts

## Estimator

**Carrier gate.** A carrier-season is a (player, team, season) with `l_season >= 0.24`
(season TSA / season on-court possessions, as given) and at least 40 games appeared for
that team. 0.24 is the point where a player takes roughly a quarter of the possessions
he is on the floor for, i.e. well above the 0.20 that five equal players would split;
40 games guarantees enough W games to pin down the teammates' baseline. A carrier-season
also needs at least 3 sit-out games with a usable possession base, otherwise the
per-season ratio is undefined or pure noise.

**Sit-outs.** Team games inside the window [first appearance, last appearance] for that
team (by game id order) in which the carrier has no box-score row. Games with no on-court
possession data (`p_g` null) are dropped on both sides.

**Base.** Everything is per **team possession**. Team possessions for a game =
sum of `p_g` over the team's rows / 5 (five players are on the floor for every possession).
The same construction is used for W and O games, so pace changes between the two sets do
not enter the comparison.

**Identity.** Per team possession, with him: `pts_W = pts_c + pts_tm`, `tsa_W = tsa_c + tsa_tm`.
Without him: `pts_O`, `tsa_O`. Assume teammates keep the return they already had on the
attempts they were already taking, and every extra attempt that appears when he sits
(`Delta = tsa_O - tsa_tm`, the redistributed attempts, including whoever replaces him)
returns `r`:

    pts_O = pts_tm + r * (tsa_O - tsa_tm)     =>     r = (pts_O - pts_tm) / (tsa_O - tsa_tm)

Any efficiency change on the teammates' pre-existing attempts (tougher defensive attention,
worse shot quality) is loaded onto `r` by construction; `r` is the full marginal return of
moving the carrier's volume onto the rest of the roster. All sums are ratio-of-sums over
games (not means of per-game ratios). `gap = P - r` with P = league points per shooting
attempt that season (sum pts / sum tsa over every player-game).

**Regression.** `gap_i = beta * L_i` through the origin, weight = number of sit-out games,
`L = l_season` as given. Bootstrap: 500 resamples of carrier-seasons, percentile CI.
The with-intercept WLS is reported alongside because a through-origin slope is positive
whenever the mean gap is positive, regardless of whether it grows with L.

**Placebo.** For each carrier-season, the same number of games as his real sit-outs is
drawn at random (seed 0) from games he actually played, labelled "absent", and the whole
pipeline is rerun. Team totals in those games still contain the carrier, so `Delta` is his
own volume and `r` recovers his own return; the placebo gap is `P - own return`. Real gap
minus placebo gap is therefore the cost of redistribution net of "who took the shots".
A placebo slope near the real slope would mean the estimator is only measuring carrier
efficiency, not the redistribution.

**Secondary variant (on-court).** Same identity, but the W side is restricted with stint
data to the carrier's on-court possessions, with counted possessions
(FGA + 0.44 FTA + TOV - OREB from the stint tallies) as the base on both sides. Only games
with stint coverage enter. This puts the redistributed volume at his full on-court share
instead of the minutes-diluted whole-game share.

## 2015-16 to 2025-26

Eligible carrier-seasons (l_season >= 0.24, >= 40 games for the team): 563; of which 469 have >= 3 sit-out games with a possession base.

### Primary: whole-game W vs O
- carrier-seasons: **469**; sit-out games: **6156** (carrier-seasons dropped because Delta <= 0: 0)
- gap quantiles over carrier-seasons min / 5% / median / 95% / max: -1.796 / -0.321 / 0.094 / 0.482 / 1.291
- league price P (mean over carrier-seasons, weighted by sit-out games): **1.1440** pts per shooting attempt
- return per redistributed attempt, pooled ratio of sums (total marginal points / total redistributed attempts): **1.0506**
- return per redistributed attempt, sit-out-game-weighted mean of per-season r_i: **1.0521**
- gap P - r (sit-out-game-weighted mean): **0.0919** pts per redistributed attempt
- redistributed attempts per team possession (mean Delta): 0.1860
- carrier's own return on the W side (pts per attempt): 1.1551
- slope of gap on L through the origin, weight = sit-out games: **0.3304**, bootstrap 95% CI [0.2534, 0.4089] (500 resamples over carrier-seasons)
- with intercept: slope 0.2305 [-0.5149, 0.9112], intercept 0.0280
- robustness, carrier-seasons with >= 10 sit-out games only: slope through origin 0.3152 [0.2101, 0.4072] (n = 263, sit-out games = 4930)

| share band | carrier-seasons | sit-out games | gap = P - r | r (pts/attempt) | P |
|---|---|---|---|---|---|
| 24-27% | 208 | 2729 | 0.0814 | 1.0589 | 1.1402 |
| 27-30% | 155 | 2112 | 0.0999 | 1.0471 | 1.1470 |
| 30-34% | 87 | 1062 | 0.1026 | 1.0443 | 1.1469 |
| 34%+ | 19 | 253 | 0.0950 | 1.0527 | 1.1477 |

### Placebo: fake sit-outs drawn from played games
- carrier-seasons: **469**; sit-out games: **5636** (carrier-seasons dropped because Delta <= 0: 0)
- gap quantiles over carrier-seasons min / 5% / median / 95% / max: -1.105 / -0.491 / -0.019 / 0.421 / 1.139
- league price P (mean over carrier-seasons, weighted by sit-out games): **1.1432** pts per shooting attempt
- return per redistributed attempt, pooled ratio of sums (total marginal points / total redistributed attempts): **1.1546**
- return per redistributed attempt, sit-out-game-weighted mean of per-season r_i: **1.1582**
- gap P - r (sit-out-game-weighted mean): **-0.0150** pts per redistributed attempt
- redistributed attempts per team possession (mean Delta): 0.1854
- carrier's own return on the W side (pts per attempt): 1.1556
- slope of gap on L through the origin, weight = sit-out games: **-0.0607**, bootstrap 95% CI [-0.1516, 0.0225] (500 resamples over carrier-seasons)
- with intercept: slope -0.6835 [-1.4933, 0.0632], intercept 0.1749
- robustness, carrier-seasons with >= 10 sit-out games only: slope through origin -0.0484 [-0.1549, 0.0562] (n = 263, sit-out games = 4410)

| share band | carrier-seasons | sit-out games | gap = P - r | r (pts/attempt) | P |
|---|---|---|---|---|---|
| 24-27% | 208 | 2479 | 0.0005 | 1.1391 | 1.1395 |
| 27-30% | 155 | 1929 | -0.0122 | 1.1582 | 1.1460 |
| 30-34% | 87 | 987 | -0.0498 | 1.1957 | 1.1459 |
| 34%+ | 19 | 241 | -0.0535 | 1.2008 | 1.1473 |

### Secondary: on-court W (stints) vs O
- carrier-seasons: **448**; sit-out games: **5678** (carrier-seasons dropped because Delta <= 0: 0)
- gap quantiles over carrier-seasons min / 5% / median / 95% / max: -0.549 / -0.166 / 0.088 / 0.348 / 0.680
- league price P (mean over carrier-seasons, weighted by sit-out games): **1.1437** pts per shooting attempt
- return per redistributed attempt, pooled ratio of sums (total marginal points / total redistributed attempts): **1.0580**
- return per redistributed attempt, sit-out-game-weighted mean of per-season r_i: **1.0604**
- gap P - r (sit-out-game-weighted mean): **0.0834** pts per redistributed attempt
- redistributed attempts per team possession (mean Delta): 0.2832
- carrier's own return on the W side (pts per attempt): 1.1569
- slope of gap on L through the origin, weight = sit-out games: **0.3044**, bootstrap 95% CI [0.2496, 0.3541] (500 resamples over carrier-seasons)
- with intercept: slope 0.7177 [0.2027, 1.2138], intercept -0.1161
- robustness, carrier-seasons with >= 10 sit-out games only: slope through origin 0.2903 [0.2297, 0.3522] (n = 245, sit-out games = 4470)

| share band | carrier-seasons | sit-out games | gap = P - r | r (pts/attempt) | P |
|---|---|---|---|---|---|
| 24-27% | 197 | 2462 | 0.0570 | 1.0825 | 1.1395 |
| 27-30% | 149 | 1982 | 0.1025 | 1.0447 | 1.1472 |
| 30-34% | 83 | 981 | 0.1025 | 1.0436 | 1.1461 |
| 34%+ | 19 | 253 | 0.1159 | 1.0317 | 1.1477 |

## 2000-01 to 2014-15

Eligible carrier-seasons (l_season >= 0.24, >= 40 games for the team): 631; of which 397 have >= 3 sit-out games with a possession base.

### Primary: whole-game W vs O
- carrier-seasons: **397**; sit-out games: **4369** (carrier-seasons dropped because Delta <= 0: 0)
- gap quantiles over carrier-seasons min / 5% / median / 95% / max: -1.030 / -0.387 / 0.063 / 0.468 / 0.868
- league price P (mean over carrier-seasons, weighted by sit-out games): **1.0702** pts per shooting attempt
- return per redistributed attempt, pooled ratio of sums (total marginal points / total redistributed attempts): **0.9967**
- return per redistributed attempt, sit-out-game-weighted mean of per-season r_i: **1.0151**
- gap P - r (sit-out-game-weighted mean): **0.0551** pts per redistributed attempt
- redistributed attempts per team possession (mean Delta): 0.1811
- carrier's own return on the W side (pts per attempt): 1.0717
- slope of gap on L through the origin, weight = sit-out games: **0.2056**, bootstrap 95% CI [0.1096, 0.2941] (500 resamples over carrier-seasons)
- with intercept: slope 0.4094 [-0.3075, 1.1659], intercept -0.0559
- robustness, carrier-seasons with >= 10 sit-out games only: slope through origin 0.1707 [0.0663, 0.2783] (n = 175, sit-out games = 3140)

| share band | carrier-seasons | sit-out games | gap = P - r | r (pts/attempt) | P |
|---|---|---|---|---|---|
| 24-27% | 238 | 2633 | 0.0501 | 1.0216 | 1.0717 |
| 27-30% | 98 | 1086 | 0.0497 | 1.0188 | 1.0686 |
| 30-34% | 49 | 513 | 0.0727 | 0.9957 | 1.0684 |
| 34%+ | 12 | 137 | 0.1293 | 0.9330 | 1.0623 |

### Placebo: fake sit-outs drawn from played games
- carrier-seasons: **397**; sit-out games: **4111** (carrier-seasons dropped because Delta <= 0: 0)
- gap quantiles over carrier-seasons min / 5% / median / 95% / max: -0.870 / -0.500 / -0.034 / 0.433 / 0.628
- league price P (mean over carrier-seasons, weighted by sit-out games): **1.0706** pts per shooting attempt
- return per redistributed attempt, pooled ratio of sums (total marginal points / total redistributed attempts): **1.1002**
- return per redistributed attempt, sit-out-game-weighted mean of per-season r_i: **1.1082**
- gap P - r (sit-out-game-weighted mean): **-0.0377** pts per redistributed attempt
- redistributed attempts per team possession (mean Delta): 0.1834
- carrier's own return on the W side (pts per attempt): 1.0745
- slope of gap on L through the origin, weight = sit-out games: **-0.1403**, bootstrap 95% CI [-0.2418, -0.0455] (500 resamples over carrier-seasons)
- with intercept: slope -0.2633 [-0.9821, 0.4163], intercept 0.0337
- robustness, carrier-seasons with >= 10 sit-out games only: slope through origin -0.1811 [-0.3056, -0.0549] (n = 175, sit-out games = 2882)

| share band | carrier-seasons | sit-out games | gap = P - r | r (pts/attempt) | P |
|---|---|---|---|---|---|
| 24-27% | 238 | 2514 | -0.0339 | 1.1059 | 1.0720 |
| 27-30% | 98 | 982 | -0.0596 | 1.1290 | 1.0694 |
| 30-34% | 49 | 492 | -0.0128 | 1.0805 | 1.0676 |
| 34%+ | 12 | 123 | -0.0381 | 1.1004 | 1.0622 |

### Secondary: on-court W (stints) vs O
- carrier-seasons: **391**; sit-out games: **4289** (carrier-seasons dropped because Delta <= 0: 0)
- gap quantiles over carrier-seasons min / 5% / median / 95% / max: -0.669 / -0.224 / 0.077 / 0.357 / 0.683
- league price P (mean over carrier-seasons, weighted by sit-out games): **1.0704** pts per shooting attempt
- return per redistributed attempt, pooled ratio of sums (total marginal points / total redistributed attempts): **1.0011**
- return per redistributed attempt, sit-out-game-weighted mean of per-season r_i: **1.0060**
- gap P - r (sit-out-game-weighted mean): **0.0644** pts per redistributed attempt
- redistributed attempts per team possession (mean Delta): 0.2694
- carrier's own return on the W side (pts per attempt): 1.0719
- slope of gap on L through the origin, weight = sit-out games: **0.2417**, bootstrap 95% CI [0.1858, 0.2933] (500 resamples over carrier-seasons)
- with intercept: slope 0.5996 [0.0520, 1.1234], intercept -0.0981
- robustness, carrier-seasons with >= 10 sit-out games only: slope through origin 0.2021 [0.1280, 0.2780] (n = 172, sit-out games = 3077)

| share band | carrier-seasons | sit-out games | gap = P - r | r (pts/attempt) | P |
|---|---|---|---|---|---|
| 24-27% | 235 | 2613 | 0.0564 | 1.0153 | 1.0717 |
| 27-30% | 96 | 1038 | 0.0657 | 1.0037 | 1.0695 |
| 30-34% | 48 | 501 | 0.0849 | 0.9834 | 1.0683 |
| 34%+ | 12 | 137 | 0.1328 | 0.9295 | 1.0623 |

## pooled 2000-01 to 2025-26

Eligible carrier-seasons (l_season >= 0.24, >= 40 games for the team): 1194; of which 866 have >= 3 sit-out games with a possession base.

### Primary: whole-game W vs O
- carrier-seasons: **866**; sit-out games: **10525** (carrier-seasons dropped because Delta <= 0: 0)
- gap quantiles over carrier-seasons min / 5% / median / 95% / max: -1.796 / -0.358 / 0.075 / 0.473 / 1.291
- league price P (mean over carrier-seasons, weighted by sit-out games): **1.1134** pts per shooting attempt
- return per redistributed attempt, pooled ratio of sums (total marginal points / total redistributed attempts): **1.0294**
- return per redistributed attempt, sit-out-game-weighted mean of per-season r_i: **1.0367**
- gap P - r (sit-out-game-weighted mean): **0.0767** pts per redistributed attempt
- redistributed attempts per team possession (mean Delta): 0.1839
- carrier's own return on the W side (pts per attempt): 1.1205
- slope of gap on L through the origin, weight = sit-out games: **0.2800**, bootstrap 95% CI [0.2174, 0.3388] (500 resamples over carrier-seasons)
- with intercept: slope 0.3689 [-0.1695, 0.8790], intercept -0.0247
- robustness, carrier-seasons with >= 10 sit-out games only: slope through origin 0.2605 [0.1775, 0.3360] (n = 438, sit-out games = 8070)

| share band | carrier-seasons | sit-out games | gap = P - r | r (pts/attempt) | P |
|---|---|---|---|---|---|
| 24-27% | 446 | 5362 | 0.0660 | 1.0406 | 1.1066 |
| 27-30% | 253 | 3198 | 0.0828 | 1.0375 | 1.1204 |
| 30-34% | 136 | 1575 | 0.0929 | 1.0284 | 1.1213 |
| 34%+ | 31 | 390 | 0.1070 | 1.0106 | 1.1177 |

### Placebo: fake sit-outs drawn from played games
- carrier-seasons: **866**; sit-out games: **9747** (carrier-seasons dropped because Delta <= 0: 0)
- gap quantiles over carrier-seasons min / 5% / median / 95% / max: -1.105 / -0.493 / -0.022 / 0.427 / 1.139
- league price P (mean over carrier-seasons, weighted by sit-out games): **1.1126** pts per shooting attempt
- return per redistributed attempt, pooled ratio of sums (total marginal points / total redistributed attempts): **1.1326**
- return per redistributed attempt, sit-out-game-weighted mean of per-season r_i: **1.1371**
- gap P - r (sit-out-game-weighted mean): **-0.0245** pts per redistributed attempt
- redistributed attempts per team possession (mean Delta): 0.1846
- carrier's own return on the W side (pts per attempt): 1.1214
- slope of gap on L through the origin, weight = sit-out games: **-0.0933**, bootstrap 95% CI [-0.1586, -0.0226] (500 resamples over carrier-seasons)
- with intercept: slope -0.4578 [-1.0509, 0.1240], intercept 0.1013
- robustness, carrier-seasons with >= 10 sit-out games only: slope through origin -0.0991 [-0.1837, -0.0086] (n = 438, sit-out games = 7292)

| share band | carrier-seasons | sit-out games | gap = P - r | r (pts/attempt) | P |
|---|---|---|---|---|---|
| 24-27% | 446 | 4993 | -0.0168 | 1.1224 | 1.1055 |
| 27-30% | 253 | 2911 | -0.0282 | 1.1484 | 1.1202 |
| 30-34% | 136 | 1479 | -0.0375 | 1.1573 | 1.1198 |
| 34%+ | 31 | 364 | -0.0483 | 1.1669 | 1.1186 |

### Secondary: on-court W (stints) vs O
- carrier-seasons: **839**; sit-out games: **9967** (carrier-seasons dropped because Delta <= 0: 0)
- gap quantiles over carrier-seasons min / 5% / median / 95% / max: -0.669 / -0.195 / 0.084 / 0.357 / 0.683
- league price P (mean over carrier-seasons, weighted by sit-out games): **1.1122** pts per shooting attempt
- return per redistributed attempt, pooled ratio of sums (total marginal points / total redistributed attempts): **1.0350**
- return per redistributed attempt, sit-out-game-weighted mean of per-season r_i: **1.0370**
- gap P - r (sit-out-game-weighted mean): **0.0752** pts per redistributed attempt
- redistributed attempts per team possession (mean Delta): 0.2773
- carrier's own return on the W side (pts per attempt): 1.1203
- slope of gap on L through the origin, weight = sit-out games: **0.2782**, bootstrap 95% CI [0.2355, 0.3201] (500 resamples over carrier-seasons)
- with intercept: slope 0.6972 [0.3409, 1.0291], intercept -0.1165
- robustness, carrier-seasons with >= 10 sit-out games only: slope through origin 0.2555 [0.2060, 0.3058] (n = 417, sit-out games = 7547)

| share band | carrier-seasons | sit-out games | gap = P - r | r (pts/attempt) | P |
|---|---|---|---|---|---|
| 24-27% | 432 | 5075 | 0.0567 | 1.0479 | 1.1046 |
| 27-30% | 245 | 3020 | 0.0899 | 1.0306 | 1.1205 |
| 30-34% | 131 | 1482 | 0.0965 | 1.0233 | 1.1198 |
| 34%+ | 31 | 390 | 0.1218 | 0.9958 | 1.1177 |

## Headline

| window | carrier-seasons | sit-out games | P | r (pooled) | gap | slope (origin) | 95% CI | placebo slope | placebo 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| 2015-16 to 2025-26 | 469 | 6156 | 1.1440 | 1.0506 | 0.0919 | 0.3304 | [0.2534, 0.4089] | -0.0607 | [-0.1516, 0.0225] |
| 2000-01 to 2014-15 | 397 | 4369 | 1.0702 | 0.9967 | 0.0551 | 0.2056 | [0.1096, 0.2941] | -0.1403 | [-0.2418, -0.0455] |
| pooled 2000-01 to 2025-26 | 866 | 10525 | 1.1134 | 1.0294 | 0.0767 | 0.2800 | [0.2174, 0.3388] | -0.0933 | [-0.1586, -0.0226] |

**Reading.** In every window the attempts that move off the carrier come back below the
league price: the pooled return on a redistributed attempt is 5-9 hundredths of a point below
P (about 4-8% of an attempt's value), while the carrier's own return on those same attempts was
at or slightly above P. The placebo -- same game-splitting, same ratio, but the carrier still
takes the shots -- lands on roughly his own return (placebo r is within 0.001-0.026 of the
"own return" line), gives a gap around zero to slightly negative, and a slope of the opposite
sign. So the positive gap and slope are produced by the absence, not by the machinery. The gap
rises across the share bands in the pooled sample and in 2000-15 (flat between the first two
bands there); in 2015-26 it is flat above 27%. The through-origin slope (gap ~ 0.28-0.33 x L
pooled/modern) is well determined; the with-intercept slope is positive but its CI includes
zero for the primary variant, and is clearly positive for the on-court variant -- the data
support "gap rises with share" more firmly when the W side is his actual on-court possessions.

**Caveats.** Sit-outs are not random: injuries, rest days and late-season shutdowns cluster on
back-to-backs and on teams out of contention, and opponent strength is not controlled. The
per-season ratio r_i is noisy when Delta is small (5%/95% gap quantiles are roughly -0.35/+0.47
per attempt); the ratio-of-sums and the >= 10 sit-out robustness line are the numbers to trust.
Team possessions are inferred as sum(p_g)/5, which undercounts by the fraction of on-court
coverage (~1-2%) identically on both sides. 2025-26 is a live, partial season.
