# PAC — the replacement price of a scoring possession

**What a possession returns when somebody else takes it, measured instead of assumed.**

Every load-adjusted scoring statistic in basketball values an attempt against league-average
efficiency. True shooting added, relative true shooting, points above average: each one asserts
that if a replacement had taken the shot, it would have returned the league rate. Nobody measured
whether that is true.

This repository measures it.

```
s = 0.245        95% CI [0.204, 0.284]
```

For every 10 percentage points of his team's offense a player carries, the shots that replace him
come back about 2.5 points per 100 worse than league average.

Author: Ayomide Awoyemi (LUMA Basketball Research, court-share.com/luma).
Data: NBA play-by-play, 1997-98 through 2025-26. 688,049 player-games, 198,480 lineup stints.
All public sources. No tracking data.

---

## The formula

```
L        = TSA / P                      season share of on-court possessions
TSA      = FGA + 0.44 * FTA             shooting possessions consumed
price(L) = 2*lgTS - s*L                 measured return on a redistributed possession
PAC      = PTS - TSA * price(L)         Points Above Cost
```

Break-even falls out by setting PAC to zero:

```
rTS = -(s/2) * L
```

At 35% share that is -4.4 true shooting points. A player carrying a third of his team's offense
breaks even while shooting below league average, because the shots that replace him are worse.

`L` is the **season** share, never the night's share. The price is a property of the player's
role; the quantity is per game. Pricing per night breaks the identity that per-game values sum to
season values.

---

## What the estimand is, and what it is not

This is **not** the usage-efficiency skill curve. That asks what happens to *his* efficiency when
he carries more load. Oliver, Witus, and the usage-versus-efficiency literature answer that
question.

This is **not** the optimal-stopping question either. That asks what a possession is worth if he
passes and the clock keeps running. Goldman and Rao, Skinner, and Sandholtz answer that one.

This asks a third thing: **he is absent for the game. What does the team recover on the
possessions his teammates must now absorb, as a function of how much he was carrying?** The
population being measured is the teammates, not the star.

---

## Reproduce the coefficient

```bash
# the headline number, held-out prediction, 739 carriers and 10,676 absences
python scripts/_calib_sitout.py

# pooled WLS through the origin, plus era splits and the placebo
python scripts/_sitout_slope.py 2000 2025
python scripts/_sitout_slope.py 2000 2025 --placebo

# the in-game rest design and its placebo
python scripts/_ingame_slope.py 2000 2025
python scripts/_ingame_slope.py 2000 2025 --placebo

# an independent implementation written without sight of the numbers above
python scripts/_blind_sitout.py

# build PAC for any season at the measured price
python scripts/_sv_pergame.py --years 2025
```

`scripts/_sv_pergame.py` carries `SLOPE = 0.25` and a provenance header. It writes
`_sv_pergame_{code}.json` per player-game and `_sv_season_{code}.json` per player-season.

Raw play-by-play is not redistributed here. The scripts read `nba_data_raw/nbastats_{year}.csv`
and the stint files built from it; see `docs/DATA.md`.

---

## The evidence

| design | data | events | s | 95% CI |
|---|---|---|---|---|
| **held-out prediction** | stints + box | 739 carrier-seasons, 10,676 absences | **0.245** | [0.204, 0.284] |
| 10-fold CV, 20 shuffles | same | same | 0.233 (sd .006) | folds [0.214, 0.251] |
| leave-one-season-out | same | 26 folds | 0.22-0.24 | beats 0 in 23/26 |
| pooled WLS through origin | same | same | 0.233 | [0.192, 0.275] |
| independent reimplementation | stints + box | 866 carriers, 10,525 absences | 0.28 | [0.22, 0.34] |
| in-game rest | stints | 1,194 carriers, 2.2M bench attempts | 0.324 | [0.293, 0.353] |
| teammate-absence instrument | box, 480,013 player-games | — | k = -0.30 TS/share | [-0.38, -0.22] |
| next-season team efficiency | box, 832 pairs | — | 0.27 | flat |
| **placebo**, fake absences from games played | stints + box | 5,677 | **0.004** | [-0.05, 0.06] |
| **placebo**, random split of own stints | stints | 1,194 | **-0.066** | [-0.09, -0.03] |

### The held-out curve

Nothing is fitted. For each candidate price, predict team points per attempt in the games he sat,
using only teammates' on-court rates, and score against what happened.

| s | weighted MSE x 1e-3 |
|---|---|
| 0.00 | 2.01 |
| 0.10 | 1.76 |
| 0.20 | 1.64 |
| **0.245** | **1.62** |
| 0.30 | 1.64 |
| 0.42 | 1.82 |
| 0.50 | 2.04 |
| 0.60 | 2.43 |

**Zero is rejected. So is 0.50.** Mean residual crosses zero at 0.234.

### Functional form is checked, not assumed

Price minus observed return per attempt, by usage band:

| band | absence design | in-game | placebo |
|---|---|---|---|
| 24-27% | .042 | .066 | .004 |
| 27-30% | .080 | .110 | .016 |
| 30-34% | .099 | .133 | -.008 |
| 34%+ | .098 | .148 | -.075 |

Monotone, no bend at the top.

### The naive regression has the wrong sign

Regressing efficiency on usage directly gives **+0.04**. Hot nights get more shots, so the raw
scatter everyone runs is selection. Instrumenting a player's share with his absent teammates'
attempts across 480,013 player-games gives **-0.30 [-0.38, -0.22]**. First stage: 15 absent
teammate attempts raise your share by about 0.9 points.

### Two designs, same carriers, different counterfactuals

The absence and rest designs share **all 739 carriers**, yet per-carrier implied slopes correlate
only **0.346** and the pooled estimates differ, **0.233 against 0.301** on that identical sample.
A shared confounder moves both together; it cannot produce a stable wedge on identical units.
Absence possessions go to the rotation, rest possessions go to the bench unit. PAC's counterfactual
is the first, so 0.245 is the matched estimate and 0.324 is an upper bound.

---

## What the price does to the leaderboard

| player-season | PPG rank | TS Add rank | PAC rank | of |
|---|---|---|---|---|
| Allen Iverson 2001-02 | 1 | **214** | **60** | 216 |
| Russell Westbrook 2016-17 | 1 | 92 | 15 | 240 |
| DeMar DeRozan 2016-17 | 5 | 107 | 28 | 240 |
| Jaylen Brown 2025-26 | 4 | 168 | 33 | 236 |
| Tyson Chandler 2011-12 | 102 | 4 | 5 | 179 |
| Terry Rozier 2022-23 | 39 | 233 | 230 | 234 |

DeRozan's nine Toronto seasons: **TS Add -132 points, PAC +730.** Same box scores.

Four of 218 player-seasons above 30% usage fall below break-even: T-Mac 2007-08, Kobe 2015-16,
John Wall 2020-21, Michael Jordan 2001-02.

---

## Every statistic implies a price

Recovered by regressing each target on true-shooting-added and the load term as two free
variables. This is what each statistic *pays* for load.

| statistic | implied s | inside [0.20, 0.28] |
|---|---|---|
| TS Add / relative TS / points above average | 0.00 | no, by assumption |
| net regularized plus-minus | 0.30 | marginal |
| offensive RAPM | 0.36-0.62 | no |
| offensive box plus-minus | 0.77 | no |
| **this work** | **0.245** | by measurement |

The counterfactual price is 0.25. Offensive impact pays about 0.45 for load because usage carries
non-scoring value — creation, gravity, foul drawing — that this deliberately does not price.

**The coefficient was never chosen by validation against an impact target.** Sweeping `s` against
impact rewards load monotonically and would have produced a larger number. Measure on outcomes,
then validate, in that order.

---

## Limitations, stated plainly

1. **Absences are not random.** Injury type, rest, tanking, opponent game-planning. The in-game
   design uses no absences at all; the placebo returns 0.004; the instrument uses teammates'
   absences rather than his.
2. **Teams may adapt to anticipated absences.** Not yet tested. The planned test compares
   announced absences against same-day scratches.
3. **Linearity is checked by four usage bands**, which is thin for a curvature claim.
4. **The price is a league average conditional on share**, not roster-specific. Per-player
   estimates have 5 to 15 absences each and are noise. Same simplification replacement level makes
   in WAR.
5. **PAC is a scoring column, not a value metric.** No defense, no passing, no gravity. Jaylen
   Brown won 2024 Finals MVP while ranking third on his own team in PAC.
6. **`FGA + 0.44*FTA` is an approximation.** Rebuilding the denominator from play-by-play shot
   events is the leading robustness check.
7. **Gross points only.** The opponent's next possession is not netted, and offensive-rebound
   continuations are not modelled.
8. 2025-26 on-court coverage is 81% of games, so pace from summed on-court possessions is
   unreliable that season.

---

## A retraction, kept public on purpose

An earlier version of this work reported **s = 0.42**, computed in a notebook and never saved as a
script. On re-verification it could not be reproduced by any design: three rebuilds, an
independent reimplementation, and a held-out calibration. It sits outside every interval above.
Likely causes were absences counted outside the player's tenure window, and share measured against
team attempts rather than on-court possessions, roughly a 1.4x unit mismatch.

**Do not reuse 0.42.**

The rule that came out of it, and that governs this repository: any coefficient that reaches a
leaderboard needs a script that regenerates it from raw files, a placebo, a held-out prediction
curve, and an independent reimplementation by someone given the data and the question but not the
number. Notebook results are provisional until the script exists.

---

## Citation

```
Awoyemi, A. (2026). The replacement price of a scoring possession in the NBA.
LUMA Basketball Research. https://github.com/aayoawoyemi/pac
```

## License

Code MIT. Derived results free to use with attribution.
