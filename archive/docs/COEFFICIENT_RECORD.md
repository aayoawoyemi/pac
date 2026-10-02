# Coefficient record

Extracted from the project master document, 2026-09-21.

# PART II — THE COEFFICIENT

## II-1. The answer

**s = 0.245, bootstrap 95% interval [0.204, 0.284].** Reported as 0.25 [0.20, 0.28].

Plainly: **for every 10 percentage points of offense a player carries, the shots that replace him
come back about 2.5 points per 100 worse.** It is a penalty on the baseline, not on the player.
The more you carry, the worse the alternative to you, so the bar you are measured against drops.

## II-2. The designs

| # | design | data | events | s | 95% CI | script |
|---|---|---|---|---|---|---|
| 1 | **held-out prediction** — predict team pts/attempt in games he sat from teammates' on-court rates and the priced return; fit-free, weighted MSE | stints + box | 739 carrier-seasons, 10,676 DNPs, 2000-01→2025-26 | **0.245** | [0.204, 0.284] | `_calib_sitout.py` (blind agent) |
| 1b | 10-fold CV, 20 shuffles | same | same | 0.233, sd .006 | folds [0.214, 0.251] | kernel |
| 1c | leave-one-season-out | same | 26 folds | 0.22–0.24 | beats 0.42 in 20/26, beats 0 in 23/26 | kernel |
| 2 | sit-out WLS through origin | same | same | 0.233 | [0.192, 0.275] | `_sitout_slope.py` |
| 2a | split 2000-14 / 2015-25 | same | 320 / 419 | 0.181 / 0.266 | [.12,.24] / [.21,.32] | `_sitout_slope.py` |
| 3 | blind replication, independent design | stints + box | 866 carriers, 10,525 DNPs | 0.28 | [0.22, 0.34] | `_blind_sitout.py` |
| 4 | **in-game rest** — absent side is bench stints in games he played | stints | 1,194 carriers, 2.2M bench attempts | 0.324 | [0.293, 0.353] | `_ingame_slope.py` |
| 5 | teammate-absence IV | box, 480,013 player-games | absent teammates' attempts as instrument | k = −0.30 TS/share | [−0.38, −0.22] | kernel, reproduced exactly |
| 6 | next-season team efficiency | box, 832 pairs | — | 0.27 | flat | `_calib_events.py` |
| 7 | roster events (trades, season-ending injuries) | box | 278 events | 0.01 | [−0.14, 0.16] | `_calib_events.py` |
| 8 | **placebo**, fake DNPs from games played | stints + box | 5,677 | **0.004** | [−0.05, 0.06] | `_sitout_slope.py --placebo` |
| 9 | **placebo**, random split of his own stints | stints | 1,194 | **−0.066** | [−0.09, −0.03] | `_ingame_slope.py --placebo` |

## II-3. The held-out curve — why this is a measurement and not a fit

For each candidate price, predict the team's points per attempt in games he sat, using only
teammates' on-court rates and the priced return. Score by weighted MSE against what happened.
Nothing is fitted; every candidate is scored on games the fit never saw.

| s | weighted MSE ×10⁻³ |
|---|---|
| 0.00 | 2.01 |
| 0.10 | 1.76 |
| 0.20 | 1.64 |
| **0.245** | **1.62** |
| 0.30 | 1.64 |
| 0.35 | 1.69 |
| 0.42 | 1.82 |
| 0.50 | 2.04 |
| 0.60 | 2.43 |

Mean residual crosses zero at 0.234. **Zero is rejected. 0.50 is already worse than zero.** Both
tails matter: a U-shaped held-out curve is the difference between measuring a parameter and
picking one.

## II-4. Non-parametric check on functional form

Price minus observed return, per attempt, by usage band:

| band | DNP design | in-game | placebo |
|---|---|---|---|
| 24–27% | .042 | .066 | .004 |
| 27–30% | .080 | .110 | .016 |
| 30–34% | .099 | .133 | −.008 |
| 34%+ | .098 | .148 | −.075 |

Monotone, does not bend at the top. **The earlier "cap the priced share at 0.34" caveat is
withdrawn.** Linearity is checked, not merely assumed, and this table is the check.

## II-5. Why the naive regression is worthless

Regressing efficiency on usage directly gives **+0.04**, the wrong sign. Hot nights get more
shots, so the raw scatter is selection. The instrument fixes it: 15 absent-teammate attempts
raise your own share by about 0.9 points (first stage), reduced form −0.27 TS, IV estimate
−0.30 [−0.38, −0.22] on 480,013 player-games. Same in both eras, −0.29 pre-2015 and −0.31 post.

This matters rhetorically as much as statistically. The scatter everyone runs on Twitter shows
usage and efficiency positively related. That scatter is the selection artifact, and naming it
is part of the paper.

## II-6. DNP versus in-game: two prices, one counterfactual

0.245 and 0.324 do not overlap, and that is a feature rather than a problem. When a carrier
rests mid-game his possessions go to **the bench unit on the floor**. When he misses a game they
go to **the rotation**, with starters absorbing and a rotation player moving up. PAC's stated
counterfactual is the second, so 0.245 is the matched estimate and 0.324 is an upper bound with
cleaner selection. Report both.

## II-7. The designs share the population, not the variation

Computed directly this session:

| quantity | value |
|---|---|
| carrier-seasons in both designs | 739, **100% of the DNP sample** |
| correlation of per-carrier implied slope | **Pearson 0.346, Spearman 0.325** |
| correlation of the per-carrier price gap | 0.362 |
| pooled WLS on the identical 739 carriers, DNP | 0.233 |
| pooled WLS on the identical 739 carriers, in-game | 0.301 |

Same players, same seasons, per-carrier estimates correlating 0.35, so roughly 88% of the
carrier-level variance in one is unshared with the other.

This turns the referee's objection into evidence. Holding the carrier set exactly fixed, the two
designs return 0.233 and 0.301. A shared confounder — injury selection, tanking, opponent
game-planning — would have to move both together. It cannot produce a stable 0.07 wedge on
identical units. The divergence is what a difference in counterfactual looks like and the
opposite of what a common artifact looks like.

Correct abstract wording: **"partially non-overlapping."** Never "do not share a failure mode."

## II-8. Era

DNP 0.18 early versus 0.27 modern, CIs touching. In-game 0.317 versus 0.330, no gradient. The
DNP gradient is probably sit-out selection, 2000s absences being more injury-heavy, rather than a
change in the price. **One pooled slope. No era parameter.** The earlier "0.54 pre-2015" claim is
withdrawn.

## II-9. Impact weight is not the price

Sweeping `s` against impact targets rewards load monotonically: split-half and year-over-year
never peak, pure oRAPM plateaus 0.42–0.60, OBPM prefers 0.8. Regressing each target on TS Add and
the load term as two free variables gives the weight each target *pays* for load:

| target | implied s | note |
|---|---|---|
| TS Add / rTS / points above average | 0.00 | by assumption |
| pure net RAPM | 0.30 | — |
| pure oRAPM (4 anchors, α=10000) | 0.36–0.62 | wide, mostly excludes 0.25 |
| OBPM | 0.77 [0.75, 0.80] | contains an explicit usage term |
| DBPM (negative control) | load coef −0.09, R² .04 | clean |

**The counterfactual price is 0.25. Offensive impact pays about 0.45 for load; net impact about
0.30.** The excess over 0.25 is the non-scoring value of usage — creation, gravity, foul drawing —
that oRAPM sees and PAC deliberately does not price. Net RAPM pays less because high-usage
scorers cost something on defense.

**0.42 was where offensive impact starts to plateau. It was never the price.**

Choosing `s` by validation against impact is the ScoreVal move: a coefficient fitted to the
target. PAC's `s` is measured on outcomes first and validated second, in that order, on purpose.

## II-10. The 0.42 retraction

A prior session reported s = 0.42 from a sit-out build computed in-kernel and never saved as a
script. On re-verification it could not be reproduced by any design: three rebuilds, one blind
replication, one held-out calibration. It sits outside every interval above.

Likely causes: sit-outs counted outside the carrier's tenure window (games he was not on the
roster, 10,511 claimed versus 6,423 same-window), and share measured on team TSA rather than
on-court possessions, roughly a 1.4× unit mismatch on the x-axis. The claimed "2015-26" window
also matches the pooled 2000-25 count almost exactly, so the original was probably a pooled
sample mislabeled.

**Process rule, now binding.** Any coefficient that reaches a leaderboard needs:

1. a script in `courtshare-backend/` that regenerates it from raw files,
2. a placebo,
3. a held-out prediction curve,
4. a blind replication by an agent given the data and the question but not the number.

In-kernel results are provisional until (1) exists. This rule would have caught 0.42 a session
earlier.

---

