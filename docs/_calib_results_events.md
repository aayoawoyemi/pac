## 2. Roster-change events (season-ending absence / trade away)

Event = player-team-season with `l_season >= 0.22`, >= 25 appearances for the team, and last appearance with >= 15 team games still to play. `before` = team pts/TSA over his appeared games (whole-game box); `after` = team pts/TSA over the remaining games he missed. Realized change = after - before. Predicted change = -L*(lambda - (price - s*L)), lambda = his pts/TSA over the before games, L = his share of *team whole-game TSA* over the before games (not the on-court `l_season`; the on-court value is only the screen). Loss = weighted MSE(predicted - realized), weight = remaining games.

- n events = 278 (pre-2015 seasons: 164, 2015-16 onward: 114); seasons 9798..2526 (29 seasons).
- Sum of weights (remaining games) = 8461; mean remaining games = 30.4; mean L (team-TSA share) = 0.152; mean lambda - price = -0.0313.
- Weighted mean realized change (after - before, pts/TSA) = +0.00725; unweighted = +0.00668; SD of realized = 0.0408 (n = 278).

| s | weighted MSE (all) | wMSE pre-2015 | wMSE 2015+ | mean predicted (weighted) | mean realized (weighted) | mean pred - real |
|---|---|---|---|---|---|---|
| 0 | 0.001409 | 0.001296 | 0.001580 | +0.00330 | +0.00725 | -0.00395 |
| 0.1 | 0.001416 | 0.001322 | 0.001557 | +0.00076 | +0.00725 | -0.00649 |
| 0.2 | 0.001440 | 0.001367 | 0.001550 | -0.00177 | +0.00725 | -0.00902 |
| 0.25 | 0.001458 | 0.001396 | 0.001552 | -0.00304 | +0.00725 | -0.01029 |
| 0.3 | 0.001481 | 0.001430 | 0.001558 | -0.00431 | +0.00725 | -0.01156 |
| 0.35 | 0.001508 | 0.001468 | 0.001569 | -0.00558 | +0.00725 | -0.01283 |
| 0.42 | 0.001553 | 0.001529 | 0.001590 | -0.00736 | +0.00725 | -0.01461 |
| 0.5 | 0.001615 | 0.001610 | 0.001624 | -0.00939 | +0.00725 | -0.01664 |
| 0.6 | 0.001708 | 0.001727 | 0.001681 | -0.01193 | +0.00725 | -0.01917 |
| 0.8 | 0.001946 | 0.002016 | 0.001842 | -0.01700 | +0.00725 | -0.02425 |
| 1.0 | 0.002254 | 0.002377 | 0.002068 | -0.02208 | +0.00725 | -0.02933 |

- Grid arg-min (all events, n = 278): s = 0. Closed-form weighted-LS arg-min (loss is quadratic in s): s* = 0.011.
- Pre-2015 (n = 164): grid arg-min s = 0, closed-form s* = -0.096. 2015+ (n = 114): grid arg-min s = 0.2, closed-form s* = 0.195.
- Bootstrap over events (2000 resamples of n = 278): SE(s*) = 0.076, 95% interval [-0.138, 0.156]; SE(grid arg-min) = 0.052, grid arg-min distribution: 0: 0.71, 0.1: 0.26, 0.2: 0.03, 0.25: 0.00
- Loss at s = 0 is 0.001409; loss at s* is 0.001409 (0.01% lower). Predicting zero change gives 0.001576. corr(predicted at s=0, realized) = +0.275 (unweighted Pearson, n = 278).
- Variant with the return term using on-court `l_season` instead of whole-game L (pred = -L*(lambda - (price - s*l_season)), n = 278): s=0: 0.001409, s=0.1: 0.001432, s=0.2: 0.001490, s=0.25: 0.001531, s=0.3: 0.001581, s=0.35: 0.001640, s=0.42: 0.001736, s=0.5: 0.001866, s=0.6: 0.002059, s=0.8: 0.002548, s=1.0: 0.003173. Grid arg-min s = 0, closed-form s* = -0.019, bootstrap SE = 0.053, 95% interval [-0.123, 0.088].

Largest-share events (sanity):

| season | player | L (team TSA) | lambda | price | before | after | realized | n before | n after |
|---|---|---|---|---|---|---|---|---|---|
| 0506 | Tracy McGrady | 0.278 | 0.991 | 1.075 | 1.051 | 1.040 | -0.0112 | 47 | 20 |
| 1617 | DeMarcus Cousins | 0.267 | 1.130 | 1.108 | 1.118 | 1.121 | +0.0037 | 55 | 25 |
| 0607 | Ray Allen | 0.253 | 1.144 | 1.088 | 1.094 | 1.100 | +0.0057 | 55 | 16 |
| 9798 | Glenn Robinson | 0.252 | 1.046 | 1.051 | 1.069 | 1.038 | -0.0307 | 56 | 24 |
| 0708 | Dwyane Wade | 0.252 | 1.099 | 1.084 | 1.070 | 1.010 | -0.0601 | 51 | 21 |
| 1415 | Carmelo Anthony | 0.251 | 1.067 | 1.071 | 1.048 | 0.993 | -0.0550 | 40 | 29 |
| 0607 | Joe Johnson | 0.249 | 1.127 | 1.088 | 1.044 | 1.073 | +0.0293 | 57 | 21 |
| 1011 | Carmelo Anthony | 0.245 | 1.100 | 1.087 | 1.158 | 1.141 | -0.0171 | 50 | 25 |

## 5. Forecasting next-season outcomes (non-load outcome)

Team-season score_s = [ sum_i TSA_i*(lambda_i - price) + s * sum_i TSA_i*L_i ] / team games, summed over every player who logged a shooting attempt for that team that season (lambda_i, TSA_i from his games with that team; L_i = `l_season`, his season on-court share; price = league pts/TSA that season). The first term (A) is team points added at s = 0 (it equals team pts - price*team TSA); the second (B) is the concentration term. Outcome = the same team id's win% and pts/TSA in season t+1 (consecutive seasons only; win decided by box-score pts). Pearson correlations.

- n team-season pairs = 832 (seasons 9798..2425 -> next season; 28 transitions).
- Reference: corr(next-year win%, this-year win%) = +0.6097 (n = 832).
- Era drift: mean B/G = 18.88 in 9798 vs 20.75 in 2425; league price = 1.051 vs 1.165 (2526). Raw pts/TSA and B both trend with era, so the raw block below is followed by a season-demeaned block (season fixed effects; next-year pts/TSA taken relative to next-year league price).

### 5a. Raw (as specified)

| s | corr(score_s, next-year win%) | corr(score_s, next-year pts/TSA) |
|---|---|---|
| 0 | +0.4585 | +0.3844 |
| 0.1 | +0.4622 | +0.4004 |
| 0.2 | +0.4651 | +0.4158 |
| 0.25 | +0.4663 | +0.4233 |
| 0.3 | +0.4674 | +0.4305 |
| 0.35 | +0.4682 | +0.4375 |
| 0.42 | +0.4691 | +0.4471 |
| 0.5 | +0.4697 | +0.4575 |
| 0.6 | +0.4699 | +0.4697 |
| 0.8 | +0.4685 | +0.4917 |
| 1.0 | +0.4648 | +0.5103 |

- Concentration term alone (B/G): corr with next-year win% = +0.1101, with next-year pts/TSA = +0.4266, with the s = 0 score (A/G) = +0.0155; corr(A/G, same-year win%) = +0.7138 (n = 832).
- Grid arg-max: s = 0.6 for next-year win%, s = 1.0 for next-year pts/TSA. Fine-grid (step 0.01, s in [-3, 5]) arg-max: s = 0.58 (win%), s = 2.86 (next-year pts/TSA).
- OLS next-year win% ~ A/G + B/G (n = 832): beta_A = +0.02091 per pt/game, beta_B = +0.01212 per (TSA*L)/game; implied s = beta_B/beta_A = +0.580. Same for next-year pts/TSA: beta_A = +0.006175, beta_B = +0.017666, implied s = +2.861.
- Bootstrap (2000 resamples of n = 832 pairs): SE(implied s, win%) = 0.188, 95% interval [0.225, 0.970]; grid arg-max (win%) distribution: 0: 0.00, 0.1: 0.01, 0.2: 0.02, 0.25: 0.03, 0.3: 0.03, 0.35: 0.06, 0.42: 0.11, 0.5: 0.18, 0.6: 0.30, 0.8: 0.22, 1.0: 0.05.
- Season-cluster bootstrap (2000 resamples of 28 seasons): SE(implied s, win%) = 0.171, 95% interval [0.248, 0.915].
- Partial corr(B/G, next-year win% | A/G) = +0.1160; partial corr(B/G, next-year pts/TSA | A/G) = +0.4557 (n = 832).

### 5b. Season-demeaned (A, B, and next-year pts/TSA - next-year price, each minus its season mean)

| s | corr(score_s, next-year win%) | corr(score_s, next-year pts/TSA rel. to price) |
|---|---|---|
| 0 | +0.4585 | +0.6008 |
| 0.1 | +0.4624 | +0.6021 |
| 0.2 | +0.4658 | +0.6028 |
| 0.25 | +0.4673 | +0.6029 |
| 0.3 | +0.4687 | +0.6029 |
| 0.35 | +0.4700 | +0.6027 |
| 0.42 | +0.4716 | +0.6023 |
| 0.5 | +0.4731 | +0.6014 |
| 0.6 | +0.4746 | +0.5998 |
| 0.8 | +0.4763 | +0.5951 |
| 1.0 | +0.4764 | +0.5885 |

- Concentration term alone (Bd/G): corr with next-year win% = +0.1392, with next-year pts/TSA rel. to price = +0.0625, with the s = 0 score (Ad/G) = +0.0203; corr(Ad/G, same-year win%) = +0.7137 (n = 832).
- Grid arg-max: s = 1.0 for next-year win%, s = 0.25 for next-year pts/TSA rel. to price. Fine-grid (step 0.01, s in [-3, 5]) arg-max: s = 0.91 (win%), s = 0.27 (next-year pts/TSA rel. to price).
- OLS next-year win% ~ Ad/G + Bd/G (n = 832): beta_A = +0.02086 per pt/game, beta_B = +0.01891 per (TSA*L)/game; implied s = beta_B/beta_A = +0.906. Same for next-year pts/TSA rel. to price: beta_A = +0.006294, beta_B = +0.001678, implied s = +0.267.
- Bootstrap (2000 resamples of n = 832 pairs): SE(implied s, win%) = 0.229, 95% interval [0.470, 1.381]; grid arg-max (win%) distribution: 0.1: 0.00, 0.2: 0.00, 0.25: 0.00, 0.3: 0.00, 0.35: 0.00, 0.42: 0.01, 0.5: 0.03, 0.6: 0.12, 0.8: 0.31, 1.0: 0.51.
- Season-cluster bootstrap (2000 resamples of 28 seasons): SE(implied s, win%) = 0.248, 95% interval [0.406, 1.370].
- Partial corr(Bd/G, next-year win% | Ad/G) = +0.1462; partial corr(Bd/G, next-year pts/TSA rel. to price | Ad/G) = +0.0629 (n = 832).

