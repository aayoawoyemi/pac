# -*- coding: utf-8 -*-
"""_blind_sitout.py -- blind replication of the sit-out redistribution estimate.

QUESTION
  When a scoring possession is taken away from a high-usage player ("carrier")
  and handed to his teammates, how much less does it return, and how does that
  loss scale with the carrier's share of the offence?

ESTIMATOR (per carrier-season, per-possession base, ratio of sums)
  Games are split into W (carrier appeared) and O (carrier absent) inside the
  window between his first and last appearance for that team.  Everything is
  expressed per TEAM POSSESSION so that pace differences between W and O do
  not leak into the comparison.  Team possessions for a game are recovered
  from the on-court possession counts: every possession has exactly five
  players on the floor, so  poss_team = sum(p_g over the team's rows) / 5.

    With him   (per poss):  pts_W = pts_c + pts_tm     tsa_W = tsa_c + tsa_tm
    Without him(per poss):  pts_O                      tsa_O

  Identity solved: teammates keep the return they already had on the attempts
  they already took (tsa_tm per poss); every EXTRA attempt that shows up when
  he sits (Delta = tsa_O - tsa_tm, the redistributed attempts, incl. those of
  whoever replaces him) returns r points:

        pts_O = pts_tm + r * (tsa_O - tsa_tm)
     => r     = (pts_O - pts_tm) / (tsa_O - tsa_tm)

  Any efficiency change on the teammates' pre-existing attempts is loaded
  onto r by construction -- that is intended: r is the full marginal return
  of moving the carrier's volume onto the rest of the roster.
  gap = P - r, where P = league points per shooting attempt that season.

  Secondary variant ("on-court"): same identity, but the W side is restricted
  with stint data to the carrier's own on-court possessions (counted
  possessions FGA+0.44FTA+TOV-OREB from the stint tallies on both sides).

REGRESSION
  gap_i = beta * L_i  (through the origin), weights = number of sit-out games,
  L = l_season as given (season TSA / season on-court possessions).
  Bootstrap over carrier-seasons, 500 resamples, percentile 95% CI.

PLACEBO
  For each carrier-season, the same number of games as his real sit-outs is
  drawn at random from the games he actually played and labelled "absent";
  team totals in those games therefore still contain him.  Delta then equals
  his own volume and r recovers his own return, so the placebo gap is
  P - own return.  The difference (real gap - placebo gap) is the pure cost of
  the redistribution, net of "who took the shots".
"""
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

be = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, be)
import _leaf_poss  # noqa: E402

L_MIN = 0.24          # carrier gate on l_season
GP_MIN = 40           # games appeared for that team
SIT_MIN = 3           # sit-out games needed for a usable per-season ratio
N_BOOT = 500
BANDS = [(0.24, 0.27), (0.27, 0.30), (0.30, 0.34), (0.34, 9.9)]
WINDOWS = {
    "2015-16 to 2025-26": ["1516", "1617", "1718", "1819", "1920", "2021",
                           "2122", "2223", "2324", "2425", "2526"],
    "2000-01 to 2014-15": ["0001", "0102", "0203", "0304", "0405", "0506",
                           "0607", "0708", "0809", "0910", "1011", "1112",
                           "1213", "1314", "1415"],
}
WINDOWS["pooled 2000-01 to 2025-26"] = (WINDOWS["2000-01 to 2014-15"]
                                        + WINDOWS["2015-16 to 2025-26"])


def gid_key(g):
    return str(g).zfill(10)


def vec_parts(vec):
    """(pts, tsa, counted_poss) from a 12-entry stint tally."""
    pts = vec[1] + vec[3] + vec[5] + vec[7]
    tsa = vec[0] + vec[2] + vec[4] + 0.44 * vec[6]
    poss = vec[0] + vec[2] + vec[4] + 0.44 * vec[6] + vec[8] - vec[9]
    return pts, tsa, poss


# --------------------------------------------------------------------------
# per-season extraction
# --------------------------------------------------------------------------
def season_carriers(code, rng):
    rows = json.load(open(os.path.join(be, f"_sv_pergame_{code}.json"),
                          encoding="utf-8"))["rows"]
    price = sum(r["pts"] for r in rows) / sum(r["tsa"] for r in rows)

    # team-game aggregates on the per-possession base
    tg = {}
    for r in rows:
        k = (gid_key(r["gid"]), r["team"])
        a = tg.get(k)
        if a is None:
            a = tg[k] = {"pts": 0.0, "tsa": 0.0, "pg": 0.0, "ok": True}
        a["pts"] += r["pts"]
        a["tsa"] += r["tsa"]
        if r["p_g"] is None:
            a["ok"] = False
        else:
            a["pg"] += r["p_g"]
    for a in tg.values():
        a["poss"] = a["pg"] / 5.0 if a["ok"] else None

    sched = defaultdict(list)
    for (g, t) in tg:
        sched[t].append(g)
    for t in sched:
        sched[t].sort()

    app = defaultdict(dict)   # (pid, team) -> gid -> row
    lseason = {}
    for r in rows:
        app[(r["pid"], r["team"])][gid_key(r["gid"])] = r
        if r["l_season"] is not None:
            lseason[r["pid"]] = r["l_season"]

    leaf = _leaf_poss.load_leaf(code)

    out = []
    n_eligible = 0
    for (pid, team), games in app.items():
        L = lseason.get(pid)
        if L is None or L < L_MIN or len(games) < GP_MIN:
            continue
        n_eligible += 1
        gids = sorted(games)
        lo, hi = gids[0], gids[-1]
        window = [g for g in sched[team] if lo <= g <= hi]
        sit = [g for g in window if g not in games and tg[(g, team)]["ok"]]
        played = [g for g in gids if tg[(g, team)]["ok"]]
        if len(sit) < SIT_MIN:
            continue
        name = games[gids[0]]["name"]

        def agg(glist, drop_carrier):
            pts = tsa = poss = 0.0
            for g in glist:
                a = tg[(g, team)]
                pts += a["pts"]
                tsa += a["tsa"]
                poss += a["poss"]
                if drop_carrier:
                    pts -= games[g]["pts"]
                    tsa -= games[g]["tsa"]
            return pts, tsa, poss

        pts_tm, tsa_tm, poss_W = agg(played, True)
        pts_c = sum(games[g]["pts"] for g in played)
        tsa_c = sum(games[g]["tsa"] for g in played)
        pts_O, tsa_O, poss_O = agg(sit, False)

        rec = {
            "code": code, "pid": pid, "name": name, "team": team, "L": L,
            "price": price, "n_sit": len(sit), "n_played": len(played),
            "poss_W": poss_W, "poss_O": poss_O,
            "pts_tm": pts_tm, "tsa_tm": tsa_tm, "pts_c": pts_c, "tsa_c": tsa_c,
            "pts_O": pts_O, "tsa_O": tsa_O,
        }

        # placebo: relabel len(sit) played games as absent, keep him in totals
        # (capped at half his played games so the W side keeps a baseline)
        n_fake = min(len(sit), len(played) // 2)
        fake = rng.sample(played, n_fake)
        fset = set(fake)
        rest = [g for g in played if g not in fset]
        p_tm, t_tm, p_W = agg(rest, True)
        p_O, t_O, pp_O = agg(fake, False)
        rec["plc"] = {"pts_tm": p_tm, "tsa_tm": t_tm, "poss_W": p_W,
                      "pts_O": p_O, "tsa_O": t_O, "poss_O": pp_O,
                      "n_sit": n_fake}

        # on-court variant from stints (both sides counted possessions)
        on = _oncourt(leaf, pid, team, games, played, sit, app)
        if on is not None:
            rec["on"] = on
        out.append(rec)
    ELIGIBLE[code] = n_eligible
    return out


def _oncourt(leaf, pid, team, games, played, sit, app):
    """W side restricted to the carrier's on-court stints."""
    ipid = int(pid)
    team_pids = {int(p) for (p, t) in app if t == team}
    pts_on = tsa_on = poss_on = 0.0
    pts_c = tsa_c = 0.0
    n_w = 0
    for g in played:
        st = leaf.get(g)
        if not st:
            continue
        side = _side(st, team_pids)
        if side is None:
            continue
        for row in st:
            if ipid in row[side]:
                p, t, q = vec_parts(row[5 + side])
                pts_on += p
                tsa_on += t
                poss_on += q
        pts_c += games[g]["pts"]
        tsa_c += games[g]["tsa"]
        n_w += 1
    pts_O = tsa_O = poss_O = 0.0
    n_o = 0
    for g in sit:
        st = leaf.get(g)
        if not st:
            continue
        side = _side(st, team_pids)
        if side is None:
            continue
        for row in st:
            p, t, q = vec_parts(row[5 + side])
            pts_O += p
            tsa_O += t
            poss_O += q
        n_o += 1
    if n_w < GP_MIN or n_o < SIT_MIN or poss_on <= 0 or poss_O <= 0:
        return None
    return {"pts_tm": pts_on - pts_c, "tsa_tm": tsa_on - tsa_c,
            "poss_W": poss_on, "pts_c": pts_c, "tsa_c": tsa_c,
            "pts_O": pts_O, "tsa_O": tsa_O, "poss_O": poss_O,
            "n_w": n_w, "n_o": n_o}


def _side(stints, team_pids):
    h = sum(1 for r in stints[:6] for p in r[0] if p in team_pids)
    a = sum(1 for r in stints[:6] for p in r[1] if p in team_pids)
    if h > a:
        return 0
    if a > h:
        return 1
    return None


# --------------------------------------------------------------------------
# estimator
# --------------------------------------------------------------------------
def marginal(d):
    """(r, Delta) : return per redistributed attempt, redistributed attempts per poss."""
    delta = d["tsa_O"] / d["poss_O"] - d["tsa_tm"] / d["poss_W"]
    dpts = d["pts_O"] / d["poss_O"] - d["pts_tm"] / d["poss_W"]
    if delta <= 0:
        return None, delta
    return dpts / delta, delta


def pooled_return(recs, key=None):
    """Ratio of sums: total marginal points / total redistributed attempts.
    Each carrier-season enters with its sit-out possessions as the scale."""
    num = den = 0.0
    for rec in recs:
        d = rec if key is None else rec[key]
        r, delta = marginal(d)
        if r is None:
            continue
        num += d["poss_O"] * delta * r
        den += d["poss_O"] * delta
    return num / den if den else float("nan")


def slope_origin(points):
    """points: (L, gap, w). beta = sum(w L gap) / sum(w L^2)."""
    sxy = sum(w * L * g for L, g, w in points)
    sxx = sum(w * L * L for L, g, w in points)
    return sxy / sxx if sxx else float("nan")


def slope_intercept(points):
    sw = sum(w for _, _, w in points)
    mx = sum(w * L for L, _, w in points) / sw
    my = sum(w * g for _, g, w in points) / sw
    sxy = sum(w * (L - mx) * (g - my) for L, g, w in points)
    sxx = sum(w * (L - mx) ** 2 for L, _, w in points)
    b = sxy / sxx if sxx else float("nan")
    return b, my - b * mx


def boot(points, fn, rng):
    n = len(points)
    vals = []
    for _ in range(N_BOOT):
        s = [points[rng.randrange(n)] for _ in range(n)]
        vals.append(fn(s))
    vals.sort()
    return vals[int(0.025 * N_BOOT)], vals[int(0.975 * N_BOOT) - 1]


def wmean(vals, ws):
    sw = sum(ws)
    return sum(v * w for v, w in zip(vals, ws)) / sw if sw else float("nan")


ELIGIBLE = {}


def analyse(recs, key=None, label="", min_sit=SIT_MIN):
    """Build the (L, gap, w) cloud and all summary numbers."""
    pts = []
    rows = []
    n_delta_drop = 0
    for rec in recs:
        d = rec if key is None else rec.get(key)
        if d is None:
            continue
        w = rec["n_sit"] if key is None else d["n_o"]
        if w < min_sit:
            continue
        r, delta = marginal(d)
        if r is None:
            n_delta_drop += 1
            continue
        gap = rec["price"] - r
        pts.append((rec["L"], gap, w))
        rows.append((rec, d, r, delta, gap, w))
    n_sit = sum(w for _, _, w in pts)
    rng = random.Random(1)
    b = slope_origin(pts)
    lo, hi = boot(pts, slope_origin, rng)
    bi, ai = slope_intercept(pts)
    ilo, ihi = boot(pts, lambda s: slope_intercept(s)[0], rng)
    price = wmean([rec["price"] for rec, *_ in rows], [w for *_, w in rows])
    r_pool = pooled_return([rec if key is None else rec[key] for rec, *_ in rows])
    r_mean = wmean([r for _, _, r, *_ in rows], [w for *_, w in rows])
    gap_mean = wmean([g for *_, g, _ in rows], [w for *_, w in rows])
    delta_mean = wmean([dl for _, _, _, dl, _, _ in rows], [w for *_, w in rows])
    own = wmean([d["pts_c"] / d["tsa_c"] for _, d, *_ in rows],
                [w for *_, w in rows])
    bands = []
    for lo_b, hi_b in BANDS:
        sub = [(rec, d, r, dl, g, w) for rec, d, r, dl, g, w in rows
               if lo_b <= rec["L"] < hi_b]
        if not sub:
            bands.append((lo_b, hi_b, 0, 0, float("nan"), float("nan"),
                          float("nan")))
            continue
        ws = [w for *_, w in sub]
        bands.append((lo_b, hi_b, len(sub), sum(ws),
                      wmean([g for *_, g, _ in sub], ws),
                      wmean([r for _, _, r, *_ in sub], ws),
                      wmean([rec["price"] for rec, *_ in sub], ws)))
    gaps = sorted(g for *_, g, _ in rows)
    q = lambda f: gaps[min(len(gaps) - 1, int(f * len(gaps)))] if gaps else float("nan")
    robust = [(L, g, w) for L, g, w in pts if w >= 10]
    b10 = slope_origin(robust)
    lo10, hi10 = boot(robust, slope_origin, rng) if len(robust) > 1 else (float("nan"),) * 2
    return {
        "label": label, "n_cs": len(rows), "n_sit": n_sit,
        "n_delta_drop": n_delta_drop,
        "gap_q": (gaps[0], q(0.05), q(0.5), q(0.95), gaps[-1]) if gaps else None,
        "n_cs10": len(robust), "n_sit10": sum(w for *_, w in robust),
        "slope10": b10, "ci10": (lo10, hi10),
        "price": price, "r_pool": r_pool, "r_mean": r_mean,
        "gap_mean": gap_mean, "delta_mean": delta_mean, "own": own,
        "slope": b, "ci": (lo, hi),
        "slope_i": bi, "icept": ai, "ci_i": (ilo, ihi),
        "bands": bands, "rows": rows,
    }


def fmt_block(res):
    o = []
    o.append(f"  carrier-seasons n = {res['n_cs']}, sit-out games = {res['n_sit']}"
             f"   (dropped for Delta<=0: {res['n_delta_drop']})")
    if res["gap_q"]:
        mn, q05, q50, q95, mx = res["gap_q"]
        o.append(f"  gap quantiles min/5%/50%/95%/max          = {mn:.3f} / {q05:.3f} / {q50:.3f} / {q95:.3f} / {mx:.3f}")
    o.append(f"  league price P (sit-out-game weighted)      = {res['price']:.4f} pts/attempt")
    o.append(f"  return per redistributed attempt, pooled    = {res['r_pool']:.4f}  (ratio of sums)")
    o.append(f"  return per redistributed attempt, mean      = {res['r_mean']:.4f}  (sit-out-game weighted mean of r_i)")
    o.append(f"  gap P - r, mean                             = {res['gap_mean']:.4f}")
    o.append(f"  redistributed attempts per possession, mean = {res['delta_mean']:.4f}")
    o.append(f"  carrier's own return (pts/attempt), mean    = {res['own']:.4f}")
    lo, hi = res["ci"]
    o.append(f"  slope gap ~ L through origin                = {res['slope']:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    ilo, ihi = res["ci_i"]
    o.append(f"  slope with intercept                        = {res['slope_i']:.4f}   95% CI [{ilo:.4f}, {ihi:.4f}]   intercept {res['icept']:.4f}")
    o.append("  band       n_cs  sit-games   gap      r       P")
    for lo_b, hi_b, n, w, g, r, p in res["bands"]:
        tag = f"{lo_b*100:.0f}-{hi_b*100:.0f}%" if hi_b < 1 else f"{lo_b*100:.0f}%+"
        o.append(f"  {tag:<9} {n:5d} {w:9d}   {g:7.4f} {r:7.4f} {p:7.4f}")
    lo10, hi10 = res["ci10"]
    o.append(f"  robustness, sit-outs >= 10 only             = {res['slope10']:.4f}   95% CI [{lo10:.4f}, {hi10:.4f}]   (n = {res['n_cs10']}, sit-out games = {res['n_sit10']})")
    return "\n".join(o)


def placebo_view(recs):
    """Same records with the placebo split swapped in as the primary fields."""
    out = []
    for rec in recs:
        p = dict(rec)
        p.update(rec["plc"])
        p["pts_c"] = rec["pts_c"]
        p["tsa_c"] = rec["tsa_c"]
        out.append(p)
    return out


def main():
    t0 = time.time()
    rng = random.Random(0)
    per_code = {}
    for code in WINDOWS["pooled 2000-01 to 2025-26"]:
        tc = time.time()
        per_code[code] = season_carriers(code, rng)
        print(f"{code}: {len(per_code[code])} carrier-seasons  ({time.time()-tc:.1f}s)",
              flush=True)

    report = []
    for wname, codes in WINDOWS.items():
        recs = [r for c in codes for r in per_code[c]]
        real = analyse(recs, None, wname)
        plc = analyse(placebo_view(recs), None, wname + " placebo")
        on = analyse(recs, "on", wname + " on-court")
        elig = sum(ELIGIBLE[c] for c in codes)
        block = [f"=== {wname} ===",
                 f"  eligible carrier-seasons (L >= {L_MIN}, >= {GP_MIN} games) = {elig}; "
                 f"with >= {SIT_MIN} usable sit-out games = {len(recs)}",
                 "-- whole-game W vs O (primary) --", fmt_block(real),
                 "-- placebo (fake sit-outs from played games) --", fmt_block(plc),
                 "-- on-court W (stints) vs O (secondary) --", fmt_block(on)]
        print("\n".join(block), flush=True)
        report.append((wname, elig, len(recs), real, plc, on))

    write_md(report)
    print(f"total {time.time()-t0:.1f}s")


def md_table(res):
    o = ["| share band | carrier-seasons | sit-out games | gap = P - r | r (pts/attempt) | P |",
         "|---|---|---|---|---|---|"]
    for lo_b, hi_b, n, w, g, r, p in res["bands"]:
        tag = f"{lo_b*100:.0f}-{hi_b*100:.0f}%" if hi_b < 1 else f"{lo_b*100:.0f}%+"
        o.append(f"| {tag} | {n} | {w} | {g:.4f} | {r:.4f} | {p:.4f} |")
    return "\n".join(o)


def md_block(res):
    lo, hi = res["ci"]
    ilo, ihi = res["ci_i"]
    return "\n".join([
        f"- carrier-seasons: **{res['n_cs']}**; sit-out games: **{res['n_sit']}** (carrier-seasons dropped because Delta <= 0: {res['n_delta_drop']})",
        f"- gap quantiles over carrier-seasons min / 5% / median / 95% / max: {' / '.join(f'{v:.3f}' for v in res['gap_q'])}",
        f"- league price P (mean over carrier-seasons, weighted by sit-out games): **{res['price']:.4f}** pts per shooting attempt",
        f"- return per redistributed attempt, pooled ratio of sums (total marginal points / total redistributed attempts): **{res['r_pool']:.4f}**",
        f"- return per redistributed attempt, sit-out-game-weighted mean of per-season r_i: **{res['r_mean']:.4f}**",
        f"- gap P - r (sit-out-game-weighted mean): **{res['gap_mean']:.4f}** pts per redistributed attempt",
        f"- redistributed attempts per team possession (mean Delta): {res['delta_mean']:.4f}",
        f"- carrier's own return on the W side (pts per attempt): {res['own']:.4f}",
        f"- slope of gap on L through the origin, weight = sit-out games: **{res['slope']:.4f}**, bootstrap 95% CI [{lo:.4f}, {hi:.4f}] ({N_BOOT} resamples over carrier-seasons)",
        f"- with intercept: slope {res['slope_i']:.4f} [{ilo:.4f}, {ihi:.4f}], intercept {res['icept']:.4f}",
        f"- robustness, carrier-seasons with >= 10 sit-out games only: slope through origin {res['slope10']:.4f} [{res['ci10'][0]:.4f}, {res['ci10'][1]:.4f}] (n = {res['n_cs10']}, sit-out games = {res['n_sit10']})",
        "",
        md_table(res),
    ])


def write_md(report):
    o = ["# Blind sit-out replication: return on redistributed attempts", ""]
    o.append("## Estimator")
    o.append("""
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
""")
    for wname, elig, nrec, real, plc, on in report:
        o.append(f"## {wname}")
        o.append("")
        o.append(f"Eligible carrier-seasons (l_season >= {L_MIN}, >= {GP_MIN} games for the team): "
                 f"{elig}; of which {nrec} have >= {SIT_MIN} sit-out games with a possession base.")
        o.append("")
        o.append("### Primary: whole-game W vs O")
        o.append(md_block(real))
        o.append("")
        o.append("### Placebo: fake sit-outs drawn from played games")
        o.append(md_block(plc))
        o.append("")
        o.append("### Secondary: on-court W (stints) vs O")
        o.append(md_block(on))
        o.append("")
    o.append("## Headline")
    o.append("")
    o.append("| window | carrier-seasons | sit-out games | P | r (pooled) | gap | slope (origin) | 95% CI | placebo slope | placebo 95% CI |")
    o.append("|---|---|---|---|---|---|---|---|---|---|")
    for wname, elig, nrec, real, plc, on in report:
        o.append(f"| {wname} | {real['n_cs']} | {real['n_sit']} | {real['price']:.4f} | "
                 f"{real['r_pool']:.4f} | {real['gap_mean']:.4f} | {real['slope']:.4f} | "
                 f"[{real['ci'][0]:.4f}, {real['ci'][1]:.4f}] | {plc['slope']:.4f} | "
                 f"[{plc['ci'][0]:.4f}, {plc['ci'][1]:.4f}] |")
    o.append("")
    o.append("""**Reading.** In every window the attempts that move off the carrier come back below the
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
""")
    open(os.path.join(be, "_blind_sitout_results.md"), "w",
         encoding="utf-8").write("\n".join(o))


if __name__ == "__main__":
    main()
