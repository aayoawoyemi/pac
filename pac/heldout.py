# -*- coding: utf-8 -*-
"""pac/heldout.py -- the held-out design (second design in the paper): s = 0.245 [0.204, 0.284].

Two stages:
  build    (scripts/03_build_heldout_carriers.py) needs raw lineup stints and per-game rows; writes
           data/heldout/carriers.json (739 carrier-seasons) and data/heldout/build_stats.json.
  estimate (scripts/11_heldout_design.py) reads only those two files; writes results/heldout.md.


Model under test: when a carrier (l_season >= 0.24) is absent, the shooting
possessions he used come back to teammates at  price - s*L  points per attempt.
s = 0 is the null (league-average return).

Test 1: predict OFF-game team efficiency from ON-game stint data, no fitting.
Test 3: break-even regression, intercept vs s.

Sections 1 and 3 of the original calibration report (2, 4, 5 belonged to retired designs).
"""
import os, json, sys, random
import numpy as np
from collections import defaultdict
from pac import paths

CODES = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2000, 2026)]
S_GRID = [0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.42, 0.5, 0.6, 0.8, 1.0]
L_MIN, GP_MIN, OFF_MIN = 0.24, 40, 5
STINT_TOL = 0.03      # game's stint side-pts must match box team pts within 3%
N_BOOT = 2000
rng = random.Random(7)
BUILD_STATS = os.path.join(paths.DATA, "heldout", "build_stats.json")


def side_tsa(v):
    return v[0] + v[2] + v[4] + 0.44 * v[6]


def side_pts(v):
    return v[1] + v[3] + v[5] + v[7]


def norm_gid(g):
    return str(g).zfill(10)


def build_carriers(code):
    """One record per (pid, team, season) carrier with ON/OFF measurements."""
    from pac.leaf_poss import load_leaf   # build stage only: needs raw lineup stints
    d = json.load(open(paths.raw(f"_sv_pergame_{code}.json"), encoding="utf-8"))
    rows = d["rows"]
    leaf = load_leaf(code)
    if not leaf:
        return [], None, {}
    price = sum(r["pts"] for r in rows) / sum(r["tsa"] for r in rows)

    team_games = defaultdict(dict)          # team -> gid -> {pts, tsa}
    player_games = defaultdict(dict)        # (pid, team) -> gid -> row
    for r in rows:
        tg = team_games[r["team"]].setdefault(r["gid"], {"pts": 0.0, "tsa": 0.0})
        tg["pts"] += r["pts"]
        tg["tsa"] += r["tsa"]
        player_games[(r["pid"], r["team"])][r["gid"]] = r

    stats = {"cand": 0, "off_short": 0, "on_bad_stints": 0, "kept": 0}
    out = []
    for (pid, team), games in player_games.items():
        lseason = next(iter(games.values()))["l_season"]
        if lseason is None or lseason < L_MIN or len(games) < GP_MIN:
            continue
        stats["cand"] += 1
        sched = sorted(team_games[team])
        gids = sorted(games)
        lo, hi = gids[0], gids[-1]
        window = [g for g in sched if lo <= g <= hi]
        off = [g for g in window if g not in games]
        if len(off) < OFF_MIN:
            stats["off_short"] += 1
            continue
        ipid = int(pid)
        side_p = side_t = own_p = own_t = 0.0
        n_on_used = n_on_bad = 0
        for g in gids:
            st = leaf.get(norm_gid(g))
            if not st:
                n_on_bad += 1
                continue
            # locate his side and check stint coverage against box
            hp = ap = 0.0
            sp = stx = 0.0
            side = None
            for row in st:
                h5, a5, dur, _, _, hv, av = row[:7]
                hp += side_pts(hv)
                ap += side_pts(av)
                if ipid in h5:
                    side = "h" if side is None else side
                    sp += side_pts(hv); stx += side_tsa(hv)
                elif ipid in a5:
                    side = "a" if side is None else side
                    sp += side_pts(av); stx += side_tsa(av)
            box = team_games[team][g]["pts"]
            tot = hp if side == "h" else ap if side == "a" else None
            if side is None or tot is None or abs(tot - box) > STINT_TOL * max(box, 1):
                n_on_bad += 1
                continue
            r = games[g]
            if r["tsa"] > stx + 1e-9:
                n_on_bad += 1
                continue
            side_p += sp; side_t += stx
            own_p += r["pts"]; own_t += r["tsa"]
            n_on_used += 1
        if n_on_used < 20 or side_t - own_t <= 0:
            stats["on_bad_stints"] += 1
            continue
        t_on = (side_p - own_p) / (side_t - own_t)
        L_on = own_t / side_t
        op = sum(team_games[team][g]["pts"] for g in off)
        ot = sum(team_games[team][g]["tsa"] for g in off)
        t_off = op / ot
        stats["kept"] += 1
        out.append({
            "code": code, "year": 2000 + int(code[:2]) if int(code[:2]) < 90 else 1900 + int(code[:2]),
            "pid": pid, "name": next(iter(games.values()))["name"], "team": team,
            "L": lseason, "L_on": L_on, "t_on": t_on, "t_off": t_off, "price": price,
            "own_ts": own_p / own_t, "n_on": n_on_used, "n_on_bad": n_on_bad, "n_off": len(off),
        })
    return out, price, stats


class Arr:
    """Column arrays for a carrier set; every statistic below is closed-form."""
    def __init__(self, c):
        self.n = len(c)
        self.w = np.array([x["n_off"] for x in c], float)
        L_on = np.array([x["L_on"] for x in c]); L = np.array([x["L"] for x in c])
        price = np.array([x["price"] for x in c]); t_on = np.array([x["t_on"] for x in c])
        t_off = np.array([x["t_off"] for x in c]); own = np.array([x["own_ts"] for x in c])
        # test 1: resid(s) = a - s*b
        self.a = L_on * price + (1 - L_on) * t_on - t_off
        self.b = L_on * L
        # test 3: x(s) = x0 + s*k, y = t_off - t_on
        self.y = t_off - t_on
        self.x0 = (own - price) / 2 * 100
        self.k = L / 2 * 100

    def sub(self, idx):
        o = Arr.__new__(Arr)
        o.n = len(idx)
        for f in ("w", "a", "b", "y", "x0", "k"):
            setattr(o, f, getattr(self, f)[idx])
        return o


def loss(A, s):
    r = A.a - s * A.b
    W = A.w.sum()
    return float((A.w * r * r).sum() / W), float((A.w * r).sum() / W)


def argmin_s(A):
    """(MSE arg-min, zero-mean-residual s)."""
    return (float((A.w * A.a * A.b).sum() / (A.w * A.b * A.b).sum()),
            float((A.w * A.a).sum() / (A.w * A.b).sum()))


def _moments(A):
    W = A.w.sum()
    m0 = (A.w * A.x0).sum() / W; mk = (A.w * A.k).sum() / W; my = (A.w * A.y).sum() / W
    d0 = A.x0 - m0; dk = A.k - mk; dy = A.y - my
    return dict(m0=m0, mk=mk, my=my,
                V00=(A.w * d0 * d0).sum() / W, V0k=(A.w * d0 * dk).sum() / W, Vkk=(A.w * dk * dk).sum() / W,
                C0y=(A.w * d0 * dy).sum() / W, Cky=(A.w * dk * dy).sum() / W)


def wls_intercept(A, s):
    M = _moments(A)
    var = M["V00"] + 2 * s * M["V0k"] + s * s * M["Vkk"]
    cov = M["C0y"] + s * M["Cky"]
    beta = cov / var
    return float(M["my"] - beta * (M["m0"] + s * M["mk"])), float(beta)


def intercept_crossing(A, hi=6.0):
    """s where the WLS intercept is zero.

    intercept(s) = 0  <=>  my*var(s) - cov(s)*xbar(s) = 0, a quadratic in s.
    The intercept is negative at s=0 in every set and the quadratic also has a
    spurious negative root where the fitted slope has flipped sign, so the
    calibration point is the smallest root in (0, hi]: the first s at which the
    intercept, moving up from the null, reaches zero. None if no such root.
    """
    M = _moments(A)
    q2 = M["my"] * M["Vkk"] - M["Cky"] * M["mk"]
    q1 = 2 * M["my"] * M["V0k"] - M["C0y"] * M["mk"] - M["Cky"] * M["m0"]
    q0 = M["my"] * M["V00"] - M["C0y"] * M["m0"]
    roots = np.roots([q2, q1, q0]) if abs(q2) > 1e-18 else (np.array([-q0 / q1]) if q1 != 0 else np.array([]))
    roots = [float(r.real) for r in roots if abs(r.imag) < 1e-9 and 0 < r.real <= hi]
    return min(roots) if roots else None


def boot(A, fn, n=N_BOOT):
    rs = np.random.RandomState(7)
    vals = []
    for _ in range(n):
        v = fn(A.sub(rs.randint(0, A.n, A.n)))
        if v is not None:
            vals.append(v)
    v = np.array(vals)
    return (float(v.mean()), float(v.std(ddof=1)), float(np.percentile(v, 2.5)),
            float(np.percentile(v, 97.5)), len(vals))


def fmt(x, d=5):
    return "n/a" if x is None else f"{x:.{d}f}"


def build():
    carriers, stats_all = [], defaultdict(int)
    for code in CODES:
        c, price, st = build_carriers(code)
        for k, v in st.items():
            stats_all[k] += v
        carriers += c
    with open(paths.HELDOUT_CARRIERS, "w", encoding="utf-8") as f:
        json.dump(carriers, f)
    with open(BUILD_STATS, "w", encoding="utf-8") as f:
        json.dump(dict(stats_all), f, indent=1)
    print(f"carrier-seasons kept: {len(carriers)}  stats: {dict(stats_all)}")


def estimate():
    with open(paths.HELDOUT_CARRIERS, encoding="utf-8") as f:
        carriers = json.load(f)
    with open(BUILD_STATS, encoding="utf-8") as f:
        stats_all = defaultdict(int, json.load(f))
    # every season has carriers, and price is constant within a season, so the per-season table is exact
    per_season = []
    for code in CODES:
        c = [x for x in carriers if x["code"] == code]
        if c:
            per_season.append((code, c[0]["price"], len(c), sum(x["n_off"] for x in c)))
    print(f"carrier-seasons kept: {len(carriers)}  stats: {dict(stats_all)}")
    n_off_tot = sum(x["n_off"] for x in carriers)
    n_on_tot = sum(x["n_on"] for x in carriers)

    eras = {
        "pooled": carriers,
        "2000-2014": [x for x in carriers if x["year"] <= 2014],
        "2015-2025": [x for x in carriers if x["year"] >= 2015],
    }
    A = {k: Arr(v) for k, v in eras.items()}
    noff = {k: int(A[k].w.sum()) for k in eras}

    L = []
    L.append("# Sit-out slope calibration -- blind, out-of-sample\n")
    L.append("Generated by `scripts/11_heldout_design.py` from `data/heldout/carriers.json`.\n")
    L.append("## 1. Held-out sit-out prediction (fit-free)\n")
    L.append("**Setup.** Carrier = (player, team, season) with `l_season >= 0.24` and >= 40 games appeared for that team, "
             f"seasons {CODES[0]}..{CODES[-1]} (all seasons with leaf stint files). Window = first..last appearance for that team; "
             "OFF = team games in window he did not appear in (>= 5 required). ON-game stint data: `t_on` = teammates' pts / teammates' TSA "
             "while he is on the floor (side totals over his on-floor stints minus his own box pts/TSA); `L_on` = his TSA / side TSA while on. "
             "ON games are used only when the game has stints, the stint side-points reconcile with the box team points within 3%, and his box TSA "
             "does not exceed on-floor side TSA; a carrier needs >= 20 usable ON games. `t_off` = team pts / team TSA in OFF games (box). "
             "`price` = season league pts/TSA over all per-game rows.\n")
    L.append("Prediction: `t_off_pred(s) = L_on*(price - s*L) + (1 - L_on)*t_on`. Loss = weighted MSE of `t_off_pred - t_off` across carrier-seasons, "
             "weight = number of OFF games. Nothing is fit, so every carrier is held-out for every `s`; the loss is quadratic in `s` so the arg-min is closed-form "
             "and bootstrapped by resampling carrier-seasons (2000 draws).\n")
    L.append(f"Denominators: {len(carriers)} carrier-seasons; {n_off_tot} OFF games; {n_on_tot} usable ON games. "
             f"Candidates meeting L/GP thresholds: {stats_all['cand']}; dropped for < {OFF_MIN} OFF games: {stats_all['off_short']}; "
             f"dropped for insufficient reconciled stints: {stats_all['on_bad_stints']}.\n")
    L.append("| season | price (pts/TSA) | carrier-seasons | OFF games |\n|---|---|---|---|")
    for code, price, n, n_off_s in per_season:
        L.append(f"| {code} | {price:.4f} | {n} | {n_off_s} |")
    L.append("")

    hdr = "| s | " + " | ".join(f"{k} (n={len(v)}, OFF={noff[k]})" for k, v in eras.items()) + " |"
    L.append("**Weighted MSE of `t_off_pred(s) - t_off` (pts per TSA, squared) and weighted mean residual (pred - actual):**\n")
    L.append(hdr)
    L.append("|---|" + "---|" * len(eras))
    for s in S_GRID:
        cells = []
        for k, v in eras.items():
            mse, mr = loss(A[k], s)
            cells.append(f"MSE {mse:.6f}, mean resid {mr:+.5f}")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("**Arg-min and calibration point (bootstrap over carrier-seasons, 2000 draws):**\n")
    L.append("| set | n carriers | OFF games | arg-min s (MSE) | boot SE | 95% CI | s at mean-resid = 0 | boot SE | 95% CI | RMSE at arg-min | RMSE at s=0 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, v in eras.items():
        sm, sc = argmin_s(A[k])
        bm = boot(A[k], lambda c: argmin_s(c)[0])
        bc = boot(A[k], lambda c: argmin_s(c)[1])
        rm = loss(A[k], sm)[0] ** 0.5
        r0 = loss(A[k], 0)[0] ** 0.5
        L.append(f"| {k} | {len(v)} | {noff[k]} | {sm:.3f} | {bm[1]:.3f} | [{bm[2]:.3f}, {bm[3]:.3f}] | "
                 f"{sc:.3f} | {bc[1]:.3f} | [{bc[2]:.3f}, {bc[3]:.3f}] | {rm:.5f} | {r0:.5f} |")
    L.append("")
    L.append("The MSE arg-min and the zero-mean-residual point coincide only if residual and `L_on*L` are uncorrelated across carriers; both are reported.\n")

    L.append("## 2. Break-even calibration\n")
    L.append("`rTSplus(s) = (his pts/TSA - price)/2*100 + (s/2)*L*100` (TS-point units). For each `s`, WLS of realized `(t_off - t_on)` on `rTSplus(s)` "
             "across carrier-seasons, weights = OFF games. The calibrated `s` is where the intercept crosses zero. Same carrier set and denominators as section 1.\n")
    L.append("| s | " + " | ".join(f"{k} (n={len(v)}) intercept / slope" for k, v in eras.items()) + " |")
    L.append("|---|" + "---|" * len(eras))
    for s in S_GRID:
        cells = []
        for k, v in eras.items():
            ic, be_ = wls_intercept(A[k], s)
            cells.append(f"{ic:+.5f} / {be_:+.5f}")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("| set | n carriers | OFF games | s at intercept = 0 (first root > 0) | boot SE | 95% CI | boot draws with a root in (0, 6] |")
    L.append("|---|---|---|---|---|---|---|")
    for k, v in eras.items():
        x0 = intercept_crossing(A[k])
        bx = boot(A[k], intercept_crossing)
        L.append(f"| {k} | {len(v)} | {noff[k]} | {fmt(x0, 3)} | {bx[1]:.3f} | [{bx[2]:.3f}, {bx[3]:.3f}] | {bx[4]}/{N_BOOT} |")
    L.append("")
    L.append("The intercept depends on `s` only through `x-bar(s)` and the fitted slope; with slope ~ -0.001 to -0.0017 pts/TSA per TS-point the "
             "intercept is a shallow curve in `s`, so the crossing is identified far less sharply than the direct prediction test in section 1 "
             "(the intercept at s=0.25 is -0.016 pts/TSA, i.e. the regression is not the same statistic as the prediction residual). "
             "The quadratic also has a spurious negative root (pooled ~ -1.8) where the fitted slope has changed sign; it is excluded.\n")

    txt = "\n".join(L)
    print(txt)
    with open(paths.result("heldout.md"), "w", encoding="utf-8") as f:
        f.write(txt)


if __name__ == "__main__":
    estimate()
