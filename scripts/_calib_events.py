"""Blind calibration of s via (2) roster-change events and (5) next-season forecasting.

Model under test: a high-usage player's shooting possessions, when redistributed to
teammates, return  price - s*L  points per attempt.  s = 0 is the null model.

Run:  python _calib_events.py      (writes _calib_results_events.md)
"""
import glob
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

DIR = os.path.dirname(os.path.abspath(__file__))
S_GRID = [0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.42, 0.5, 0.6, 0.8, 1.0]
N_BOOT = 2000
random.seed(12345)


def start_year(code):
    y = int(code[:2])
    return 1900 + y if y >= 90 else 2000 + y


def codes():
    cs = [os.path.basename(f)[12:16] for f in glob.glob(os.path.join(DIR, "_sv_pergame_*.json"))]
    return sorted(cs, key=start_year)


def load(code):
    with open(os.path.join(DIR, f"_sv_pergame_{code}.json")) as f:
        rows = json.load(f)["rows"]
    pts = sum(r["pts"] for r in rows)
    tsa = sum(r["tsa"] for r in rows)
    return rows, pts / tsa


def wmean(xs, ws):
    sw = sum(ws)
    return sum(x * w for x, w in zip(xs, ws)) / sw


def corr(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy)


def sd(xs):
    n = len(xs)
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


# ----------------------------------------------------------------------------------
# Test 2: roster-change events
# ----------------------------------------------------------------------------------
L_MIN = 0.22
MIN_BEFORE = 25
MIN_AFTER = 15


def season_events(code, rows, price):
    team_game = defaultdict(lambda: [0.0, 0.0])          # (team,gid) -> [pts, tsa]
    sched = defaultdict(set)                             # team -> {gid}
    pg = defaultdict(list)                               # (pid,team) -> [(gid, pts, tsa)]
    lseason = {}
    name = {}
    for r in rows:
        k = (r["team"], r["gid"])
        team_game[k][0] += r["pts"]
        team_game[k][1] += r["tsa"]
        sched[r["team"]].add(r["gid"])
        pg[(r["pid"], r["team"])].append((r["gid"], r["pts"], r["tsa"]))
        lseason[r["pid"]] = r["l_season"]
        name[r["pid"]] = r["name"]
    sched = {t: sorted(g) for t, g in sched.items()}
    events = []
    for (pid, team), games in pg.items():
        if lseason[pid] < L_MIN:
            continue
        games.sort()
        last = games[-1][0]
        s = sched[team]
        idx = s.index(last)
        remaining = s[idx + 1:]
        if len(games) < MIN_BEFORE or len(remaining) < MIN_AFTER:
            continue
        bp = bt = 0.0
        pp = pt = 0.0
        for gid, p, t in games:
            bp += team_game[(team, gid)][0]
            bt += team_game[(team, gid)][1]
            pp += p
            pt += t
        ap = at = 0.0
        for gid in remaining:
            ap += team_game[(team, gid)][0]
            at += team_game[(team, gid)][1]
        if pt == 0:
            continue
        before = bp / bt
        after = ap / at
        lam = pp / pt
        L = pt / bt                                      # whole-game share of team TSA
        events.append(dict(
            code=code, pid=pid, name=name[pid], team=team, price=price,
            n_before=len(games), n_after=len(remaining),
            before=before, after=after, real=after - before,
            lam=lam, L=L, l_season=lseason[pid],
            a=-L * (lam - price),                        # predicted change at s = 0
            m=L * L,                                     # d(pred)/d(-s): return term uses whole-game L
            m2=L * lseason[pid],                         # variant: return term uses on-court l_season
        ))
    return events


def pred(e, s, k="m"):
    return e["a"] - s * e[k]


def loss(events, s, k="m"):
    ws = [e["n_after"] for e in events]
    return wmean([(pred(e, s, k) - e["real"]) ** 2 for e in events], ws)


def closed_form_s(events, k="m"):
    # weighted LS of (a - real) on m:  argmin_s sum w (a - s m - real)^2
    num = sum(e["n_after"] * (e["a"] - e["real"]) * e[k] for e in events)
    den = sum(e["n_after"] * e[k] ** 2 for e in events)
    return num / den


def grid_argmin(events, k="m"):
    return min(S_GRID, key=lambda s: loss(events, s, k))


def test2(all_rows):
    events = []
    for code, (rows, price) in all_rows.items():
        events.extend(season_events(code, rows, price))
    n = len(events)
    pre = [e for e in events if start_year(e["code"]) < 2015]
    post = [e for e in events if start_year(e["code"]) >= 2015]
    ws = [e["n_after"] for e in events]
    out = []
    out.append("## 2. Roster-change events (season-ending absence / trade away)")
    out.append("")
    out.append(f"Event = player-team-season with `l_season >= {L_MIN}`, >= {MIN_BEFORE} appearances for the team, "
               f"and last appearance with >= {MIN_AFTER} team games still to play. "
               f"`before` = team pts/TSA over his appeared games (whole-game box); `after` = team pts/TSA over the remaining games he missed. "
               f"Realized change = after - before. Predicted change = -L*(lambda - (price - s*L)), lambda = his pts/TSA over the before games, "
               f"L = his share of *team whole-game TSA* over the before games (not the on-court `l_season`; the on-court value is only the screen). "
               f"Loss = weighted MSE(predicted - realized), weight = remaining games.")
    out.append("")
    cs = sorted(all_rows, key=start_year)
    out.append(f"- n events = {n} (pre-2015 seasons: {len(pre)}, 2015-16 onward: {len(post)}); seasons {cs[0]}..{cs[-1]} ({len(cs)} seasons).")
    out.append(f"- Sum of weights (remaining games) = {sum(ws)}; mean remaining games = {sum(ws)/n:.1f}; mean L (team-TSA share) = {wmean([e['L'] for e in events], ws):.3f}; "
               f"mean lambda - price = {wmean([e['lam']-e['price'] for e in events], ws):+.4f}.")
    out.append(f"- Weighted mean realized change (after - before, pts/TSA) = {wmean([e['real'] for e in events], ws):+.5f}; "
               f"unweighted = {sum(e['real'] for e in events)/n:+.5f}; SD of realized = {sd([e['real'] for e in events]):.4f} (n = {n}).")
    out.append("")
    out.append("| s | weighted MSE (all) | wMSE pre-2015 | wMSE 2015+ | mean predicted (weighted) | mean realized (weighted) | mean pred - real |")
    out.append("|---|---|---|---|---|---|---|")
    mr = wmean([e["real"] for e in events], ws)
    for s in S_GRID:
        mp = wmean([pred(e, s) for e in events], ws)
        out.append(f"| {s} | {loss(events, s):.6f} | {loss(pre, s):.6f} | {loss(post, s):.6f} | {mp:+.5f} | {mr:+.5f} | {mp-mr:+.5f} |")
    out.append("")
    s_star = closed_form_s(events)
    out.append(f"- Grid arg-min (all events, n = {n}): s = {grid_argmin(events)}. Closed-form weighted-LS arg-min (loss is quadratic in s): s* = {s_star:.3f}.")
    out.append(f"- Pre-2015 (n = {len(pre)}): grid arg-min s = {grid_argmin(pre)}, closed-form s* = {closed_form_s(pre):.3f}. "
               f"2015+ (n = {len(post)}): grid arg-min s = {grid_argmin(post)}, closed-form s* = {closed_form_s(post):.3f}.")
    # bootstrap
    boot_cf, boot_grid = [], []
    for _ in range(N_BOOT):
        samp = [events[random.randrange(n)] for _ in range(n)]
        boot_cf.append(closed_form_s(samp))
        boot_grid.append(grid_argmin(samp))
    boot_cf.sort()
    lo, hi = boot_cf[int(0.025 * N_BOOT)], boot_cf[int(0.975 * N_BOOT)]
    out.append(f"- Bootstrap over events ({N_BOOT} resamples of n = {n}): SE(s*) = {sd(boot_cf):.3f}, 95% interval [{lo:.3f}, {hi:.3f}]; "
               f"SE(grid arg-min) = {sd(boot_grid):.3f}, grid arg-min distribution: " +
               ", ".join(f"{s}: {boot_grid.count(s)/N_BOOT:.2f}" for s in S_GRID if boot_grid.count(s)))
    # loss ratio diagnostics
    l0 = loss(events, 0)
    out.append(f"- Loss at s = 0 is {l0:.6f}; loss at s* is {loss(events, s_star):.6f} ({100*(1-loss(events, s_star)/l0):.2f}% lower). "
               f"Predicting zero change gives {wmean([e['real']**2 for e in events], ws):.6f}. "
               f"corr(predicted at s=0, realized) = {corr([e['a'] for e in events], [e['real'] for e in events]):+.3f} (unweighted Pearson, n = {n}).")
    # variant: return term uses the on-court l_season the model names, pred = -L*(lambda - (price - s*l_season))
    boot_v = []
    for _ in range(N_BOOT):
        samp = [events[random.randrange(n)] for _ in range(n)]
        boot_v.append(closed_form_s(samp, "m2"))
    boot_v.sort()
    out.append(f"- Variant with the return term using on-court `l_season` instead of whole-game L (pred = -L*(lambda - (price - s*l_season)), n = {n}): "
               + ", ".join(f"s={s}: {loss(events, s, 'm2'):.6f}" for s in S_GRID)
               + f". Grid arg-min s = {grid_argmin(events, 'm2')}, closed-form s* = {closed_form_s(events, 'm2'):.3f}, "
               f"bootstrap SE = {sd(boot_v):.3f}, 95% interval [{boot_v[int(0.025*N_BOOT)]:.3f}, {boot_v[int(0.975*N_BOOT)]:.3f}].")
    # largest events for reader sanity
    ev_sorted = sorted(events, key=lambda e: -e["L"])[:8]
    out.append("")
    out.append("Largest-share events (sanity):")
    out.append("")
    out.append("| season | player | L (team TSA) | lambda | price | before | after | realized | n before | n after |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for e in ev_sorted:
        out.append(f"| {e['code']} | {e['name']} | {e['L']:.3f} | {e['lam']:.3f} | {e['price']:.3f} | {e['before']:.3f} | {e['after']:.3f} | {e['real']:+.4f} | {e['n_before']} | {e['n_after']} |")
    out.append("")
    return out, events


# ----------------------------------------------------------------------------------
# Test 5: forecasting next-season win% with a non-load outcome
# ----------------------------------------------------------------------------------
def season_teams(code, rows, price):
    tg = defaultdict(lambda: [0.0, 0.0])                 # (team,gid) -> pts, tsa
    pt = defaultdict(lambda: [0.0, 0.0, 0.0])            # (pid,team) -> pts, tsa, l_season
    for r in rows:
        k = (r["team"], r["gid"])
        tg[k][0] += r["pts"]
        tg[k][1] += r["tsa"]
        p = pt[(r["pid"], r["team"])]
        p[0] += r["pts"]
        p[1] += r["tsa"]
        p[2] = r["l_season"]
    by_gid = defaultdict(list)
    for (team, gid), (p, t) in tg.items():
        by_gid[gid].append((team, p))
    teams = defaultdict(lambda: dict(g=0, w=0, pts=0.0, tsa=0.0, A=0.0, B=0.0))
    for gid, sides in by_gid.items():
        assert len(sides) == 2
        (t1, p1), (t2, p2) = sides
        teams[t1]["g"] += 1
        teams[t2]["g"] += 1
        teams[t1 if p1 > p2 else t2]["w"] += 1
    for (team, gid), (p, t) in tg.items():
        teams[team]["pts"] += p
        teams[team]["tsa"] += t
    for (pid, team), (p, t, l) in pt.items():
        if t == 0:
            continue
        teams[team]["A"] += t * (p / t - price)          # TSA_i * (lambda_i - price)
        teams[team]["B"] += t * l                        # TSA_i * L_i  (L_i = l_season)
    return teams


def test5(all_rows):
    cs = sorted(all_rows, key=start_year)
    per = {c: season_teams(c, *all_rows[c]) for c in cs}
    pairs = []
    for c0, c1 in zip(cs, cs[1:]):
        if start_year(c1) != start_year(c0) + 1:
            continue
        for team, d in per[c0].items():
            if team not in per[c1]:
                continue
            n = per[c1][team]
            pairs.append(dict(code=c0, team=team, A=d["A"] / d["g"], B=d["B"] / d["g"],
                              win=d["w"] / d["g"], eff=d["pts"] / d["tsa"],
                              nwin=n["w"] / n["g"], neff=n["pts"] / n["tsa"],
                              neff_rel=n["pts"] / n["tsa"] - all_rows[c1][1]))
    # season-demeaned versions (season fixed effects): B/G and raw pts/TSA both trend across eras
    by_c = defaultdict(list)
    for p in pairs:
        by_c[p["code"]].append(p)
    for c, ps in by_c.items():
        m = len(ps)
        for key in ("A", "B", "neff_rel"):
            mu = sum(p[key] for p in ps) / m
            for p in ps:
                p[key + "d"] = p[key] - mu
    n = len(pairs)
    A = [p["A"] for p in pairs]
    B = [p["B"] for p in pairs]
    nwin = [p["nwin"] for p in pairs]
    neff = [p["neff"] for p in pairs]
    win = [p["win"] for p in pairs]

    def score(s, ps, ak, bk):
        return [p[ak] + s * p[bk] for p in ps]

    def ols_implied(ps, ykey, ak, bk):
        xa = [p[ak] for p in ps]
        xb = [p[bk] for p in ps]
        y = [p[ykey] for p in ps]
        m = len(ps)
        ma, mb, my = sum(xa) / m, sum(xb) / m, sum(y) / m
        saa = sum((a - ma) ** 2 for a in xa)
        sbb = sum((b - mb) ** 2 for b in xb)
        sab = sum((a - ma) * (b - mb) for a, b in zip(xa, xb))
        say = sum((a - ma) * (yy - my) for a, yy in zip(xa, y))
        sby = sum((b - mb) * (yy - my) for b, yy in zip(xb, y))
        det = saa * sbb - sab * sab
        ba = (sbb * say - sab * sby) / det
        bb = (saa * sby - sab * say) / det
        return ba, bb, (bb / ba if ba != 0 else float("nan"))

    seasons = sorted(set(p["code"] for p in pairs))
    by_season = {c: [p for p in pairs if p["code"] == c] for c in seasons}
    fine = [x / 100 for x in range(-300, 501)]

    def block(title, ak, bk, ek, elabel):
        Ak = [p[ak] for p in pairs]
        Bk = [p[bk] for p in pairs]
        Ek = [p[ek] for p in pairs]
        o = [f"### {title}", ""]
        o.append(f"| s | corr(score_s, next-year win%) | corr(score_s, {elabel}) |")
        o.append("|---|---|---|")
        for s in S_GRID:
            sc = score(s, pairs, ak, bk)
            o.append(f"| {s} | {corr(sc, nwin):+.4f} | {corr(sc, Ek):+.4f} |")
        o.append("")
        o.append(f"- Concentration term alone ({bk}/G): corr with next-year win% = {corr(Bk, nwin):+.4f}, with {elabel} = {corr(Bk, Ek):+.4f}, "
                 f"with the s = 0 score ({ak}/G) = {corr(Ak, Bk):+.4f}; corr({ak}/G, same-year win%) = {corr(Ak, win):+.4f} (n = {n}).")
        best_w = max(S_GRID, key=lambda s: corr(score(s, pairs, ak, bk), nwin))
        best_e = max(S_GRID, key=lambda s: corr(score(s, pairs, ak, bk), Ek))
        fw = max(fine, key=lambda s: corr(score(s, pairs, ak, bk), nwin))
        fe = max(fine, key=lambda s: corr(score(s, pairs, ak, bk), Ek))
        o.append(f"- Grid arg-max: s = {best_w} for next-year win%, s = {best_e} for {elabel}. "
                 f"Fine-grid (step 0.01, s in [-3, 5]) arg-max: s = {fw:.2f} (win%), s = {fe:.2f} ({elabel}).")
        ba, bb, s_imp = ols_implied(pairs, "nwin", ak, bk)
        ba2, bb2, s_imp2 = ols_implied(pairs, ek, ak, bk)
        o.append(f"- OLS next-year win% ~ {ak}/G + {bk}/G (n = {n}): beta_A = {ba:+.5f} per pt/game, beta_B = {bb:+.5f} per (TSA*L)/game; "
                 f"implied s = beta_B/beta_A = {s_imp:+.3f}. Same for {elabel}: beta_A = {ba2:+.6f}, beta_B = {bb2:+.6f}, implied s = {s_imp2:+.3f}.")
        boot_s, boot_arg, boot_cl = [], [], []
        for _ in range(N_BOOT):
            samp = [pairs[random.randrange(n)] for _ in range(n)]
            boot_s.append(ols_implied(samp, "nwin", ak, bk)[2])
            yw = [p["nwin"] for p in samp]
            boot_arg.append(max(S_GRID, key=lambda s: corr(score(s, samp, ak, bk), yw)))
            cl = []
            for _ in range(len(seasons)):
                cl.extend(by_season[seasons[random.randrange(len(seasons))]])
            boot_cl.append(ols_implied(cl, "nwin", ak, bk)[2])
        bs = sorted(boot_s)
        bc = sorted(boot_cl)
        o.append(f"- Bootstrap ({N_BOOT} resamples of n = {n} pairs): SE(implied s, win%) = {sd(boot_s):.3f}, 95% interval [{bs[int(0.025*N_BOOT)]:.3f}, {bs[int(0.975*N_BOOT)]:.3f}]; "
                 f"grid arg-max (win%) distribution: " + ", ".join(f"{s}: {boot_arg.count(s)/N_BOOT:.2f}" for s in S_GRID if boot_arg.count(s)) + ".")
        o.append(f"- Season-cluster bootstrap ({N_BOOT} resamples of {len(seasons)} seasons): SE(implied s, win%) = {sd(boot_cl):.3f}, "
                 f"95% interval [{bc[int(0.025*N_BOOT)]:.3f}, {bc[int(0.975*N_BOOT)]:.3f}].")
        o.append(f"- Partial corr({bk}/G, next-year win% | {ak}/G) = {partial_corr(Bk, nwin, Ak):+.4f}; partial corr({bk}/G, {elabel} | {ak}/G) = {partial_corr(Bk, Ek, Ak):+.4f} (n = {n}).")
        o.append("")
        return o

    out = []
    out.append("## 5. Forecasting next-season outcomes (non-load outcome)")
    out.append("")
    out.append("Team-season score_s = [ sum_i TSA_i*(lambda_i - price) + s * sum_i TSA_i*L_i ] / team games, summed over every player who logged "
               "a shooting attempt for that team that season (lambda_i, TSA_i from his games with that team; L_i = `l_season`, his season on-court share; "
               "price = league pts/TSA that season). The first term (A) is team points added at s = 0 (it equals team pts - price*team TSA); the second (B) is the concentration term. "
               "Outcome = the same team id's win% and pts/TSA in season t+1 (consecutive seasons only; win decided by box-score pts). Pearson correlations.")
    out.append("")
    out.append(f"- n team-season pairs = {n} (seasons {cs[0]}..{cs[-2]} -> next season; {len(cs)-1} transitions).")
    out.append(f"- Reference: corr(next-year win%, this-year win%) = {corr(win, nwin):+.4f} (n = {n}).")
    first, last = by_season[cs[0]], by_season[cs[-2]]
    out.append(f"- Era drift: mean B/G = {sum(p['B'] for p in first)/len(first):.2f} in {cs[0]} vs {sum(p['B'] for p in last)/len(last):.2f} in {cs[-2]}; "
               f"league price = {all_rows[cs[0]][1]:.3f} vs {all_rows[cs[-1]][1]:.3f} ({cs[-1]}). Raw pts/TSA and B both trend with era, so the raw block below is "
               f"followed by a season-demeaned block (season fixed effects; next-year pts/TSA taken relative to next-year league price).")
    out.append("")
    out.extend(block("5a. Raw (as specified)", "A", "B", "neff", "next-year pts/TSA"))
    out.extend(block("5b. Season-demeaned (A, B, and next-year pts/TSA - next-year price, each minus its season mean)", "Ad", "Bd", "neff_reld", "next-year pts/TSA rel. to price"))
    return out, pairs


def partial_corr(x, y, z):
    def resid(v, z):
        n = len(v)
        mv, mz = sum(v) / n, sum(z) / n
        b = sum((a - mv) * (c - mz) for a, c in zip(v, z)) / sum((c - mz) ** 2 for c in z)
        return [a - mv - b * (c - mz) for a, c in zip(v, z)]
    return corr(resid(x, z), resid(y, z))


def main():
    t0 = time.time()
    all_rows = {}
    for c in codes():
        all_rows[c] = load(c)
    print(f"loaded {len(all_rows)} seasons in {time.time()-t0:.1f}s", file=sys.stderr)
    out2, events = test2(all_rows)
    print(f"test2 done {time.time()-t0:.1f}s", file=sys.stderr)
    out5, pairs = test5(all_rows)
    print(f"test5 done {time.time()-t0:.1f}s", file=sys.stderr)
    text = "\n".join(out2 + out5)
    print(text)
    with open(os.path.join(DIR, "_calib_results_events.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(DIR, "_calib_events_data.json"), "w") as f:
        json.dump(dict(events=events, pairs=pairs), f)


if __name__ == "__main__":
    main()
