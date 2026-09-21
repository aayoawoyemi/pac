# -*- coding: utf-8 -*-
# _sitout_slope.py — stint-based sit-out WOWY, all in pts-per-TSA units (no per-game base mismatch).
#   ON  (his on-court stints):  t_on = teammates' pts / teammates' TSA while he is on the floor
#                               L_on = his TSA / side TSA while he is on the floor
#   OFF (games he sat, box):    t_off = team pts / team TSA
#   Identity: t_off = L_on * q_bar + (1 - L_on) * t_on   ->   q_bar = (t_off - (1-L_on) t_on) / L_on
#   q_bar = return per redistributed attempt. price = 2*lgTS. slope = through-origin WLS of (price - q_bar) on L.
# Usage: python _sitout_slope.py [yr_lo] [yr_hi] [--placebo]
import sys, os, json, collections
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
be = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, be)
from _leaf_poss import load_leaf

yr_lo = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
yr_hi = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
PLACEBO = "--placebo" in sys.argv
rng = np.random.default_rng(0)
def code_of(yr): return f"{yr%100:02d}{(yr+1)%100:02d}"
def norm(g): s = str(g); return s if len(s) == 10 else s.zfill(10)
def side_tsa(v): return float(v[0]) + float(v[2]) + float(v[4]) + 0.44 * float(v[6])
def side_pts(v): return float(v[1]) + float(v[3]) + float(v[5]) + float(v[7])

rows = []
for yr in range(yr_lo, yr_hi + 1):
    code = code_of(yr)
    fs = os.path.join(be, f"_sv_season_{code}.json"); fg = os.path.join(be, f"_sv_pergame_{code}.json")
    if not (os.path.exists(fs) and os.path.exists(fg)): continue
    leaf = load_leaf(code)
    if not leaf: print(f"{code}: no leaf", flush=True); continue
    S = json.load(open(fs, encoding="utf-8")); S = S if isinstance(S, list) else S.get("rows", list(S.values()))
    G = json.load(open(fg, encoding="utf-8")); G = G if isinstance(G, list) else G.get("rows", list(G.values()))
    price = sum(r["pts"] for r in S) / sum(r["tsa"] for r in S)   # = 2*lgTS, pts per TSA
    team_games = collections.defaultdict(dict); pl = collections.defaultdict(dict); share = {}
    for r in G:
        t, p, g = int(r["team"]), int(r["pid"]), norm(r["gid"])
        tp, tt = team_games[t].get(g, (0.0, 0.0)); team_games[t][g] = (tp + r["pts"], tt + r["tsa"])
        pl[(t, p)][g] = (r["pts"], r["tsa"]); share[(t, p)] = r.get("l_season", 0) or 0
    # on-court side totals per (gid, pid): pts and TSA of the side while pid is on the floor
    oncourt = collections.defaultdict(lambda: [0.0, 0.0])
    for g, stints in leaf.items():
        for st in stints:
            for side in (0, 1):
                v = st[5 + side]
                if not v or len(v) < 8: continue
                sp, sa = side_pts(v), side_tsa(v)
                for p in st[side]:
                    a = oncourt[(g, int(p))]; a[0] += sp; a[1] += sa
    for (t, p), games in pl.items():
        L = share[(t, p)] if "--rawL" not in sys.argv else None
        if L is None:
            L = sum(v[1] for v in games.values()) / sum(team_games[t][g][1] for g in games if g in team_games[t])
        if L < 0.24 or len(games) < 40: continue
        tg = sorted(team_games[t].keys()); order = {g: i for i, g in enumerate(tg)}
        played = set(games); pi = [order[g] for g in played if g in order]
        if not pi: continue
        g0, g1 = min(pi), max(pi)
        window = [g for g in tg if g0 <= order[g] <= g1]
        on = [g for g in window if g in played]; off = [g for g in window if g not in played]
        if PLACEBO:
            k = max(len(off), 5); idx = rng.permutation(len(on)); off = [on[i] for i in idx[:k]]; on = [on[i] for i in idx[k:]]
        if len(off) < 5 or len(on) < 20: continue
        sp = sa = hp = ht = 0.0; n_on = 0
        for g in on:
            oc = oncourt.get((g, p))
            if not oc or oc[1] <= 0: continue
            sp += oc[0]; sa += oc[1]; hp += games[g][0]; ht += games[g][1]; n_on += 1
        if n_on < 20 or ht <= 0 or sa - ht <= 0: continue
        t_on = (sp - hp) / (sa - ht); L_on = ht / sa; lam = hp / ht
        op = sum(team_games[t][g][0] for g in off); oa = sum(team_games[t][g][1] for g in off)
        if oa <= 0: continue
        t_off = op / oa
        qbar = (t_off - (1 - L_on) * t_on) / L_on
        rows.append(dict(code=code, team=t, pid=p, L=L, L_on=L_on, lam=lam, t_on=t_on, t_off=t_off, qbar=qbar, price=price, n_off=len(off), n_on=n_on))
    print(f"{code}: carriers so far {len(rows)}", flush=True)

W = pd.DataFrame(rows)
tag = f"{yr_lo}_{yr_hi}" + ("_placebo" if PLACEBO else "")
W.to_json(os.path.join(be, f"_sitout_{tag}.json"), orient="records")
w = W.n_off.values; L = W.L.values; gap = (W.price - W.qbar).values; g2 = (W.lam - W.qbar).values
s = np.sum(w * L * gap) / np.sum(w * L * L)
bs = [ (lambda i: np.sum(w[i]*L[i]*gap[i])/np.sum(w[i]*L[i]**2))(rng.integers(0, len(W), len(W))) for _ in range(500) ]
print(f"\n=== {tag}: n={len(W)} sit-outs={int(w.sum())}")
print(f"mean: price {np.average(W.price,weights=w):.3f}  lam {np.average(W.lam,weights=w):.3f}  t_on {np.average(W.t_on,weights=w):.3f}  t_off {np.average(W.t_off,weights=w):.3f}  qbar {np.average(W.qbar,weights=w):.3f}")
print(f"price-qbar mean {np.average(gap,weights=w):.3f}   lam-qbar mean {np.average(g2,weights=w):.3f}")
print(f"through-origin slope (price-qbar on L): {s:.3f}  95% CI [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]")
X = np.vstack([np.ones(len(W)), L]).T; sw = np.sqrt(w)
b0, b1 = np.linalg.lstsq(X * sw[:, None], g2 * sw, rcond=None)[0]
print(f"WLS (lam-qbar) on L: intercept {b0:.3f} slope {b1:.3f}")
W["band"] = pd.cut(W.L, [0.24, 0.27, 0.30, 0.34, 1.0])
print("\nusage bands (price-qbar per attempt):")
print(W.groupby("band", observed=True).apply(lambda x: pd.Series({"n": len(x), "credit": np.average(x.price - x.qbar, weights=x.n_off)}), include_groups=False).round(3).to_string())
