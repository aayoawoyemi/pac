# -*- coding: utf-8 -*-
# _ingame_slope.py — within-game rest estimator. Same identity as the sit-out design, but the
# "absent" side is the team's stints in games he PLAYED where he is on the bench.
#   ON : t_on = teammates' pts / teammates' TSA while he is on the floor; L_on = his TSA / side TSA
#   OFF: t_off = team pts / team TSA in stints of the same games where he is NOT on the floor
#   q_bar = (t_off - (1-L_on) t_on) / L_on ; gap = price - q_bar ; slope = through-origin WLS on L
# Also held-out prediction loss curve across s. Placebo: randomly split his ON stints into pseudo on/off.
# Usage: python _ingame_slope.py [yr_lo] [yr_hi] [--placebo]
import sys, os, json, collections
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
be = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, be)
from _leaf_poss import load_leaf
yr_lo = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
yr_hi = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
PLACEBO = "--placebo" in sys.argv
rng = np.random.default_rng(0)
def code_of(yr): return f"{yr%100:02d}{(yr+1)%100:02d}"
def norm(g): s = str(g); return s if len(s) == 10 else s.zfill(10)
def stsa(v): return float(v[0]) + float(v[2]) + float(v[4]) + 0.44 * float(v[6])
def spts(v): return float(v[1]) + float(v[3]) + float(v[5]) + float(v[7])
GRID = [0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.42, 0.5, 0.6, 0.8, 1.0]
rows = []
for yr in range(yr_lo, yr_hi + 1):
    code = code_of(yr)
    fs = os.path.join(be, f"_sv_season_{code}.json"); fg = os.path.join(be, f"_sv_pergame_{code}.json")
    if not (os.path.exists(fs) and os.path.exists(fg)): continue
    leaf = load_leaf(code)
    if not leaf: continue
    S = json.load(open(fs, encoding="utf-8")); S = S if isinstance(S, list) else S.get("rows", list(S.values()))
    G = json.load(open(fg, encoding="utf-8")); G = G if isinstance(G, list) else G.get("rows", list(G.values()))
    price = sum(r["pts"] for r in S) / sum(r["tsa"] for r in S)
    pl = collections.defaultdict(dict); share = {}; team_of = {}
    for r in G:
        t, p, g = int(r["team"]), int(r["pid"]), norm(r["gid"])
        pl[(t, p)][g] = (r["pts"], r["tsa"]); share[(t, p)] = r.get("l_season", 0) or 0
        team_of.setdefault(g, {}); team_of[g][p] = t
    # per (gid, side): list of (pids_set, pts, tsa)
    for (t, p), games in pl.items():
        L = share[(t, p)]
        if L < 0.24 or len(games) < 40: continue
        on_p = on_a = off_p = off_a = hp = ht = 0.0; n_on = n_off = 0
        for g, (hpts, htsa) in games.items():
            stints = leaf.get(g)
            if not stints: continue
            on_list = []; off_list = []
            for st in stints:
                for side in (0, 1):
                    five = [int(x) for x in st[side]]
                    # which side is his team? his pid present -> on; else if any teammate present -> off
                    if p in five: on_list.append(st[5 + side])
                    elif any(team_of.get(g, {}).get(q) == t for q in five): off_list.append(st[5 + side])
            if PLACEBO:
                allon = on_list; k = len(allon) // 2; idx = rng.permutation(len(allon))
                off_list = [allon[i] for i in idx[:k]]; on_list = [allon[i] for i in idx[k:]]
            sp = sum(spts(v) for v in on_list if v and len(v) >= 8); sa = sum(stsa(v) for v in on_list if v and len(v) >= 8)
            op = sum(spts(v) for v in off_list if v and len(v) >= 8); oa = sum(stsa(v) for v in off_list if v and len(v) >= 8)
            if sa <= 0 or oa <= 0: continue
            if PLACEBO:
                # his pts/tsa are in both halves; scale by TSA fraction
                f = sa / (sa + oa); hp_on, ht_on = hpts * f, htsa * f
            else:
                hp_on, ht_on = hpts, htsa
            if ht_on > sa: continue
            on_p += sp; on_a += sa; off_p += op; off_a += oa; hp += hp_on; ht += ht_on; n_on += 1
        if n_on < 20 or ht <= 0 or on_a - ht <= 0: continue
        t_on = (on_p - hp) / (on_a - ht); L_on = ht / on_a; lam = hp / ht; t_off = off_p / off_a
        qbar = (t_off - (1 - L_on) * t_on) / L_on
        rows.append(dict(code=code, team=t, pid=p, L=L, L_on=L_on, lam=lam, t_on=t_on, t_off=t_off, qbar=qbar, price=price, w=off_a, n=n_on))
    print(f"{code}: {len(rows)}", flush=True)
W = pd.DataFrame(rows); tag = f"{yr_lo}_{yr_hi}" + ("_placebo" if PLACEBO else "")
W.to_json(os.path.join(be, f"_ingame_{tag}.json"), orient="records")
w = W.w.values; L = W.L.values; gap = (W.price - W.qbar).values
s = np.sum(w * L * gap) / np.sum(w * L * L)
bs = [(lambda i: np.sum(w[i]*L[i]*gap[i])/np.sum(w[i]*L[i]**2))(rng.integers(0, len(W), len(W))) for _ in range(500)]
print(f"\n=== in-game {tag}: n={len(W)} off-attempts={int(w.sum())}")
print(f"mean: price {np.average(W.price,weights=w):.3f} lam {np.average(W.lam,weights=w):.3f} t_on {np.average(W.t_on,weights=w):.3f} t_off {np.average(W.t_off,weights=w):.3f} qbar {np.average(W.qbar,weights=w):.3f}")
print(f"price-qbar mean {np.average(gap,weights=w):.3f} | through-origin slope {s:.3f} CI [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]")
# held-out prediction loss
print("held-out MSE by s:", end=" ")
best = None
for sv in GRID:
    pred = W.L_on * (W.price - sv * W.L) + (1 - W.L_on) * W.t_on
    mse = np.average((pred - W.t_off) ** 2, weights=w); print(f"{sv}:{mse*1000:.3f}", end="  ")
    if best is None or mse < best[1]: best = (sv, mse)
print(f"\nargmin grid s={best[0]}")
W["band"] = pd.cut(W.L, [0.24, 0.27, 0.30, 0.34, 1.0])
print(W.groupby("band", observed=True).apply(lambda x: pd.Series({"n": len(x), "credit": np.average(x.price - x.qbar, weights=x.w)}), include_groups=False).round(3).to_string())
