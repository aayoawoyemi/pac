# -*- coding: utf-8 -*-
"""_pac_predict_games.py -- does PAC predict winning?

Part 1. vPTS and PTS+ under the validated price (2025-26), and how close each is to PPG.
Part 2. Team-season: correlation of team PAC with win% (same season, second half from first half, next season),
        against PPG, points per shooting possession, TS Add, and point margin (the ceiling). PAC is offense only,
        so "net" versions (for minus against) are reported too.
Part 3. Game-level: predict held-out games with team offense / opponent defense ratings, adjusting for who is
        missing by the absent players' per-game value under each price (none, TS Add, published PAC, validated PAC).
        Games are split 50/50 within each team-season (cross-fitting, 20 random splits); scored on points RMSE and
        winner accuracy, overall and in games with a big absence gap. A free scale k on the absence adjustment tests
        PAC's per-game meaning: if PAC/g is "what his team's scoring loses without him", k should be ~1.
Writes _pac_predict_games.md.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
YEARS = list(range(1997, 2026))


def code_of(y):
    return f"{y % 100:02d}{(y + 1) % 100:02d}"


def gap_val(L, s_h, L0):
    return s_h * np.maximum(0.0, np.nan_to_num(L) - L0)


def load_rows(s_h, L0):
    """player-game rows with PAC under each price; team-game table with results."""
    rows, tgs = [], []
    for y in YEARS:
        code = code_of(y)
        with open(os.path.join(HERE, f"_sv_pergame_{code}.json"), encoding="utf-8") as f:
            R = pd.DataFrame(json.load(f)["rows"])[["gid", "pid", "team", "pts", "tsa", "l_season"]]
        R["season"] = y
        price = R.pts.sum() / R.tsa.sum()
        L = R.l_season.fillna(0).values
        R["tsadd"] = R.pts - price * R.tsa
        R["pac_pub"] = R.tsadd + R.tsa * 0.25 * L
        R["pac_val"] = R.tsadd + R.tsa * gap_val(L, s_h, L0)
        rows.append(R)
        with open(os.path.join(HERE, f"_pacgl_season_{code}.pkl"), "rb") as f:
            tg = pickle.load(f)["tg"][["gid", "team", "opp", "home", "pts", "fga", "fta", "game_no"]].copy()
        tg["season"] = y
        tgs.append(tg)
    R = pd.concat(rows, ignore_index=True)
    R["team"] = R.team.astype("int64")
    R["gid"] = R.gid.astype("int64")
    R["pid"] = R.pid.astype("int64")
    tg = pd.concat(tgs, ignore_index=True)
    opp_pts = dict(zip(zip(tg.gid, tg.team), tg.pts))
    tg["opp_pts"] = [opp_pts[(g, o)] for g, o in zip(tg.gid, tg.opp)]
    tg["win"] = (tg.pts > tg.opp_pts).astype(int)
    return R, tg


def part1(R, s_h, L0):
    y = 2025
    D = R[R.season == y].copy()
    tsum = D.groupby(["gid", "team"]).agg(T=("tsa", "sum"), Pv=("pac_val", "sum"), Pp=("pac_pub", "sum"))
    D = D.join(tsum, on=["gid", "team"])
    D["vpts_val"] = D.pts + D.pac_val - (D.tsa / D["T"]) * D.Pv
    D["vpts_pub"] = D.pts + D.pac_pub - (D.tsa / D["T"]) * D.Pp
    with open(os.path.join(HERE, f"_sv_season_{code_of(y)}.json"), encoding="utf-8") as f:
        S = pd.DataFrame(json.load(f)["rows"])[["pid", "name", "gp", "poss"]]
    S["pid"] = S.pid.astype("int64")
    agg = D.groupby("pid")[["pts", "pac_val", "pac_pub", "vpts_val", "vpts_pub"]].sum().reset_index().merge(S, on="pid")
    q = agg[(agg.gp >= 40) & (agg.poss >= 2500)].copy()
    for c in ("pts", "pac_val", "pac_pub", "vpts_val", "vpts_pub"):
        q[c + "_g"] = q[c] / q.gp
    q["ptsplus_val_g"] = q.pts_g + q.pac_val_g
    q["ptsplus_pub_g"] = q.pts_g + q.pac_pub_g
    rk = lambda c: q[c].rank()
    out = ["## Part 1. vPTS and PTS+ under the validated price, 2025-26\n",
           "PTS+ = PTS + PAC. vPTS = PTS + PAC - (TSA share of the team's attempts) x team PAC, so a team's vPTS sums to its actual points.\n",
           "| player | PPG | PAC/g (validated) | PTS+/g published -> validated | vPTS/g published -> validated |\n|---|---|---|---|---|"]
    for _, r in q.sort_values("pac_val_g", ascending=False).head(10).iterrows():
        out.append(f"| {r['name']} | {r.pts_g:.1f} | {r.pac_val_g:+.2f} | {r.ptsplus_pub_g:.1f} -> **{r.ptsplus_val_g:.1f}** | {r.vpts_pub_g:.1f} -> **{r.vpts_val_g:.1f}** |")
    out.append("")
    out.append("| column | Spearman with PPG (published) | Spearman with PPG (validated) |\n|---|---|---|")
    for lab, a, b in (("PAC/g", "pac_pub_g", "pac_val_g"), ("PTS+/g", "ptsplus_pub_g", "ptsplus_val_g"), ("vPTS/g", "vpts_pub_g", "vpts_val_g")):
        out.append(f"| {lab} | {rk('pts_g').corr(rk(a)):.3f} | {rk('pts_g').corr(rk(b)):.3f} |")
    out.append("")
    q["d_vpts"] = q.vpts_val_g - q.pts_g
    q["d_plus"] = q.ptsplus_val_g - q.pts_g
    top_ppg = set(q.nlargest(10, "pts_g").pid)
    out.append(f"How far they move from points (validated, 2025-26 qualified, n = {len(q)}): mean |vPTS - PPG| = {q.d_vpts.abs().mean():.2f} per game, "
               f"max +{q.d_vpts.max():.2f} / {q.d_vpts.min():.2f}; top-10 overlap with the PPG top 10: vPTS "
               f"{len(top_ppg & set(q.nlargest(10, 'vpts_val_g').pid))}/10, PTS+ {len(top_ppg & set(q.nlargest(10, 'ptsplus_val_g').pid))}/10, "
               f"PAC {len(top_ppg & set(q.nlargest(10, 'pac_val_g').pid))}/10. Every team's vPTS sums to its points; PTS+ adds about "
               f"{D.groupby(['gid','team']).pac_val.sum().mean():.1f} points per team-game on top of the scoreboard.\n")
    out.append("Biggest vPTS gains over PPG: " + "; ".join(f"{r['name']} {r.pts_g:.1f} -> {r.vpts_val_g:.1f}" for _, r in q.nlargest(5, "d_vpts").iterrows()))
    out.append("")
    out.append("Biggest vPTS losses vs PPG: " + "; ".join(f"{r['name']} {r.pts_g:.1f} -> {r.vpts_val_g:.1f}" for _, r in q.nsmallest(5, "d_vpts").iterrows()))
    out.append("")
    return out


def part2(R, tg):
    tg = tg.copy()
    tg["tsa"] = tg.fga + 0.44 * tg.fta
    pac = R.groupby(["gid", "team"])[["tsadd", "pac_pub", "pac_val"]].sum()
    tg = tg.join(pac, on=["gid", "team"])
    opp = tg.set_index(["gid", "team"])[["tsadd", "pac_pub", "pac_val", "pts", "tsa"]]
    oj = opp.rename(columns=lambda c: "opp_" + c)
    tg = tg.join(oj, on=["gid", "opp"], rsuffix="_o")
    tg["half"] = np.where(tg.game_no <= tg.groupby(["team", "season"]).game_no.transform("max") / 2, 1, 2)

    def agg(g):
        return pd.Series(dict(winpct=g.win.mean(), ppg=g.pts.mean(), ppa=g.pts.sum() / g.tsa.sum(),
                              tsadd=g.tsadd.mean(), pac_pub=g.pac_pub.mean(), pac_val=g.pac_val.mean(),
                              net_tsadd=(g.tsadd - g.opp_tsadd).mean(), net_pac_val=(g.pac_val - g.opp_pac_val).mean(),
                              net_ppa=g.pts.sum() / g.tsa.sum() - g.opp_pts.sum() / g.opp_tsa.sum(),
                              margin=(g.pts - g.opp_pts).mean()))
    T = tg.groupby(["team", "season"]).apply(agg, include_groups=False).reset_index()
    H = tg.groupby(["team", "season", "half"]).apply(agg, include_groups=False).reset_index()
    cols = ["ppg", "ppa", "tsadd", "pac_pub", "pac_val", "net_ppa", "net_tsadd", "net_pac_val", "margin"]
    labels = dict(ppg="points per game", ppa="points per shooting possession", tsadd="TS Add / game",
                  pac_pub="PAC / game (published)", pac_val="PAC / game (validated)", net_ppa="net points per shooting possession",
                  net_tsadd="net TS Add / game", net_pac_val="net PAC / game (validated)", margin="point margin (ceiling)")

    def z(df, c):   # within-season standardization removes era
        return df.groupby("season")[c].transform(lambda s: (s - s.mean()) / s.std())
    for c in cols + ["winpct"]:
        T[c + "_z"] = z(T, c)
        H[c + "_z"] = H.groupby(["season", "half"])[c].transform(lambda s: (s - s.mean()) / s.std())
    nxt = T.copy()
    nxt["season"] = nxt.season - 1
    TN = T.merge(nxt[["team", "season", "winpct_z"]].rename(columns={"winpct_z": "win_next"}), on=["team", "season"])
    H1 = H[H.half == 1].set_index(["team", "season"])
    H2 = H[H.half == 2].set_index(["team", "season"])
    HH = H1.join(H2[["winpct_z"]].rename(columns={"winpct_z": "win_h2"}), how="inner")
    out = ["## Part 2. Team-season: correlation with win% (within-season z-scores, 1997-98 to 2025-26)\n",
           f"{len(T)} team-seasons. PAC is offense only; the net rows subtract the same quantity for opponents.\n",
           "| measure | same season | second half from first half | next season |\n|---|---|---|---|"]
    for c in cols:
        out.append(f"| {labels[c]} | {T[c + '_z'].corr(T.winpct_z):.3f} | {HH[c + '_z'].corr(HH.win_h2):.3f} | {TN[c + '_z'].corr(TN.win_next):.3f} |")
    out.append("")
    # how different is team PAC from team TS Add?
    out.append(f"Correlation of team PAC/g (validated) with team TS Add/g, within season: {T.pac_val_z.corr(T.tsadd_z):.3f}. "
               f"Team-level load credit (PAC - TS Add) per game: mean {(T.pac_val - T.tsadd).mean():.2f}, "
               f"SD within season {(T.pac_val - T.tsadd).groupby(T.season).std().mean():.2f} points.\n")
    return out


def part3(R, s_h, L0, n_splits=20):
    import _pac_gamelevel as G
    A = G.assemble(ft_w=0.44, outcome="tsa", min_apps=10, min_den=4.0)
    tg, P, ev = A["tg"], A["P"], A["ev"]
    # per-game value of each absent player under each price: his own season per-game numbers (games he played)
    vals = []
    for y in YEARS:
        with open(os.path.join(HERE, f"_sv_season_{code_of(y)}.json"), encoding="utf-8") as f:
            S = pd.DataFrame(json.load(f)["rows"])[["pid", "gp", "tsa", "tsadd", "l"]]
        S["season"] = y
        vals.append(S)
    V = pd.concat(vals, ignore_index=True)
    V["pid"] = V.pid.astype("int64")
    V["v_tsadd"] = V.tsadd / V.gp
    V["v_pub"] = (V.tsadd + V.tsa * 0.25 * V.l.fillna(0)) / V.gp
    V["v_val"] = (V.tsadd + V.tsa * gap_val(V.l.values, s_h, L0)) / V.gp
    Rv = R.copy()
    tsum = Rv.groupby(["gid", "team"]).agg(T=("tsa", "sum"), Pv=("pac_val", "sum"))
    Rv = Rv.join(tsum, on=["gid", "team"])
    Rv["vpts"] = Rv.pts + Rv.pac_val - (Rv.tsa / Rv["T"]) * Rv.Pv
    pv = Rv.groupby(["pid", "season"])[["pts", "vpts"]].sum().reset_index()
    V = V.merge(pv, on=["pid", "season"], how="left", suffixes=("", "_r"))
    V["v_ppg"] = V.pts / V.gp
    V["v_vpts"] = V.vpts / V.gp
    V["v_plus"] = V.v_ppg + V.v_val
    Pm = P[["pid", "season"]].copy()
    Pm["pid"] = Pm.pid.astype("int64")
    Pm = Pm.merge(V[["pid", "season", "v_tsadd", "v_pub", "v_val", "v_ppg", "v_vpts", "v_plus"]], on=["pid", "season"], how="left").fillna(0.0)
    n = len(tg)
    adj = {}
    for k in ("v_tsadd", "v_pub", "v_val", "v_ppg", "v_vpts", "v_plus"):
        adj[k] = np.bincount(ev["absent_rows"], weights=Pm[k].values[ev["absent_pl"]], minlength=n)
    adj["none"] = np.zeros(n)
    ts = pd.factorize(tg.team.astype(str) + "_" + tg.season.astype(str))[0]
    os_ = pd.factorize(tg.opp.astype(str) + "_" + tg.season.astype(str))[0]
    y = tg.pts.values.astype(float)
    home = tg.home.values.astype(float)
    key = dict(zip(zip(tg.gid.values, tg.team.values), range(n)))
    opp_row = np.array([key[(g, o)] for g, o in zip(tg.gid.values, tg.opp.values)])
    gids = tg.gid.values
    NT, NO = ts.max() + 1, os_.max() + 1

    def design(idx, a, free):
        m = len(idx)
        cols = [sp.csr_matrix((np.ones(m), (np.arange(m), ts[idx])), shape=(m, NT)),
                sp.csr_matrix((np.ones(m), (np.arange(m), os_[idx])), shape=(m, NO)),
                sp.csr_matrix(home[idx][:, None])]
        if free:
            cols.append(sp.csr_matrix(-a[idx][:, None]))
        return sp.hstack(cols).tocsr()

    rng = np.random.default_rng(11)
    KEYS = ("none", "v_ppg", "v_plus", "v_vpts", "v_tsadd", "v_pub", "v_val")
    res = {k: dict(rmse=[], acc=[], acc_big=[], k=[], rmse_f=[], acc_f=[], acc_big_f=[]) for k in KEYS}
    big_gap = np.abs(adj["v_val"] - adj["v_val"][opp_row]) >= 4.0     # a team-game where one side is missing >= 4 PAC/g more
    for it in range(n_splits):
        ug = np.unique(gids)
        fold_of_game = dict(zip(ug, rng.integers(0, 2, len(ug))))
        fold = np.array([fold_of_game[g] for g in gids])
        for kname in res:
            a = adj[kname]
            for free in (False, True):
                if kname == "none" and free:
                    continue
                preds = np.zeros(n)
                kk = []
                for f in (0, 1):
                    tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
                    Xtr = design(tr, a, free)
                    ytr = y[tr] + (0 if free else a[tr])          # offset: team scores its baseline minus the absent value
                    beta = lsqr(Xtr, ytr, atol=1e-10, btol=1e-10, iter_lim=20000)[0]
                    Xte = design(te, a, free)
                    preds[te] = Xte @ beta - (0 if free else a[te])
                    if free:
                        kk.append(beta[-1])
                err = y - preds
                pm = preds - preds[opp_row]
                am = y - y[opp_row]
                homeside = home == 1
                correct = np.sign(pm[homeside]) == np.sign(am[homeside])
                sfx = "_f" if free else ""
                if free:
                    res[kname]["k"].append(float(np.mean(kk)))
                res[kname]["rmse" + sfx].append(float(np.sqrt(np.mean(err ** 2))))
                res[kname]["acc" + sfx].append(float(correct.mean()))
                res[kname]["acc_big" + sfx].append(float(correct[big_gap[homeside]].mean()))
        print(f"  split {it + 1}/{n_splits}", flush=True)
    out = ["## Part 3. Game-level prediction when players are missing (cross-fitted within team-season)\n",
           f"{n // 2:,} games, {n_splits} random 50/50 splits. Ratings: team-season offense + opponent-season defense + home, fit on one half, "
           "predicting the other. Absent players (rotation, inside tenure windows) lower their team's predicted points by their per-game value. "
           f"'Big absence gap' = games where one side is missing at least 4 validated PAC/g more than the other ({int(big_gap[home == 1].sum()):,} games).\n",
           "Two ways to use each number: at face value (the team loses exactly that many points per game), and with a free scale k "
           "fitted on the training half (k = 1 means the number is the true per-game cost).\n",
           "| a missing player costs his team... | face value: RMSE / winner acc. / acc. in big-gap games | fitted k | with fitted k: RMSE / winner acc. / acc. in big-gap games |\n|---|---|---|---|"]
    names = dict(none="nothing (ignore who is missing)", v_ppg="his points per game", v_plus="his PTS+ per game", v_vpts="his vPTS per game",
                 v_tsadd="his TS Add per game (league-average price)", v_pub="his PAC per game, published price", v_val="his PAC per game, validated price")
    for kname in KEYS:
        r = res[kname]
        face = f"{np.mean(r['rmse']):.3f} / {100 * np.mean(r['acc']):.2f}% / {100 * np.mean(r['acc_big']):.2f}%"
        if r["k"]:
            fit = f"{np.mean(r['rmse_f']):.3f} / {100 * np.mean(r['acc_f']):.2f}% / {100 * np.mean(r['acc_big_f']):.2f}%"
            kk = f"{np.mean(r['k']):.2f}"
        else:
            fit, kk = "-", "-"
        out.append(f"| {names[kname]} | {face} | {kk} | {fit} |")
    out.append("")
    return out


def main():
    with open(os.path.join(HERE, "_pac_gamelevel_results.json"), encoding="utf-8") as f:
        h = json.load(f)["hinge"]
    s_h, L0 = h["s_h"], h["L0"]
    R, tg = load_rows(s_h, L0)
    out = ["# Does PAC predict winning?\n", f"Generated by `_pac_predict_games.py`. Validated price gap = {s_h:.3f} max(0, L - {L0:.2f}).\n"]
    out += part1(R, s_h, L0)
    out += part2(R, tg)
    out += part3(R, s_h, L0)
    txt = "\n".join(out)
    with open(os.path.join(HERE, "_pac_predict_games.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
