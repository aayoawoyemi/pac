# -*- coding: utf-8 -*-
"""_pac_gamelevel.py -- game-level estimation and validation of the replacement price of a scoring possession.

Question. When a player is absent, the shooting possessions he would have used are taken by others.
What do they return, per attempt, relative to league average, as a function of the share he carried?

Unit of observation: team-game (regular seasons 1997-98 .. 2025-26, built by _pacgl_build.py).
Outcome: Y = team points / team shooting possessions (TSA = FGA + 0.44 FTA), whole game.

For every rotation player i on team t (>= MIN_APPS appearances, >= MIN_DEN TSA per appearance) and every
team game inside his tenure window (first..last appearance for that team) that he did not appear in:

    Y_off - Y_full = L_full_i * (q_i - lam_i)                     (accounting identity, whole game)
    q_i            = price - gap(L_i),   gap(L) = c + s*L          (price schedule under test)

  L_full_i : his share of team TSA in games he played (whole games, bench minutes included)
  lam_i    : his own points per TSA in games he played
  L_i      : season share of on-court possessions, TSA/POSS -- the variable PAC prices on
  price    : league points per TSA that season (= 2*lgTS)

so, summing over the absent players in a team-game,

    Y = FE + X*b + a*Z1 - c*Z0 - s*Z2 + e
    Z1 = sum L_full*(price - lam)   (mechanical: his own efficiency leaves with him; a = 1 by identity)
    Z0 = sum L_full                 Z2 = sum L_full*L

PAC as published imposes c = 0 ("through the origin"). Here c is estimated.
q_i absorbs everything that changes when he is gone: the extra attempts others take AND any change in
their efficiency on the attempts they would have taken anyway (lost creation / spacing). It is the
total change in team scoring efficiency, expressed per attempt he used.

Fixed effects: roster spell (team-season split whenever a core player's tenure starts or ends, so
trades and season-ending injuries never identify anything), opponent-season, season x calendar month.
Controls: home, own and opponent rest (b2b / 2 / 3+ / unknown vs 1 day), opponent absence load.
Weights: team TSA. Inference: cluster-robust by team-season.

Outputs: _pac_gamelevel_results.md, _pac_gamelevel_results.json, pac_price_schedule.png
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
YEARS = list(range(1997, 2026))
BIN_EDGES = [0.0, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 1.0]
EVAL_L = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]


def code_of(y: int) -> str:
    return f"{y % 100:02d}{(y + 1) % 100:02d}"


# ----------------------------------------------------------------------------------------------
# design
# ----------------------------------------------------------------------------------------------
def season_design(y, ft_w=0.44, outcome="tsa", min_apps=10, min_den=4.0, core_apps=20, core_L=0.10):
    with open(os.path.join(HERE, f"_pacgl_season_{code_of(y)}.pkl"), "rb") as f:
        d = pickle.load(f)
    pg, tg = d["pg"].copy(), d["tg"].copy()
    extra_pg = pg.tov if outcome == "poss" else 0.0
    extra_tg = tg.tov if outcome == "poss" else 0.0
    pg["den"] = pg.fga + ft_w * pg.fta + extra_pg
    pg["tsa_w"] = pg.fga + ft_w * pg.fta
    pg["tsa44"] = pg.fga + 0.44 * pg.fta
    tg["den"] = tg.fga + ft_w * tg.fta + extra_tg
    tg = tg[tg.den > 0].copy()
    price = pg.pts.sum() / pg.den.sum()

    # game order within team: date (undated games take the previous game's date), then id
    tg["date"] = pd.to_datetime(tg.date)
    tg = tg.sort_values(["team", "gid"])
    tg["date_f"] = tg.groupby("team").date.ffill()
    tg["date_f"] = tg.groupby("team").date_f.bfill()
    tg = tg.sort_values(["team", "date_f", "gid"]).reset_index(drop=True)
    tg["k"] = tg.groupby("team").cumcount()
    prev = tg.groupby("team").date.shift(1)
    tg["rest"] = (tg.date - prev).dt.days - 1
    rmap = dict(zip(zip(tg.gid, tg.team), tg.rest))
    tg["opp_rest"] = [rmap.get((g, o), np.nan) for g, o in zip(tg.gid, tg.opp)]
    tg["month"] = tg.date_f.dt.month
    n_team_games = tg.groupby("team").k.max() + 1

    # players: whole-game shares and own efficiency in games they played
    pgm = pg.merge(tg[["gid", "team", "k", "den", "pts"]].rename(columns={"den": "den_team", "pts": "pts_team"}),
                   on=["gid", "team"], how="inner")
    agg = pgm.groupby(["pid", "team"]).agg(n_apps=("gid", "size"), pts=("pts", "sum"), den=("den", "sum"),
                                           den_team=("den_team", "sum"), pts_team=("pts_team", "sum"),
                                           kmin=("k", "min"), kmax=("k", "max")).reset_index()
    # mechanism: each shooter at his own leave-one-game-out efficiency (within team-season)
    tp = pgm.groupby(["pid", "team"])[["pts", "den"]].transform("sum")
    rest_den = tp.den - pgm.den
    base = np.where(rest_den > 0, (tp.pts - pgm.pts) / rest_den.where(rest_den > 0, 1.0), price)
    pgm["mix_num"] = pgm.den * base
    mix = pgm.groupby(["gid", "team"]).mix_num.sum()
    tg["Y_mix"] = [mix.get((g, t), np.nan) for g, t in zip(tg.gid, tg.team)]
    tg["Y_mix"] = tg.Y_mix / tg.den
    tot = pg.groupby("pid")[["tsa_w", "tsa44"]].sum()
    lseason = pg.dropna(subset=["l_season"]).drop_duplicates("pid").set_index("pid").l_season
    agg["L"] = agg.pid.map(lseason) * agg.pid.map(tot.tsa_w / tot.tsa44)
    agg["L_full"] = agg.den / agg.den_team
    agg["lam"] = agg.pts / agg.den
    agg["t_mates"] = (agg.pts_team - agg.pts) / (agg.den_team - agg.den)
    agg["den_app"] = agg.den / agg.n_apps
    agg["modeled"] = (agg.n_apps >= min_apps) & (agg.den_app >= min_den) & agg.L.notna() & (agg.L_full > 0)
    agg["core"] = agg.modeled & (agg.n_apps >= core_apps) & (agg.L_full >= core_L)
    n_unpriced = int(((agg.n_apps >= min_apps) & (agg.den_app >= min_den) & agg.L.isna()).sum())

    # row lookup: (team, k) -> row
    row_of = {}
    for t, grp in tg.groupby("team"):
        arr = np.full(int(grp.k.max()) + 1, -1, dtype=np.int64)
        arr[grp.k.values] = grp.index.values
        row_of[t] = arr
    apps_k = pgm.groupby(["pid", "team"]).k.apply(lambda s: np.sort(s.values)).to_dict()

    # spells: team-season split at every core player's tenure start / end
    tg["spell"] = 0
    for t, grp in agg[agg.core].groupby("team"):
        b = np.unique(np.concatenate([grp.kmin.values, grp.kmax.values + 1]))
        rows = row_of[t]
        ks = tg.loc[rows[rows >= 0], "k"].values
        tg.loc[rows[rows >= 0], "spell"] = np.searchsorted(b, ks, side="right")

    players, absent_rows, absent_pl, run_pos, run_len = [], [], [], [], []
    win_rows, win_pl = [], []
    pre1_rows, pre1_pl, pre2_rows, pre2_pl, post_rows, post_pl = [], [], [], [], [], []
    mod = agg[agg.modeled].reset_index(drop=True)
    for i, r in enumerate(mod.itertuples(index=False)):
        rows = row_of[r.team]
        win = np.arange(r.kmin, r.kmax + 1)
        ak = apps_k[(r.pid, r.team)]
        absent = np.setdiff1d(win, ak, assume_unique=True)
        players.append(dict(pid=r.pid, team=r.team, season=y, n_apps=r.n_apps, n_absent=len(absent), L=r.L,
                            L_full=r.L_full, lam=r.lam, t_mates=r.t_mates, price=price, core=bool(r.core),
                            window=len(win), team_games=int(n_team_games[r.team])))
        win_rows.append(rows[win])
        win_pl.append(np.full(len(win), i))
        if len(absent):
            absent_rows.append(rows[absent])
            absent_pl.append(np.full(len(absent), i))
            # runs of consecutive absent team games
            brk = np.where(np.diff(absent) != 1)[0]
            starts = np.concatenate([[0], brk + 1])
            ends = np.concatenate([brk, [len(absent) - 1]])
            pos = np.empty(len(absent), dtype=np.int64)
            ln = np.empty(len(absent), dtype=np.int64)
            akset = set(ak.tolist())
            for s0, e0 in zip(starts, ends):
                pos[s0:e0 + 1] = np.arange(1, e0 - s0 + 2)
                ln[s0:e0 + 1] = e0 - s0 + 1
                k0, k1 = absent[s0], absent[e0]
                pre1_rows.append(rows[k0 - 1]); pre1_pl.append(i)          # window starts with an appearance
                if k0 - 2 >= r.kmin and (k0 - 2) in akset:
                    pre2_rows.append(rows[k0 - 2]); pre2_pl.append(i)
                post_rows.append(rows[k1 + 1]); post_pl.append(i)          # window ends with an appearance
            run_pos.append(pos)
            run_len.append(ln)
    P = pd.DataFrame(players)
    cat = lambda xs, dt=np.int64: np.concatenate(xs).astype(dt) if xs else np.array([], dtype=dt)
    ev = dict(absent_rows=cat(absent_rows), absent_pl=cat(absent_pl), run_pos=cat(run_pos), run_len=cat(run_len),
              win_rows=cat(win_rows), win_pl=cat(win_pl),
              pre1_rows=np.array(pre1_rows, dtype=np.int64), pre1_pl=np.array(pre1_pl, dtype=np.int64),
              pre2_rows=np.array(pre2_rows, dtype=np.int64), pre2_pl=np.array(pre2_pl, dtype=np.int64),
              post_rows=np.array(post_rows, dtype=np.int64), post_pl=np.array(post_pl, dtype=np.int64))
    tg["season"] = y
    tg["Y"] = tg.pts / tg.den
    meta = dict(season=y, price=price, n_unpriced_rotation=n_unpriced, n_modeled=len(P), n_core=int(P.core.sum()) if len(P) else 0,
                n_spells=int(tg.groupby("team").spell.nunique().sum()))
    return tg, P, ev, meta


def build_design(**kw):
    tgs, Ps, evs, metas = [], [], [], []
    row_off = pl_off = 0
    for y in YEARS:
        tg, P, ev, meta = season_design(y, **kw)
        for key in ev:
            if key.endswith("_rows"):
                ev[key] = ev[key] + row_off
            elif key.endswith("_pl"):
                ev[key] = ev[key] + pl_off
        row_off += len(tg)
        pl_off += len(P)
        tgs.append(tg); Ps.append(P); evs.append(ev); metas.append(meta)
    tg = pd.concat(tgs, ignore_index=True)
    P = pd.concat(Ps, ignore_index=True)
    ev = {k: np.concatenate([e[k] for e in evs]) for k in evs[0]}
    return tg, P, ev, metas


def zsum(rows, weights, n):
    return np.bincount(rows, weights=weights, minlength=n)


def absence_terms(tg, P, ev, rows=None, pl=None):
    """Z1, Z0, Z2 per team-game from (row, player) absence entries."""
    n = len(tg)
    rows = ev["absent_rows"] if rows is None else rows
    pl = ev["absent_pl"] if pl is None else pl
    Lf, L = P.L_full.values[pl], P.L.values[pl]
    gapm = (P.price.values - P.lam.values)[pl]
    return zsum(rows, Lf * gapm, n), zsum(rows, Lf, n), zsum(rows, Lf * L, n)


def pair_terms(P, rows, pl, n, mask=None):
    """(sum L_full, sum L_full*L) for a set of (row, player) entries."""
    if mask is not None:
        rows, pl = rows[mask], pl[mask]
    Lf = P.L_full.values[pl]
    return zsum(rows, Lf, n), zsum(rows, Lf * P.L.values[pl], n)


def controls(tg, Z0):
    X, names = [], []
    X.append(tg.home.values.astype(float)); names.append("home")
    for col, lab in (("rest", "rest"), ("opp_rest", "opp_rest")):
        r = tg[col].values
        X.append((r == 0).astype(float)); names.append(f"{lab}_b2b")
        X.append((r == 2).astype(float)); names.append(f"{lab}_2")
        X.append((r >= 3).astype(float)); names.append(f"{lab}_3plus")
        X.append(np.isnan(r).astype(float)); names.append(f"{lab}_unknown")
    zmap = dict(zip(zip(tg.gid.values, tg.team.values), Z0))
    X.append(np.array([zmap.get((g, o), 0.0) for g, o in zip(tg.gid.values, tg.opp.values)])); names.append("opp_absence_load")
    return np.column_stack(X), names


def fe_groups(tg):
    s = tg.season.astype(str)
    team_season = tg.team.astype(str) + "_" + s
    spell = team_season + "_" + tg.spell.astype(str)
    return dict(
        season=s.values, team_season=team_season.values, spell=spell.values,
        opp_season=(tg.opp.astype(str) + "_" + s).values,
        season_month=(s + "_" + tg.month.astype(str)).values,
        spell_month=(spell + "_" + tg.month.astype(str)).values,
        franchise=tg.team.astype(str).values,
    )


# ----------------------------------------------------------------------------------------------
# estimation
# ----------------------------------------------------------------------------------------------
class Absorber:
    """Weighted alternating projections onto several sets of fixed effects (sparse group-sum operators)."""

    def __init__(self, groups, w):
        import scipy.sparse as sp
        self.w = np.asarray(w, dtype=float)
        self.ops = []
        n = len(self.w)
        for g in groups:
            c = pd.factorize(g)[0]
            S = sp.csr_matrix((self.w, (c, np.arange(n))), shape=(c.max() + 1, n))   # weighted group sums
            den = np.asarray(S.sum(axis=1)).ravel()
            self.ops.append((S, den, c))
        self.iters = 0

    def __call__(self, M, tol=1e-11, maxit=5000):
        M = np.array(M, dtype=float, copy=True)
        if M.ndim == 1:
            M = M[:, None]
        if not self.ops:
            return M
        for it in range(maxit):
            delta = 0.0
            for S, den, c in self.ops:
                mu = (S @ M) / den[:, None]
                M -= mu[c]
                delta = max(delta, float(np.abs(mu).max()))
            if delta < tol:
                break
        self.iters = it + 1
        return M


def wls(y, X, w, cl):
    XtW = X.T * w
    Ai = np.linalg.pinv(XtW @ X)
    beta = Ai @ (XtW @ y)
    e = y - X @ beta
    g = pd.factorize(cl)[0]
    G = g.max() + 1
    S = np.zeros((G, X.shape[1]))
    we = w * e
    for j in range(X.shape[1]):
        S[:, j] = np.bincount(g, weights=X[:, j] * we, minlength=G)
    n, k = X.shape
    V = Ai @ (S.T @ S) @ Ai * (G / (G - 1)) * ((n - 1) / max(n - k, 1))
    return beta, V


class Model:
    """Absorbs one fixed-effect structure once; fits absence specifications against it."""

    def __init__(self, w, cl, groups, Xc=None):
        self.w, self.cl = w, cl
        self.ab = Absorber(groups, w)
        self.Xd = self.ab(Xc) if Xc is not None and Xc.shape[1] else np.zeros((len(w), 0))

    def demean(self, cols):
        return self.ab(np.column_stack(cols))

    def fit(self, y, Z, offset=None):
        yy = y - (offset if offset is not None else 0.0)
        D = self.ab(np.column_stack([yy] + list(Z)))
        X = np.column_stack([D[:, 1:], self.Xd]) if self.Xd.shape[1] else D[:, 1:]
        return wls(D[:, 0], X, self.w, self.cl)


def cs(beta, V, i0, i2):
    """coefficient on sum(L_full) is -c, on sum(L_full*L) is -s."""
    return -beta[i0], -beta[i2], np.array([[V[i0, i0], V[i0, i2]], [V[i2, i0], V[i2, i2]]])


def gap_ci(c, s, Vcs, L):
    return c + s * L, float(np.sqrt(Vcs[0, 0] + 2 * L * Vcs[0, 1] + L * L * Vcs[1, 1]))


def core(M, y, Z1, Z0, Z2):
    """The four numbers that summarize a specification."""
    b0, V0 = M.fit(y, [Z2], offset=Z1)
    b, V = M.fit(y, [Z0, Z2], offset=Z1)
    c, s, Vcs = cs(b, V, 0, 1)
    bf, Vf = M.fit(y, [Z1, Z0, Z2])
    cf, sf, Vcsf = cs(bf, Vf, 1, 2)
    g25, e25 = gap_ci(c, s, Vcs, 0.25)
    g30, e30 = gap_ci(c, s, Vcs, 0.30)
    g30f, e30f = gap_ci(cf, sf, Vcsf, 0.30)
    return dict(s_origin=float(-b0[0]), s_origin_se=float(np.sqrt(V0[0, 0])), c=float(c), c_se=float(np.sqrt(Vcs[0, 0])),
                s=float(s), s_se=float(np.sqrt(Vcs[1, 1])), cov_cs=float(Vcs[0, 1]), gap25=float(g25), gap25_se=e25,
                gap30=float(g30), gap30_se=e30, a_free=float(bf[0]), a_free_se=float(np.sqrt(Vf[0, 0])),
                c_afree=float(cf), c_afree_se=float(np.sqrt(Vcsf[0, 0])), s_afree=float(sf), s_afree_se=float(np.sqrt(Vcsf[1, 1])),
                gap30_afree=float(g30f), gap30_afree_se=e30f)


# ----------------------------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------------------------
MAIN_FE = ["spell", "opp_season", "season_month"]


def assemble(**kw):
    tg, P, ev, metas = build_design(**kw)
    Z1, Z0, Z2 = absence_terms(tg, P, ev)
    Xc, xnames = controls(tg, Z0)
    return dict(tg=tg, P=P, ev=ev, metas=metas, Z1=Z1, Z0=Z0, Z2=Z2, Xc=Xc, xnames=xnames, G=fe_groups(tg),
                y=tg.Y.values, w=tg.den.values.astype(float))


def model_for(A, fe=MAIN_FE, mask=None, weighted=True, cluster="team_season", use_controls=True):
    G = A["G"]
    m = np.ones(len(A["y"]), bool) if mask is None else mask
    w = A["w"][m] if weighted else np.ones(int(m.sum()))
    Xc = A["Xc"][m] if use_controls else None
    return Model(w, G[cluster][m], [G[f][m] for f in fe], Xc), m


def run_core(A, **mkw):
    M, m = model_for(A, **mkw)
    return core(M, A["y"][m], A["Z1"][m], A["Z0"][m], A["Z2"][m])


def analysis(args):
    t0 = time.time()
    out = {}
    A = assemble(ft_w=0.44, outcome="tsa", min_apps=10, min_den=4.0)
    tg, P, ev = A["tg"], A["P"], A["ev"]
    y, Z1, Z0, Z2, n = A["y"], A["Z1"], A["Z0"], A["Z2"], len(A["y"])
    ab_pl, ab_rows = ev["absent_pl"], ev["absent_rows"]
    Lab = P.L.values[ab_pl]
    print(f"design: {n} team-games, {len(P)} rotation player-team-seasons, {len(ab_pl)} absences ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- descriptives
    desc = dict(team_games=n, games=int(tg.gid.nunique()), modeled_player_team_seasons=len(P), absences=int(len(ab_pl)),
                team_games_with_absence=int((Z0 > 0).sum()), spells=int(sum(m["n_spells"] for m in A["metas"])),
                clusters=int(len(set(A["G"]["team_season"]))), L_min=float(P.L.min()), L_max=float(P.L.max()),
                L_p5_p50_p95_absences=[float(x) for x in np.percentile(Lab, [5, 50, 95])])
    bins = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        mm = (Lab >= lo) & (Lab < hi)
        pm = (P.L >= lo) & (P.L < hi)
        bins.append(dict(lo=lo, hi=hi, absences=int(mm.sum()), players=int((pm & (P.n_absent > 0)).sum()),
                         mean_L=float(Lab[mm].mean()) if mm.any() else None))
    desc["bins"] = bins
    car = P.L >= 0.24
    desc["carriers_L24"] = dict(player_team_seasons=int(car.sum()), with_absence=int((car & (P.n_absent > 0)).sum()),
                                absences=int(P.n_absent[car].sum()))
    out["descriptives"] = desc

    # ---------------- controls ladder
    ladder_specs = [
        ("season FE only (no team comparison)", ["season"], False),
        ("team-season FE", ["team_season"], False),
        ("roster-spell FE", ["spell"], False),
        ("+ opponent-season FE", ["spell", "opp_season"], False),
        ("+ home, rest, opponent absences", ["spell", "opp_season"], True),
        ("+ season x month FE  [MAIN]", MAIN_FE, True),
    ]
    out["ladder"] = []
    for name, fe, usec in ladder_specs:
        r = run_core(A, fe=fe, use_controls=usec)
        r["spec"] = name
        out["ladder"].append(r)
        print(f"  ladder | {name}: s0={r['s_origin']:.3f} c={r['c']:.3f} s={r['s']:.3f} gap30={r['gap30']:.4f}", flush=True)

    # ---------------- main model
    M, _ = model_for(A)
    main = core(M, y, Z1, Z0, Z2)
    b, V = M.fit(y, [Z0, Z2], offset=Z1)
    c, s, Vcs = cs(b, V, 0, 1)
    main["controls"] = {nm: (float(b[2 + j]), float(np.sqrt(V[2 + j, 2 + j]))) for j, nm in enumerate(A["xnames"])}
    main["gaps"] = {f"{L:.2f}": dict(zip(("gap", "se"), map(float, gap_ci(c, s, Vcs, L)))) for L in EVAL_L}
    b0, V0 = M.fit(y, [Z2], offset=Z1)
    main["gaps_origin"] = {f"{L:.2f}": dict(gap=float(-b0[0] * L), se=float(np.sqrt(V0[0, 0]) * L)) for L in EVAL_L}
    Lstar = -c / s
    gvec = np.array([-1 / s, c / s ** 2])
    main["L_zero_gap"] = float(Lstar)
    main["L_zero_gap_se"] = float(np.sqrt(gvec @ Vcs @ gvec))
    main["t_c"] = float(c / np.sqrt(Vcs[0, 0]))
    main["t_a_equals_1"] = float((main["a_free"] - 1) / main["a_free_se"])
    out["main"] = main
    print(f"MAIN s0={main['s_origin']:.3f}({main['s_origin_se']:.3f}) c={c:.3f}({np.sqrt(Vcs[0,0]):.3f}) s={s:.3f}({np.sqrt(Vcs[1,1]):.3f}) "
          f"a_free={main['a_free']:.3f}({main['a_free_se']:.3f})", flush=True)

    # ---------------- non-parametric schedule (share bins), identity offset and free mechanical coefficient
    Zb = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        mm = (Lab >= lo) & (Lab < hi)
        Zb.append(zsum(ab_rows[mm], P.L_full.values[ab_pl][mm], n))
    bb, Vb = M.fit(y, Zb, offset=Z1)
    bbf, Vbf = M.fit(y, [Z1] + Zb)
    out["nonparametric"] = [dict(lo=lo, hi=hi, mean_L=bins[j]["mean_L"], absences=bins[j]["absences"],
                                 gap=float(-bb[j]), se=float(np.sqrt(Vb[j, j])),
                                 gap_afree=float(-bbf[1 + j]), se_afree=float(np.sqrt(Vbf[1 + j, 1 + j])))
                            for j, (lo, hi) in enumerate(zip(BIN_EDGES[:-1], BIN_EDGES[1:]))]
    out["nonparametric_a_free"] = dict(a=float(bbf[0]), se=float(np.sqrt(Vbf[0, 0])))

    # ---------------- hinge schedule: gap = s_h * max(0, L - L0), L0 profiled (L0 = 0 is PAC's through-origin form)
    L0_grid = [0.0, 0.04, 0.06, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.20, 0.22]
    Zh = [zsum(ab_rows, P.L_full.values[ab_pl] * np.maximum(0.0, Lab - L0), n) for L0 in L0_grid]
    prof = []
    for L0, zh in zip(L0_grid, Zh):
        bh, Vh = M.fit(y, [zh], offset=Z1)
        Dh = M.demean([y - Z1, zh])
        X = np.column_stack([Dh[:, 1:], M.Xd])
        e = Dh[:, 0] - X @ bh
        prof.append(dict(L0=L0, s_h=float(-bh[0]), s_h_se=float(np.sqrt(Vh[0, 0])), wssr=float(np.sum(A["w"] * e ** 2))))
    best = min(prof, key=lambda r: r["wssr"])
    for r in prof:
        r["d_wssr"] = r["wssr"] - best["wssr"]
    out["hinge"] = dict(profile=prof, L0=best["L0"], s_h=best["s_h"], s_h_se=best["s_h_se"],
                        gap30=best["s_h"] * max(0.0, 0.30 - best["L0"]))
    print(f"hinge: L0={best['L0']:.2f} s_h={best['s_h']:.3f} ({best['s_h_se']:.3f})", flush=True)

    # ---------------- mechanism: who takes the shots (mix) vs how well they shoot (rate)
    ymix = tg.Y_mix.values
    assert np.isfinite(ymix).all()
    mix = core(M, ymix, Z1, Z0, Z2)
    rate_b, rate_V = M.fit(y - ymix, [Z0, Z2])
    rc, rs, rV = cs(rate_b, rate_V, 0, 1)
    rate_b0, rate_V0 = M.fit(y - ymix, [Z2])
    mech_free = {}
    for lab_, yy in (("total", y), ("mix", ymix), ("rate", y - ymix)):
        bq, Vq = M.fit(yy, [Z1, Z0, Z2])
        cq, sq, Vcq = cs(bq, Vq, 1, 2)
        g30q, e30q = gap_ci(cq, sq, Vcq, 0.30)
        mech_free[lab_] = dict(a=float(bq[0]), a_se=float(np.sqrt(Vq[0, 0])), c=float(cq), c_se=float(np.sqrt(Vcq[0, 0])),
                               s=float(sq), s_se=float(np.sqrt(Vcq[1, 1])), gap30=float(g30q), gap30_se=e30q)
    out["mechanism"] = dict(
        mix=dict(c=mix["c"], s=mix["s"], gap25=mix["gap25"], gap25_se=mix["gap25_se"], gap30=mix["gap30"], gap30_se=mix["gap30_se"], s_origin=mix["s_origin"]),
        rate=dict(c=float(rc), s=float(rs), gap25=float(gap_ci(rc, rs, rV, .25)[0]), gap25_se=gap_ci(rc, rs, rV, .25)[1],
                  gap30=float(gap_ci(rc, rs, rV, .30)[0]), gap30_se=gap_ci(rc, rs, rV, .30)[1], s_origin=float(-rate_b0[0])),
        free=mech_free)
    print("mechanism:", {k: round(v["gap30"], 4) for k, v in out["mechanism"].items() if k != "free"},
          {k: (round(v["a"], 3), round(v["s"], 3)) for k, v in mech_free.items()}, flush=True)

    # ---------------- injuries vs single-game absences, by share
    ln_ = ev["run_len"]
    edges = [0.0, 0.14, 0.18, 0.22, 0.26, 0.30, 1.0]
    Zs, labs_ = [], []
    for gname, gm in (("single-game absence", ln_ == 1), ("absence run of 2+ games", ln_ >= 2)):
        for lo, hi in zip(edges[:-1], edges[1:]):
            mm = gm & (Lab >= lo) & (Lab < hi)
            Zs.append(zsum(ab_rows[mm], P.L_full.values[ab_pl][mm], n))
            labs_.append(dict(group=gname, lo=lo, hi=hi, n=int(mm.sum()), mean_L=float(Lab[mm].mean()) if mm.any() else None))
    bz, Vz = M.fit(y, Zs, offset=Z1)
    for j, r in enumerate(labs_):
        r["gap"], r["se"] = float(-bz[j]), float(np.sqrt(Vz[j, j]))
    single = np.zeros(n, bool)
    single[ab_rows[ln_ == 1]] = True
    out["injury_runs"] = dict(bins=labs_, multi_game_only=run_core(A, mask=~single))

    # ---------------- event study: played -2, played -1, absent (1st, 2nd-3rd, 4th+ game of the run), played +1
    pos = ev["run_pos"]
    terms = []
    labels = ["played, 2 games before", "played, game before", "absent, 1st game", "absent, games 2-3", "absent, games 4+",
              "played, first game back"]
    terms += pair_terms(P, ev["pre2_rows"], ev["pre2_pl"], n)
    terms += pair_terms(P, ev["pre1_rows"], ev["pre1_pl"], n)
    for mm in (pos == 1, (pos >= 2) & (pos <= 3), pos >= 4):
        terms += pair_terms(P, ab_rows, ab_pl, n, mm)
    terms += pair_terms(P, ev["post_rows"], ev["post_pl"], n)
    be, Ve = M.fit(y, terms, offset=Z1)
    counts = [len(ev["pre2_rows"]), len(ev["pre1_rows"]), int((pos == 1).sum()), int(((pos >= 2) & (pos <= 3)).sum()),
              int((pos >= 4).sum()), len(ev["post_rows"])]
    out["event_study"] = []
    for j, lab in enumerate(labels):
        cc, ss, VV = cs(be, Ve, 2 * j, 2 * j + 1)
        g25, e25 = gap_ci(cc, ss, VV, 0.25)
        g30, e30 = gap_ci(cc, ss, VV, 0.30)
        out["event_study"].append(dict(term=lab, n=counts[j], gap25=float(g25), gap25_se=e25, gap30=float(g30), gap30_se=e30))
    print("event study gap30:", [(r["term"], round(r["gap30"], 4), round(r["gap30_se"], 4)) for r in out["event_study"]], flush=True)

    # ---------------- run length (planned rest days vs injury spells)
    ln = ev["run_len"]
    terms = []
    groups_ = [("1-game absence", ln == 1), ("2-5 game run", (ln >= 2) & (ln <= 5)), ("6+ game run", ln >= 6)]
    for _, mm in groups_:
        terms += pair_terms(P, ab_rows, ab_pl, n, mm)
    br, Vr = M.fit(y, terms, offset=Z1)
    out["run_length"] = []
    for j, (lab, mm) in enumerate(groups_):
        cc, ss, VV = cs(br, Vr, 2 * j, 2 * j + 1)
        g25, e25 = gap_ci(cc, ss, VV, 0.25)
        g30, e30 = gap_ci(cc, ss, VV, 0.30)
        out["run_length"].append(dict(group=lab, n=int(mm.sum()), gap25=float(g25), gap25_se=e25, gap30=float(g30), gap30_se=e30))

    # ---------------- eras (all FE nested within season)
    out["eras"] = []
    for lab, lo, hi in (("1997-98 to 2006-07", 1997, 2006), ("2007-08 to 2015-16", 2007, 2015), ("2016-17 to 2025-26", 2016, 2025)):
        mm = (tg.season.values >= lo) & (tg.season.values <= hi)
        r = run_core(A, mask=mm)
        r["era"] = lab
        out["eras"].append(r)

    # ---------------- robustness
    rob = []
    def add(name, r):
        r["spec"] = name
        rob.append(r)
        print(f"  robust | {name}: s0={r['s_origin']:.3f} c={r['c']:.3f} s={r['s']:.3f} gap30={r['gap30']:.4f} a={r['a_free']:.3f}", flush=True)
    add("main", dict(main))
    add("unweighted", run_core(A, weighted=False))
    add("cluster by franchise (30 clusters)", run_core(A, cluster="franchise"))
    add("team-month FE (spell x month)", run_core(A, fe=["spell_month", "opp_season", "season_month"]))
    kfrom_end = tg.groupby(["team", "season"]).k.transform("max").values - tg.k.values
    add("drop each team's last 15 games", run_core(A, mask=kfrom_end >= 15))
    add("drop 2019-20 and 2020-21", run_core(A, mask=~np.isin(tg.season.values, [2019, 2020])))
    longrun = np.zeros(n, bool)
    longrun[ab_rows[ev["run_len"] >= 10]] = True
    add("drop team-games with an absence run of 10+ games", run_core(A, mask=~longrun))
    for nm, kw in (("FT weight 0.40", dict(ft_w=0.40)), ("FT weight 0.475", dict(ft_w=0.475)),
                   ("outcome per possession used (TSA + TOV)", dict(outcome="poss")),
                   ("stricter rotation filter (20+ games, 6+ TSA/game)", dict(min_apps=20, min_den=6.0)),
                   ("looser rotation filter (5+ games, 2.5+ TSA/game)", dict(min_apps=5, min_den=2.5))):
        base = dict(ft_w=0.44, outcome="tsa", min_apps=10, min_den=4.0)
        base.update(kw)
        add(nm, run_core(assemble(**base)))
    out["robustness"] = rob

    # ---------------- leave-one-season-out prediction of absence games
    # every FE is nested within season: absorbed variables are season-by-season projections, so fitting on the
    # other seasons and projecting the held-out season on its own FE is exact
    D = M.demean([y, y - Z1, Z0, Z2, Z1] + Zb + Zh)
    yraw, yoff, Z0d, Z2d, Z1d = D[:, 0], D[:, 1], D[:, 2], D[:, 3], D[:, 4]
    Zbd = D[:, 5:5 + len(Zb)]
    Zhd = D[:, 5 + len(Zb):]
    w, Xd, seas, has_abs = A["w"], M.Xd, tg.season.values, Z0 > 0
    models = ["no absence effect", "league-average replacement (c = s = 0)", "published PAC (c = 0, s = 0.25)",
              "through origin, trained", "free intercept, trained", "share bins, trained", "free intercept + free mechanical, trained",
              "hinge (threshold and slope trained)"]
    sse = {k: 0.0 for k in models}
    wins = {k: 0 for k in models}
    for S in YEARS:
        tr, te = seas != S, (seas == S) & has_abs
        def resid(yd, Zc):
            X = np.column_stack(Zc + [Xd])
            bet = np.linalg.lstsq(X[tr] * np.sqrt(w[tr])[:, None], yd[tr] * np.sqrt(w[tr]), rcond=None)[0]
            return yd[te] - X[te] @ bet
        # hinge: threshold chosen on the training seasons only
        ssr_best, zh_best = None, None
        for zh in Zhd.T:
            X = np.column_stack([zh, Xd])
            bet = np.linalg.lstsq(X[tr] * np.sqrt(w[tr])[:, None], yoff[tr] * np.sqrt(w[tr]), rcond=None)[0]
            ssr = float(np.sum(w[tr] * (yoff[tr] - X[tr] @ bet) ** 2))
            if ssr_best is None or ssr < ssr_best:
                ssr_best, zh_best = ssr, zh
        res = {models[0]: resid(yraw, []), models[1]: resid(yoff, []), models[2]: resid(yoff + 0.25 * Z2d, []),
               models[3]: resid(yoff, [Z2d]), models[4]: resid(yoff, [Z0d, Z2d]),
               models[5]: resid(yoff, [Zbd[:, j] for j in range(Zbd.shape[1])]), models[6]: resid(yraw, [Z1d, Z0d, Z2d]),
               models[7]: resid(yoff, [zh_best])}
        ms = {k: float(np.sum(w[te] * r_ ** 2) / np.sum(w[te])) for k, r_ in res.items()}
        for k in models:
            sse[k] += float(np.sum(w[te] * res[k] ** 2))
            wins[k] += int(ms[k] < ms[models[1]])
    wte = float(np.sum(w[has_abs]))
    out["loso"] = dict(models=models, mse={k: sse[k] / wte for k in models}, beats_league_avg_seasons=wins, seasons=len(YEARS))
    print("LOSO mse x1e3:", {k: round(1e3 * v, 5) for k, v in out["loso"]["mse"].items()}, wins, flush=True)

    # ---------------- permutation placebo on the free specification (no identity offset: under the null
    # the fake-absent player actually played, so nothing is removed mechanically and every coefficient is 0)
    real_f = M.fit(y, [Z1, Z0, Z2])[0]
    real_f0 = M.fit(y, [Z1, Z2])[0]
    rng = np.random.default_rng(20260928)
    wr, wp = ev["win_rows"], ev["win_pl"]
    starts = np.searchsorted(wp, np.arange(len(P)))
    n_abs = P.n_absent.values
    perm = []
    for it in range(args.perms):
        order = np.lexsort((rng.random(len(wr)), wp))
        rank = np.arange(len(wr)) - starts[wp[order]]
        sel = order[rank < n_abs[wp[order]]]
        pZ1, pZ0, pZ2 = absence_terms(tg, P, ev, rows=wr[sel], pl=wp[sel])
        bf = M.fit(y, [pZ1, pZ0, pZ2])[0]
        bf0 = M.fit(y, [pZ1, pZ2])[0]
        perm.append(dict(a=bf[0], c=-bf[1], s=-bf[2], gap30=-bf[1] - 0.30 * bf[2], s_origin=-bf0[1], a_origin=bf0[0]))
        if (it + 1) % 100 == 0:
            print(f"  permutations {it + 1}/{args.perms}", flush=True)
    pdf = pd.DataFrame(perm)
    real = dict(a=real_f[0], c=-real_f[1], s=-real_f[2], gap30=-real_f[1] - 0.30 * real_f[2], s_origin=-real_f0[1], a_origin=real_f0[0])
    out["permutation"] = {"n": args.perms}
    for k in real:
        v = pdf[k].values
        out["permutation"][k] = dict(real=float(real[k]), mean=float(v.mean()), sd=float(v.std(ddof=1)),
                                     lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)),
                                     p=float((np.sum(np.abs(v) >= abs(real[k])) + 1) / (len(v) + 1)))
    out["permutation_draws"] = pdf.to_dict(orient="list")
    print("permutation:", {k: (round(v['real'], 4), round(v['lo'], 4), round(v['hi'], 4)) for k, v in out["permutation"].items() if isinstance(v, dict)}, flush=True)

    # ---------------- cluster bootstrap over team-seasons (checks the analytic SEs; interval for the hinge threshold)
    if args.boot:
        cl = pd.factorize(A["G"]["team_season"])[0]
        ncl = cl.max() + 1
        order_ = np.argsort(cl, kind="stable")
        bounds = np.searchsorted(cl[order_], np.arange(ncl + 1))
        spell_c = pd.factorize(A["G"]["spell"])[0]
        opp_c = pd.factorize(A["G"]["opp_season"])[0]
        sm_c = pd.factorize(A["G"]["season_month"])[0]
        base = np.column_stack([y - Z1, Z0, Z2] + Zh + [A["Xc"]])
        nx = A["Xc"].shape[1]
        rngb = np.random.default_rng(7)
        boots = []
        for it in range(args.boot):
            pick = rngb.integers(0, ncl, ncl)
            idx = np.concatenate([order_[bounds[k]:bounds[k + 1]] for k in pick])
            copy_id = np.repeat(np.arange(ncl), [bounds[k + 1] - bounds[k] for k in pick])
            ab = Absorber([spell_c[idx] + copy_id * (spell_c.max() + 1), opp_c[idx], sm_c[idx]], A["w"][idx])
            Db = ab(base[idx], tol=1e-9)
            wb = A["w"][idx]
            yb, Z0b, Z2b, Zhb, Xb = Db[:, 0], Db[:, 1], Db[:, 2], Db[:, 3:3 + len(Zh)], Db[:, 3 + len(Zh):]
            def ols(cols):
                X = np.column_stack(cols + [Xb])
                sw = np.sqrt(wb)
                bet = np.linalg.lstsq(X * sw[:, None], yb * sw, rcond=None)[0]
                return bet, float(np.sum(wb * (yb - X @ bet) ** 2))
            b0b, _ = ols([Z2b])
            bfb, _ = ols([Z0b, Z2b])
            hs = [ols([Zhb[:, j]]) for j in range(len(Zh))]
            jbest = int(np.argmin([h[1] for h in hs]))
            boots.append(dict(s_origin=-b0b[0], c=-bfb[0], s=-bfb[1], gap30=-bfb[0] - 0.30 * bfb[1],
                              L0=L0_grid[jbest], s_h=-hs[jbest][0][0]))
            if (it + 1) % 50 == 0:
                print(f"  bootstrap {it + 1}/{args.boot}", flush=True)
        bdf = pd.DataFrame(boots)
        out["bootstrap"] = {"n": args.boot}
        for k in bdf.columns:
            v = bdf[k].values
            out["bootstrap"][k] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), lo=float(np.percentile(v, 2.5)),
                                       hi=float(np.percentile(v, 97.5)))
        out["bootstrap"]["L0_distribution"] = {f"{k:.2f}": int(v) for k, v in bdf.L0.value_counts().sort_index().items()}
        print("bootstrap:", {k: (round(v['sd'], 4), round(v['lo'], 4), round(v['hi'], 4)) for k, v in out["bootstrap"].items() if isinstance(v, dict) and 'sd' in v}, flush=True)
    out["meta"] = dict(seconds=time.time() - t0, seasons=A["metas"], bin_edges=BIN_EDGES)
    return out


# ----------------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------------
def ci(x, se, d=3):
    return f"{x:.{d}f} [{x - 1.96 * se:.{d}f}, {x + 1.96 * se:.{d}f}]"


def report(o):
    d, m = o["descriptives"], o["main"]
    L = []
    L.append("# Replacement price of a scoring possession: game-level estimates\n")
    L.append("Generated by `_pac_gamelevel.py` from `_pacgl_season_*.pkl` (built by `_pacgl_build.py`). Every number below is "
             "regenerated by `python _pac_gamelevel.py`.\n")
    L.append("**Units.** `gap` = league points per shooting possession minus what the possessions a player would have used return "
             "when he is absent, in points per TSA (TSA = FGA + 0.44 FTA). Multiply by 50 for true-shooting points, by 100 for "
             "points per 100 attempts. `L` = the absent player's season share of on-court possessions (TSA / POSS), the variable PAC prices on. "
             "Brackets are 95% intervals, cluster-robust by team-season unless stated.\n")
    L.append("## Sample\n")
    L.append(f"- {d['team_games']:,} team-games ({d['games']:,} games), regular seasons 1997-98 through 2025-26")
    L.append(f"- {d['modeled_player_team_seasons']:,} rotation player-team-seasons (10+ appearances, 4+ TSA per appearance)")
    L.append(f"- {d['absences']:,} absences inside tenure windows; {d['team_games_with_absence']:,} team-games with at least one")
    L.append(f"- {d['spells']:,} roster spells, {d['clusters']} team-season clusters")
    L.append(f"- share of absent players: 5th / 50th / 95th percentile = " + " / ".join(f"{x:.3f}" for x in d["L_p5_p50_p95_absences"]))
    L.append(f"- players at L >= 0.24: {d['carriers_L24']['player_team_seasons']:,} player-team-seasons, {d['carriers_L24']['absences']:,} absences\n")

    L.append("## 1. Main estimates\n")
    L.append("| quantity | estimate |\n|---|---|")
    L.append(f"| through-origin slope s (PAC's functional form, c = 0) | {ci(m['s_origin'], m['s_origin_se'])} |")
    L.append(f"| free intercept c | {ci(m['c'], m['c_se'])} (t = {m['t_c']:.1f}) |")
    L.append(f"| free slope s | {ci(m['s'], m['s_se'])} |")
    L.append(f"| share at which the gap is zero, -c/s | {ci(m['L_zero_gap'], m['L_zero_gap_se'])} |")
    L.append(f"| free coefficient on the mechanical term (identity says 1) | {ci(m['a_free'], m['a_free_se'])} (t vs 1 = {m['t_a_equals_1']:.1f}) |")
    L.append(f"| c, s with the mechanical coefficient free | {m['c_afree']:.3f}, {m['s_afree']:.3f} |")
    L.append("")
    L.append("| L | gap, free intercept | gap, through origin | TS points (free intercept) |\n|---|---|---|---|")
    for k, v in m["gaps"].items():
        go = m["gaps_origin"][k]
        L.append(f"| {k} | {ci(v['gap'], v['se'])} | {ci(go['gap'], go['se'])} | {50 * v['gap']:.1f} |")
    L.append("")
    L.append("Controls (main model): " + ", ".join(f"{k} {v[0]:+.4f} ({v[1]:.4f})" for k, v in m["controls"].items()) + "\n")

    L.append("## 2. Price schedule by share (non-parametric)\n")
    L.append(f"| L bin | mean L | absences | gap (identity) | gap (mechanical coef free, a = {o['nonparametric_a_free']['a']:.3f}) | through-origin fit | free-intercept fit |\n|---|---|---|---|---|---|---|")
    for r in o["nonparametric"]:
        mL = r["mean_L"] or 0
        L.append(f"| {r['lo']:.2f}-{r['hi']:.2f} | {mL:.3f} | {r['absences']:,} | {ci(r['gap'], r['se'])} | {ci(r['gap_afree'], r['se_afree'])} | "
                 f"{m['s_origin'] * mL:.3f} | {m['c'] + m['s'] * mL:.3f} |")
    L.append("")

    h = o["hinge"]
    L.append("## 2b. Hinge schedule: gap = s_h * max(0, L - L0)\n")
    L.append(f"Best threshold L0 = {h['L0']:.2f}, slope s_h = {ci(h['s_h'], h['s_h_se'])} (SE conditional on L0); "
             f"gap at L = 0.30: {h['gap30']:.3f}. L0 = 0 is PAC's through-origin form.\n")
    L.append("| L0 | s_h | weighted SSR minus best |\n|---|---|---|")
    for r in h["profile"]:
        L.append(f"| {r['L0']:.2f} | {ci(r['s_h'], r['s_h_se'])} | {r['d_wssr']:.4f} |")
    L.append("")
    if "bootstrap" in o:
        bt = o["bootstrap"]
        L.append(f"## 2c. Cluster bootstrap over team-seasons ({bt['n']} draws)\n")
        L.append("| parameter | analytic estimate (SE) | bootstrap SD | bootstrap 95% percentile interval |\n|---|---|---|---|")
        an = dict(s_origin=(m["s_origin"], m["s_origin_se"]), c=(m["c"], m["c_se"]), s=(m["s"], m["s_se"]), gap30=(m["gap30"], m["gap30_se"]),
                  L0=(h["L0"], float("nan")), s_h=(h["s_h"], h["s_h_se"]))
        for k in ("s_origin", "c", "s", "gap30", "L0", "s_h"):
            v = bt[k]
            L.append(f"| {k} | {an[k][0]:.3f} ({an[k][1]:.3f}) | {v['sd']:.3f} | [{v['lo']:.3f}, {v['hi']:.3f}] |")
        L.append("")
        L.append("Bootstrap distribution of the chosen threshold: " + ", ".join(f"{k}: {v}" for k, v in bt["L0_distribution"].items()) + "\n")

    L.append("## 3. Controls ladder\n")
    L.append("| specification | s (through origin) | c | s | gap at L = 0.30 |\n|---|---|---|---|---|")
    for r in o["ladder"]:
        L.append(f"| {r['spec']} | {ci(r['s_origin'], r['s_origin_se'])} | {ci(r['c'], r['c_se'])} | {ci(r['s'], r['s_se'])} | {ci(r['gap30'], r['gap30_se'])} |")
    L.append("")

    L.append("## 4. Placebo: absences re-drawn at random inside each player's own tenure window\n")
    p = o["permutation"]
    L.append(f"{p['n']} permutations, free specification (no identity offset, so every coefficient is zero under the null).\n")
    L.append("| parameter | real | placebo mean | placebo 95% range | permutation p |\n|---|---|---|---|---|")
    for k, lab in (("s_origin", "s, through origin"), ("c", "c"), ("s", "s"), ("gap30", "gap at L = 0.30"), ("a", "mechanical coefficient")):
        v = p[k]
        L.append(f"| {lab} | {v['real']:.3f} | {v['mean']:.3f} | [{v['lo']:.3f}, {v['hi']:.3f}] | {v['p']:.4f} |")
    L.append("")

    L.append("## 5. Event study (gap at L = 0.30; games he played are placebo / partial-treatment terms)\n")
    L.append("| term | n | gap at L = 0.25 | gap at L = 0.30 |\n|---|---|---|---|")
    for r in o["event_study"]:
        L.append(f"| {r['term']} | {r['n']:,} | {ci(r['gap25'], r['gap25_se'])} | {ci(r['gap30'], r['gap30_se'])} |")
    L.append("")
    L.append("| absence run | n | gap at L = 0.25 | gap at L = 0.30 |\n|---|---|---|---|")
    for r in o["run_length"]:
        L.append(f"| {r['group']} | {r['n']:,} | {ci(r['gap25'], r['gap25_se'])} | {ci(r['gap30'], r['gap30_se'])} |")
    L.append("")

    L.append("## 6. Mechanism: who takes the shots vs how well they shoot\n")
    L.append("`mix` = each shooter valued at his own leave-one-game-out efficiency (change in who shoots). `rate` = actual minus that "
             "(change in how well the same shooters shoot). The two add to the total.\n")
    L.append("| component | gap at L = 0.25 | gap at L = 0.30 | through-origin s |\n|---|---|---|---|")
    for k in ("mix", "rate"):
        r = o["mechanism"][k]
        L.append(f"| {k} | {ci(r['gap25'], r['gap25_se'])} | {ci(r['gap30'], r['gap30_se'])} | {r['s_origin']:.3f} |")
    L.append(f"| total (main) | {ci(m['gap25'], m['gap25_se'])} | {ci(m['gap30'], m['gap30_se'])} | {m['s_origin']:.3f} |")
    L.append("")
    L.append("Free specification (mechanical coefficient a estimated, no offset), same decomposition:\n")
    L.append("| component | a (loading on his own efficiency edge) | c | s (loading on share) | gap at L = 0.30 |\n|---|---|---|---|---|")
    for k in ("total", "mix", "rate"):
        r = o["mechanism"]["free"][k]
        L.append(f"| {k} | {ci(r['a'], r['a_se'])} | {ci(r['c'], r['c_se'])} | {ci(r['s'], r['s_se'])} | {ci(r['gap30'], r['gap30_se'])} |")
    L.append("")

    L.append("## 6b. Injury runs vs single-game absences, by share\n")
    L.append("| absence type | L bin | mean L | absences | gap |\n|---|---|---|---|---|")
    for r in o["injury_runs"]["bins"]:
        L.append(f"| {r['group']} | {r['lo']:.2f}-{r['hi']:.2f} | {(r['mean_L'] or 0):.3f} | {r['n']:,} | {ci(r['gap'], r['se'])} |")
    r = o["injury_runs"]["multi_game_only"]
    L.append("")
    L.append(f"Dropping every team-game that contains a single-game absence: s (origin) {ci(r['s_origin'], r['s_origin_se'])}, "
             f"c {ci(r['c'], r['c_se'])}, s {ci(r['s'], r['s_se'])}, gap at 0.30 {ci(r['gap30'], r['gap30_se'])}.\n")

    L.append("## 7. Eras\n")
    L.append("| era | s (through origin) | c | s | gap at L = 0.30 |\n|---|---|---|---|---|")
    for r in o["eras"]:
        L.append(f"| {r['era']} | {ci(r['s_origin'], r['s_origin_se'])} | {ci(r['c'], r['c_se'])} | {ci(r['s'], r['s_se'])} | {ci(r['gap30'], r['gap30_se'])} |")
    L.append("")

    L.append("## 8. Robustness\n")
    L.append("| specification | s (through origin) | c | s | gap at L = 0.30 | mechanical coef |\n|---|---|---|---|---|---|")
    for r in o["robustness"]:
        L.append(f"| {r['spec']} | {ci(r['s_origin'], r['s_origin_se'])} | {ci(r['c'], r['c_se'])} | {ci(r['s'], r['s_se'])} | "
                 f"{ci(r['gap30'], r['gap30_se'])} | {r['a_free']:.3f} |")
    L.append("")

    L.append("## 9. Leave-one-season-out prediction of absence games\n")
    lo = o["loso"]
    base = lo["mse"][lo["models"][1]]
    L.append("Weighted MSE of team points per TSA in held-out absence games (x 10^-3), after the held-out season's own fixed effects.\n")
    L.append("| model | MSE x 10^-3 | vs league-average replacement | held-out seasons beating it |\n|---|---|---|---|")
    for k in lo["models"]:
        L.append(f"| {k} | {1e3 * lo['mse'][k]:.4f} | {1e3 * (lo['mse'][k] - base):+.4f} | {lo['beats_league_avg_seasons'][k]} / {lo['seasons']} |")
    L.append("")
    return "\n".join(L)


def figure(o, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = o["main"]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw=dict(width_ratios=[1.35, 1.1, 0.9]))
    ax = axes[0]
    npar = [r for r in o["nonparametric"] if r["mean_L"]]
    xs = np.array([r["mean_L"] for r in npar]); gs = np.array([r["gap"] for r in npar]); es = np.array([r["se"] for r in npar])
    Lg = np.linspace(0.08, 0.38, 100)
    Vcs = np.array([[m["c_se"] ** 2, m["cov_cs"]], [m["cov_cs"], m["s_se"] ** 2]])
    band = np.array([gap_ci(m["c"], m["s"], Vcs, L)[1] for L in Lg])
    ax.axhline(0, color="#888", lw=1)
    ax.fill_between(Lg, m["c"] + m["s"] * Lg - 1.96 * band, m["c"] + m["s"] * Lg + 1.96 * band, color="#2c6fbb", alpha=0.15, lw=0)
    ax.plot(Lg, m["c"] + m["s"] * Lg, color="#2c6fbb", lw=2.5, label=f"free intercept: {m['c']:.3f} + {m['s']:.2f}·L")
    ax.plot(Lg, m["s_origin"] * Lg, color="#c0392b", lw=2, ls="--", label=f"through origin (PAC form): {m['s_origin']:.3f}·L")
    ax.errorbar(xs, gs, yerr=1.96 * es, fmt="o", color="black", ms=7, capsize=4, lw=1.5, label="share bins (non-parametric)")
    for x, r in zip(xs, npar):
        ax.annotate(f"{r['absences']:,}", (x, -0.085), ha="center", fontsize=8, color="#555")
    ax.set_xlabel("absent player's share of on-court possessions, L")
    ax.set_ylabel("replacement gap (points per attempt below league)")
    ax.set_ylim(-0.1, 0.16)
    sec = ax.secondary_yaxis("right", functions=(lambda v: 50 * v, lambda v: v / 50))
    sec.set_ylabel("true-shooting points")
    ax.set_title("A. What his possessions return when he sits", loc="left", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.text(0.99, 0.02, "numbers under points: absences per bin", transform=ax.transAxes, ha="right", fontsize=8, color="#555")

    ax = axes[1]
    es_ = o["event_study"]
    lab = ["played\n-2", "played\n-1", "out\n1st", "out\n2-3", "out\n4+", "played\n+1"]
    g = np.array([r["gap30"] for r in es_]); e = np.array([r["gap30_se"] for r in es_])
    col = ["#888", "#888", "#2c6fbb", "#2c6fbb", "#2c6fbb", "#888"]
    ax.axhline(0, color="#888", lw=1)
    for i in range(len(g)):
        ax.errorbar(i, g[i], yerr=1.96 * e[i], fmt="o", color=col[i], ms=8, capsize=4, lw=1.5)
    ax.set_xticks(range(len(g))); ax.set_xticklabels(lab)
    ax.set_ylabel("gap at L = 0.30 (points per attempt)")
    ax.set_title("B. Event time around an absence run", loc="left", fontweight="bold")

    ax = axes[2]
    pd_ = np.array(o["permutation_draws"]["gap30"])
    ax.hist(pd_, bins=30, color="#bbb", edgecolor="white")
    ax.axvline(o["permutation"]["gap30"]["real"], color="#2c6fbb", lw=3)
    ax.set_xlabel("gap at L = 0.30, free specification")
    ax.set_ylabel("permutations")
    ax.set_title("C. Placebo: absences shuffled in-window", loc="left", fontweight="bold")
    ax.annotate(f"real {o['permutation']['gap30']['real']:.3f}", (o["permutation"]["gap30"]["real"], ax.get_ylim()[1] * 0.9),
                xytext=(-8, 0), textcoords="offset points", ha="right", color="#2c6fbb", fontweight="bold")
    d = o["descriptives"]
    fig.text(0.01, 0.005, f"{d['team_games']:,} team-games 1997-98 to 2025-26, {d['absences']:,} absences of {d['modeled_player_team_seasons']:,} rotation player-seasons. "
             "Fixed effects: roster spell, opponent-season, season-month; controls: home, rest, opponent absences. 95% CIs clustered by team-season.",
             fontsize=8.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--tag", default="results")
    args = ap.parse_args()
    o = analysis(args)
    with open(os.path.join(HERE, f"_pac_gamelevel_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(o, f, indent=1, default=float)
    md = report(o)
    with open(os.path.join(HERE, f"_pac_gamelevel_{args.tag}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    figure(o, os.path.join(HERE, "pac_price_schedule.png"))
    print(f"done in {o['meta']['seconds']:.0f}s")


if __name__ == "__main__":
    main()
