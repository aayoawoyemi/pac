"""Neutral benchmark: a family of box-score scoring statistics vs external impact targets.

Run:  python _bench_scoring.py
Writes: _bench_scoring_results.md

Inputs (same directory):
  _sv_pergame_{code}.json   per player-game rows (uses gid,pid,name,team,pts,fga,fta,tsa,p_g,l_season)
  _obpm_dbpm_{code}.json    Basketball-Reference OBPM/DBPM keyed by player name
  _pure_rapm_{code}.json    pure RAPM (no box prior), 3-season window ending at code

Family (per player-season; PTS, TSA, P=sum p_g, L=l_season, gp=games, price=league pts per TSA):
  PPG, TS, rTS, TSAdd (sum / per game / per 100), UWrTS,
  F(s)=TSAdd + s*TSA*L for a grid of s (sum / per game / per 100), LOADONLY=TSA*L (sum / per game / per 100).
Gate: gp >= 40 and P >= 2500.
"""
import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict, Counter

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_bench_scoring_results.md")

S_GRID = [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.42, 0.5, 0.6, 0.8, 1.0]
GP_MIN = 40
P_MIN = 2500
PS_MIN = 2000
N_BOOT = 500
FLAT_TOL = 0.02
RNG = np.random.default_rng(20260921)


def season_codes():
    codes = []
    for y in range(1997, 2026):
        codes.append(f"{y % 100:02d}{(y + 1) % 100:02d}")
    return codes


def next_code(code):
    a = int(code[:2])
    b = int(code[2:])
    return f"{b:02d}{(b + 1) % 100:02d}"


def season_label(code):
    a = int(code[:2])
    y = 1900 + a if a >= 90 else 2000 + a
    return f"{y}-{code[2:]}"


# --------------------------------------------------------------------------- loading

def load_pergame(code):
    with open(os.path.join(HERE, f"_sv_pergame_{code}.json"), encoding="utf-8") as f:
        return json.load(f)["rows"]


def load_obpm(code):
    p = os.path.join(HERE, f"_obpm_dbpm_{code}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_rapm(code):
    with open(os.path.join(HERE, f"_pure_rapm_{code}.json"), encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- names

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
_TRANSLIT = str.maketrans({"ı": "i", "ø": "o", "đ": "d", "ł": "l", "ß": "ss", "æ": "ae", "œ": "oe"})


def norm_name(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().translate(_TRANSLIT).replace(".", "").replace("'", "").replace("-", " ")
    return " ".join(s.split())


def norm_name_loose(s):
    s = _SUFFIX.sub(" ", norm_name(s).replace(",", " "))
    return " ".join(s.split())


class NameMatcher:
    """Match per-game player names to a name-keyed dict (OBPM).

    1. exact normalized match; 2. suffix-stripped match; 3. same last name and one first name is a prefix
    of the other (truncated / shortened first names), accepted only if unique.
    """

    def __init__(self, keyed):
        self.exact = {}
        self.loose = {}
        self.by_last = defaultdict(list)
        for k, v in keyed.items():
            self.exact.setdefault(norm_name(k), v)
            lk = norm_name_loose(k)
            self.loose.setdefault(lk, v)
            parts = lk.split()
            if len(parts) >= 2:
                self.by_last[parts[-1]].append((parts[0], v))
        self.n_exact = 0
        self.n_loose = 0
        self.n_prefix = 0
        self.n_miss = 0

    def get(self, name):
        k = norm_name(name)
        if k in self.exact:
            self.n_exact += 1
            return self.exact[k]
        k = norm_name_loose(name)
        if k in self.loose:
            self.n_loose += 1
            return self.loose[k]
        parts = k.split()
        if len(parts) >= 2:
            first = parts[0]
            cands = [v for f, v in self.by_last[parts[-1]] if f.startswith(first) or first.startswith(f)]
            if len(cands) == 1:
                self.n_prefix += 1
                return cands[0]
        self.n_miss += 1
        return None


# --------------------------------------------------------------------------- aggregation

def aggregate(rows):
    """Return (price, players, teams, coverage).

    players[pid] = dict(name, PTS, TSA, P, gp, cov, L)
    teams[team]  = dict(pts, tsa, P, load(sum tsa*L), games, wins)
    coverage     = dict(games, covered_games)

    p_g is null for games without on-court data (whole games). P = covered sum + (gp - cov) * (covered sum / cov):
    uncovered games get the player's mean covered on-court possessions. Players with cov == 0 have P = nan.
    """
    tot_pts = 0.0
    tot_tsa = 0.0
    pl = {}
    games = defaultdict(set)
    names = defaultdict(Counter)
    tm = defaultdict(lambda: {"pts": 0.0, "tsa": 0.0, "P": 0.0, "load": 0.0, "games": 0, "wins": 0})
    gscore = defaultdict(dict)
    covered_gids = set()
    for r in rows:
        pid = str(r["pid"])
        pts = float(r["pts"])
        tsa = float(r["tsa"])
        pg = r["p_g"]
        L = float(r["l_season"])
        tot_pts += pts
        tot_tsa += tsa
        d = pl.get(pid)
        if d is None:
            d = pl[pid] = {"PTS": 0.0, "TSA": 0.0, "Pcov": 0.0, "cov": 0, "L": L, "Lmin": L, "Lmax": L, "teams": Counter()}
        d["PTS"] += pts
        d["TSA"] += tsa
        if pg is not None:
            d["Pcov"] += float(pg)
            d["cov"] += 1
            covered_gids.add(r["gid"])
        if L < d["Lmin"]:
            d["Lmin"] = L
        if L > d["Lmax"]:
            d["Lmax"] = L
        games[pid].add(r["gid"])
        names[pid][r["name"]] += 1
        team = str(r["team"])
        d["teams"][team] += 1
        t = tm[team]
        t["pts"] += pts
        t["tsa"] += tsa
        t["load"] += tsa * L
        gscore[r["gid"]][team] = gscore[r["gid"]].get(team, 0.0) + pts
    price = tot_pts / tot_tsa
    for pid, d in pl.items():
        d["gp"] = len(games[pid])
        d["name"] = names[pid].most_common(1)[0][0]
        if d["cov"] > 0:
            d["P"] = d["Pcov"] * d["gp"] / d["cov"]
        else:
            d["P"] = float("nan")
        # team P: distribute the player's imputed P across teams in proportion to games played for each
        for team, ng in d["teams"].items():
            if d["cov"] > 0:
                tm[team]["P"] += d["P"] * ng / d["gp"]
    for gid, sc in gscore.items():
        if len(sc) != 2:
            continue
        (a, pa), (b, pb) = sc.items()
        tm[a]["games"] += 1
        tm[b]["games"] += 1
        if pa > pb:
            tm[a]["wins"] += 1
        elif pb > pa:
            tm[b]["wins"] += 1
        else:
            tm[a]["wins"] += 0.5
            tm[b]["wins"] += 0.5
    coverage = {"games": len(gscore), "covered_games": len(covered_gids)}
    return price, pl, dict(tm), coverage


def family_names():
    names = ["PPG", "TS", "rTS", "UWrTS", "TSAdd_sum", "TSAdd_pg", "TSAdd_p100"]
    for s in S_GRID:
        names.append(f"F({s:g})_sum")
    for s in S_GRID:
        names.append(f"F({s:g})_pg")
    for s in S_GRID:
        names.append(f"F({s:g})_p100")
    names += ["LOADONLY_sum", "LOADONLY_pg", "LOADONLY_p100"]
    return names


FAMILY = family_names()


def compute_family(PTS, TSA, P, gp, L, price, ur_mean):
    """All arrays aligned. ur_mean = league mean of usage rate TSA/P*100 (scalar). Returns dict name->array."""
    lgTS = price / 2.0
    TS = PTS / (2.0 * TSA)
    rTS = 100.0 * (TS - lgTS)
    UR = TSA / P * 100.0
    TSAdd = PTS - price * TSA
    LOAD = TSA * L
    out = {
        "PPG": PTS / gp,
        "TS": TS,
        "rTS": rTS,
        "UWrTS": rTS * UR / ur_mean,
        "TSAdd_sum": TSAdd,
        "TSAdd_pg": TSAdd / gp,
        "TSAdd_p100": TSAdd / P * 100.0,
    }
    for s in S_GRID:
        Fs = TSAdd + s * LOAD
        out[f"F({s:g})_sum"] = Fs
    for s in S_GRID:
        out[f"F({s:g})_pg"] = out[f"F({s:g})_sum"] / gp
    for s in S_GRID:
        out[f"F({s:g})_p100"] = out[f"F({s:g})_sum"] / P * 100.0
    out["LOADONLY_sum"] = LOAD
    out["LOADONLY_pg"] = LOAD / gp
    out["LOADONLY_p100"] = LOAD / P * 100.0
    return out


class Season:
    def __init__(self, code):
        self.code = code
        rows = load_pergame(code)
        self.price, self.players, self.teams, self.coverage = aggregate(rows)
        self.n_rows = len(rows)
        pids = sorted(self.players)
        self.gated = [p for p in pids if self.players[p]["gp"] >= GP_MIN
                      and self.players[p]["cov"] > 0 and self.players[p]["P"] >= P_MIN]
        g = self.gated
        PTS = np.array([self.players[p]["PTS"] for p in g])
        TSA = np.array([self.players[p]["TSA"] for p in g])
        P = np.array([self.players[p]["P"] for p in g])
        gp = np.array([self.players[p]["gp"] for p in g], dtype=float)
        L = np.array([self.players[p]["L"] for p in g])
        self.ur_mean = float(np.mean(TSA / P * 100.0)) if len(g) else float("nan")
        self.fam = compute_family(PTS, TSA, P, gp, L, self.price, self.ur_mean) if len(g) else None
        self.idx = {p: i for i, p in enumerate(g)}
        self.L_dev = max((self.players[p]["Lmax"] - self.players[p]["Lmin"]) for p in pids) if pids else 0.0
        self.L_vs_ratio = float(np.max(np.abs(L - TSA / P))) if len(g) else float("nan")
        # team family
        tids = sorted(self.teams)
        self.team_ids = tids
        tpts = np.array([self.teams[t]["pts"] for t in tids])
        ttsa = np.array([self.teams[t]["tsa"] for t in tids])
        tP = np.array([self.teams[t]["P"] for t in tids])
        tg = np.array([self.teams[t]["games"] for t in tids], dtype=float)
        tload = np.array([self.teams[t]["load"] for t in tids])
        self.team_winpct = np.array([self.teams[t]["wins"] / self.teams[t]["games"] for t in tids])
        self.team_games = tg
        t_ur_mean = float(np.mean(ttsa / tP * 100.0))
        # team L := team load / team TSA so that TSA*L reproduces the summed row load
        tL = tload / ttsa
        self.team_fam = compute_family(tpts, ttsa, tP, tg, tL, self.price, t_ur_mean)


# --------------------------------------------------------------------------- correlation helpers

def pearson_cols(X, y):
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    num = Xc.T @ yc
    den = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def spearman_cols(X, y):
    Xr = rankdata(X, axis=0)
    yr = rankdata(y)
    return pearson_cols(Xr, yr)


def demean_by_group(M, groups):
    """Subtract group means (rows grouped by `groups`) from each column of M (2-D) or vector."""
    M = np.asarray(M, dtype=float)
    out = M.copy()
    for gval in np.unique(groups):
        m = groups == gval
        if M.ndim == 1:
            out[m] = M[m] - M[m].mean()
        else:
            out[m] = M[m] - M[m].mean(axis=0)
    return out


def stat_matrix(fam_list, index_list):
    """Stack family values for selected indices across seasons. fam_list: list of (fam dict, idx array)."""
    cols = {name: [] for name in FAMILY}
    for fam, idx in fam_list:
        for name in FAMILY:
            cols[name].append(fam[name][idx])
    return np.column_stack([np.concatenate(cols[name]) for name in FAMILY])


def evaluate(X, y, groups=None):
    """Return dict with per-member pearson/spearman, n. X: n x |FAMILY|. If groups given, season-demean."""
    n = len(y)
    if groups is not None:
        X = demean_by_group(X, groups)
        y = demean_by_group(y, groups)
    pr = pearson_cols(X, y)
    sp = spearman_cols(X, y)
    return {"n": n, "pearson": dict(zip(FAMILY, pr)), "spearman": dict(zip(FAMILY, sp)), "X": X, "y": y}


def sweep_bootstrap(X, y, suffix):
    """Arg-max s over grid for one variant (suffix in _sum/_pg/_p100), point estimate + bootstrap."""
    cols = [FAMILY.index(f"F({s:g}){suffix}") for s in S_GRID]
    Xs = X[:, cols]
    n = len(y)
    pr = pearson_cols(Xs, y)
    sp = spearman_cols(Xs, y)
    res = {}
    for label, curve, fn in (("pearson", pr, pearson_cols), ("spearman", sp, spearman_cols)):
        k = int(np.nanargmax(curve))
        boots = np.empty(N_BOOT)
        for b in range(N_BOOT):
            ii = RNG.integers(0, n, n)
            c = fn(Xs[ii], y[ii])
            boots[b] = S_GRID[int(np.nanargmax(c))]
        lo = np.percentile(boots, 2.5, method="lower")
        hi = np.percentile(boots, 97.5, method="higher")
        rng_ = float(np.nanmax(curve) - np.nanmin(curve))
        res[label] = {
            "argmax_s": S_GRID[k],
            "r_at_max": float(curve[k]),
            "r_at_0": float(curve[0]),
            "r_at_1": float(curve[-1]),
            "range": rng_,
            "flat": rng_ < FLAT_TOL,
            "ci": (float(lo), float(hi)),
            "boot_hist": Counter(boots.tolist()),
        }
    return res


# --------------------------------------------------------------------------- markdown

def fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    return f"{x:+.3f}"


def table_md(ev, extra=None, title=None):
    """extra: optional second evaluate() dict (e.g. DBPM) to add columns."""
    lines = []
    if title:
        lines.append(f"**{title}** (n = {ev['n']})")
        lines.append("")
    if extra is None:
        lines.append("| member | Pearson | Spearman | n |")
        lines.append("|---|---:|---:|---:|")
        for name in FAMILY:
            lines.append(f"| {name} | {fmt(ev['pearson'][name])} | {fmt(ev['spearman'][name])} | {ev['n']} |")
    else:
        lines.append("| member | Pearson (OBPM) | Spearman (OBPM) | Pearson (DBPM) | Spearman (DBPM) | n |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for name in FAMILY:
            lines.append(
                f"| {name} | {fmt(ev['pearson'][name])} | {fmt(ev['spearman'][name])} | "
                f"{fmt(extra['pearson'][name])} | {fmt(extra['spearman'][name])} | {ev['n']} |"
            )
    lines.append("")
    return lines


def best_md(ev):
    lines = []
    for label in ("pearson", "spearman"):
        d = ev[label]
        order = sorted(FAMILY, key=lambda k: -abs(d[k]) if not np.isnan(d[k]) else 0)
        top = order[:5]
        lines.append(f"- Best by |{label}|: " + ", ".join(f"{k} ({fmt(d[k])})" for k in top))
    lines.append(f"- LOADONLY: sum {fmt(ev['pearson']['LOADONLY_sum'])}/{fmt(ev['spearman']['LOADONLY_sum'])}, "
                 f"pg {fmt(ev['pearson']['LOADONLY_pg'])}/{fmt(ev['spearman']['LOADONLY_pg'])}, "
                 f"p100 {fmt(ev['pearson']['LOADONLY_p100'])}/{fmt(ev['spearman']['LOADONLY_p100'])} (Pearson/Spearman)")
    return lines


def sweep_md(X, y, n):
    lines = ["| variant | metric | argmax s | r(argmax) | r(s=0) | r(s=1) | max-min | shape | bootstrap 95% s | bootstrap argmax distribution |",
             "|---|---|---:|---:|---:|---:|---:|---|---|---|"]
    for suffix in ("_sum", "_pg", "_p100"):
        res = sweep_bootstrap(X, y, suffix)
        for label in ("pearson", "spearman"):
            r = res[label]
            hist = ", ".join(f"{s:g}:{c}" for s, c in sorted(r["boot_hist"].items()))
            lines.append(
                f"| F(s){suffix} | {label} | {r['argmax_s']:g}{'*' if r['argmax_s'] in (S_GRID[0], S_GRID[-1]) else ''} | {fmt(r['r_at_max'])} | {fmt(r['r_at_0'])} | {fmt(r['r_at_1'])} | "
                f"{r['range']:.3f} | {'flat' if r['flat'] else 'peaked'} | [{r['ci'][0]:g}, {r['ci'][1]:g}] | {hist} |"
            )
    lines.append("")
    lines.append(f"(`*` = arg-max at the edge of the s grid [{S_GRID[0]:g}, {S_GRID[-1]:g}]; the optimum may lie outside the grid)")
    lines.append("")
    lines.append(f"(bootstrap: {N_BOOT} resamples over the {n} player-seasons / team-seasons in the table; "
                 f"'flat' = max-min of the correlation curve across s < {FLAT_TOL})")
    lines.append("")
    return lines


def identity_md(X, label):
    """Report numerically identical and rank-identical (|Spearman| >= 0.999) family pairs on sample X."""
    lines = [f"Identity check on the {label} sample (n = {X.shape[0]}):", ""]
    R = rankdata(X, axis=0)
    Rc = R - R.mean(axis=0)
    Rn = Rc / np.sqrt((Rc ** 2).sum(axis=0))
    S = Rn.T @ Rn
    k = len(FAMILY)
    parent = list(range(k))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    numeric_pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            if abs(S[i, j]) >= 0.999:
                parent[find(i)] = find(j)
                if np.allclose(X[:, i], X[:, j], rtol=1e-9, atol=1e-9):
                    numeric_pairs.append((FAMILY[i], FAMILY[j]))
    groups = defaultdict(list)
    for i in range(k):
        groups[find(i)].append(FAMILY[i])
    groups = [g for g in groups.values() if len(g) > 1]
    if numeric_pairs:
        lines.append("- Numerically identical: " + "; ".join(f"{a} == {b}" for a, b in numeric_pairs))
    else:
        lines.append("- Numerically identical: none")
    if groups:
        lines.append("- Rank-identical groups (|Spearman| >= 0.999 pairwise-connected):")
        for g in groups:
            lines.append("  - " + ", ".join(g))
    else:
        lines.append("- Rank-identical groups: none")
    # near-identical summary for the F(s) sweep neighbours
    lines.append("")
    return lines


# --------------------------------------------------------------------------- main

def main():
    t0 = time.time()
    md = []
    md.append("# Scoring-statistic family benchmark")
    md.append("")
    md.append(f"Generated by `_bench_scoring.py`. Gate: gp >= {GP_MIN} and on-court possessions P >= {P_MIN} "
              f"(player-seasons); pure RAPM gate ps >= {PS_MIN}. s grid: {S_GRID}. "
              f"Bootstrap {N_BOOT} resamples. All correlations are Pearson / Spearman with n stated per table.")
    md.append("")
    md.append("Family definitions (per player-season): PPG = PTS/gp; TS = PTS/(2 TSA); rTS = 100 (TS - lgTS); "
              "TSAdd = PTS - price*TSA; UWrTS = rTS * (TSA/P*100) / mean over gated player-seasons of (TSA/P*100); "
              "F(s) = TSAdd + s*TSA*L; LOADONLY = TSA*L. `_sum` = season total, `_pg` = / games, `_p100` = / P * 100. "
              "price = season sum(pts)/sum(tsa) over all rows; lgTS = price/2. L = l_season from the per-game file. "
              "Team-level: the same formulas on team totals (sum over the team's rows, no player gate; "
              "team L := sum(tsa*L)/sum(tsa) so team LOADONLY = sum of row-level tsa*L; per game = / team games).")
    md.append("")

    codes = season_codes()
    print("loading seasons ...", flush=True)
    seasons = {}
    for c in codes:
        seasons[c] = Season(c)
        print(f"  {c}: rows={seasons[c].n_rows} players={len(seasons[c].players)} gated={len(seasons[c].gated)} "
              f"price={seasons[c].price:.4f} teams={len(seasons[c].teams)}", flush=True)

    # data sanity
    md.append("## Data sanity")
    md.append("")
    md.append("| season | rows | games | games with on-court data | players | gated player-seasons | price (pts/TSA) | lgTS | max within-player l_season spread | max |L - TSA/P| (gated) | teams | team games check |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for c in codes:
        S = seasons[c]
        tg = sorted(set(int(x) for x in S.team_games))
        md.append(f"| {season_label(c)} | {S.n_rows} | {S.coverage['games']} | {S.coverage['covered_games']} | {len(S.players)} | {len(S.gated)} | {S.price:.4f} | {S.price/2:.4f} | "
                  f"{S.L_dev:.4f} | {S.L_vs_ratio:.4f} | {len(S.teams)} | games/team {tg[0]}..{tg[-1]} |")
    md.append("")
    md.append("`p_g` is null for every row of a game without on-court data. P for a player = covered sum of p_g scaled by gp/covered games "
              "(mean covered on-court possessions imputed to uncovered games); this P feeds the gate, `_p100`, and UWrTS. "
              "l_season is taken as given from the file.")
    md.append("")
    md.append("Team win% is derived from the per-game rows: team points = sum of player pts per gid; higher total wins. "
              "(the `team_check.winner_higher` field in the source summaries is the upstream check of that same construction.)")
    md.append("")

    # ---------------------------------------------------------------- target 1: same-season OBPM
    print("target 1: same-season OBPM/DBPM", flush=True)
    obpm_codes = [c for c in codes if (int(c[:2]) < 90)]  # 2000-01 .. 2025-26
    match_rows = []
    per_season = {}
    pooled_X, pooled_o, pooled_d, pooled_g = [], [], [], []
    for c in obpm_codes:
        S = seasons[c]
        ob = load_obpm(c)
        if ob is None or S.fam is None:
            continue
        mo = NameMatcher(ob["obpm"])
        md_ = NameMatcher(ob["dbpm"])
        idx, o, d = [], [], []
        for p in S.gated:
            v = mo.get(S.players[p]["name"])
            w = md_.get(S.players[p]["name"])
            if v is None or w is None:
                continue
            idx.append(S.idx[p])
            o.append(float(v))
            d.append(float(w))
        idx = np.array(idx, dtype=int)
        o = np.array(o)
        d = np.array(d)
        match_rows.append((c, len(S.gated), mo.n_exact, mo.n_loose, mo.n_prefix, mo.n_miss, len(ob["obpm"])))
        if len(idx) < 10:
            continue
        X = stat_matrix([(S.fam, idx)], None)
        per_season[c] = (evaluate(X, o), evaluate(X, d))
        pooled_X.append(X)
        pooled_o.append(o)
        pooled_d.append(d)
        pooled_g.append(np.full(len(o), int(c)))
    Xp = np.vstack(pooled_X)
    op = np.concatenate(pooled_o)
    dp = np.concatenate(pooled_d)
    gp_ = np.concatenate(pooled_g)
    ev_o = evaluate(Xp, op, gp_)
    ev_d = evaluate(Xp, dp, gp_)

    md.append("## Name matching (per-game names -> Basketball-Reference OBPM/DBPM)")
    md.append("")
    md.append("Normalization: NFKD, strip combining marks, transliterate ı/ø/đ/ł/ß, lowercase, drop `.`/`'`, `-`->space, collapse whitespace. "
              "Fallbacks (only when exact fails): (a) strip suffixes jr/sr/ii/iii/iv/v; (b) same last name and one first name a prefix of the other "
              "(e.g. 'Clar. Weatherspoon' -> 'Clarence Weatherspoon'), accepted only when unique.")
    md.append("")
    md.append("| season | gated player-seasons | exact match | suffix-fallback match | first-name-prefix fallback | unmatched | match rate | BBRef players in file |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for c, ng, ne, nl, npf, nm, nb in match_rows:
        md.append(f"| {season_label(c)} | {ng} | {ne} | {nl} | {npf} | {nm} | {(ne+nl+npf)/ng:.3f} | {nb} |")
    tot = np.array([(r[1], r[2], r[3], r[4], r[5]) for r in match_rows]).sum(axis=0)
    md.append(f"| **all** | {tot[0]} | {tot[1]} | {tot[2]} | {tot[3]} | {tot[4]} | {(tot[1]+tot[2]+tot[3])/tot[0]:.3f} | |")
    md.append("")
    unmatched_names = Counter()
    for c in obpm_codes:
        S = seasons[c]
        ob = load_obpm(c)
        if ob is None:
            continue
        mo = NameMatcher(ob["obpm"])
        for p in S.gated:
            if mo.get(S.players[p]["name"]) is None:
                unmatched_names[S.players[p]["name"]] += 1
    if unmatched_names:
        md.append("Unmatched gated names (count of seasons): " + ", ".join(f"{k} ({v})" for k, v in unmatched_names.most_common(40)))
        md.append("")

    md.append("## Target 1: same-season OBPM (DBPM as negative control)")
    md.append("")
    md.append("### Pooled 2000-01 .. 2025-26, season-demeaned (stat and target both demeaned within season)")
    md.append("")
    md.extend(table_md(ev_o, ev_d))
    md.extend(best_md(ev_o))
    dn = ev_d["pearson"]
    md.append(f"- DBPM negative control (pooled, Pearson): PPG {fmt(dn['PPG'])}, rTS {fmt(dn['rTS'])}, TSAdd_sum {fmt(dn['TSAdd_sum'])}, "
              f"F(0.42)_sum {fmt(dn['F(0.42)_sum'])}, F(1)_sum {fmt(dn['F(1)_sum'])}, LOADONLY_sum {fmt(dn['LOADONLY_sum'])}; "
              f"max |r| over family = {max(abs(v) for v in dn.values()):.3f} ({max(dn, key=lambda k: abs(dn[k]))})")
    md.append("")
    md.append("#### F(s) sweep vs pooled OBPM")
    md.append("")
    md.extend(sweep_md(ev_o["X"], ev_o["y"], ev_o["n"]))
    md.append("#### F(s) sweep vs pooled DBPM (negative control)")
    md.append("")
    md.extend(sweep_md(ev_d["X"], ev_d["y"], ev_d["n"]))
    md.extend(identity_md(ev_o["X"], "pooled season-demeaned OBPM"))

    md.append("### Per-season summary (same-season OBPM)")
    md.append("")
    md.append("| season | n | best Pearson member | best Spearman member | argmax s (sum / pg / p100, Pearson) | r F(0)_sum | r F(1)_sum | LOADONLY_sum r | rTS r | PPG r | DBPM max |r| |")
    md.append("|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|")
    for c in obpm_codes:
        if c not in per_season:
            continue
        eo, ed = per_season[c]
        pr = eo["pearson"]
        sp = eo["spearman"]
        bp = max(FAMILY, key=lambda k: pr[k] if not np.isnan(pr[k]) else -9)
        bs = max(FAMILY, key=lambda k: sp[k] if not np.isnan(sp[k]) else -9)
        am = []
        for suf in ("_sum", "_pg", "_p100"):
            curve = [pr[f"F({s:g}){suf}"] for s in S_GRID]
            am.append(f"{S_GRID[int(np.nanargmax(curve))]:g}")
        dmax = max(abs(v) for v in ed["pearson"].values())
        md.append(f"| {season_label(c)} | {eo['n']} | {bp} ({fmt(pr[bp])}) | {bs} ({fmt(sp[bs])}) | {' / '.join(am)} | "
                  f"{fmt(pr['F(0)_sum'])} | {fmt(pr['F(1)_sum'])} | {fmt(pr['LOADONLY_sum'])} | {fmt(pr['rTS'])} | {fmt(pr['PPG'])} | {dmax:.3f} |")
    md.append("")
    md.append("Full per-season tables are in the appendix.")
    md.append("")

    # ---------------------------------------------------------------- target 2: same-window pure oRAPM
    print("target 2: same-window pure oRAPM", flush=True)
    anchors = ["2223", "2324", "2425", "2526"]
    md.append("## Target 2: same-window pure oRAPM (pO), pN secondary")
    md.append("")
    md.append(f"Stat from the anchor season's per-game rows (gated); target = pure RAPM with 3-season window ending at the anchor, ps >= {PS_MIN}. "
              "Join on pid. The window actually stored in each file is listed below (the 2223 file carries a 2-season window). "
              "Adjacent anchors share seasons, so the pooled n counts player-seasons whose targets are not independent across anchors.")
    md.append("")
    md.append("| anchor | window | alpha | rapm players | ps>=2000 | gated stat players | joined n |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    rapm = {}
    t2_X, t2_pO, t2_pN, t2_g = [], [], [], []
    t2_ev = {}
    for a in anchors:
        R = load_rapm(a)
        rapm[a] = R
        S = seasons[a]
        keep = {str(p["pid"]): p for p in R["players"] if p["ps"] >= PS_MIN}
        idx, pO, pN = [], [], []
        for p in S.gated:
            q = keep.get(p)
            if q is None:
                continue
            idx.append(S.idx[p])
            pO.append(float(q["pO"]))
            pN.append(float(q["pN"]))
        idx = np.array(idx, dtype=int)
        pO = np.array(pO)
        pN = np.array(pN)
        X = stat_matrix([(S.fam, idx)], None)
        t2_ev[a] = (evaluate(X, pO), evaluate(X, pN))
        t2_X.append(X)
        t2_pO.append(pO)
        t2_pN.append(pN)
        t2_g.append(np.full(len(pO), int(a)))
        md.append(f"| {a} | {R['window']} | {R['alpha']:g} | {len(R['players'])} | {len(keep)} | {len(S.gated)} | {len(idx)} |")
    md.append("")
    X2 = np.vstack(t2_X)
    g2 = np.concatenate(t2_g)
    ev2O = evaluate(X2, np.concatenate(t2_pO), g2)
    ev2N = evaluate(X2, np.concatenate(t2_pN), g2)
    md.append("### Pooled over the 4 anchors, anchor-demeaned -- target pO")
    md.append("")
    md.extend(table_md(ev2O))
    md.extend(best_md(ev2O))
    md.append("")
    md.append("#### F(s) sweep vs pooled same-window pO")
    md.append("")
    md.extend(sweep_md(ev2O["X"], ev2O["y"], ev2O["n"]))
    md.append("### Pooled over the 4 anchors, anchor-demeaned -- secondary target pN")
    md.append("")
    md.extend(table_md(ev2N))
    md.extend(best_md(ev2N))
    md.append("")
    md.append("#### F(s) sweep vs pooled same-window pN")
    md.append("")
    md.extend(sweep_md(ev2N["X"], ev2N["y"], ev2N["n"]))
    md.append("### Per-anchor summary (pO)")
    md.append("")
    md.append("| anchor | n | best Pearson | best Spearman | argmax s (sum/pg/p100, Pearson) | r F(0)_sum | r F(1)_sum | LOADONLY_sum r | rTS r | PPG r |")
    md.append("|---|---:|---|---|---|---:|---:|---:|---:|---:|")
    for a in anchors:
        eo, en = t2_ev[a]
        pr = eo["pearson"]
        sp = eo["spearman"]
        bp = max(FAMILY, key=lambda k: pr[k])
        bs = max(FAMILY, key=lambda k: sp[k])
        am = [f"{S_GRID[int(np.nanargmax([pr[f'F({s:g}){suf}'] for s in S_GRID]))]:g}" for suf in ("_sum", "_pg", "_p100")]
        md.append(f"| {a} | {eo['n']} | {bp} ({fmt(pr[bp])}) | {bs} ({fmt(sp[bs])}) | {' / '.join(am)} | {fmt(pr['F(0)_sum'])} | "
                  f"{fmt(pr['F(1)_sum'])} | {fmt(pr['LOADONLY_sum'])} | {fmt(pr['rTS'])} | {fmt(pr['PPG'])} |")
    md.append("")
    md.append("Full per-anchor tables (pO and pN) are in the appendix.")
    md.append("")

    # ---------------------------------------------------------------- target 3: OOS pure oRAPM
    print("target 3: out-of-sample pure oRAPM", flush=True)
    md.append("## Target 3: out-of-sample pure oRAPM -- stat from 2022-23, target = pure RAPM anchored 2025-26 (window 2324-2526, disjoint)")
    md.append("")
    S = seasons["2223"]
    R = rapm["2526"]
    keep = {str(p["pid"]): p for p in R["players"] if p["ps"] >= PS_MIN}
    idx, pO, pN = [], [], []
    for p in S.gated:
        q = keep.get(p)
        if q is None:
            continue
        idx.append(S.idx[p])
        pO.append(float(q["pO"]))
        pN.append(float(q["pN"]))
    idx = np.array(idx, dtype=int)
    X3 = stat_matrix([(S.fam, idx)], None)
    ev3O = evaluate(X3, np.array(pO))
    ev3N = evaluate(X3, np.array(pN))
    md.append(f"Gated 2022-23 player-seasons: {len(S.gated)}; 2526-anchor RAPM players with ps >= {PS_MIN}: {len(keep)}; joined n = {len(idx)}. "
              "2324 -> 2526 is skipped (window overlaps). Single season, no demeaning.")
    md.append("")
    md.append("### Target pO (2526 anchor)")
    md.append("")
    md.extend(table_md(ev3O))
    md.extend(best_md(ev3O))
    md.append("")
    md.append("#### F(s) sweep vs OOS pO")
    md.append("")
    md.extend(sweep_md(ev3O["X"], ev3O["y"], ev3O["n"]))
    md.append("### Secondary target pN (2526 anchor)")
    md.append("")
    md.extend(table_md(ev3N))
    md.extend(best_md(ev3N))
    md.append("")
    md.append("#### F(s) sweep vs OOS pN")
    md.append("")
    md.extend(sweep_md(ev3N["X"], ev3N["y"], ev3N["n"]))

    # ---------------------------------------------------------------- target 4: next-season OBPM
    print("target 4: next-season OBPM", flush=True)
    md.append("## Target 4: next-season OBPM (stat in t -> OBPM in t+1, same player), pooled, season-t-demeaned")
    md.append("")
    md.append("Primary: t gated, t+1 only requires an OBPM entry (matched by the player's t+1 name if they appear in the t+1 per-game file, else the t name). "
              f"Secondary: t+1 also gated (gp >= {GP_MIN}, P >= {P_MIN}).")
    md.append("")
    md.append("| season t | t+1 | gated in t | in t+1 per-game file | OBPM(t+1) matched | also gated in t+1 |")
    md.append("|---|---|---:|---:|---:|---:|")
    X4, y4, g4, X4b, y4b, g4b = [], [], [], [], [], []
    for c in obpm_codes:
        nc = next_code(c)
        if nc not in seasons:
            continue
        S = seasons[c]
        S2 = seasons[nc]
        ob = load_obpm(nc)
        if ob is None or S.fam is None:
            continue
        mo = NameMatcher(ob["obpm"])
        idx, y, idxb, yb = [], [], [], []
        in_next = 0
        for p in S.gated:
            nm = S2.players[p]["name"] if p in S2.players else S.players[p]["name"]
            if p in S2.players:
                in_next += 1
            v = mo.get(nm)
            if v is None and p in S2.players:
                v = mo.get(S.players[p]["name"])
            if v is None:
                continue
            idx.append(S.idx[p])
            y.append(float(v))
            if p in S2.idx:
                idxb.append(S.idx[p])
                yb.append(float(v))
        md.append(f"| {season_label(c)} | {season_label(nc)} | {len(S.gated)} | {in_next} | {len(idx)} | {len(idxb)} |")
        if len(idx) >= 10:
            X4.append(stat_matrix([(S.fam, np.array(idx, dtype=int))], None))
            y4.append(np.array(y))
            g4.append(np.full(len(y), int(c)))
        if len(idxb) >= 10:
            X4b.append(stat_matrix([(S.fam, np.array(idxb, dtype=int))], None))
            y4b.append(np.array(yb))
            g4b.append(np.full(len(yb), int(c)))
    md.append("")
    ev4 = evaluate(np.vstack(X4), np.concatenate(y4), np.concatenate(g4))
    ev4b = evaluate(np.vstack(X4b), np.concatenate(y4b), np.concatenate(g4b))
    md.append("### Primary (t gated only)")
    md.append("")
    md.extend(table_md(ev4))
    md.extend(best_md(ev4))
    md.append("")
    md.append("#### F(s) sweep vs next-season OBPM (primary)")
    md.append("")
    md.extend(sweep_md(ev4["X"], ev4["y"], ev4["n"]))
    md.append("### Secondary (t and t+1 both gated)")
    md.append("")
    md.extend(table_md(ev4b))
    md.extend(best_md(ev4b))
    md.append("")
    md.append("#### F(s) sweep vs next-season OBPM (both gated)")
    md.append("")
    md.extend(sweep_md(ev4b["X"], ev4b["y"], ev4b["n"]))

    # ---------------------------------------------------------------- target 5: team level
    print("target 5: team level", flush=True)
    md.append("## Target 5: team-level -- team totals of each family member vs team win% (same season, next season)")
    md.append("")
    md.append("Team totals use every row of the team (no player gate). Rate members (TS, rTS, UWrTS, PPG, `_p100`) are the team-total versions "
              "(team PTS/(2 team TSA), team TSA/P*100 where P = sum of player on-court possessions, PPG = team pts per game). Pooled over all seasons, season-demeaned.")
    md.append("")
    tX, ty, tg_ = [], [], []
    tXn, tyn, tgn = [], [], []
    team_rows = []
    for c in codes:
        S = seasons[c]
        X = np.column_stack([S.team_fam[name] for name in FAMILY])
        tX.append(X)
        ty.append(S.team_winpct)
        tg_.append(np.full(len(S.team_ids), int(c)))
        nc = next_code(c)
        nn = 0
        if nc in seasons:
            S2 = seasons[nc]
            pos = {t: i for i, t in enumerate(S2.team_ids)}
            keep = [i for i, t in enumerate(S.team_ids) if t in pos]
            if keep:
                tXn.append(X[keep])
                tyn.append(np.array([S2.team_winpct[pos[S.team_ids[i]]] for i in keep]))
                tgn.append(np.full(len(keep), int(c)))
                nn = len(keep)
        team_rows.append((c, len(S.team_ids), nn))
    md.append("| season | teams | teams with next-season match |")
    md.append("|---|---:|---:|")
    for c, nt, nn in team_rows:
        md.append(f"| {season_label(c)} | {nt} | {nn} |")
    md.append("")
    ev5 = evaluate(np.vstack(tX), np.concatenate(ty), np.concatenate(tg_))
    ev5n = evaluate(np.vstack(tXn), np.concatenate(tyn), np.concatenate(tgn))
    md.append("### Same-season win%")
    md.append("")
    md.extend(table_md(ev5))
    md.extend(best_md(ev5))
    md.append("")
    md.append("#### F(s) sweep vs same-season team win%")
    md.append("")
    md.extend(sweep_md(ev5["X"], ev5["y"], ev5["n"]))
    md.append("### Next-season win% (same franchise id)")
    md.append("")
    md.extend(table_md(ev5n))
    md.extend(best_md(ev5n))
    md.append("")
    md.append("#### F(s) sweep vs next-season team win%")
    md.append("")
    md.extend(sweep_md(ev5n["X"], ev5n["y"], ev5n["n"]))
    md.extend(identity_md(ev5["X"], "pooled season-demeaned team-season"))

    # ---------------------------------------------------------------- cross-target summary
    md.append("## Cross-target summary")
    md.append("")
    md.append("| target | n | best Pearson | best Spearman | argmax s sum / pg / p100 (Pearson) | shape (sum/pg/p100) | LOADONLY_sum r | TSAdd_sum r | rTS r | PPG r |")
    md.append("|---|---:|---|---|---|---|---:|---:|---:|---:|")
    summary_targets = [
        ("Same-season OBPM (pooled)", ev_o),
        ("Same-season DBPM (control, pooled)", ev_d),
        ("Same-window pure oRAPM pO (pooled)", ev2O),
        ("Same-window pure pN (pooled)", ev2N),
        ("OOS pure pO (2223 -> 2526)", ev3O),
        ("OOS pure pN (2223 -> 2526)", ev3N),
        ("Next-season OBPM (primary)", ev4),
        ("Next-season OBPM (both gated)", ev4b),
        ("Team win% same season", ev5),
        ("Team win% next season", ev5n),
    ]
    for label, ev in summary_targets:
        pr = ev["pearson"]
        sp = ev["spearman"]
        bp = max(FAMILY, key=lambda k: pr[k] if not np.isnan(pr[k]) else -9)
        bs = max(FAMILY, key=lambda k: sp[k] if not np.isnan(sp[k]) else -9)
        am, shp = [], []
        for suf in ("_sum", "_pg", "_p100"):
            curve = np.array([pr[f"F({s:g}){suf}"] for s in S_GRID])
            k = int(np.nanargmax(curve))
            am.append(f"{S_GRID[k]:g}{'*' if k in (0, len(S_GRID) - 1) else ''}")
            shp.append("flat" if (np.nanmax(curve) - np.nanmin(curve)) < FLAT_TOL else "peaked")
        md.append(f"| {label} | {ev['n']} | {bp} ({fmt(pr[bp])}) | {bs} ({fmt(sp[bs])}) | {' / '.join(am)} | {' / '.join(shp)} | "
                  f"{fmt(pr['LOADONLY_sum'])} | {fmt(pr['TSAdd_sum'])} | {fmt(pr['rTS'])} | {fmt(pr['PPG'])} |")
    md.append("")
    md.append("`*` = arg-max at the edge of the s grid. Identities: TSAdd == F(0) (all three denominators); UWrTS is a season-constant multiple of "
              "TSAdd_p100 (rTS * TSA/P = 50 * TSAdd / P), so they are rank-identical; TS and rTS are affine within a season, so rank-identical after season-demeaning.")
    md.append("")
    md.append("Bootstrap intervals for each arg-max s are in the per-target sweep tables above.")
    md.append("")

    # ---------------------------------------------------------------- appendix
    md.append("## Appendix A: per-season same-season OBPM / DBPM tables")
    md.append("")
    for c in obpm_codes:
        if c not in per_season:
            continue
        eo, ed = per_season[c]
        md.extend(table_md(eo, ed, title=f"{season_label(c)} same-season OBPM / DBPM"))
    md.append("## Appendix B: per-anchor same-window pure RAPM tables")
    md.append("")
    for a in anchors:
        eo, en = t2_ev[a]
        md.append(f"**anchor {a} (window {rapm[a]['window']})** (n = {eo['n']})")
        md.append("")
        md.append("| member | Pearson (pO) | Spearman (pO) | Pearson (pN) | Spearman (pN) | n |")
        md.append("|---|---:|---:|---:|---:|---:|")
        for name in FAMILY:
            md.append(f"| {name} | {fmt(eo['pearson'][name])} | {fmt(eo['spearman'][name])} | "
                      f"{fmt(en['pearson'][name])} | {fmt(en['spearman'][name])} | {eo['n']} |")
        md.append("")

    md.append(f"_Runtime: {time.time() - t0:.1f} s_")
    md.append("")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"wrote {OUT} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
