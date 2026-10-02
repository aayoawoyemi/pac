# -*- coding: utf-8 -*-
"""PAC (Points Above Cost), CAP (Consumption-Adjusted Points) and vPTS per player-game, from PBP + on-court possessions.

    OC_ig = TSA_ig * qbar(L_i)               opportunity cost of the scoring possessions consumed
    qbar(L) = 2*lgTS_s - SLOPE*L             (points a possession returns when reallocated away
                                              from a player carrying share L of the budget)
    L_i    = TSA_i / POSS_i                  (season share of on-court possessions, _nba_pl_adv_{code}.json)

    PAC_ig = PTS_ig - OC_ig                  Points Above Cost: the metric (leaderboards, paper)
    CAP_ig = PTS_ig + PAC_ig = 2*PTS - OC    Consumption-Adjusted Points: points-unit column; PAC=0 -> CAP=PTS
    vPTS_ig = PTS + PAC - (TSA/sum TSA)*sum PAC   team score re-divided by who earned it; sums to team PTS exactly

SLOPE = 0.25 (2026-09-21). Measured, not fitted. Held-out prediction of team pts/attempt in games a
>=24%-share carrier sat (739 carrier-seasons, 10,676 sit-outs, 2000-01 -> 2025-26, _calib_sitout.py):
arg-min 0.245, bootstrap SE 0.020, 95% CI [0.204, 0.284]; 10-fold CV fitted 0.233 (sd 0.006 across
folds); leave-one-season-out 0.22-0.24 on every fold. Independent blind replication (_blind_sitout.py)
0.28 [0.22, 0.34]. In-game rest variant (_ingame_slope.py) 0.32 [0.29, 0.35] = upper bound (bench
units absorb, not the rotation). Placebo (fake sit-outs from games played) 0.004 [-0.05, 0.06].
Teammate-absence IV on 480,013 player-games: own skill-curve slope -0.30 TS pts per unit share.
History: an earlier in-kernel build reported 0.42 (never saved as a script). It is outside every
CI above; the likely causes were sit-outs counted outside the carrier's tenure window and/or share
measured on team TSA rather than on-court possessions. Do not reuse 0.42.
The 1:1 weight on PAC inside CAP is a convention (every k in PTS + k*PAC satisfies the break-even
calibration); PAC is the fully-measured object.

The price is fixed per player-season (his role); the quantity (TSA) is per game, so per-game
values sum exactly to the season values. Per-game on-court possessions P_ig (from _leaf_stints)
and L_ig = TSA_ig/P_ig are carried on each row for display only.

Outputs per season:
    _sv_pergame_{code}.json   rows: gid, pid, name, team, pts, fga, fgm, fta, ftm, ft_trips, tov,
                              tsa, p_g, l_g, l_season, qbar, oc, pac, cap, vpts
    _sv_season_{code}.json    per player: gp, games_with_attempts, pts, tsa, poss, l, ts, rts, tsadd,
                              credit, pac, pac_g, pac100, cap, cap_g
and prints the team-sum check on PAC: corr(sum SS home - away, margin) and % of games the winner is higher.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pac import paths  # noqa: E402
from pac.pbp_events import iter_games, fill_legacy_teams_and_rebounds  # noqa: E402
HERE = paths.RAW                       # NBA.com player JSON dumps (_nba_pl_adv_*, _nba_pl_base_*) live in raw/
RAW = paths.raw("nba_data_raw")        # play-by-play CSVs
SLOPE = 0.25
SCOPES = ("tsa", "tsatov")   # possessions charged: scoring attempts only, or attempts + turnovers
FT_WEIGHT = 0.44
FT_TRIP_RE = re.compile(r"(\d) of (\d)")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def code_of(year: int) -> str:
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


def load_json(name: str):
    with open(os.path.join(HERE, name), encoding="utf-8") as handle:
        return json.load(handle)


def columnar_rows(data: dict) -> list[dict]:
    keys = list(data.keys())
    n = len(data[keys[0]])
    return [{k: data[k][str(i)] if str(i) in data[k] else data[k].get(i) for k in keys} for i in range(n)]


def pbp_path(year: int) -> tuple[str, bool]:
    modern = year >= 2021
    path = os.path.join(RAW, f"{'cdnnba' if modern else 'nbastats'}_{year}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path, modern


def player_games(year: int):
    """(gid, pid) -> Counter and (gid, team) -> Counter from PBP. gid is int."""
    path, modern = pbp_path(year)
    pg: dict[tuple[int, str], collections.Counter] = collections.defaultdict(collections.Counter)
    tg: dict[tuple[int, str], collections.Counter] = collections.defaultdict(collections.Counter)
    pteam: dict[tuple[int, str], str] = {}
    for gid_s, events in iter_games(path, modern):
        try:
            gid = int(gid_s)
        except ValueError:
            continue
        if not modern:
            fill_legacy_teams_and_rebounds(events)
        for ev in events:
            if ev.kind not in {"fg", "ft", "turnover", "oreb"}:
                continue
            team = ev.team
            if ev.kind == "fg":
                tg[(gid, team)]["fga"] += 1
                tg[(gid, team)]["pts"] += ev.points
                if not ev.is_team and ev.player:
                    c = pg[(gid, ev.player)]
                    c["fga"] += 1
                    c["fgm"] += int(ev.made)
                    c["pts"] += ev.points
                    pteam[(gid, ev.player)] = team
            elif ev.kind == "ft":
                detail_l = ev.detail.lower()
                technical = "technical" in detail_l
                if not technical:
                    tg[(gid, team)]["fta"] += 1
                tg[(gid, team)]["pts"] += ev.points
                if not ev.is_team and ev.player:
                    c = pg[(gid, ev.player)]
                    c["pts"] += ev.points
                    c["ftm"] += int(ev.made)
                    if not technical:
                        c["fta"] += 1
                        m = FT_TRIP_RE.search(ev.detail)
                        if m and m.group(1) == m.group(2):
                            c["ft_trips"] += 1
                    pteam[(gid, ev.player)] = team
            elif ev.kind == "turnover":
                tg[(gid, team)]["tov"] += 1
                if not ev.is_team and ev.player:
                    pg[(gid, ev.player)]["tov"] += 1
                    pteam[(gid, ev.player)] = team
            elif ev.kind == "oreb":
                tg[(gid, team)]["oreb"] += 1
    return pg, tg, pteam


def oncourt_possessions(code: str):
    """(gid, pid) -> on-court possessions and (gid, side_team_key) from _leaf_stints. gid is int."""
    from pac import leaf_poss as lp  # noqa: WPS433

    try:
        leaf = lp.load_leaf(code)
    except Exception:
        return {}
    out: dict[tuple[int, int], float] = collections.defaultdict(float)
    for gid_s, stints in leaf.items():
        gid = int(gid_s)
        for row in stints:
            home, away, hv, av = row[0], row[1], row[5], row[6]
            if not hv or not av or len(hv) < 10:
                continue
            for lineup, v in ((home, hv), (away, av)):
                poss = (v[0] + v[2] + v[4]) + FT_WEIGHT * v[6] + v[8] - v[9]
                for pid in lineup:
                    out[(gid, int(pid))] += poss
    return out


def season_shares(code: str, pg) -> tuple[dict[str, float], dict[str, str], dict[str, float], dict[str, int]]:
    """L_i from season TSA (PBP) / on-court POSS (adv file); names; season possessions; official GP."""
    adv = columnar_rows(load_json(f"_nba_pl_adv_{code}.json"))
    base = columnar_rows(load_json(f"_nba_pl_base_{code}.json"))
    poss = {str(int(float(r["PLAYER_ID"]))): float(r["POSS"]) for r in adv if r.get("POSS")}
    names = {str(int(float(r["PLAYER_ID"]))): str(r["PLAYER_NAME"]) for r in base}
    gp_official = {str(int(float(r["PLAYER_ID"]))): int(float(r["GP"])) for r in base if r.get("GP")}
    tsa_season: collections.Counter = collections.Counter()
    for (_, pid), c in pg.items():
        tsa_season[pid] += c["fga"] + FT_WEIGHT * c["fta"]
    shares = {pid: (tsa_season[pid] / poss[pid]) for pid in tsa_season if poss.get(pid, 0) > 0}
    return shares, names, poss, gp_official


def run_season(year: int, slope: float, scope: str = "tsa", write: bool = True) -> dict:
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    code = code_of(year)
    pg, tg, pteam = player_games(year)
    oncourt = oncourt_possessions(code)
    shares, names, poss, gp_official = season_shares(code, pg)

    lg_pts = sum(c["pts"] for c in tg.values())
    lg_tsa = sum(c["fga"] + FT_WEIGHT * c["fta"] for c in tg.values())
    lg_ts = lg_pts / (2 * lg_tsa)
    lam = 2 * lg_ts

    rows = []
    team_sv: dict[tuple[int, str], float] = collections.defaultdict(float)
    team_tsa: dict[tuple[int, str], float] = collections.defaultdict(float)
    for (gid, pid), c in pg.items():
        tsa = c["fga"] + FT_WEIGHT * c["fta"]
        l_season = shares.get(pid)
        credit = slope * l_season if l_season is not None else 0.0
        qbar = lam - max(credit, 0.0)
        charged = tsa + (c["tov"] if scope == "tsatov" else 0)
        oc = charged * qbar
        pac = c["pts"] - oc
        cap = c["pts"] + pac
        p_g = oncourt.get((gid, int(pid)))
        team = pteam.get((gid, pid), "")
        team_sv[(gid, team)] += pac
        team_tsa[(gid, team)] += tsa
        rows.append(
            dict(
                gid=gid, pid=pid, name=names.get(pid, ""), team=team,
                pts=c["pts"], fga=c["fga"], fgm=c["fgm"], fta=c["fta"], ftm=c["ftm"],
                ft_trips=c["ft_trips"], tov=c["tov"], tsa=round(tsa, 2),
                p_g=round(p_g, 1) if p_g is not None else None,
                l_g=round(tsa / p_g, 3) if p_g else None,
                l_season=round(l_season, 3) if l_season is not None else None,
                qbar=round(qbar, 4), oc=round(oc, 2), pac=round(pac, 2), cap=round(cap, 2), _pac=pac,
            )
        )
    # vPTS: re-divide the team's actual points by who earned them; sums to team PTS exactly
    for r in rows:
        key = (r["gid"], r["team"])
        share = r["tsa"] / team_tsa[key] if team_tsa[key] else 0.0
        r["vpts"] = round(r["pts"] + r.pop("_pac") - share * team_sv[key], 2)

    # team-sum check: winner's SV total higher?
    by_game: dict[int, list[tuple[str, float, float]]] = collections.defaultdict(list)
    for (gid, team), c in tg.items():
        if team:
            by_game[gid].append((team, float(c["pts"]), team_sv[(gid, team)]))
    agree = n = 0
    xs: list[float] = []
    ys: list[float] = []
    for gid, sides in by_game.items():
        if len(sides) != 2 or sides[0][1] == sides[1][1]:
            continue
        margin = sides[0][1] - sides[1][1]
        d_sv = sides[0][2] - sides[1][2]
        n += 1
        agree += int((margin > 0) == (d_sv > 0))
        xs.append(margin)
        ys.append(d_sv)
    corr = None
    if n > 2:
        mx, my = sum(xs) / n, sum(ys) / n
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        corr = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (vx * vy) ** 0.5 if vx and vy else None

    # season aggregates
    agg: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in rows:
        a = agg[r["pid"]]
        a["games_with_attempts"] += 1
        a["pts"] += r["pts"]
        a["tsa"] += r["tsa"]
        a["pac"] += r["pac"]
    season = []
    for pid, a in agg.items():
        p = poss.get(pid, 0.0)
        ts = a["pts"] / (2 * a["tsa"]) if a["tsa"] else 0.0
        l = shares.get(pid)
        gp = gp_official.get(pid) or a["games_with_attempts"]   # zero-attempt games carry SV 0 but still count
        season.append(
            dict(
                pid=pid, name=names.get(pid, ""), gp=gp, games_with_attempts=a["games_with_attempts"],
                pts=a["pts"], tsa=round(a["tsa"], 1),
                poss=p, l=round(l, 3) if l is not None else None,
                ts=round(100 * ts, 1), rts=round(100 * (ts - lg_ts), 1),
                tsadd=round(a["pts"] - lam * a["tsa"], 1),
                credit=round(slope * l * a["tsa"], 1) if l is not None else 0.0,
                pac=round(a["pac"], 1), pac_g=round(a["pac"] / gp, 2),
                pac100=round(100 * a["pac"] / p, 2) if p else None,
                cap=round(a["pts"] + a["pac"], 1), cap_g=round((a["pts"] + a["pac"]) / gp, 2),
            )
        )
    season.sort(key=lambda r: -r["pac"])

    summary = dict(
        code=code, games=len(by_game), player_games=len(rows), lg_ts=round(100 * lg_ts, 2), slope=slope, scope=scope,
        oncourt_coverage=round(sum(r["p_g"] is not None for r in rows) / max(len(rows), 1), 3),
        team_check=dict(n=n, winner_higher=round(agree / n, 4) if n else None, corr_margin=round(corr, 4) if corr else None),
    )
    if write:
        with open(os.path.join(HERE, f"_sv_pergame_{code}.json"), "w", encoding="utf-8") as h:
            json.dump(dict(summary=summary, rows=rows), h)
        with open(paths.pac_season(code), "w", encoding="utf-8") as h:
            json.dump(dict(summary=summary, rows=season), h, indent=1)
    print(json.dumps(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, required=True, help="season start years, e.g. 2005 2025")
    parser.add_argument("--slope", type=float, default=SLOPE)
    parser.add_argument("--scope", choices=SCOPES, default="tsa")
    args = parser.parse_args()
    for year in args.years:
        run_season(year, args.slope, args.scope)


if __name__ == "__main__":
    main()
