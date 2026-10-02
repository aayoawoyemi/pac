# -*- coding: utf-8 -*-
"""_pacgl_build.py -- stage 1 of the game-level PAC price estimation: build the team-game panel.

Per regular season 1997-98 .. 2025-26:
  * participation: every (game, player, team) with ANY play-by-play event (shots, fouls, rebounds,
    substitutions, ...), so zero-stat appearances are appearances, not absences.
  * home/away: NBA PBP person types (4 = home player, 5 = visitor player); 2025-26 CDN feed: score deltas.
  * dates: ESPN schedules (_pacgl_espn_sched.json), matched to NBA game ids by (home, away) pair in
    game-id order; used only for rest-day controls.
  * box: player-game points / FGA / FTA / TOV from _sv_pergame_{code}.json (same source PAC is built on).

Output: _pacgl_season_{code}.pkl  with
    pg : player-game rows for every appearance (zero-stat appearances get zeros)
    tg : team-game rows (gid, team, opp, home, date, rest, opp_rest, pts, fga, fta, tov)
    meta: match diagnostics
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "nba_data_raw")
YEARS = range(1997, 2026)   # season start years

NAME_KEYS = [  # (substring of ESPN displayName, NBA franchise id); first match wins
    ("Charlotte Hornets", 1610612766), ("NO/Oklahoma City", 1610612740),
    ("Hawks", 1610612737), ("Celtics", 1610612738), ("Cavaliers", 1610612739), ("Orleans", 1610612740),
    ("Pelicans", 1610612740), ("Bulls", 1610612741), ("Mavericks", 1610612742), ("Nuggets", 1610612743),
    ("Warriors", 1610612744), ("Rockets", 1610612745), ("Clippers", 1610612746), ("Lakers", 1610612747),
    ("Heat", 1610612748), ("Bucks", 1610612749), ("Timberwolves", 1610612750), ("Nets", 1610612751),
    ("Knicks", 1610612752), ("Magic", 1610612753), ("Pacers", 1610612754), ("76ers", 1610612755),
    ("Suns", 1610612756), ("Trail Blazers", 1610612757), ("Kings", 1610612758), ("Spurs", 1610612759),
    ("SuperSonics", 1610612760), ("Thunder", 1610612760), ("Raptors", 1610612761), ("Jazz", 1610612762),
    ("Grizzlies", 1610612763), ("Wizards", 1610612764), ("Pistons", 1610612765), ("Charlotte Hornets", 1610612766),
    ("Bobcats", 1610612766),
]


def code_of(y: int) -> str:
    return f"{y % 100:02d}{(y + 1) % 100:02d}"


def espn_franchise(name: str | None):
    if not name:
        return None
    name = " ".join(name.split())
    for key, tid in NAME_KEYS:
        if key in name:
            return tid
    return None


def local_date(utc: str):
    # every NBA tip is between 16:00 and 06:30 UTC; shifting 7h maps each game to its US calendar date
    t = dt.datetime.strptime(utc[:16], "%Y-%m-%dT%H:%M")
    return (t - dt.timedelta(hours=7)).date()


def mode(s: pd.Series):
    return s.value_counts().index[0]


def parse_nbastats(year: int):
    path = os.path.join(RAW, f"nbastats_{year}.csv")
    cols = ["GAME_ID"] + [f"{a}{k}{b}" for k in (1, 2, 3) for a, b in (("PERSON", "TYPE"), ("PLAYER", "_ID"), ("PLAYER", "_TEAM_ID"))]
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    frames = []
    for k in (1, 2, 3):
        sub = df[["GAME_ID", f"PERSON{k}TYPE", f"PLAYER{k}_ID", f"PLAYER{k}_TEAM_ID"]].copy()
        sub.columns = ["gid", "ptype", "pid", "team"]
        frames.append(sub)
    p = pd.concat(frames, ignore_index=True)
    for c in ("gid", "ptype", "pid", "team"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p.dropna()
    p = p[p.ptype.isin([4, 5]) & (p.pid > 0) & (p.pid < 1e8) & (p.team > 1e9)].astype("int64")
    home = p[p.ptype == 4].groupby("gid").team.agg(mode).to_dict()
    away = p[p.ptype == 5].groupby("gid").team.agg(mode).to_dict()
    app = p[["gid", "pid", "team"]].drop_duplicates()
    return app, home, away, {}


def parse_cdn(year: int):
    path = os.path.join(RAW, f"cdnnba_{year}.csv")
    df = pd.read_csv(path, usecols=["gameId", "personId", "teamId", "scoreHome", "scoreAway", "orderNumber", "timeActual"],
                     low_memory=False)
    df["personId"] = pd.to_numeric(df.personId, errors="coerce")
    df["teamId"] = pd.to_numeric(df.teamId, errors="coerce")
    app = df[(df.personId > 0) & (df.personId < 1e8) & (df.teamId > 1e9)][["gameId", "personId", "teamId"]]
    app = app.drop_duplicates().astype("int64")
    app.columns = ["gid", "pid", "team"]
    df = df.sort_values(["gameId", "orderNumber"])
    sh = pd.to_numeric(df.scoreHome, errors="coerce").groupby(df.gameId).ffill()
    sa = pd.to_numeric(df.scoreAway, errors="coerce").groupby(df.gameId).ffill()
    df["dh"] = sh.groupby(df.gameId).diff()
    df["da"] = sa.groupby(df.gameId).diff()
    home = df[(df.dh > 0) & (df.teamId > 1e9)].groupby("gameId").teamId.agg(mode).astype("int64").to_dict()
    away = df[(df.da > 0) & (df.teamId > 1e9)].groupby("gameId").teamId.agg(mode).astype("int64").to_dict()
    ta = df.dropna(subset=["timeActual"]).groupby("gameId").timeActual.min()
    dates = {int(g): local_date(str(t).replace(" ", "T")) for g, t in ta.items()}
    return app, home, away, dates


def match_dates(year: int, games: dict[int, tuple[int, int]], espn: dict):
    """gid -> date via ESPN events matched on (home, away) pair in id order; reversed pair as fallback."""
    evs = espn.get(str(year + 1), [])
    by_pair = collections.defaultdict(list)
    unknown = collections.Counter()
    for e in evs:
        h, a = espn_franchise(e["home"]["name"]), espn_franchise(e["away"]["name"])
        if h is None or a is None:
            unknown[(e["home"]["name"], e["away"]["name"])] += 1
            continue
        by_pair[(h, a)].append(local_date(e["date_utc"]))
    for k in by_pair:
        by_pair[k].sort()
    nba_pair = collections.defaultdict(list)
    for gid, (h, a) in games.items():
        nba_pair[(h, a)].append(gid)
    out, n_rev, n_miss = {}, 0, 0
    leftovers = []
    for pair, gids in nba_pair.items():
        gids.sort()
        ds = by_pair.get(pair, [])
        for i, gid in enumerate(gids):
            if i < len(ds):
                out[gid] = ds[i]
            else:
                leftovers.append((gid, pair))
    used_rev = collections.Counter()
    for gid, (h, a) in leftovers:
        ds = by_pair.get((a, h), [])
        n_nba_rev = len(nba_pair.get((a, h), []))
        j = n_nba_rev + used_rev[(a, h)]
        if j < len(ds):
            out[gid] = ds[j]
            used_rev[(a, h)] += 1
            n_rev += 1
        else:
            n_miss += 1
    return out, dict(espn_events=len(evs), matched=len(out), reversed=n_rev, missing=n_miss,
                     unknown_names=sum(unknown.values()))


def build(year: int, espn: dict):
    code = code_of(year)
    # 2021-22 onward: the CDN feed is the source the box rows were built from, and its tip timestamps
    # give exact dates (game ids stop being chronological once NBA Cup games are slotted in)
    if year <= 2020:
        app, home, away, cdn_dates = parse_nbastats(year)
    else:
        app, home, away, cdn_dates = parse_cdn(year)
    with open(os.path.join(HERE, f"_sv_pergame_{code}.json"), encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    box = pd.DataFrame(rows)[["gid", "pid", "team", "name", "pts", "fga", "fta", "tov", "l_season"]]
    box["pid"] = box.pid.astype("int64")
    box["team"] = box.team.astype("int64")
    box["gid"] = box.gid.astype("int64")

    # every box row must be an appearance; zero-stat appearances are added with zeros
    pg = app.merge(box, on=["gid", "pid", "team"], how="outer", indicator=True)
    n_box_not_app = int((pg._merge == "right_only").sum())
    n_app_no_box = int((pg._merge == "left_only").sum())
    pg = pg.drop(columns="_merge")
    for c in ("pts", "fga", "fta", "tov"):
        pg[c] = pg[c].fillna(0.0)
    names = box.drop_duplicates("pid").set_index("pid").name.to_dict()
    lseason = box.dropna(subset=["l_season"]).drop_duplicates("pid").set_index("pid").l_season.to_dict()
    pg["name"] = pg.pid.map(names).fillna("")
    pg["l_season"] = pg.pid.map(lseason)

    # games: the two teams from box rows; home/away from PBP
    teams_by_game = box.groupby("gid").team.agg(lambda s: sorted(set(s)))
    games, bad_ha = {}, 0
    for gid, ts in teams_by_game.items():
        if len(ts) != 2:
            continue
        h, a = home.get(gid), away.get(gid)
        if h in ts and a in ts and h != a:
            games[gid] = (h, a)
        elif h in ts:
            games[gid] = (h, ts[0] if ts[1] == h else ts[1])
        elif a in ts:
            games[gid] = (ts[0] if ts[1] == a else ts[1], a)
        else:
            bad_ha += 1
    dates, dmeta = match_dates(year, games, espn)
    if cdn_dates:   # 2025-26: CDN actual tip times are authoritative; ESPN as fallback
        agree = sum(1 for g in games if g in dates and g in cdn_dates and dates[g] == cdn_dates[g])
        dmeta["cdn_espn_agree"] = agree
        for g in games:
            if g in cdn_dates:
                dates[g] = cdn_dates[g]

    # team-game table
    tot = pg.groupby(["gid", "team"])[["pts", "fga", "fta", "tov"]].sum().reset_index()
    recs = []
    for r in tot.itertuples(index=False):
        if r.gid not in games:
            continue
        h, a = games[r.gid]
        recs.append(dict(gid=r.gid, team=r.team, opp=a if r.team == h else h, home=int(r.team == h),
                         date=dates.get(r.gid), pts=r.pts, fga=r.fga, fta=r.fta, tov=r.tov))
    tg = pd.DataFrame(recs)
    # rest days (days between games minus one); NaN for first game or unknown date
    tg = tg.sort_values(["team", "date", "gid"]).reset_index(drop=True)
    d = pd.to_datetime(tg.date)
    prev = d.groupby(tg.team).shift(1)
    tg["rest"] = (d - prev).dt.days - 1
    tg["game_no"] = tg.groupby("team").cumcount() + 1
    rest_map = tg.set_index(["gid", "team"]).rest
    tg["opp_rest"] = [rest_map.get((g, o), np.nan) for g, o in zip(tg.gid, tg.opp)]
    meta = dict(code=code, games=len(games), bad_home_away=bad_ha, box_rows=len(box), appearances=len(pg),
                box_not_in_pbp=n_box_not_app, zero_stat_appearances=n_app_no_box,
                dated=int(tg.date.notna().sum()), team_games=len(tg), **dmeta)
    with open(os.path.join(HERE, f"_pacgl_season_{code}.pkl"), "wb") as f:
        pickle.dump(dict(pg=pg, tg=tg, meta=meta), f)
    return meta


def main():
    years = [int(a) for a in sys.argv[1:]] or list(YEARS)
    with open(os.path.join(HERE, "_pacgl_espn_sched.json"), encoding="utf-8") as f:
        espn = json.load(f)
    for y in years:
        m = build(y, espn)
        print(json.dumps(m, default=str), flush=True)


if __name__ == "__main__":
    main()
