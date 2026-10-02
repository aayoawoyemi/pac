# -*- coding: utf-8 -*-
"""_pacgl_fetch_espn.py -- regular-season schedules (date, home, away) for 1997-98 .. 2025-26 from ESPN.

Used only for game dates (rest / back-to-back controls). Home/away and participation come from NBA PBP.
Output: _pacgl_espn_sched.json  {season_end_year: [{id, date_utc, home: {id, abbr, name}, away: {...}}]}
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
H = {"User-Agent": "Mozilla/5.0"}
YEARS = range(1998, 2027)          # ESPN season = end year
TEAM_IDS = range(1, 31)


def fetch(args):
    yr, tid = args
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{tid}/schedule?season={yr}&seasontype=2"
    for attempt in range(4):
        try:
            r = requests.get(url, headers=H, timeout=30)
            if r.status_code == 200:
                out = []
                for ev in r.json().get("events", []):
                    comp = ev["competitions"][0]
                    side = {}
                    for c in comp["competitors"]:
                        t = c["team"]
                        side[c["homeAway"]] = dict(id=t.get("id"), abbr=t.get("abbreviation"), name=t.get("displayName"))
                    if "home" in side and "away" in side:
                        out.append(dict(id=ev["id"], date_utc=ev.get("date", ""), home=side["home"], away=side["away"]))
                return yr, tid, out
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return yr, tid, None


def main():
    jobs = [(y, t) for y in YEARS for t in TEAM_IDS]
    res = {}
    failed = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for yr, tid, evs in ex.map(fetch, jobs):
            if evs is None:
                failed.append((yr, tid))
                continue
            bucket = res.setdefault(str(yr), {})
            for e in evs:
                bucket[e["id"]] = e
    out = {y: sorted(v.values(), key=lambda e: e["date_utc"]) for y, v in res.items()}
    for y in sorted(out):
        print(y, len(out[y]))
    print("failed:", failed)
    with open(os.path.join(HERE, "_pacgl_espn_sched.json"), "w", encoding="utf-8") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
