# -*- coding: utf-8 -*-
"""_leaf_poss.py — canonical leaf-stint loader with counted possessions.

WHY THIS FILE EXISTS
  ARC's O/D channels were solved against a CLOCK PROXY for possessions:

      poss = secs/60.0 * 2.0        # flat 2 possessions per minute, every lineup

  The target is points per 100 POSSESSIONS. When the denominator is a clock
  proxy, every lineup whose real pace differs from 2.0/min gets a systematic
  error in y, and the ridge hands that error to whoever was on the floor.
  Measured consequence on the shipped NBA build: corr(O,D) = -0.874, i.e. the
  defensive board was mostly negative offence wearing a defensive label.

  The fix is to COUNT possessions from events, the same construction the WNBA
  build adopted 2026-08-23:

      poss = FGA + 0.44*FTA + TOV - OREB

TWO BUILDERS, TWO KEY FORMATS  (the reason this loader is not a one-liner)
  _build_leaf_stints.py       writes 10-digit zero-padded gids ("0022300001").
                              Validates against production stints with a
                              points-reconciliation check. CANONICAL.
  _build_leaf_stints_hist.py  writes 8-digit gids ("22300001") via a gid_box()
                              shim. Older historical backfill.

  Both wrote into the same _leaf_stints_{code}.json files for 2122-2425, so
  those files carry the SAME GAME twice under two keys. Raw key counts run
  ~2,180 for an 82-game season (1,230 games). Pooling naively would DOUBLE
  COUNT roughly half the league.

  Verified on 2324 game 22300001: identical lineups, identical 48:00, points
  237 vs 236, stints 35 vs 36 — the same game segmented slightly differently
  by the two builders.

COVERAGE, measured 2026-08-28  (padded / short / union vs 1,230-game seasons)
      1516-2021   0 / ~1,000-1,145 / same     short-key only (hist builder)
      2122        1,030 / 1,156 / 1,217
      2223        1,044 / 1,144 / 1,208
      2324        1,061 / 1,156 / 1,214
      2425        1,014 / 1,164 / 1,209
      2526          996 /     0 /   996       live season, ingest incomplete
      pre-1516    none                        no leaf files at all

RESOLUTION RULE
  1. Normalise every gid to 10-digit zero-padded form.
  2. On collision prefer the PADDED original (canonical builder).
  3. Union the two sources — padded alone loses ~15% of 2122-2425, and loses
     ALL of 1516-2021, which has no padded variant at all.

  This maximises coverage while preferring the validated builder wherever both
  exist.

CLOCK FALLBACK POLICY — per season, never per row
  _wnba_od_arc.py states the principle: "Counted events are the possession; the
  clock is a proxy for it and mixing the two puts two different quantities in
  one denominator."

  Mixing WITHIN a season is therefore forbidden. But dropping seasons with no
  leaf data entirely would delete 1997-2015 from the record. So the decision is
  made once per season:

      leaf coverage >= MIN_COV  ->  counted possessions, rows without a
                                    countable vector are DROPPED
      leaf coverage <  MIN_COV  ->  clock proxy for the WHOLE season

  Each season is then internally consistent. The discontinuity is between
  seasons, documented, and visible in the returned metadata.
"""
import os, json

be = os.path.dirname(os.path.abspath(__file__))

MIN_COV = 0.60      # fraction of legacy games needing leaf data to count a season

_CACHE = {}


def _norm_gid(g):
    """All gids to 10-digit zero-padded. '22300001' -> '0022300001'."""
    s = str(g)
    if len(s) == 10 and s.startswith("00"):
        return s
    if len(s) == 8:
        return "00" + s
    return s.zfill(10)


def counted_poss(vec):
    """FGA + 0.44*FTA + TOV - OREB from a leaf tally vector.

    Vector layout (12 slots, per side) — from _build_leaf_stints.py, which is
    the file that WRITES these vectors:
        [0] FGA_rim (2pt <=4ft)   [1] PTS_rim
        [2] FGA_mid (2pt >4ft)    [3] PTS_mid
        [4] FGA_3pt               [5] PTS_3pt
        [6] FTA                   [7] PTS_ft
        [8] TOV                   [9] OREB
        [10] FB_att               [11] FB_pts

    So vec[0]+vec[2]+vec[4] is total FGA (rim + mid + three), vec[6] is FTA,
    vec[8] TOV, vec[9] OREB. The formula below is the standard possession
    estimator over those fields.

    ON FREE THROWS AND CONTINUOUS POSSESSIONS
      A trip to the line is ONE possession no matter how many FTs are shot, and
      an and-1 does not end a possession at all. Counting raw FTA would inflate
      the denominator. The 0.44 coefficient is the league-standard estimate of
      "what fraction of FTA are possession-ENDING" — it nets out and-1s, the
      first of a pair, and technical/flagrant extras.

      This is the one ESTIMATED term left in an otherwise counted formula. A
      fully event-counted version would mark possession-ending FTs directly
      from the PBP (last FT of a trip, excluding and-1 continuation). The leaf
      vectors do not currently carry that flag — [6] is raw FTA — so 0.44 stands.
      Logged as open work.

    Index scheme cross-checked on 2526: counted/clock ratio = 1.022, i.e. the
    counted total lands within 2% of the clock estimate league-wide while
    varying correctly per lineup.

    Returns None when the row carries no usable vector — the caller DROPS that
    row rather than substituting a clock estimate.
    """
    if not vec or len(vec) < 10:
        return None
    try:
        p = (float(vec[0]) + float(vec[2]) + float(vec[4])
             + 0.44 * float(vec[6]) + float(vec[8]) - float(vec[9]))
    except (TypeError, ValueError):
        return None
    return p if p > 0.5 else None


def load_leaf(code):
    """Deduped leaf stints for a season code, keyed by normalised gid.

    Returns {} when the season has no leaf file. Cached per process.
    """
    if code in _CACHE:
        return _CACHE[code]
    p = os.path.join(be, f"_leaf_stints_{code}.json")
    if not os.path.exists(p):
        _CACHE[code] = {}
        return {}
    raw = json.load(open(p, encoding="utf-8"))
    out = {}
    padded_keys = set()
    for gid, stints in raw.items():
        n = _norm_gid(gid)
        is_padded = len(str(gid)) == 10 and str(gid).startswith("00")
        # Prefer the canonical (padded) builder; a short-key game only lands
        # if no padded version of the same game exists.
        if n in out and not is_padded:
            continue
        if is_padded:
            padded_keys.add(n)
        out[n] = stints
    _CACHE[code] = out
    return out


# A season is ~1,230 games (82 x 30 / 2). Lockout years are shorter, and the
# legacy _stints_ file is MISSING for some seasons entirely (0405 has none),
# so legacy game count is not a usable denominator on its own.
EXPECTED_GAMES = 1230
LOCKOUT = {"1112": 990, "1920": 1059, "2021": 1080, "9899": 725}


def season_mode(code):
    """('counted'|'clock', coverage_fraction) for one season.

    Coverage is measured against the EXPECTED schedule, not against the legacy
    stint file. Using legacy as the denominator produced two failures:
      - 0405 has no legacy file at all, so cov computed as 0.0 and a season with
        1,197 perfectly good leaf games fell back to a clock proxy that then had
        no data to read, emitting an EMPTY BOARD.
      - 9798 has a partial legacy file (712), so cov computed as 1.601.

    Leaf coverage is capped at 1.0; a season only falls back to the clock when
    it genuinely lacks leaf data (9697).
    """
    leaf = load_leaf(code)
    expected = LOCKOUT.get(code, EXPECTED_GAMES)
    cov = min(len(leaf) / expected, 1.0) if expected else 0.0
    return ("counted" if cov >= MIN_COV else "clock"), cov
