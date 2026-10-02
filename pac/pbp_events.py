# Vendored from the LUMA backend (_nba_box_ev_decade.py): play-by-play event parser used by pac/player_values.py.
# -*- coding: utf-8 -*-
"""Build a direct-observation NBA Box+ action ledger from ten PBP seasons.

This is intentionally *not* a Box+ coefficient fitter.  It first emits the
quantities the PBP can identify without an allocation convention:

* realized points on terminal attempts (FGA, turnover, or a foul-resolved trip);
* realized value of the next attempt created by OREB/DREB/STL;
* block recovery outcomes and the next attempt after a defensive recovery;
* actual FT points by granular foul class; and
* realized points on assisted and unassisted makes.

The scorer/assister split is not identifiable from ordinary PBP because an
assist is only recorded on a make.  That split belongs in a later credit layer.

Seasons are labelled by start year.  The canonical decade is 2016-17 through
2025-26: legacy ``nbastats`` for 2016-20 and modern ``cdnnba`` for 2021-25.
"""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import json
import math
import os
from typing import Iterator


from pac import paths

HERE = paths.RAW
RAW = paths.raw("nba_data_raw")
DEFAULT_OUTPUT = paths.raw("_nba_box_ev_decade.json")
csv.field_size_limit(10**7)


def num(value: object) -> int:
    try:
        return int(float(str(value or "0")))
    except (TypeError, ValueError):
        return 0


def player_id(value: object) -> str:
    parsed = str(value or "").strip()
    return "" if parsed in {"", "0", "0.0", "None", "null"} else parsed


def joined_description(row: dict[str, str]) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in ("HOMEDESCRIPTION", "NEUTRALDESCRIPTION", "VISITORDESCRIPTION")
    ).strip()


def event_side(row: dict[str, str]) -> str:
    if row.get("HOMEDESCRIPTION"):
        return "home"
    if row.get("VISITORDESCRIPTION"):
        return "away"
    return "neutral"


def foul_class(descriptor: str, subtype: str, description: str) -> str:
    value = " ".join((descriptor, subtype, description)).lower()
    if "flagrant" in value and ("type2" in value or "type 2" in value):
        return "flagrant_2"
    if "flagrant" in value:
        return "flagrant_1"
    if "defensive-3-second" in value or "defensive 3 seconds" in value:
        return "defensive_3_seconds"
    if "technical" in value or "t.foul" in value:
        return "technical"
    if "transition take" in value:
        return "transition_take"
    if "away-from-play" in value or "away from play" in value:
        return "away_from_play"
    if "clear path" in value:
        return "clear_path"
    if "take" in value:
        return "take"
    if "charge" in value:
        return "charge"
    if "offensive" in value:
        return "offensive"
    if "loose ball" in value or "l.b.foul" in value:
        return "loose_ball"
    if descriptor.lower() == "shooting" or "s.foul" in value or "shooting" in value:
        return "shooting"
    if "double" in value:
        return "double"
    return "personal"


LATENT_OR_RETAINED_FOULS = {
    "technical", "defensive_3_seconds", "flagrant_1", "flagrant_2",
    "take", "transition_take", "away_from_play", "clear_path", "double",
}


@dataclasses.dataclass(slots=True)
class Event:
    pos: int
    period: str
    clock: str
    kind: str
    team: str = ""
    player: str = ""
    other_player: str = ""
    other_team: str = ""
    side: str = "neutral"
    points: int = 0
    shot_value: int = 0
    made: bool = False
    assist_player: str = ""
    steal_player: str = ""
    block_player: str = ""
    detail: str = ""
    is_team: bool = False

    @property
    def clock_key(self) -> tuple[str, str]:
        return (self.period, self.clock)


@dataclasses.dataclass(slots=True)
class Attempt:
    pos: int
    team: str
    kind: str
    points: int
    event: Event
    attached_ft_points: int = 0
    assisted: bool = False
    blocked: bool = False
    stolen: bool = False


@dataclasses.dataclass
class Running:
    n: int = 0
    total: float = 0.0
    total2: float = 0.0

    def add(self, value: float) -> None:
        self.n += 1
        self.total += value
        self.total2 += value * value

    def merge(self, other: "Running") -> None:
        self.n += other.n
        self.total += other.total
        self.total2 += other.total2

    def emit(self) -> dict[str, float | int | None]:
        if not self.n:
            return {"n": 0, "mean": None, "sd": None, "se": None}
        mean = self.total / self.n
        variance = max(0.0, self.total2 / self.n - mean * mean)
        sd = math.sqrt(variance)
        return {
            "n": self.n,
            "mean": round(mean, 8),
            "sd": round(sd, 8),
            "se": round(sd / math.sqrt(self.n), 8),
        }


class Ledger:
    def __init__(self) -> None:
        self.samples: collections.defaultdict[str, Running] = collections.defaultdict(Running)
        self.counts: collections.Counter[str] = collections.Counter()
        self.validation: collections.Counter[str] = collections.Counter()

    def add(self, key: str, value: float) -> None:
        self.samples[key].add(value)

    def merge(self, other: "Ledger") -> None:
        for key, value in other.samples.items():
            self.samples[key].merge(value)
        self.counts.update(other.counts)
        self.validation.update(other.validation)

    def emit(self) -> dict[str, object]:
        return {
            "validation": dict(sorted(self.validation.items())),
            "counts": dict(sorted(self.counts.items())),
            "observed_values": {
                key: value.emit() for key, value in sorted(self.samples.items())
            },
        }


def normalize_modern(row: dict[str, str], pos: int) -> Event:
    action = str(row.get("actionType") or "").lower()
    subtype = str(row.get("subType") or "").lower()
    descriptor = str(row.get("descriptor") or "").lower()
    description = str(row.get("description") or "")
    base = dict(
        pos=pos,
        period=str(row.get("period") or ""),
        clock=str(row.get("clock") or ""),
        team=str(row.get("teamId") or ""),
        player=player_id(row.get("personId")),
        is_team=not player_id(row.get("personId")),
    )
    if action in {"2pt", "3pt"}:
        made = row.get("shotResult") == "Made"
        shot_value = 3 if action == "3pt" else 2
        return Event(
            **base,
            kind="fg",
            points=shot_value if made else 0,
            shot_value=shot_value,
            made=made,
            assist_player=player_id(row.get("assistPersonId")),
            detail=description,
        )
    if action == "freethrow":
        made = row.get("shotResult") == "Made"
        return Event(**base, kind="ft", points=1 if made else 0, made=made, detail=description)
    if action == "turnover":
        return Event(
            **base,
            kind="turnover",
            steal_player=player_id(row.get("stealPersonId")),
            other_player=player_id(row.get("stealPersonId")),
            detail=" ".join((subtype, descriptor, description)),
        )
    if action == "steal":
        return Event(**base, kind="steal", detail=description)
    if action == "block":
        return Event(**base, kind="block", detail=description)
    if action == "rebound":
        return Event(**base, kind="oreb" if subtype == "offensive" else "dreb", detail=description)
    if action == "foul":
        return Event(
            **base,
            kind="foul",
            other_player=str(row.get("foulDrawnPersonId") or ""),
            other_team=str(row.get("possession") or ""),
            detail=foul_class(descriptor, subtype, description),
        )
    return Event(**base, kind=action or "other", detail=description)


def normalize_legacy(row: dict[str, str], pos: int) -> Event:
    msg = num(row.get("EVENTMSGTYPE"))
    description = joined_description(row)
    upper = description.upper()
    base = dict(
        pos=pos,
        period=str(row.get("PERIOD") or ""),
        clock=str(row.get("PCTIMESTRING") or ""),
        team=str(row.get("PLAYER1_TEAM_ID") or ""),
        player=player_id(row.get("PLAYER1_ID")),
        side=event_side(row),
        is_team=(not player_id(row.get("PLAYER1_ID")) or not str(row.get("PLAYER1_NAME") or "").strip()),
    )
    if msg in {1, 2}:
        made = msg == 1
        shot_value = 3 if "3PT" in upper else 2
        return Event(
            **base,
            kind="fg",
            points=shot_value if made else 0,
            shot_value=shot_value,
            made=made,
            assist_player=player_id(row.get("PLAYER2_ID")) if made and "AST" in upper else "",
            block_player=player_id(row.get("PLAYER3_ID")) if not made and "BLOCK" in upper else "",
            other_team=str(row.get("PLAYER3_TEAM_ID") or "") if "BLOCK" in upper else "",
            detail=description,
        )
    if msg == 3:
        made = "MISS" not in upper
        return Event(**base, kind="ft", points=1 if made else 0, made=made, detail=description)
    if msg == 4:
        return Event(**base, kind="rebound", detail=description)
    if msg == 5:
        steal_player = player_id(row.get("PLAYER2_ID")) if "STEAL" in upper else ""
        return Event(
            **base,
            kind="turnover",
            steal_player=steal_player,
            other_player=steal_player,
            other_team=str(row.get("PLAYER2_TEAM_ID") or "") if steal_player else "",
        )
    if msg == 6:
        return Event(
            **base,
            kind="foul",
            other_player=str(row.get("PLAYER2_ID") or ""),
            other_team=str(row.get("PLAYER2_TEAM_ID") or ""),
            detail=foul_class("", "", description),
        )
    return Event(**base, kind=f"msg_{msg}", detail=description)


def iter_games(path: str, modern: bool) -> Iterator[tuple[str, list[Event]]]:
    game_key = "gameId" if modern else "GAME_ID"
    normalizer = normalize_modern if modern else normalize_legacy
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        current = ""
        events: list[Event] = []
        for pos, row in enumerate(reader):
            game_id = str(row.get(game_key) or "")
            if current and game_id != current:
                yield current, events
                events = []
            current = game_id
            events.append(normalizer(row, pos))
        if current:
            yield current, events


def fill_legacy_teams_and_rebounds(events: list[Event]) -> None:
    side_team: dict[str, str] = {}
    for event in events:
        if event.side in {"home", "away"} and event.team:
            side_team[event.side] = event.team
    for event in events:
        if not event.team and event.side in side_team:
            event.team = side_team[event.side]

    last_miss_team = ""
    for event in events:
        if event.kind == "fg" and not event.made:
            last_miss_team = event.team
        elif event.kind == "ft" and not event.made:
            last_miss_team = event.team
        elif event.kind == "rebound":
            event.kind = "oreb" if event.team and event.team == last_miss_team else "dreb"


def clock_groups(events: list[Event]) -> dict[tuple[str, str], list[Event]]:
    grouped: collections.defaultdict[tuple[str, str], list[Event]] = collections.defaultdict(list)
    for event in events:
        grouped[event.clock_key].append(event)
    return grouped


def build_attempts(events: list[Event], ledger: Ledger) -> list[Attempt]:
    groups = clock_groups(events)
    attempts: list[Attempt] = []
    assigned_ft_positions: set[int] = set()

    for group in groups.values():
        fgs = [event for event in group if event.kind == "fg"]
        fts = [event for event in group if event.kind == "ft"]
        fouls = [event for event in group if event.kind == "foul"]
        live_fouls = [event for event in fouls if event.detail not in LATENT_OR_RETAINED_FOULS]
        shooting_fouls = [event for event in live_fouls if event.detail == "shooting"]
        ft_points = sum(event.points for event in fts)

        # Attach a same-clock shooting-foul trip to the one official FGA when
        # the feed records it (normally an and-one; rarely an official miss).
        attach_to_fg = len(fgs) == 1 and bool(shooting_fouls) and bool(fts)
        for fg in fgs:
            attached = ft_points if attach_to_fg else 0
            if attached:
                assigned_ft_positions.update(event.pos for event in fts)
            attempts.append(
                Attempt(
                    pos=fg.pos,
                    team=fg.team,
                    kind="fg_make" if fg.made else "fg_miss",
                    points=fg.points + attached,
                    event=fg,
                    attached_ft_points=attached,
                    assisted=bool(fg.assist_player),
                    blocked=bool(fg.block_player),
                )
            )

        # A live defensive foul with free throws but no official FGA is still
        # one terminal attempt.  Retained-possession/technical FT awards are
        # latent and are not counted as a new attempt.
        if not fgs and live_fouls and fts:
            foul = live_fouls[0]
            attempts.append(
                Attempt(
                    pos=foul.pos,
                    team=foul.other_team,
                    kind=f"foul_trip_{foul.detail}",
                    points=ft_points,
                    event=foul,
                    attached_ft_points=ft_points,
                )
            )
            assigned_ft_positions.update(event.pos for event in fts)

    for event in events:
        if event.kind == "turnover":
            attempts.append(
                Attempt(
                    pos=event.pos,
                    team=event.team,
                    kind="turnover_stolen" if event.steal_player else "turnover_dead",
                    points=0,
                    event=event,
                    stolen=bool(event.steal_player),
                )
            )

    attempts.sort(key=lambda attempt: attempt.pos)
    raw_points = sum(event.points for event in events if event.kind in {"fg", "ft"})
    attempt_points = sum(attempt.points for attempt in attempts)
    latent_points = sum(
        event.points for event in events
        if event.kind == "ft" and event.pos not in assigned_ft_positions
    )
    ledger.validation["raw_points"] += raw_points
    ledger.validation["attempt_points"] += attempt_points
    ledger.validation["latent_or_unassigned_ft_points"] += latent_points
    ledger.validation["point_reconciliation_error"] += raw_points - attempt_points - latent_points
    return attempts


def next_attempt(attempts: list[Attempt], pos: int, team: str = "") -> Attempt | None:
    for attempt in attempts:
        if attempt.pos <= pos:
            continue
        return attempt if not team or attempt.team == team else None
    return None


def next_rebound(events: list[Event], pos: int) -> Event | None:
    for event in events:
        if event.pos <= pos:
            continue
        if event.kind in {"oreb", "dreb"}:
            return event
        if event.kind in {"fg", "turnover"}:
            return None
    return None


def process_game(events: list[Event], modern: bool, ledger: Ledger) -> None:
    if not modern:
        fill_legacy_teams_and_rebounds(events)
    attempts = build_attempts(events, ledger)
    ledger.validation["games"] += 1
    ledger.validation["attempts"] += len(attempts)

    groups = clock_groups(events)
    for attempt in attempts:
        ledger.counts[f"attempt.{attempt.kind}"] += 1
        ledger.add("attempt.all.realized_points", attempt.points)
        ledger.add(f"attempt.{attempt.kind}.realized_points", attempt.points)
        if attempt.kind in {"turnover_stolen", "turnover_dead"}:
            created = next_attempt(attempts, attempt.pos)
            if created is not None:
                ledger.add(
                    f"attempt.{attempt.kind}.next_opponent_attempt_points",
                    created.points,
                )
        if attempt.kind == "fg_make":
            label = "assisted_make" if attempt.assisted else "unassisted_make"
            shot_label = "made_3" if attempt.event.shot_value == 3 else "made_2"
            ledger.counts[label] += 1
            ledger.add(f"{label}.fg_points", attempt.event.points)
            ledger.add(f"{label}.attempt_points_including_attached_fts", attempt.points)
            ledger.add(f"{label}.is_three", 1.0 if attempt.event.shot_value == 3 else 0.0)
            ledger.add(f"{label}.{shot_label}.attempt_points_including_attached_fts", attempt.points)

    for event in events:
        if event.kind in {"oreb", "dreb"}:
            actor = "team" if event.is_team else "player"
            ledger.counts[f"{event.kind}.{actor}"] += 1
            created = next_attempt(attempts, event.pos, event.team)
            if created is not None:
                ledger.add(f"{event.kind}.{actor}.next_same_team_attempt_points", created.points)
        elif event.kind == "steal":
            ledger.counts["steal.player"] += 1
            created = next_attempt(attempts, event.pos, event.team)
            if created is not None:
                ledger.add("steal.player.next_attempt_points", created.points)
        elif not modern and event.kind == "turnover" and event.steal_player:
            # Legacy feed embeds the steal in the turnover row.
            ledger.counts["steal.player"] += 1
            created = next_attempt(attempts, event.pos, event.other_team)
            if created is not None:
                ledger.add("steal.player.next_attempt_points", created.points)
        elif event.kind == "block":
            ledger.counts["block.player"] += 1
            recovery = next_rebound(events, event.pos)
            if recovery is not None:
                recovered_by_defense = recovery.team == event.team
                ledger.add("block.player.defensive_recovery", 1.0 if recovered_by_defense else 0.0)
                if recovered_by_defense:
                    created = next_attempt(attempts, recovery.pos, event.team)
                    if created is not None:
                        ledger.add("block.player.next_attempt_points_after_defensive_recovery", created.points)
        elif event.kind == "fg" and event.block_player:
            # Legacy feed embeds the blocker in the missed-FG event.
            ledger.counts["block.player"] += 1
            recovery = next_rebound(events, event.pos)
            if recovery is not None:
                recovered_by_defense = bool(event.other_team and recovery.team == event.other_team)
                ledger.add("block.player.defensive_recovery", 1.0 if recovered_by_defense else 0.0)
                if recovered_by_defense:
                    created = next_attempt(attempts, recovery.pos, event.other_team)
                    if created is not None:
                        ledger.add("block.player.next_attempt_points_after_defensive_recovery", created.points)
        elif event.kind == "foul":
            ledger.counts[f"foul.{event.detail}"] += 1
            # A drawn charge is the defensive analogue of the offensive-foul
            # turnover.  Price only the attempt that actually follows for the
            # drawing team; the erased offensive attempt remains charged to
            # the turnover, exactly as in the STL/TOV split.
            if event.detail == "charge" and event.other_player:
                ledger.counts["charge_drawn.player"] += 1
                # The feed also logs the offensive-foul turnover at this same
                # clock.  Step past the entire clock group so that value is
                # the drawing team's ensuing attempt, not the zero-point TOV.
                created = next_attempt(attempts, max(item.pos for item in groups[event.clock_key]))
                if created is not None:
                    ledger.add("charge_drawn.player.next_attempt_points", created.points)
            group = groups[event.clock_key]
            group_fouls = [item for item in group if item.kind == "foul"]
            if len(group_fouls) == 1:
                ft_points = sum(item.points for item in group if item.kind == "ft")
                ft_attempts = sum(1 for item in group if item.kind == "ft")
                ledger.add(f"foul.{event.detail}.ft_attempts_awarded", ft_attempts)
                ledger.add(f"foul.{event.detail}.ft_points_awarded", ft_points)
            else:
                ledger.counts[f"foul.{event.detail}.ambiguous_same_clock_group"] += 1

    # Modern has standalone block/steal events; legacy embeds both.
    ledger.validation["fga"] += sum(1 for event in events if event.kind == "fg")
    ledger.validation["turnovers"] += sum(1 for event in events if event.kind == "turnover")
    ledger.validation["individual_oreb"] += sum(
        1 for event in events if event.kind == "oreb" and not event.is_team
    )
    ledger.validation["individual_dreb"] += sum(
        1 for event in events if event.kind == "dreb" and not event.is_team
    )


def run_season(year: int, force_source: str = "") -> tuple[str, Ledger]:
    modern = year >= 2021
    if force_source:
        modern = force_source == "modern"
    prefix = "cdnnba" if modern else "nbastats"
    path = os.path.join(RAW, f"{prefix}_{year}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ledger = Ledger()
    for _, events in iter_games(path, modern):
        process_game(events, modern, ledger)
    return prefix, ledger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2016)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--source", choices=("", "legacy", "modern"), default="")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    pooled = Ledger()
    seasons: dict[str, object] = {}
    for year in range(args.start, args.end + 1):
        source, ledger = run_season(year, args.source)
        pooled.merge(ledger)
        emitted = ledger.emit()
        emitted["source"] = source
        emitted["season"] = f"{year}-{str(year + 1)[-2:]}"
        validation = emitted["validation"]
        assert isinstance(validation, dict)
        raw_points = int(validation.get("raw_points", 0))
        attempt_points = int(validation.get("attempt_points", 0))
        attempts = int(validation.get("attempts", 0))
        emitted["rates"] = {
            "terminal_attempt_ppp": round(attempt_points / attempts, 8) if attempts else None,
            "all_points_per_terminal_attempt": round(raw_points / attempts, 8) if attempts else None,
            "latent_point_share": round((raw_points - attempt_points) / raw_points, 8) if raw_points else None,
        }
        seasons[str(year)] = emitted
        points = int(validation.get("raw_points", 0))
        error = int(validation.get("point_reconciliation_error", 0))
        print(
            f"{year}-{str(year + 1)[-2:]} {source:<8} "
            f"games={int(validation.get('games', 0)):,} "
            f"points={points:,} attempts={attempts:,} "
            f"terminal_ppp={attempt_points / attempts:.4f} "
            f"all_in_ppp={points / attempts:.4f} reconcile={error:+d}"
        )
        if error:
            raise RuntimeError(f"point reconciliation failed for {year}: {error:+d}")

    pooled_emit = pooled.emit()
    pooled_validation = pooled_emit["validation"]
    assert isinstance(pooled_validation, dict)
    pooled_raw_points = int(pooled_validation.get("raw_points", 0))
    pooled_attempt_points = int(pooled_validation.get("attempt_points", 0))
    pooled_attempts = int(pooled_validation.get("attempts", 0))
    pooled_emit["rates"] = {
        "terminal_attempt_ppp": round(pooled_attempt_points / pooled_attempts, 8) if pooled_attempts else None,
        "all_points_per_terminal_attempt": round(pooled_raw_points / pooled_attempts, 8) if pooled_attempts else None,
        "latent_point_share": round((pooled_raw_points - pooled_attempt_points) / pooled_raw_points, 8)
        if pooled_raw_points else None,
    }
    output = {
        "spec": {
            "season_range": [args.start, args.end],
            "unit": "terminal attempt: official FGA, turnover, or live foul-resolved FT trip without FGA",
            "free_throws": "attached to the attempt they resolve; technical/retained-possession awards are latent",
            "credit_layer": "not estimated here; assisted makes remain composite events",
            "impact_used": False,
        },
        "seasons": seasons,
        "pooled": pooled_emit,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
