"""CSV store for odds snapshots (historics).

We persist per-bookmaker odds rows for each match into league-specific CSV files:
  data/odds_history/h2h_{LEAGUE}.csv

Schema (header):
timestamp,league,week,source,bookmaker,match_date,commence_time,home_team,away_team,market,outcome,line,decimal_odds,american_odds,implied_prob,book_overround


Provides helpers to:

Expected CSV columns for H2H:
 - match_date (YYYY-MM-DD or ISO datetime)
 - home_team, away_team
 - bookmaker, market (e.g., h2h)
 - outcome (home|draw|away)
 - american_odds, decimal_odds, implied_prob

Expected CSV columns for totals:
 - match_date (YYYY-MM-DD or ISO datetime)
 - home_team, away_team
 - bookmaker, market ("totals")
 - outcome (over|under)
 - line (float goals line, e.g., 2.5)
 - american_odds, decimal_odds, implied_prob

"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from datetime import datetime, timedelta
import os
from .corners_actuals_service import corners_actuals_store


BASE_DIR = Path("data/odds_history")
BASE_DIR.mkdir(parents=True, exist_ok=True)


CSV_HEADER = [
    "timestamp",
    "league",
    "week",
    "source",
    "bookmaker",
    "match_date",
    "commence_time",
    "home_team",
    "away_team",
    "market",
    "outcome",
    "line",
    "decimal_odds",
    "american_odds",
    "implied_prob",
    "book_overround",
]


def _csv_path(league: str, market: str = "h2h") -> Path:
    code = (league or "PL").upper()
    return BASE_DIR / f"{market}_{code}.csv"


def _ensure_file(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)


def _ts() -> str:
    return datetime.utcnow().isoformat() + "Z"


def append_h2h_from_bovada(
    league: str, events: Iterable[Dict[str, Any]], week: Optional[int] = None
) -> int:
    """Append H2H odds rows from Bovada parsed events.

    Expects each event to contain:
      - home_team, away_team, commence_time (ISO) or start_time_ms
      - h2h: { 'H': p, 'D': p, 'A': p } (implied probs normalized)
      - h2h_american (optional): { 'H': ml, 'D': ml, 'A': ml }
      - h2h_decimal (optional): { 'H': dec, 'D': dec, 'A': dec }
    """
    path = _csv_path(league, "h2h")
    _ensure_file(path)
    count = 0
    now = _ts()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for ev in events:
            home = ev.get("home_team")
            away = ev.get("away_team")
            if not (home and away):
                continue
            ct = ev.get("commence_time")
            if not ct:
                # build from ms if available
                ms = ev.get("start_time_ms")
                if isinstance(ms, int):
                    try:
                        ct = (
                            datetime.utcfromtimestamp(ms / 1000)
                            .isoformat()
                            .replace("+00:00", "Z")
                        )
                    except Exception:
                        ct = None
            match_date = None
            if isinstance(ct, str) and "T" in ct:
                match_date = ct.split("T")[0]
            probs = ev.get("h2h") or {}
            if not isinstance(probs, dict) or not any(probs.values()):
                continue
            amer = ev.get("h2h_american") or {}
            decs = ev.get("h2h_decimal") or {}
            book_over = None  # unknown per-bookmaker; probs are normalized
            for outcome in ("H", "D", "A"):
                p = probs.get(outcome)
                dec = decs.get(outcome)
                ml = amer.get(outcome)
                w.writerow(
                    [
                        now,
                        league,
                        week if week is not None else "",
                        "bovada",
                        "bovada",
                        match_date or "",
                        ct or "",
                        home,
                        away,
                        "h2h",
                        outcome,
                        "",
                        dec if dec is not None else "",
                        ml if ml is not None else "",
                        round(p, 6) if isinstance(p, (int, float)) else "",
                        book_over if book_over is not None else "",
                    ]
                )
                count += 1
    return count


def append_totals_from_bovada(
    league: str, events: Iterable[Dict[str, Any]], week: Optional[int] = None
) -> int:
    """Append Totals (goals) O/U rows from Bovada parsed events.

    Each event may include a list under 'totals' with items like:
      { 'line': 2.5, 'over_prob': 0.52, 'under_prob': 0.48 }

    We persist both Over and Under outcomes with implied probabilities and decimal odds (1/p) when available.
    """
    path = _csv_path(league, "totals")
    _ensure_file(path)
    now = _ts()
    cnt = 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for ev in events or []:
            home = ev.get("home_team")
            away = ev.get("away_team")
            if not (home and away):
                continue
            ct = ev.get("commence_time")
            if not ct:
                ms = ev.get("start_time_ms")
                if isinstance(ms, int):
                    try:
                        ct = (
                            datetime.utcfromtimestamp(ms / 1000)
                            .isoformat()
                            .replace("+00:00", "Z")
                        )
                    except Exception:
                        ct = None
            match_date = ct.split("T")[0] if isinstance(ct, str) and "T" in ct else ""
            totals = ev.get("totals") or []
            for t in totals:
                try:
                    line = float(t.get("line")) if t.get("line") is not None else None
                except Exception:
                    line = None
                over_p = t.get("over_prob")
                under_p = t.get("under_prob")
                if line is None or (over_p is None and under_p is None):
                    continue
                # Write Over
                if isinstance(over_p, (int, float)) and over_p > 0:
                    dec = round(1.0 / float(over_p), 4)
                    w.writerow(
                        [
                            now,
                            league,
                            week if week is not None else "",
                            "bovada",
                            "bovada",
                            match_date,
                            ct or "",
                            home,
                            away,
                            "totals",
                            "over",
                            line,
                            dec,
                            "",
                            round(float(over_p), 6),
                            "",
                        ]
                    )
                    cnt += 1
                # Write Under
                if isinstance(under_p, (int, float)) and under_p > 0:
                    dec = round(1.0 / float(under_p), 4)
                    w.writerow(
                        [
                            now,
                            league,
                            week if week is not None else "",
                            "bovada",
                            "bovada",
                            match_date,
                            ct or "",
                            home,
                            away,
                            "totals",
                            "under",
                            line,
                            dec,
                            "",
                            round(float(under_p), 6),
                            "",
                        ]
                    )
                    cnt += 1
    return cnt


def append_btts_from_bovada(
    league: str, events: Iterable[Dict[str, Any]], week: Optional[int] = None
) -> int:
    """Append BTTS (Both Teams To Score) rows from Bovada parsed events.

    Expects per-event dict possibly containing:
      btts: {
        'yes_prob': float | None,
        'no_prob': float | None,
        'yes_ml': int | None,
        'no_ml': int | None
      }

    We persist 'yes' and 'no' outcomes with implied probabilities and decimal odds derived as 1/p when available.
    """
    path = _csv_path(league, "btts")
    _ensure_file(path)
    now = _ts()
    cnt = 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for ev in events or []:
            home = ev.get("home_team")
            away = ev.get("away_team")
            if not (home and away):
                continue
            ct = ev.get("commence_time")
            if not ct:
                ms = ev.get("start_time_ms")
                if isinstance(ms, int):
                    try:
                        ct = (
                            datetime.utcfromtimestamp(ms / 1000)
                            .isoformat()
                            .replace("+00:00", "Z")
                        )
                    except Exception:
                        ct = None
            match_date = ct.split("T")[0] if isinstance(ct, str) and "T" in ct else ""
            b = ev.get("btts") or {}
            if not isinstance(b, dict):
                continue
            yes_p = b.get("yes_prob")
            no_p = b.get("no_prob")
            yes_ml = b.get("yes_ml")
            no_ml = b.get("no_ml")
            # Write YES
            if isinstance(yes_p, (int, float)) and yes_p > 0:
                dec = round(1.0 / float(yes_p), 4)
                w.writerow(
                    [
                        now,
                        league,
                        week if week is not None else "",
                        "bovada",
                        "bovada",
                        match_date,
                        ct or "",
                        home,
                        away,
                        "btts",
                        "yes",
                        "",
                        dec,
                        yes_ml if yes_ml is not None else "",
                        round(float(yes_p), 6),
                        "",
                    ]
                )
                cnt += 1
            # Write NO
            if isinstance(no_p, (int, float)) and no_p > 0:
                dec = round(1.0 / float(no_p), 4)
                w.writerow(
                    [
                        now,
                        league,
                        week if week is not None else "",
                        "bovada",
                        "bovada",
                        match_date,
                        ct or "",
                        home,
                        away,
                        "btts",
                        "no",
                        "",
                        dec,
                        no_ml if no_ml is not None else "",
                        round(float(no_p), 6),
                        "",
                    ]
                )
                cnt += 1
    return cnt


# -------- Additional market snapshotters (Bovada) --------

def _match_datetime_from_event(ev: Dict[str, Any]) -> Tuple[str, str]:
    ct = ev.get("commence_time")
    if not ct:
        ms = ev.get("start_time_ms")
        if isinstance(ms, int):
            try:
                ct = (
                    datetime.utcfromtimestamp(ms / 1000)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
            except Exception:
                ct = None
    match_date = ct.split("T")[0] if isinstance(ct, str) and "T" in ct else ""
    return (match_date, ct or "")


def _write_ou_rows(
    w: csv.writer,
    now: str,
    league: str,
    week: Optional[int],
    match_date: str,
    commence: str,
    home: str,
    away: str,
    market: str,
    line: Optional[float],
    over_prob: Optional[float],
    under_prob: Optional[float],
) -> int:
    cnt = 0
    if isinstance(over_prob, (int, float)) and over_prob > 0:
        try:
            dec = round(1.0 / float(over_prob), 4)
        except Exception:
            dec = ""
        w.writerow(
            [
                now,
                league,
                week if week is not None else "",
                "bovada",
                "bovada",
                match_date,
                commence,
                home,
                away,
                market,
                "over",
                line if line is not None else "",
                dec,
                "",
                round(float(over_prob), 6),
                "",
            ]
        )
        cnt += 1
    if isinstance(under_prob, (int, float)) and under_prob > 0:
        try:
            dec = round(1.0 / float(under_prob), 4)
        except Exception:
            dec = ""
        w.writerow(
            [
                now,
                league,
                week if week is not None else "",
                "bovada",
                "bovada",
                match_date,
                commence,
                home,
                away,
                market,
                "under",
                line if line is not None else "",
                dec,
                "",
                round(float(under_prob), 6),
                "",
            ]
        )
        cnt += 1
    return cnt


def append_first_half_totals_from_bovada(
    league: str, events: Iterable[Dict[str, Any]], week: Optional[int] = None
) -> int:
    path = _csv_path(league, "first_half_totals")
    _ensure_file(path)
    now = _ts()
    cnt = 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for ev in events or []:
            home = ev.get("home_team"); away = ev.get("away_team")
            if not (home and away):
                continue
            match_date, commence = _match_datetime_from_event(ev)
            for t in ev.get("first_half_totals") or []:
                try:
                    line = float(t.get("line")) if t.get("line") is not None else None
                except Exception:
                    line = None
                cnt += _write_ou_rows(
                    w, now, league, week, match_date, commence, home, away,
                    "first_half_totals", line, t.get("over_prob"), t.get("under_prob")
                )
    return cnt


def append_second_half_totals_from_bovada(
    league: str, events: Iterable[Dict[str, Any]], week: Optional[int] = None
) -> int:
    path = _csv_path(league, "second_half_totals")
    _ensure_file(path)
    now = _ts()
    cnt = 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for ev in events or []:
            home = ev.get("home_team"); away = ev.get("away_team")
            if not (home and away):
                continue
            match_date, commence = _match_datetime_from_event(ev)
            for t in ev.get("second_half_totals") or []:
                try:
                    line = float(t.get("line")) if t.get("line") is not None else None
                except Exception:
                    line = None
                cnt += _write_ou_rows(
                    w, now, league, week, match_date, commence, home, away,
                    "second_half_totals", line, t.get("over_prob"), t.get("under_prob")
                )
    return cnt


def append_team_goals_totals_from_bovada(
    league: str, events: Iterable[Dict[str, Any]], week: Optional[int] = None
) -> int:
    path = _csv_path(league, "team_goals_totals")
    _ensure_file(path)
    now = _ts()
    cnt = 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for ev in events or []:
            home = ev.get("home_team"); away = ev.get("away_team")
            if not (home and away):
                continue
            match_date, commence = _match_datetime_from_event(ev)
            for t in ev.get("team_totals") or []:
                side = t.get("side")
                try:
                    line = float(t.get("line")) if t.get("line") is not None else None
                except Exception:
                    line = None
                cnt += _write_ou_rows(
                    w, now, league, week, match_date, commence, home, away,
                    f"team_goals_totals_{side}" if side in ("home","away") else "team_goals_totals",
                    line, t.get("over_prob"), t.get("under_prob")
                )
    return cnt


def append_corners_totals_from_bovada(
    league: str, events: Iterable[Dict[str, Any]], week: Optional[int] = None
) -> int:
    path = _csv_path(league, "corners_totals")
    _ensure_file(path)
    now = _ts()
    cnt = 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for ev in events or []:
            home = ev.get("home_team"); away = ev.get("away_team")
            if not (home and away):
                continue
            match_date, commence = _match_datetime_from_event(ev)
            for t in ev.get("corners_totals") or []:
                try:
                    line = float(t.get("line")) if t.get("line") is not None else None
                except Exception:
                    line = None
                cnt += _write_ou_rows(
                    w, now, league, week, match_date, commence, home, away,
                    "corners_totals", line, t.get("over_prob"), t.get("under_prob")
                )
    return cnt


def append_team_corners_from_bovada(
    league: str, events: Iterable[Dict[str, Any]], week: Optional[int] = None
) -> int:
    path = _csv_path(league, "team_corners_totals")
    _ensure_file(path)
    now = _ts()
    cnt = 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for ev in events or []:
            home = ev.get("home_team"); away = ev.get("away_team")
            if not (home and away):
                continue
            match_date, commence = _match_datetime_from_event(ev)
            for t in ev.get("team_corners") or []:
                side = t.get("side")
                try:
                    line = float(t.get("line")) if t.get("line") is not None else None
                except Exception:
                    line = None
                cnt += _write_ou_rows(
                    w, now, league, week, match_date, commence, home, away,
                    f"team_corners_totals_{side}" if side in ("home","away") else "team_corners_totals",
                    line, t.get("over_prob"), t.get("under_prob")
                )
    return cnt


def append_double_chance_from_bovada(
    league: str, events: Iterable[Dict[str, Any]], week: Optional[int] = None
) -> int:
    path = _csv_path(league, "double_chance")
    _ensure_file(path)
    now = _ts()
    cnt = 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for ev in events or []:
            home = ev.get("home_team"); away = ev.get("away_team")
            if not (home and away):
                continue
            match_date, commence = _match_datetime_from_event(ev)
            dc = ev.get("double_chance") or {}
            for key in ("1X", "X2", "12"):
                prob = (dc.get(key) or {}).get("prob")
                if isinstance(prob, (int, float)) and prob > 0:
                    try:
                        dec = round(1.0 / float(prob), 4)
                    except Exception:
                        dec = ""
                    w.writerow(
                        [
                            now, league, week if week is not None else "", "bovada", "bovada",
                            match_date, commence, home, away,
                            "double_chance", key,
                            "", dec, "", round(float(prob), 6), "",
                        ]
                    )
                    cnt += 1
    return cnt


def append_dnb_from_bovada(
    league: str, events: Iterable[Dict[str, Any]], week: Optional[int] = None
) -> int:
    path = _csv_path(league, "dnb")
    _ensure_file(path)
    now = _ts()
    cnt = 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for ev in events or []:
            home = ev.get("home_team"); away = ev.get("away_team")
            if not (home and away):
                continue
            match_date, commence = _match_datetime_from_event(ev)
            dnb = ev.get("draw_no_bet") or {}
            for key in ("home_prob", "away_prob"):
                prob = dnb.get(key)
                label = "home" if key == "home_prob" else "away"
                if isinstance(prob, (int, float)) and prob > 0:
                    try:
                        dec = round(1.0 / float(prob), 4)
                    except Exception:
                        dec = ""
                    w.writerow(
                        [
                            now, league, week if week is not None else "", "bovada", "bovada",
                            match_date, commence, home, away,
                            "dnb", label,
                            "", dec, "", round(float(prob), 6), "",
                        ]
                    )
                    cnt += 1
    return cnt


def append_asian_handicap_from_bovada(
    league: str, events: Iterable[Dict[str, Any]], week: Optional[int] = None
) -> int:
    path = _csv_path(league, "asian_handicap")
    _ensure_file(path)
    now = _ts()
    cnt = 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for ev in events or []:
            home = ev.get("home_team"); away = ev.get("away_team")
            if not (home and away):
                continue
            match_date, commence = _match_datetime_from_event(ev)
            for t in ev.get("asian_handicap") or []:
                line = t.get("line")
                for side_key, prob_key in (("home", "home_prob"), ("away", "away_prob")):
                    prob = t.get(prob_key)
                    if isinstance(prob, (int, float)) and prob > 0:
                        try:
                            dec = round(1.0 / float(prob), 4)
                        except Exception:
                            dec = ""
                        w.writerow(
                            [
                                now, league, week if week is not None else "", "bovada", "bovada",
                                match_date, commence, home, away,
                                "asian_handicap", f"{side_key}",
                                line if line is not None else "", dec, "", round(float(prob), 6), "",
                            ]
                        )
                        cnt += 1
    return cnt


def append_cards_totals_from_bovada(
    league: str, events: Iterable[Dict[str, Any]], week: Optional[int] = None
) -> int:
    path = _csv_path(league, "cards_totals")
    _ensure_file(path)
    now = _ts()
    cnt = 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for ev in events or []:
            home = ev.get("home_team"); away = ev.get("away_team")
            if not (home and away):
                continue
            match_date, commence = _match_datetime_from_event(ev)
            for t in ev.get("cards_totals") or []:
                try:
                    line = float(t.get("line")) if t.get("line") is not None else None
                except Exception:
                    line = None
                cnt += _write_ou_rows(
                    w, now, league, week, match_date, commence, home, away,
                    "cards_totals", line, t.get("over_prob"), t.get("under_prob")
                )
    return cnt


def append_corners_handicap_from_bovada(
    league: str, events: Iterable[Dict[str, Any]], week: Optional[int] = None
) -> int:
    path = _csv_path(league, "corners_handicap")
    _ensure_file(path)
    now = _ts()
    cnt = 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for ev in events or []:
            home = ev.get("home_team"); away = ev.get("away_team")
            if not (home and away):
                continue
            match_date, commence = _match_datetime_from_event(ev)
            for t in ev.get("corners_handicap") or []:
                line = t.get("line")
                for side_key, prob_key in (("home", "home_prob"), ("away", "away_prob")):
                    prob = t.get(prob_key)
                    if isinstance(prob, (int, float)) and prob > 0:
                        try:
                            dec = round(1.0 / float(prob), 4)
                        except Exception:
                            dec = ""
                        w.writerow(
                            [
                                now, league, week if week is not None else "", "bovada", "bovada",
                                match_date, commence, home, away,
                                "corners_handicap", f"{side_key}",
                                line if line is not None else "", dec, "", round(float(prob), 6), "",
                            ]
                        )
                        cnt += 1
    return cnt


def append_h2h_from_oddsapi(
    league: str,
    payload: Dict[str, Any],
    week: Optional[int] = None,
    bookmaker_filter: Optional[List[str]] = None,
) -> int:
    """Append H2H odds rows from The Odds API normalized payload (from fetch_h2h_odds).

    payload structure expected:
      {
        'sport_key': ..., 'events': [ { 'commence_time', 'home_team','away_team',
           'bookmakers': [ { 'key','title','h2h': { 'raw_odds': {H,D,A}, 'implied_probabilities': {...}, 'overround': ... } } ] } ]
      }
    """
    path = _csv_path(league, "h2h")
    _ensure_file(path)
    bk_set = None
    if bookmaker_filter:
        bk_set = {b.lower() for b in bookmaker_filter}
    cnt = 0
    now = _ts()
    events = payload.get("events") or []
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Case A: payload from fetch_h2h_odds -> events with bookmakers[].h2h
        for ev in events:
            home = ev.get("home_team")
            away = ev.get("away_team")
            ct = ev.get("commence_time")
            if not (home and away and ct):
                continue
            match_date = ct.split("T")[0] if "T" in ct else ""
            for bm in ev.get("bookmakers") or []:
                key = (bm.get("key") or bm.get("title") or "").lower()
                if bk_set and key not in bk_set:
                    continue
                h2h = bm.get("h2h") or {}
                raw = h2h.get("raw_odds") or {}
                ip = h2h.get("implied_probabilities") or {}
                over = h2h.get("overround")
                for outcome in ("H", "D", "A"):
                    dec = raw.get(outcome)
                    prob = ip.get(outcome)
                    w.writerow(
                        [
                            now,
                            league,
                            week if week is not None else "",
                            "the-odds-api",
                            key or "unknown",
                            match_date,
                            ct,
                            home,
                            away,
                            "h2h",
                            outcome,
                            "",
                            dec if isinstance(dec, (int, float)) else "",
                            "",
                            round(prob, 6) if isinstance(prob, (int, float)) else "",
                            over if isinstance(over, (int, float)) else "",
                        ]
                    )
                    cnt += 1
        # Case B: payload from fetch_the_odds_api -> records with bookmakers[].markets[].outcomes
        records = payload.get("records") or []
        for rec in records:
            home = rec.get("home_team")
            away = rec.get("away_team")
            ct = rec.get("commence_time")
            if not (home and away and ct):
                continue
            match_date = ct.split("T")[0] if "T" in ct else ""
            for bm in rec.get("bookmakers") or []:
                key = (bm.get("key") or bm.get("name") or bm.get("title") or "").lower()
                if bk_set and key not in bk_set:
                    continue
                for mk in bm.get("markets") or []:
                    if mk.get("key") != "h2h":
                        continue
                    ip = mk.get("implied_probabilities") or {}
                    over = mk.get("overround")
                    # derive decimals from outcomes if present
                    dec_map: Dict[str, float] = {}
                    for o in mk.get("outcomes") or []:
                        name = (o.get("name") or "").upper()
                        try:
                            price = (
                                float(o.get("price"))
                                if o.get("price") is not None
                                else None
                            )
                        except Exception:
                            price = None
                        if price and price > 1.0:
                            if "HOME" in name:
                                dec_map["H"] = price
                            elif "AWAY" in name:
                                dec_map["A"] = price
                            elif "DRAW" in name or "TIE" in name:
                                dec_map["D"] = price
                    for outcome in ("H", "D", "A"):
                        dec = dec_map.get(outcome)
                        prob = ip.get(outcome)
                        w.writerow(
                            [
                                now,
                                league,
                                week if week is not None else "",
                                "the-odds-api",
                                key or "unknown",
                                match_date,
                                ct,
                                home,
                                away,
                                "h2h",
                                outcome,
                                "",
                                dec if isinstance(dec, (int, float)) else "",
                                "",
                                round(prob, 6)
                                if isinstance(prob, (int, float))
                                else "",
                                over if isinstance(over, (int, float)) else "",
                            ]
                        )
                        cnt += 1
    return cnt


def append_totals_from_oddsapi(
    league: str,
    payload: Dict[str, Any],
    week: Optional[int] = None,
    bookmaker_filter: Optional[List[str]] = None,
) -> int:
    """Append Totals (goals) O/U rows from The Odds API normalized payload.

    Supports both shapes:
      - events[].bookmakers[].markets[] with key == "totals" and outcomes with price/point
      - records[] (aggregated) with bookmakers[].markets[] structure
    """
    path = _csv_path(league, "totals")
    _ensure_file(path)
    now = _ts()
    cnt = 0
    bk_set = {b.lower() for b in bookmaker_filter} if bookmaker_filter else None
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Case A: payload with events list
        for ev in payload.get("events") or []:
            home = ev.get("home_team")
            away = ev.get("away_team")
            ct = ev.get("commence_time")
            if not (home and away and ct):
                continue
            match_date = ct.split("T")[0] if isinstance(ct, str) and "T" in ct else ""
            for bm in ev.get("bookmakers") or []:
                key = (bm.get("key") or bm.get("title") or "").lower()
                if bk_set and key not in bk_set:
                    continue
                for mk in bm.get("markets") or []:
                    if mk.get("key") != "totals":
                        continue
                    over = mk.get("overround")
                    for o in mk.get("outcomes") or []:
                        name = (o.get("name") or "").lower()
                        outcome = (
                            "over"
                            if "over" in name
                            else ("under" if "under" in name else None)
                        )
                        if not outcome:
                            continue
                        try:
                            price = (
                                float(o.get("price"))
                                if o.get("price") is not None
                                else None
                            )
                        except Exception:
                            price = None
                        try:
                            line = (
                                float(o.get("point"))
                                if o.get("point") is not None
                                else None
                            )
                        except Exception:
                            line = None
                        if line is None:
                            continue
                        ip = (
                            (1.0 / price)
                            if (
                                isinstance(price, (int, float))
                                and price
                                and price > 1.0
                            )
                            else None
                        )
                        w.writerow(
                            [
                                now,
                                league,
                                week if week is not None else "",
                                "the-odds-api",
                                key or "unknown",
                                match_date,
                                ct,
                                home,
                                away,
                                "totals",
                                outcome,
                                line,
                                price if isinstance(price, (int, float)) else "",
                                "",
                                round(ip, 6) if isinstance(ip, float) else "",
                                over if isinstance(over, (int, float)) else "",
                            ]
                        )
                        cnt += 1
        # Case B: payload with records list
        for rec in payload.get("records") or []:
            home = rec.get("home_team")
            away = rec.get("away_team")
            ct = rec.get("commence_time")
            if not (home and away and ct):
                continue
            match_date = ct.split("T")[0] if isinstance(ct, str) and "T" in ct else ""
            for bm in rec.get("bookmakers") or []:
                key = (bm.get("key") or bm.get("name") or bm.get("title") or "").lower()
                if bk_set and key not in bk_set:
                    continue
                for mk in bm.get("markets") or []:
                    if mk.get("key") != "totals":
                        continue
                    over = mk.get("overround")
                    for o in mk.get("outcomes") or []:
                        name = (o.get("name") or "").lower()
                        outcome = (
                            "over"
                            if "over" in name
                            else ("under" if "under" in name else None)
                        )
                        if not outcome:
                            continue
                        try:
                            price = (
                                float(o.get("price"))
                                if o.get("price") is not None
                                else None
                            )
                        except Exception:
                            price = None
                        try:
                            line = (
                                float(o.get("point"))
                                if o.get("point") is not None
                                else None
                            )
                        except Exception:
                            line = None
                        if line is None:
                            continue
                        ip = (
                            (1.0 / price)
                            if (
                                isinstance(price, (int, float))
                                and price
                                and price > 1.0
                            )
                            else None
                        )
                        w.writerow(
                            [
                                now,
                                league,
                                week if week is not None else "",
                                "the-odds-api",
                                key or "unknown",
                                match_date,
                                ct,
                                home,
                                away,
                                "totals",
                                outcome,
                                line,
                                price if isinstance(price, (int, float)) else "",
                                "",
                                round(ip, 6) if isinstance(ip, float) else "",
                                over if isinstance(over, (int, float)) else "",
                            ]
                        )
                        cnt += 1
    return cnt


# -------- Results CSV (final scores) ---------
RESULTS_HEADER = [
    "timestamp",
    "league",
    "week",
    "match_date",
    "commence_time",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "result",
    "status",
    "source",
]


def _csv_results_path(league: str) -> Path:
    code = (league or "PL").upper()
    return BASE_DIR / f"results_{code}.csv"


def append_results_from_fixtures(
    league: str, fixtures: Iterable[Dict[str, Any]], week: Optional[int] = None
) -> int:
    """Append final results rows from fixtures (Football-Data.org converted fixtures shape).

    Expects each fixture with keys: home_team, away_team, utc_date, home_score, away_score, status, matchday
    Rows are appended when scores are present (completed) or status indicates completion.
    """
    path = _csv_results_path(league)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(RESULTS_HEADER)
    now = _ts()
    cnt = 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for m in fixtures or []:
            home = m.get("home_team") or (m.get("homeTeam") or {}).get("name")
            away = m.get("away_team") or (m.get("awayTeam") or {}).get("name")
            ct = m.get("utc_date") or m.get("date")
            status = (m.get("status") or "").upper()
            hs = m.get("home_score")
            as_ = m.get("away_score")
            if home is None or away is None or ct is None:
                continue
            # Only write for completed matches (status or scores present)
            is_done = status in ("FINISHED", "COMPLETED") or (
                hs is not None and as_ is not None
            )
            if not is_done:
                continue
            date = ct.split("T")[0] if isinstance(ct, str) and "T" in ct else (ct or "")
            try:
                wk = (
                    int(m.get("matchday"))
                    if m.get("matchday") is not None
                    else (int(week) if week else "")
                )
            except Exception:
                wk = week if week is not None else ""
            # Determine categorical result
            result = None
            if isinstance(hs, int) and isinstance(as_, int):
                if hs > as_:
                    result = "H"
                elif hs < as_:
                    result = "A"
                else:
                    result = "D"
            w.writerow(
                [
                    now,
                    league,
                    wk,
                    date,
                    ct or "",
                    home,
                    away,
                    hs if hs is not None else "",
                    as_ if as_ is not None else "",
                    result or "",
                    status,
                    "football-data",
                ]
            )
            cnt += 1
    return cnt


# -------- Goals Actuals (alias of results with total_goals) ---------
GOALS_ACTUALS_HEADER = [
    "timestamp",
    "league",
    "week",
    "match_date",
    "commence_time",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "total_goals",
    "result",
    "status",
    "source",
]


def _csv_goals_actuals_path(league: str) -> Path:
    code = (league or "PL").upper()
    return BASE_DIR / f"goals_actuals_{code}.csv"


def append_goals_actuals_from_fixtures(
    league: str, fixtures: Iterable[Dict[str, Any]], week: Optional[int] = None
) -> int:
    path = _csv_goals_actuals_path(league)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(GOALS_ACTUALS_HEADER)
    now = _ts()
    cnt = 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for m in fixtures or []:
            home = m.get("home_team") or (m.get("homeTeam") or {}).get("name")
            away = m.get("away_team") or (m.get("awayTeam") or {}).get("name")
            ct = m.get("utc_date") or m.get("date")
            status = (m.get("status") or "").upper()
            hs = m.get("home_score")
            as_ = m.get("away_score")
            if not (home and away and ct):
                continue
            is_done = status in ("FINISHED", "COMPLETED") or (
                hs is not None and as_ is not None
            )
            if not is_done:
                continue
            date = ct.split("T")[0] if isinstance(ct, str) and "T" in ct else (ct or "")
            try:
                wk = (
                    int(m.get("matchday"))
                    if m.get("matchday") is not None
                    else (int(week) if week else "")
                )
            except Exception:
                wk = week if week is not None else ""
            total = (
                (hs + as_) if (isinstance(hs, int) and isinstance(as_, int)) else None
            )
            result = None
            if isinstance(hs, int) and isinstance(as_, int):
                if hs > as_:
                    result = "H"
                elif hs < as_:
                    result = "A"
                else:
                    result = "D"
            w.writerow(
                [
                    now,
                    league,
                    wk,
                    date,
                    ct or "",
                    home,
                    away,
                    hs if hs is not None else "",
                    as_ if as_ is not None else "",
                    total if total is not None else "",
                    result or "",
                    status,
                    "football-data",
                ]
            )
            cnt += 1
    return cnt


# -------- Corners Actuals (from store) ---------
CORNERS_ACTUALS_HEADER = [
    "timestamp",
    "league",
    "week",
    "match_date",
    "commence_time",
    "home_team",
    "away_team",
    "home_corners",
    "away_corners",
    "total_corners",
    "source",
]


def _csv_corners_actuals_path(league: str) -> Path:
    code = (league or "PL").upper()
    return BASE_DIR / f"corners_actuals_{code}.csv"


def append_corners_actuals_from_store(
    league: str, fixtures: Iterable[Dict[str, Any]], week: Optional[int] = None
) -> int:
    path = _csv_corners_actuals_path(league)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CORNERS_ACTUALS_HEADER)
    now = _ts()
    cnt = 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for m in fixtures or []:
            home = m.get("home_team") or (m.get("homeTeam") or {}).get("name")
            away = m.get("away_team") or (m.get("awayTeam") or {}).get("name")
            ct = m.get("utc_date") or m.get("date")
            if not (home and away and ct):
                continue
            date = ct.split("T")[0] if isinstance(ct, str) and "T" in ct else (ct or "")
            rec = corners_actuals_store.lookup(date, home, away)
            if not rec:
                continue
            try:
                wk = (
                    int(m.get("matchday"))
                    if m.get("matchday") is not None
                    else (int(week) if week else "")
                )
            except Exception:
                wk = week if week is not None else ""
            w.writerow(
                [
                    now,
                    league,
                    wk,
                    date,
                    ct or "",
                    rec.get("home_team") or home,
                    rec.get("away_team") or away,
                    rec.get("home_corners"),
                    rec.get("away_corners"),
                    rec.get("total_corners"),
                    "fbref-or-manual",
                ]
            )
            cnt += 1
    return cnt


def load_h2h_index_from_csv(
    league: str,
    days: int = 30,
    preferred_bookmakers: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load recent H2H odds rows from CSV and aggregate by (date, home, away).

    Returns index mapping both "date|home|away" and "home|away" to records with:
      - consensus_implied: {H,D,A}
      - consensus_overround: median overround across bookmakers when available
      - preferred_decimals: {H,D,A} from first preferred bookmaker seen
      - preferred_bookmaker: name if chosen
    """
    path = _csv_path(league, "h2h")
    if not path.exists():
        return {}
    # Use timezone-aware UTC for robust comparisons
    from datetime import timezone as _tz
    since = datetime.now(_tz.utc) - timedelta(days=max(days, 1))
    pref = None
    if preferred_bookmakers:
        pref = {b.lower() for b in preferred_bookmakers}
    # group rows
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for rec in r:
                ts_dt = None
                try:
                    ts = rec.get("timestamp") or ""
                    if ts:
                        ts_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        # Normalize to aware UTC for comparison
                        if ts_dt.tzinfo is None:
                            ts_dt = ts_dt.replace(tzinfo=_tz.utc)
                        else:
                            ts_dt = ts_dt.astimezone(_tz.utc)
                except Exception:
                    ts_dt = None
                try:
                    if ts_dt and ts_dt < since:
                        continue
                except Exception:
                    # If comparison fails for any reason, keep the row rather than dropping
                    pass
                rows.append(rec)
    except Exception:
        return {}
    # aggregate
    grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for rec in rows:
        league_rec = (rec.get("league") or "").upper()
        if league_rec != (league or "PL").upper():
            continue
        date = rec.get("match_date") or ""
        home = rec.get("home_team") or ""
        away = rec.get("away_team") or ""
        key = (date, home, away)
        g = grouped.setdefault(
            key,
            {
                "prob_samples": {"H": [], "D": [], "A": []},
                "over_samples": [],
                "preferred": None,
            },
        )
        outcome = rec.get("outcome")
        try:
            prob = float(rec.get("implied_prob")) if rec.get("implied_prob") else None
        except Exception:
            prob = None
        try:
            over = (
                float(rec.get("book_overround")) if rec.get("book_overround") else None
            )
        except Exception:
            over = None
        if outcome in ("H", "D", "A") and isinstance(prob, float):
            g["prob_samples"][outcome].append(prob)
        if isinstance(over, float):
            g["over_samples"].append(over)
        # preferred bookmaker capture (first seen)
        bk = (rec.get("bookmaker") or "").lower()
        if pref and not g.get("preferred") and bk in pref:
            # stash decimals for EV
            try:
                dec = (
                    float(rec.get("decimal_odds")) if rec.get("decimal_odds") else None
                )
            except Exception:
                dec = None
            if dec and outcome in ("H", "D", "A"):
                g.setdefault("preferred_decimals", {})[outcome] = dec
                g["preferred"] = bk
    # build index
    out_idx: Dict[str, Dict[str, Any]] = {}
    from statistics import median

    for (date, home, away), g in grouped.items():
        cons = {k: (median(vs) if vs else None) for k, vs in g["prob_samples"].items()}
        over_c = median(g["over_samples"]) if g["over_samples"] else None
        rec = {
            "consensus_implied": cons,
            "consensus_overround": over_c,
        }
        if g.get("preferred_decimals"):
            rec["preferred_bookmaker"] = g.get("preferred")
            rec["preferred_decimals"] = g.get("preferred_decimals")
        key_basic = f"{home.lower()}|{away.lower()}"
        out_idx[f"{date}|{key_basic}"] = rec
        out_idx[key_basic] = rec
    return out_idx
