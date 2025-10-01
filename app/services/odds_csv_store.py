"""CSV store for odds snapshots (historics).

We persist per-bookmaker odds rows for each match into league-specific CSV files:
  data/odds_history/h2h_{LEAGUE}.csv

Schema (header):
timestamp,league,week,source,bookmaker,match_date,commence_time,home_team,away_team,market,outcome,line,decimal_odds,american_odds,implied_prob,book_overround

- market: 'h2h' for 3-way moneyline; other markets can be added later.
- outcome: 'H','D','A' for h2h; for totals use 'over'/'under'.
- line: numeric for totals/handicaps; empty for h2h.

Provides helpers to:
- append rows from Bovada and The Odds API snapshots
- read back recent rows and build an index by (date, home, away) aggregating implied probabilities
- retrieve preferred bookmaker decimal odds for EV calculations
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from datetime import datetime, timedelta
import os


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
    since = datetime.utcnow() - timedelta(days=max(days, 1))
    pref = None
    if preferred_bookmakers:
        pref = {b.lower() for b in preferred_bookmakers}
    # group rows
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for rec in r:
                try:
                    ts = rec.get("timestamp") or ""
                    ts_dt = (
                        datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        if ts
                        else None
                    )
                except Exception:
                    ts_dt = None
                if ts_dt and ts_dt < since:
                    continue
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
