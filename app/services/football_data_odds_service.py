"""Football-Data.org Odds Fetch Service (best-effort).

IMPORTANT:
- Football-Data.org odds availability depends on your subscription tier.
- Many free tiers return an 'Activate plan' message or omit odds entirely.
- This service will gracefully handle absence of odds and return an informative
  structure so downstream logic can decide how to proceed.

Endpoint reference pattern (v4):
  GET https://api.football-data.org/v4/competitions/{competition}/matches?season=YYYY&matchday=N
  Each match object MAY contain an 'odds' block OR require per-match lookup:
  GET https://api.football-data.org/v4/matches/{matchId}

We implement a two-pass strategy:
1. Fetch all matches for the matchday.
2. If top-level odds info missing and deep_lookup=True, fetch each match by id.

Returned canonical odds structure per match:
{
  'match_id': int,
  'home_team': str,
  'away_team': str,
  'status': str,
  'odds_available': bool,
  'raw_overround': float | None,
  'bookmakers': [
      {
         'name': str,
         'markets': [
            {
               'key': '3way',
               'outcomes': [ {'name': 'HOME', 'price': 2.10}, ... ],
               'implied_probabilities': {...},
               'overround': 0.0612
            }
         ]
      }
  ]
}

If no odds are available, 'odds_available' is False and bookmakers list empty.
"""
from __future__ import annotations
from typing import Any, Dict, List
import os
import math
import time

import json

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

API_BASE = "https://api.football-data.org/v4"


def _calc_implied_and_overround(outcomes: List[Dict[str, Any]]):
    prices = []
    for o in outcomes:
        price = o.get("price") or o.get("odd")
        try:
            fv = float(price)
            if fv > 1.01:
                prices.append(fv)
        except Exception:
            pass
    if len(prices) < 2:
        return None, None
    inv = [1.0 / p for p in prices]
    s = sum(inv)
    if s <= 0:
        return None, None
    overround = s - 1.0
    probs = [v / s for v in inv]
    implied = {}
    for idx, o in enumerate(outcomes):
        implied[o.get("name") or f"o{idx}"] = probs[idx] if idx < len(probs) else None
    return implied, overround


def _extract_match_level_odds(match_payload: Dict[str, Any]) -> Dict[str, Any]:
    odds_block = match_payload.get("odds")
    out: Dict[str, Any] = {
        "match_id": match_payload.get("id"),
        "home_team": (match_payload.get("homeTeam") or {}).get("name"),
        "away_team": (match_payload.get("awayTeam") or {}).get("name"),
        "status": match_payload.get("status"),
        "odds_available": False,
        "raw_overround": None,
        "bookmakers": [],
    }
    if not odds_block:
        return out
    bms = odds_block.get("bookmakers") or []
    total_overround_samples = []
    norm_bookmakers = []
    for bm in bms:
        name = bm.get("name")
        markets = bm.get("bets") or bm.get("markets") or []
        norm_markets = []
        for mk in markets:
            outcomes = mk.get("values") or mk.get("outcomes") or []
            implied, overround = _calc_implied_and_overround(outcomes)
            if overround is not None:
                total_overround_samples.append(overround)
            norm_markets.append(
                {
                    "key": mk.get("label") or mk.get("type") or "unknown",
                    "outcomes": outcomes,
                    "implied_probabilities": implied,
                    "overround": overround,
                }
            )
        norm_bookmakers.append({"name": name, "markets": norm_markets})
    if norm_bookmakers:
        out["odds_available"] = True
    out["bookmakers"] = norm_bookmakers
    if total_overround_samples:
        out["raw_overround"] = sum(total_overround_samples) / len(
            total_overround_samples
        )
    return out


def fetch_week_odds(
    competition: str,
    season: int,
    matchday: int,
    deep_lookup: bool = False,
    token: str | None = None,
    throttle_seconds: float = 0.4,
) -> Dict[str, Any]:
    token = token or os.getenv("FOOTBALL_DATA_API_TOKEN")
    if requests is None:
        return {"error": "requests not installed"}
    if not token:
        return {"error": "FOOTBALL_DATA_API_TOKEN not set"}
    headers = {"X-Auth-Token": token}
    base_params = {"season": str(season), "matchday": matchday}
    url = f"{API_BASE}/competitions/{competition}/matches"
    try:
        resp = requests.get(url, headers=headers, params=base_params, timeout=15)
        if resp.status_code != 200:
            return {"error": f"status {resp.status_code}", "body": resp.text[:400]}
        data = resp.json()
        matches = data.get("matches", [])
        out_rows: List[Dict[str, Any]] = []
        for m in matches:
            odds_record = _extract_match_level_odds(m)
            out_rows.append(odds_record)
        # Deep per-match fetch for missing odds if requested
        if deep_lookup:
            for idx, r in enumerate(out_rows):
                if not r["odds_available"] and r["match_id"]:
                    detail_url = f"{API_BASE}/matches/{r['match_id']}"
                    d_resp = requests.get(detail_url, headers=headers, timeout=15)
                    if d_resp.status_code == 200:
                        detail = d_resp.json().get("match") or d_resp.json()
                        enriched = _extract_match_level_odds(detail)
                        # Preserve identity fields
                        r.update(enriched)
                    time.sleep(throttle_seconds)
        available = sum(1 for r in out_rows if r["odds_available"])
        avg_overround = None
        over_samples = [
            r["raw_overround"]
            for r in out_rows
            if isinstance(r.get("raw_overround"), (float, int))
        ]
        if over_samples:
            avg_overround = sum(over_samples) / len(over_samples)
        return {
            "competition": competition,
            "season": season,
            "matchday": matchday,
            "matches": len(out_rows),
            "with_odds": available,
            "avg_overround": avg_overround,
            "deep_lookup": deep_lookup,
            "records": out_rows,
        }
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}
