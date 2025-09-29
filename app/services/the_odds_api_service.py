"""Integration with The Odds API (https://the-odds-api.com/).

This service provides a lightweight wrapper to pull EPL (or other) odds
using an API key supplied via environment variable ODDS_API_KEY.

Free plan limits & notes:
- Rate limits and number of requests per month are restricted.
- Some bookmakers/regions may not be available on the free tier.
- Draw odds for soccer may appear under the same 'h2h' market with three outcomes.

We normalize the response into a structure similar to football_data_odds_service:
{
  'sport_key': 'soccer_epl',
  'updated': 'ISO8601',
  'bookmakers': [
     {
       'key': 'bet365',
       'title': 'Bet365',
       'markets': [
          {
            'key': 'h2h',
            'outcomes': [
               { 'name': 'HOME', 'price': 2.1 },
               { 'name': 'DRAW', 'price': 3.4 },
               { 'name': 'AWAY', 'price': 3.2 }
            ],
            'implied_probabilities': {...},
            'overround': 0.0587
          }
        ]
     }
  ]
}

We then wrap per-match odds into a list with canonical fields:
  match_id: constructed synthetic id (home-away-date or API id if provided)
"""
from __future__ import annotations
from typing import Any, Dict, List
import os
import time
import json

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

API_BASE = "https://api.the-odds-api.com/v4"

DEFAULT_REGIONS = "uk"  # could be 'uk,eu,us'
DEFAULT_MARKETS = "h2h"  # head-to-head (three-way for soccer)
DEFAULT_ODDS_FORMAT = "decimal"


def _calc_probs(outcomes: List[Dict[str, Any]]):
    prices = []
    for o in outcomes:
        try:
            p = float(o.get("price"))
            if p > 1.01:
                prices.append(p)
        except Exception:
            pass
    if len(prices) < 2:
        return None, None
    inv = [1.0 / x for x in prices]
    s = sum(inv)
    if s <= 0:
        return None, None
    over = s - 1.0
    probs = [v / s for v in inv]
    implied = {}
    for idx, o in enumerate(outcomes):
        implied[o.get("name") or f"o{idx}"] = probs[idx] if idx < len(probs) else None
    return implied, over


def fetch_the_odds_api(
    sport_key: str = "soccer_epl",
    regions: str = DEFAULT_REGIONS,
    markets: str = DEFAULT_MARKETS,
    odds_format: str = DEFAULT_ODDS_FORMAT,
    date_format: str = "iso",
    api_key: str | None = None,
) -> Dict[str, Any]:
    if requests is None:
        return {"error": "requests not installed"}
    api_key = api_key or os.getenv("ODDS_API_KEY")
    if not api_key:
        return {"error": "ODDS_API_KEY not set"}
    params = {
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
        "apiKey": api_key,
    }
    url = f"{API_BASE}/sports/{sport_key}/odds"
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return {"error": f"status {resp.status_code}", "body": resp.text[:400]}
        arr = resp.json()
        if not isinstance(arr, list):
            return {"error": "unexpected response structure"}
        out_rows: List[Dict[str, Any]] = []
        for match in arr:
            commence = match.get("commence_time")
            home = (match.get("home_team") or "").strip()
            away = (match.get("away_team") or "").strip()
            mid = match.get("id") or f"{home}_{away}_{commence}"
            bookmakers = match.get("bookmakers", [])
            bm_norm = []
            all_over = []
            for bm in bookmakers:
                markets_list = bm.get("markets", [])
                norm_markets = []
                for mk in markets_list:
                    if mk.get("key") != markets:
                        continue
                    outcomes = mk.get("outcomes") or []
                    # For soccer h2h include draw if present; some data sources only supply two-way
                    # Tag outcomes with canonical names HOME/DRAW/AWAY where possible.
                    canon_outcomes = []
                    for o in outcomes:
                        name = o.get("name")
                        price = o.get("price")
                        # Keep original outcome ordering; consumer can map further if needed.
                        canon_outcomes.append(
                            {
                                "name": name.upper() if isinstance(name, str) else name,
                                "price": price,
                            }
                        )
                    implied, over = _calc_probs(canon_outcomes)
                    if over is not None:
                        all_over.append(over)
                    norm_markets.append(
                        {
                            "key": mk.get("key"),
                            "outcomes": canon_outcomes,
                            "implied_probabilities": implied,
                            "overround": over,
                        }
                    )
                bm_norm.append(
                    {"name": bm.get("key") or bm.get("title"), "markets": norm_markets}
                )
            avg_over = sum(all_over) / len(all_over) if all_over else None
            out_rows.append(
                {
                    "match_id": mid,
                    "home_team": home,
                    "away_team": away,
                    "commence_time": commence,
                    "bookmakers": bm_norm,
                    "avg_overround": avg_over,
                    "odds_available": bool(bm_norm),
                }
            )
        return {
            "provider": "the-odds-api",
            "sport_key": sport_key,
            "matches": len(out_rows),
            "with_odds": sum(1 for r in out_rows if r["odds_available"]),
            "records": out_rows,
        }
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}
