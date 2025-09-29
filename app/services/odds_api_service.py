"""Odds API Integration Service (generic The Odds API style).

This service fetches soccer match odds from a third‑party Odds API compatible
with endpoints such as:
  https://api.the-odds-api.com/v4/sports/{sport}/odds/?regions=eu&markets=h2h

We deliberately:
- DO NOT embed the API key in code; expects environment variable ODDS_API_KEY.
- Provide graceful fallbacks and structured error messages.
- Normalize bookmaker odds into implied probabilities & overround metrics.

Environment variables:
  ODDS_API_KEY        -> required
  ODDS_API_TIMEOUT    -> optional (seconds, default 12)
  ODDS_API_BASE_URL   -> optional override (default https://api.the-odds-api.com/v4)

Example usage (offline CLI wiring added separately):
  from app.services.odds_api_service import fetch_h2h_odds
  resp = fetch_h2h_odds(sport_key='soccer_epl')

Returned structure (simplified):
{
  'sport_key': 'soccer_epl',
  'sport_title': 'EPL',
  'markets': 'h2h',
  'regions': 'eu',
  'num_events': 10,
  'events': [
     {
       'commence_time': '2025-08-16T14:00:00Z',
       'home_team': 'Manchester United',
       'away_team': 'Arsenal',
       'bookmakers': [
          {
            'key': 'bet365',
            'title': 'Bet365',
            'last_update': '2025-08-10T09:11:22Z',
            'h2h': {
               'raw_odds': {'H': 2.05, 'D': 3.60, 'A': 3.40},
               'implied_probabilities': {'H': 0.39, 'D': 0.222, 'A': 0.388},
               'overround': 0.102
            }
          }, ...
       ]
     }
  ],
  'aggregates': {
     'avg_overround': 0.074,
     'events_with_odds': 9
  }
}
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import os, json, math, time

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

# Reuse a minimal normalization (could import central version if circular not an issue)
NORMALIZATION_MAP = {
    "man utd": "Manchester United",
    "man united": "Manchester United",
    "manchester united": "Manchester United",
    "man city": "Manchester City",
    "manchester city": "Manchester City",
    "tottenham hotspur": "Tottenham",
    "spurs": "Tottenham",
    "wolverhampton wanderers": "Wolves",
    "wolves": "Wolves",
    "brighton & hove albion": "Brighton",
    "afc bournemouth": "Bournemouth",
    "nottingham forest": "Nottm Forest",
    "nottm forest": "Nottm Forest",
    "sheffield united": "Sheffield Utd",
    "sheffield utd": "Sheffield Utd",
    "sheff utd": "Sheffield Utd",
    "west ham united": "West Ham",
    "west ham": "West Ham",
    "newcastle united": "Newcastle",
}


def normalize_team(name: str | None) -> str | None:
    if not name:
        return name
    raw = name.strip()
    key = raw.lower()
    cleaned = key.replace(" fc", "").replace(" afc", "").replace("&", "and")
    if cleaned in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[cleaned]
    if key in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[key]
    return " ".join(w.capitalize() for w in raw.split())


def _implied_probs(odds: Dict[str, float]) -> Dict[str, float]:
    inv = {k: (1.0 / v if v and v > 0 else 0.0) for k, v in odds.items()}
    s = sum(inv.values())
    if s <= 0:
        return {k: 0.0 for k in odds}
    return {k: inv[k] / s for k in inv}


def _overround(odds: Dict[str, float]) -> float:
    inv_sum = sum((1.0 / v if v and v > 0 else 0.0) for v in odds.values())
    if inv_sum <= 0:
        return 0.0
    return inv_sum - 1.0


def fetch_h2h_odds(
    sport_key: str = "soccer_epl",
    regions: str = "uk,eu,us",
    markets: str = "h2h",
    date_format: str = "iso",
    odds_format: str = "decimal",
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Fetch H2H odds for a sport (default EPL) from the Odds API.

    Returns structured data; if API key missing or request fails, includes error.
    """
    if requests is None:
        return {"error": "requests library not available"}
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        return {"error": "ODDS_API_KEY not set in environment"}
    base_url = os.getenv("ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4")
    timeout = timeout or int(os.getenv("ODDS_API_TIMEOUT", "12"))

    # Allow env override for regions (e.g., 'uk,eu,us') to target specific bookmakers
    regions = os.getenv("ODDS_API_REGIONS", regions)
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }
    url = f"{base_url}/sports/{sport_key}/odds"
    try:
        r = requests.get(url, params=params, timeout=timeout)
        status = r.status_code
        if status != 200:
            return {"error": f"status {status}", "body": r.text[:400]}
        arr = r.json()
        out_events: List[Dict[str, Any]] = []
        overround_samples: List[float] = []
        events_with_odds = 0
        for ev in arr:
            home = normalize_team(ev.get("home_team"))
            away = normalize_team(ev.get("away_team"))
            commence = ev.get("commence_time")
            bookmakers = ev.get("bookmakers") or []
            norm_bookmakers = []
            for bm in bookmakers:
                key = bm.get("key")
                title = bm.get("title")
                last_update = bm.get("last_update")
                markets_list = bm.get("markets") or []
                h2h_payload = {}
                for mk in markets_list:
                    if mk.get("key") == "h2h":
                        outcomes = mk.get("outcomes") or []
                        odds_map: Dict[str, float] = {}
                        for o in outcomes:
                            name = o.get("name")
                            price = o.get("price")
                            if name and price and price > 1.01 and price < 1000:
                                # Map names: Home / Away / Draw variations
                                lname = name.lower()
                                if lname in ("home", home.lower() if home else "home"):
                                    odds_map["H"] = float(price)
                                elif lname in (
                                    "away",
                                    away.lower() if away else "away",
                                ):
                                    odds_map["A"] = float(price)
                                elif lname in ("draw", "tie"):  # draw naming
                                    odds_map["D"] = float(price)
                                else:
                                    # Attempt fuzzy: if equals home/away team names
                                    if home and lname == home.lower():
                                        odds_map["H"] = float(price)
                                    elif away and lname == away.lower():
                                        odds_map["A"] = float(price)
                        if len(odds_map) >= 2:  # need at least two to compute something
                            probs = _implied_probs(odds_map)
                            over = _overround(odds_map)
                            overround_samples.append(over)
                            h2h_payload = {
                                "raw_odds": odds_map,
                                "implied_probabilities": {
                                    k: round(v, 4) for k, v in probs.items()
                                },
                                "overround": round(over, 4),
                            }
                norm_bookmakers.append(
                    {
                        "key": key,
                        "title": title,
                        "last_update": last_update,
                        "h2h": h2h_payload,
                    }
                )
            if any(bm.get("h2h") for bm in norm_bookmakers):
                events_with_odds += 1
            out_events.append(
                {
                    "commence_time": commence,
                    "home_team": home,
                    "away_team": away,
                    "bookmakers": norm_bookmakers,
                }
            )
        avg_over = None
        if overround_samples:
            avg_over = sum(overround_samples) / len(overround_samples)
        return {
            "sport_key": sport_key,
            "markets": markets,
            "regions": regions,
            "events": out_events,
            "num_events": len(out_events),
            "events_with_odds": events_with_odds,
            "avg_overround": round(avg_over, 4) if avg_over is not None else None,
        }
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}
