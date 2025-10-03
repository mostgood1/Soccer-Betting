"""Odds API Goals Totals Markets Fetcher

Fetch total goals Over/Under markets via an Odds API-compatible service.
Returns flattened records suitable for persisting to CSV or loading into the
goals_market_service store.

Notes:
- Some providers expose totals under market keys like 'totals', 'total_goals',
  'alternate_totals', etc. We'll try a few candidates and also fall back to
  per-event odds endpoints when needed. Historical endpoints are optionally used
  when enabled via env flag ODDS_API_ALLOW_HISTORICAL=1 or when `historical=True`.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import os

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

from .team_name_normalizer import normalize_team_name


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key)
    return v if v is not None else default


def _fetch(url: str, params: Dict[str, Any], timeout: int) -> Tuple[Optional[int], Any]:
    if requests is None:
        return None, {"error": "requests library not available"}
    try:
        r = requests.get(url, params=params, timeout=timeout)
        return r.status_code, (
            r.json()
            if r.headers.get("Content-Type", "").startswith("application/json")
            else r.text
        )
    except Exception as e:  # pragma: no cover
        return None, {"error": str(e)}


def _parse_totals_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse a list of Odds API-style events and extract totals markets.

    Produces flattened rows: {date, home_team, away_team, line, over_odds, under_odds, bookmaker}
    where odds are decimal and line is a float.
    """
    out: List[Dict[str, Any]] = []
    for ev in events or []:
        date = ev.get("commence_time")
        home = normalize_team_name(ev.get("home_team"))
        away = normalize_team_name(ev.get("away_team"))
        for bm in ev.get("bookmakers") or []:
            bm_title = bm.get("title") or bm.get("key") or bm.get("name")
            for mk in bm.get("markets") or []:
                mkey = str(mk.get("key") or "").lower()
                if mkey not in ("totals", "total_goals", "alternate_totals"):
                    continue
                outcomes = mk.get("outcomes") or []
                by_point: Dict[float, Dict[str, float]] = {}
                # There might also be a top-level 'point' on the market; we'll rely on per-outcome points if present
                for o in outcomes:
                    name = (o.get("name") or "").strip().lower()
                    price = o.get("price")
                    pt = (
                        o.get("point")
                        if o.get("point") is not None
                        else mk.get("point")
                    )
                    if pt is None or not isinstance(price, (int, float)):
                        continue
                    try:
                        ptf = float(pt)
                    except Exception:
                        continue
                    by_point.setdefault(ptf, {})
                    if name == "over":
                        by_point[ptf]["over_odds"] = float(price)
                    elif name == "under":
                        by_point[ptf]["under_odds"] = float(price)
                for pt, pair in by_point.items():
                    over_odds = pair.get("over_odds")
                    under_odds = pair.get("under_odds")
                    if home and away and date and (over_odds and under_odds):
                        out.append(
                            {
                                "date": str(date),
                                "home_team": home,
                                "away_team": away,
                                "line": float(pt),
                                "over_odds": float(over_odds),
                                "under_odds": float(under_odds),
                                "bookmaker": bm_title,
                            }
                        )
    return out


def _parse_event_odds_payload(payload: Any) -> List[Dict[str, Any]]:
    """Parse the response from /events/{id}/odds which typically returns a list with one event."""
    events = []
    if isinstance(payload, list):
        events = payload
    elif isinstance(payload, dict) and "bookmakers" in payload:
        events = [payload]
    else:
        events = payload if isinstance(payload, list) else []
    return _parse_totals_events(events)


def _parse_event_odds_history_payload(
    payload: Any, home: Optional[str], away: Optional[str], commence: Optional[str]
) -> List[Dict[str, Any]]:
    """Parse the response from /events/{id}/odds-history to flattened records for totals."""
    out_events: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        for snap in payload:
            if not isinstance(snap, dict):
                continue
            bms = snap.get("bookmakers") or []
            if not bms:
                continue
            ev = {
                "home_team": home,
                "away_team": away,
                "commence_time": commence,
                "bookmakers": bms,
            }
            out_events.append(ev)
    elif isinstance(payload, dict) and "bookmakers" in payload:
        ev = {
            "home_team": home,
            "away_team": away,
            "commence_time": commence,
            "bookmakers": payload.get("bookmakers") or [],
        }
        out_events.append(ev)
    return _parse_totals_events(out_events)


def fetch_goals_totals_from_odds_api(
    sport_key: str = "soccer_epl",
    regions: str = "eu",
    timeout: Optional[int] = None,
    historical: bool = False,
    ts_from: Optional[str] = None,
    ts_to: Optional[str] = None,
    markets: Optional[str] = None,
    bookmakers: Optional[str] = None,
) -> Dict[str, Any]:
    api_key = _env("ODDS_API_KEY")
    if not api_key:
        return {"error": "ODDS_API_KEY not set in environment"}
    base_url = _env("ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4")
    timeout = int(timeout or int(_env("ODDS_API_TIMEOUT", "12") or "12"))
    regions = _env("ODDS_API_REGIONS", regions)
    markets_try = markets or _env("ODDS_API_GOALS_MARKETS", "totals")
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets_try,
        "oddsFormat": _env("ODDS_API_ODDS_FORMAT", "decimal"),
        "dateFormat": _env("ODDS_API_DATE_FORMAT", "iso"),
    }
    if bookmakers or _env("ODDS_API_BOOKMAKERS"):
        params["bookmakers"] = bookmakers or _env("ODDS_API_BOOKMAKERS")
    if historical or _env("ODDS_API_ALLOW_HISTORICAL", "0") == "1":
        url = f"{base_url}/sports/{sport_key}/odds-history"
        if ts_from:
            params["from"] = ts_from
        if ts_to:
            params["to"] = ts_to
    else:
        url = f"{base_url}/sports/{sport_key}/odds"

    def _attempt(url: str, params: Dict[str, Any]):
        return _fetch(url, params, timeout)

    status, payload = _attempt(url, dict(params))
    # Fallback to try known totals market keys if invalid/unsupported
    if (
        status == 422
        and isinstance(payload, dict)
        and "INVALID_MARKET" in str(payload).upper()
    ):
        for alt in [
            "totals",
            "total_goals",
            "alternate_totals",
        ]:
            if params.get("markets") == alt:
                continue
            p2 = dict(params)
            p2["markets"] = alt
            status, payload = _attempt(url, p2)
            if status == 200:
                markets_try = alt
                break
    if status != 200:
        # We'll proceed to the per-event fallback below
        pass
    events = (
        payload
        if isinstance(payload, list)
        else payload.get("events")
        if isinstance(payload, dict)
        else []
    )
    rows = _parse_totals_events(events)
    # If zero rows (e.g., /odds suppressed totals), try per-event odds fetch
    if not rows:
        rows_accum: List[Dict[str, Any]] = []
        id_events: List[Dict[str, Any]] = []
        # Build event list with ids using h2h list endpoints (cheap) or historical list
        if (
            historical
            or _env("ODDS_API_ALLOW_HISTORICAL", "0") == "1"
            or ts_from
            or ts_to
        ):
            list_params_hist = {
                "apiKey": api_key,
                "regions": regions,
                "markets": "h2h",
                "oddsFormat": _env("ODDS_API_ODDS_FORMAT", "decimal"),
                "dateFormat": _env("ODDS_API_DATE_FORMAT", "iso"),
            }
            if ts_from:
                list_params_hist["from"] = ts_from
            if ts_to:
                list_params_hist["to"] = ts_to
            list_url_hist = f"{base_url}/sports/{sport_key}/odds-history"
            st_e, payload_e = _fetch(list_url_hist, list_params_hist, timeout)
            if st_e == 200 and isinstance(payload_e, list):
                id_events = payload_e
        else:
            list_params = {
                "apiKey": api_key,
                "regions": regions,
                "markets": "h2h",
                "oddsFormat": _env("ODDS_API_ODDS_FORMAT", "decimal"),
                "dateFormat": _env("ODDS_API_DATE_FORMAT", "iso"),
            }
            list_url = f"{base_url}/sports/{sport_key}/odds"
            st_e, payload_e = _fetch(list_url, list_params, timeout)
            if st_e == 200 and isinstance(payload_e, list):
                id_events = payload_e

        # Query per-event odds
        for ev in id_events:
            ev_id = ev.get("id") or ev.get("event_id")
            if not ev_id:
                continue
            home = normalize_team_name(ev.get("home_team"))
            away = normalize_team_name(ev.get("away_team"))
            commence = ev.get("commence_time")
            ev_params = {
                "apiKey": api_key,
                "regions": regions,
                "markets": markets_try,
                "oddsFormat": _env("ODDS_API_ODDS_FORMAT", "decimal"),
                "dateFormat": _env("ODDS_API_DATE_FORMAT", "iso"),
            }
            if bookmakers or _env("ODDS_API_BOOKMAKERS"):
                ev_params["bookmakers"] = bookmakers or _env("ODDS_API_BOOKMAKERS")
            if (
                historical
                or _env("ODDS_API_ALLOW_HISTORICAL", "0") == "1"
                or ts_from
                or ts_to
            ):
                if ts_from:
                    ev_params["from"] = ts_from
                if ts_to:
                    ev_params["to"] = ts_to
                ev_url = f"{base_url}/sports/{sport_key}/events/{ev_id}/odds-history"
                st_ev, payload_ev = _fetch(ev_url, ev_params, timeout)
                rows_ev: List[Dict[str, Any]] = []
                if st_ev == 200:
                    rows_ev = _parse_event_odds_history_payload(
                        payload_ev, home, away, commence
                    )
                if not rows_ev:
                    try_params = dict(ev_params)
                    try_params.pop("markets", None)
                    st2, payload2 = _fetch(ev_url, try_params, timeout)
                    if st2 == 200:
                        rows_ev = _parse_event_odds_history_payload(
                            payload2, home, away, commence
                        )
                if not rows_ev:
                    continue
            else:
                ev_url = f"{base_url}/sports/{sport_key}/events/{ev_id}/odds"
                st_ev, payload_ev = _fetch(ev_url, ev_params, timeout)
                rows_ev = []
                if st_ev == 200:
                    # Enrich payload with teams/date when missing
                    if isinstance(payload_ev, list) and payload_ev:
                        for rec in payload_ev:
                            rec.setdefault("home_team", home)
                            rec.setdefault("away_team", away)
                            rec.setdefault("commence_time", commence)
                    rows_ev = _parse_event_odds_payload(payload_ev)
                if not rows_ev:
                    try_params = dict(ev_params)
                    try_params.pop("markets", None)
                    st2, payload2 = _fetch(ev_url, try_params, timeout)
                    if st2 == 200:
                        if isinstance(payload2, list) and payload2:
                            for rec in payload2:
                                rec.setdefault("home_team", home)
                                rec.setdefault("away_team", away)
                                rec.setdefault("commence_time", commence)
                        rows_ev = _parse_event_odds_payload(payload2)
                if not rows_ev:
                    continue
            rows_accum.extend(rows_ev)
        if rows_accum:
            rows = rows_accum
    return {
        "records": rows,
        "count": len(rows),
        "source": "oddsapi",
        "historical": bool(historical or _env("ODDS_API_ALLOW_HISTORICAL", "0") == "1"),
        "markets_param": markets_try,
    }
