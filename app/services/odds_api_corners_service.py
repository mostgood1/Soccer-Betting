"""Odds API Corners Markets Fetcher

Fetch corners totals (and optionally team corners) odds via an Odds API-compatible
service, if supported by the provider/plan. Produces records compatible with
corners_market_service (date, teams, line, over_odds, under_odds, bookmaker, market_type, side?).

Notes:
- Not all Odds API providers expose corners markets; this service attempts to
  fetch markets whose key contains the substring 'corner'.
- Historical odds may require a paid plan and a different endpoint. If enabled
  via env flag ODDS_API_ALLOW_HISTORICAL=1, we'll try an odds-history endpoint.
  Otherwise we fetch current/near-term markets.
"""
from __future__ import annotations
import os
from typing import Dict, Any, List, Optional, Tuple

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


def _parse_corners_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ev in events or []:
        date = ev.get("commence_time")
        home = normalize_team_name(ev.get("home_team"))
        away = normalize_team_name(ev.get("away_team"))
        for bm in ev.get("bookmakers") or []:
            bm_title = bm.get("title") or bm.get("key")
            for mk in bm.get("markets") or []:
                mkey = str(mk.get("key") or "").lower()
                if "corner" not in mkey:
                    continue
                # Build per-point map of over/under
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
                                "market_type": "total",
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
        # Some providers wrap the single event in an array anyway
        events = payload if isinstance(payload, list) else []
    # Inject synthetic home/away/date if provided at top-level
    out: List[Dict[str, Any]] = []
    for ev in events:
        out.append(ev)
    return _parse_corners_events(out)


def _parse_event_odds_history_payload(
    payload: Any, home: Optional[str], away: Optional[str], commence: Optional[str]
) -> List[Dict[str, Any]]:
    """Parse the response from /events/{id}/odds-history.
    Expected shape: a list of snapshots, each with 'bookmakers' array similar to /odds payloads, plus a 'timestamp'.
    We normalize to a synthetic single-event-per-snapshot structure and reuse the corners parser.
    """
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
    # Reuse standard parser
    return _parse_corners_events(out_events)


def fetch_corners_totals_from_odds_api(
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
    # Markets to try: allow param override; otherwise env; fallback to alternate_totals_corners
    markets_try = markets or _env(
        "ODDS_API_CORNERS_MARKETS", "alternate_totals_corners"
    )
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
    # Fallback to try known corners market keys if invalid market
    if (
        status == 422
        and isinstance(payload, dict)
        and "INVALID_MARKET" in str(payload).upper()
    ):
        for alt in [
            "alternate_totals_corners",
            "totals_corners",
            "total_corners",
            "corners",
            "alternate_total_corners",
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
        # Try event-level fallback: fetch event ids with h2h then query per-event markets
        pass
    events = (
        payload
        if isinstance(payload, list)
        else payload.get("events")
        if isinstance(payload, dict)
        else []
    )
    rows = _parse_corners_events(events)
    # If zero rows (e.g., because the /odds endpoint suppresses alternate markets), try per-event odds fetch
    if not rows:
        # 1) Fetch event list with ids via h2h (lightweight). For historical, use odds-history to get past events.
        rows_accum: List[Dict[str, Any]] = []
        id_events: List[Dict[str, Any]] = []
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

        # 2) For each id, query event odds for the desired markets
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
                # Retry without markets filter if empty or non-200
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
                # Retry without markets filter if empty or non-200
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
        "events": len(events) if isinstance(events, list) else None,
        "records": rows,
        "count": len(rows),
        "source": "oddsapi",
        "historical": bool(historical or _env("ODDS_API_ALLOW_HISTORICAL", "0") == "1"),
        "markets_param": markets_try,
    }
