"""Bovada Odds Service (backup provider).

Fetches EPL (Premier League) markets from Bovada's public coupon API and converts
odds to implied probabilities. Intended as a backup when primary odds sources
are unavailable.

Endpoints (observed; may change):
  - https://www.bovada.lv/services/sports/event/coupon/events/A/description/soccer/england/premier-league

Notes:
  - We attempt to parse:
      * Match Result / 3-Way Moneyline (H/D/A)
      * Total Goals Over/Under (full game)
      * First Half Total Goals Over/Under
      * Second Half Total Goals Over/Under
      * Team Total Goals Over/Under (home/away)
  - Output is normalized to sum to 1 per market (removes overround).
  - Be resilient to missing fields or format changes.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import os, time

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore


def _normalize_team(name: Optional[str]) -> Optional[str]:
    if not name:
        return name
    raw = name.strip()
    key = raw.lower()
    repl = {
        "man utd": "Manchester United",
        "man united": "Manchester United",
        "manchester united": "Manchester United",
        "man city": "Manchester City",
        "manchester city": "Manchester City",
        "spurs": "Tottenham",
        "tottenham hotspur": "Tottenham",
        "wolves": "Wolves",
        "wolverhampton wanderers": "Wolves",
        "afc bournemouth": "Bournemouth",
        "brighton & hove albion": "Brighton",
        "nottingham forest": "Nottm Forest",
        "nottm forest": "Nottm Forest",
        "sheffield united": "Sheffield Utd",
        "sheffield utd": "Sheffield Utd",
        "west ham united": "West Ham",
        "newcastle united": "Newcastle",
    }
    cleaned = key.replace(" fc", "").replace(" afc", "").replace("&", "and")
    if cleaned in repl:
        return repl[cleaned]
    if key in repl:
        return repl[key]
    return " ".join(w.capitalize() for w in raw.split())


def _american_to_decimal(american: Optional[float]) -> Optional[float]:
    if american is None:
        return None
    try:
        a = float(american)
        if a > 0:
            return 1.0 + (a / 100.0)
        elif a < 0:
            return 1.0 + (100.0 / (-a))
        else:
            return None
    except Exception:
        return None


def _implied_from_decimal(dec: Optional[float]) -> Optional[float]:
    if dec is None or dec <= 1.0:
        return None
    return 1.0 / dec


def _normalize_probs(values: Dict[str, Optional[float]]) -> Dict[str, float]:
    inv = {
        k: (v if isinstance(v, (int, float)) and v > 0 else 0.0)
        for k, v in values.items()
    }
    s = sum(inv.values())
    if s <= 0:
        return {k: 0.0 for k in values}
    return {k: inv[k] / s for k in values}


def _fetch_bovada_coupon(
    url_suffix: str, timeout: Optional[int] = None
) -> Dict[str, Any]:
    """Fetch a Bovada coupon by URL suffix and parse to normalized events structure.
    Example suffixes: '/soccer/england/premier-league', '/soccer/germany/1-bundesliga'
    Tries the primary base (BOVADA_BASE_URL) then optional alternates (BOVADA_ALT_BASES)
    such as Bodog in case of regional blocks.
    """
    if requests is None:
        return {"error": "requests not available"}
    # Build candidate base URLs list
    primary = os.getenv("BOVADA_BASE_URL", "https://www.bovada.lv")
    alts_env = os.getenv("BOVADA_ALT_BASES", "")
    alt_list = [s.strip() for s in alts_env.split(",") if s.strip()] or [
        "https://www.bodog.eu"
    ]
    bases = [primary] + [b for b in alt_list if b != primary]
    timeout = timeout or int(os.getenv("BOVADA_TIMEOUT", "12"))
    headers = {
        "Accept": "application/json,text/plain,*/*",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }
    last_err: Dict[str, Any] = {}
    for base in bases:
        # Try both coupon groups (A/B) as Bovada sometimes switches group
        arr = None
        for grp in ("A", "B"):
            # Include query params to retrieve pre-match markets and sufficient events
            url = (
                base.rstrip("/")
                + f"/services/sports/event/coupon/events/{grp}/description"
                + url_suffix
                + "?marketFilterId=def&preMatchOnly=true&lang=en&eventsLimit=2000"
            )
            try:
                r = requests.get(url, headers=headers, timeout=timeout)
                if r.status_code != 200:
                    last_err = {
                        "error": f"status {r.status_code}",
                        "body": r.text[:400],
                        "base": base,
                        "group": grp,
                    }
                    continue
                arr = r.json()
                break
            except Exception as e:  # pragma: no cover
                last_err = {"error": str(e), "base": base, "group": grp}
                continue
        if arr is None:
            continue
        events_out: List[Dict[str, Any]] = []

        # Response can be nested; flatten out event list
        def iter_events(payload: Any):
            if isinstance(payload, list):
                for x in payload:
                    yield from iter_events(x)
            elif isinstance(payload, dict):
                # Some Bovada payloads have key 'events'
                if "events" in payload and isinstance(payload["events"], list):
                    for ev in payload["events"]:
                        yield ev
                # Some contain direct event fields
                elif "description" in payload and "competitors" in payload:
                    yield payload

        def parse_event(ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            # Teams
            comps = ev.get("competitors") or []
            home_name = None
            away_name = None
            for c in comps:
                if c.get("home") is True:
                    home_name = c.get("name") or c.get("description")
                else:
                    away_name = c.get("name") or c.get("description")
            home = _normalize_team(home_name)
            away = _normalize_team(away_name)
            if not (home and away):
                return None
            # Kickoff time (Bovada may provide seconds or milliseconds). Normalize to ms and ISO8601 UTC.
            commence_raw = ev.get("startTimeMillis") or ev.get("startTime")
            start_time_ms: Optional[int] = None
            utc_date: Optional[str] = None
            try:
                if commence_raw is not None:
                    v = int(commence_raw)
                    # Heuristic: if looks like seconds, convert to ms
                    if v < 10**12:
                        v = v * 1000
                    start_time_ms = v
                    try:
                        from datetime import datetime, timezone

                        utc_date = (
                            datetime.fromtimestamp(v / 1000, tz=timezone.utc)
                            .isoformat()
                            .replace("+00:00", "Z")
                        )
                    except Exception:
                        utc_date = None
            except Exception:
                start_time_ms = None
                utc_date = None
            # Markets reside in displayGroups -> markets -> outcomes
            display_groups = ev.get("displayGroups") or []
            h2h_probs: Optional[Dict[str, float]] = None
            h2h_decimals: Dict[str, Optional[float]] = {"H": None, "D": None, "A": None}
            h2h_american: Dict[str, Optional[int]] = {"H": None, "D": None, "A": None}
            totals_markets: List[Dict[str, Any]] = []
            fh_totals: List[Dict[str, Any]] = []
            sh_totals: List[Dict[str, Any]] = []
            team_totals: List[Dict[str, Any]] = []
            corners_totals: List[Dict[str, Any]] = []
            team_corners: List[Dict[str, Any]] = []
            btts: Optional[Dict[str, Any]] = None
            double_chance: Optional[Dict[str, Any]] = None
            draw_no_bet: Optional[Dict[str, Any]] = None
            asian_handicap: List[Dict[str, Any]] = []
            corners_handicap: List[Dict[str, Any]] = []
            cards_totals: List[Dict[str, Any]] = []
            for dg in display_groups:
                markets = dg.get("markets") or []
                for mk in markets:
                    desc = (mk.get("description") or mk.get("displayKey") or "").lower()
                    outcomes = mk.get("outcomes") or []
                    # H2H 3-way: Match Result / 3-Way Moneyline
                    if (
                        any(
                            k in desc
                            for k in (
                                "match result",
                                "3-way moneyline",
                                "moneyline 3-way",
                                "result",
                            )
                        )
                        and len(outcomes) >= 2
                    ):
                        probs: Dict[str, Optional[float]] = {
                            "H": None,
                            "D": None,
                            "A": None,
                        }
                        for o in outcomes:
                            oname = (
                                o.get("description") or o.get("name") or ""
                            ).lower()
                            price = o.get("price") or {}
                            american = price.get("american")
                            dec = (
                                _american_to_decimal(float(american))
                                if american is not None
                                else None
                            )
                            imp = _implied_from_decimal(dec)
                            if "draw" in oname:
                                probs["D"] = imp
                                h2h_decimals["D"] = dec
                                try:
                                    h2h_american["D"] = (
                                        int(american)
                                        if isinstance(american, (int, float))
                                        else None
                                    )
                                except Exception:
                                    h2h_american["D"] = None
                            elif home_name and _normalize_team(
                                oname
                            ) == _normalize_team(home_name):
                                probs["H"] = imp
                                h2h_decimals["H"] = dec
                                try:
                                    h2h_american["H"] = (
                                        int(american)
                                        if isinstance(american, (int, float))
                                        else None
                                    )
                                except Exception:
                                    h2h_american["H"] = None
                            elif away_name and _normalize_team(
                                oname
                            ) == _normalize_team(away_name):
                                probs["A"] = imp
                                h2h_decimals["A"] = dec
                                try:
                                    h2h_american["A"] = (
                                        int(american)
                                        if isinstance(american, (int, float))
                                        else None
                                    )
                                except Exception:
                                    h2h_american["A"] = None
                            elif "home" in oname:
                                probs["H"] = imp
                                h2h_decimals["H"] = dec
                                try:
                                    h2h_american["H"] = (
                                        int(american)
                                        if isinstance(american, (int, float))
                                        else None
                                    )
                                except Exception:
                                    h2h_american["H"] = None
                            elif "away" in oname:
                                probs["A"] = imp
                                h2h_decimals["A"] = dec
                                try:
                                    h2h_american["A"] = (
                                        int(american)
                                        if isinstance(american, (int, float))
                                        else None
                                    )
                                except Exception:
                                    h2h_american["A"] = None
                        h2h_probs = _normalize_probs(probs)
                    # Totals (Full game): look for Over/Under with a handicap/line
                    if ("total" in desc and "half" not in desc) and len(outcomes) >= 2:
                        over_imp = None
                        under_imp = None
                        over_ml = None
                        under_ml = None
                        line_val: Optional[float] = None
                        for o in outcomes:
                            oname = (
                                o.get("description") or o.get("name") or ""
                            ).lower()
                            price = o.get("price") or {}
                            american = price.get("american")
                            dec = (
                                _american_to_decimal(float(american))
                                if american is not None
                                else None
                            )
                            imp = _implied_from_decimal(dec)
                            # Handicap may appear at market or outcome level
                            hcap = (
                                o.get("price", {}).get("handicap")
                                or mk.get("handicap")
                                or o.get("handicap")
                            )
                            try:
                                if hcap is not None:
                                    line_val = float(hcap)
                            except Exception:
                                pass
                            if "over" in oname:
                                over_imp = imp
                                over_ml = (
                                    int(american)
                                    if isinstance(american, (int, float))
                                    else None
                                )
                            elif "under" in oname:
                                under_imp = imp
                                under_ml = (
                                    int(american)
                                    if isinstance(american, (int, float))
                                    else None
                                )
                        if over_imp is not None or under_imp is not None:
                            probs = _normalize_probs(
                                {"over": over_imp, "under": under_imp}
                            )
                            totals_markets.append(
                                {
                                    "line": line_val,
                                    "over_prob": probs.get("over"),
                                    "under_prob": probs.get("under"),
                                    "over_ml": over_ml,
                                    "under_ml": under_ml,
                                }
                            )
                    # Both Teams To Score (Yes/No)
                    if ("both teams to score" in desc or "btts" in desc) and len(
                        outcomes
                    ) >= 2:
                        yes_imp = None
                        no_imp = None
                        yes_ml = None
                        no_ml = None
                        for o in outcomes:
                            oname = (
                                o.get("description") or o.get("name") or ""
                            ).lower()
                            price = o.get("price") or {}
                            american = price.get("american")
                            dec = (
                                _american_to_decimal(float(american))
                                if american is not None
                                else None
                            )
                            imp = _implied_from_decimal(dec)
                            if "yes" in oname:
                                yes_imp = imp
                                yes_ml = (
                                    int(american)
                                    if isinstance(american, (int, float))
                                    else None
                                )
                            elif "no" in oname:
                                no_imp = imp
                                no_ml = (
                                    int(american)
                                    if isinstance(american, (int, float))
                                    else None
                                )
                        if yes_imp is not None or no_imp is not None:
                            btts = {
                                "yes_prob": yes_imp,
                                "no_prob": no_imp,
                                "yes_ml": yes_ml,
                                "no_ml": no_ml,
                            }
                    # Double Chance (1X, X2, 12)
                    if ("double chance" in desc) and len(outcomes) >= 2:
                        dc_map: Dict[str, Dict[str, Optional[float]]] = {}
                        for o in outcomes:
                            oname = (
                                o.get("description") or o.get("name") or ""
                            ).lower()
                            price = o.get("price") or {}
                            american = price.get("american")
                            dec = (
                                _american_to_decimal(float(american))
                                if american is not None
                                else None
                            )
                            imp = _implied_from_decimal(dec)
                            key = None
                            if "home or draw" in oname or "1x" in oname:
                                key = "1X"
                            elif "draw or away" in oname or "x2" in oname:
                                key = "X2"
                            elif "home or away" in oname or "12" in oname:
                                key = "12"
                            if key:
                                dc_map[key] = {
                                    "prob": imp,
                                    "ml": int(american)
                                    if isinstance(american, (int, float))
                                    else None,
                                }
                        if dc_map:
                            double_chance = {
                                "1X": dc_map.get("1X"),
                                "X2": dc_map.get("X2"),
                                "12": dc_map.get("12"),
                            }
                    # Draw No Bet (2-way)
                    if (
                        ("draw no bet" in desc)
                        or ("moneyline" in desc and "2-way" in desc)
                        or ("2 way" in desc)
                    ) and len(outcomes) >= 2:
                        dh_imp = None
                        da_imp = None
                        dh_ml = None
                        da_ml = None
                        for o in outcomes:
                            oname = (
                                o.get("description") or o.get("name") or ""
                            ).lower()
                            price = o.get("price") or {}
                            american = price.get("american")
                            dec = (
                                _american_to_decimal(float(american))
                                if american is not None
                                else None
                            )
                            imp = _implied_from_decimal(dec)
                            name_norm = _normalize_team(oname)
                            if home_name and name_norm == _normalize_team(home_name):
                                dh_imp = imp
                                dh_ml = (
                                    int(american)
                                    if isinstance(american, (int, float))
                                    else None
                                )
                            elif away_name and name_norm == _normalize_team(away_name):
                                da_imp = imp
                                da_ml = (
                                    int(american)
                                    if isinstance(american, (int, float))
                                    else None
                                )
                            elif "home" in oname:
                                dh_imp = imp
                                dh_ml = (
                                    int(american)
                                    if isinstance(american, (int, float))
                                    else None
                                )
                            elif "away" in oname:
                                da_imp = imp
                                da_ml = (
                                    int(american)
                                    if isinstance(american, (int, float))
                                    else None
                                )
                        if dh_imp is not None or da_imp is not None:
                            draw_no_bet = {
                                "home_prob": dh_imp,
                                "away_prob": da_imp,
                                "home_ml": dh_ml,
                                "away_ml": da_ml,
                            }
                    # Total Corners (Full game)
                    if (
                        "corner" in desc and "total" in desc and "team" not in desc
                    ) and len(outcomes) >= 2:
                        c_over_imp = None
                        c_under_imp = None
                        c_over_ml = None
                        c_under_ml = None
                        c_line_val: Optional[float] = None
                        for o in outcomes:
                            oname = (
                                o.get("description") or o.get("name") or ""
                            ).lower()
                            price = o.get("price") or {}
                            american = price.get("american")
                            dec = (
                                _american_to_decimal(float(american))
                                if american is not None
                                else None
                            )
                            imp = _implied_from_decimal(dec)
                            hcap = (
                                o.get("price", {}).get("handicap")
                                or mk.get("handicap")
                                or o.get("handicap")
                            )
                            try:
                                if hcap is not None:
                                    c_line_val = float(hcap)
                            except Exception:
                                pass
                            if "over" in oname:
                                c_over_imp = imp
                                c_over_ml = (
                                    int(american)
                                    if isinstance(american, (int, float))
                                    else None
                                )
                            elif "under" in oname:
                                c_under_imp = imp
                                c_under_ml = (
                                    int(american)
                                    if isinstance(american, (int, float))
                                    else None
                                )
                        if c_over_imp is not None or c_under_imp is not None:
                            probs = _normalize_probs(
                                {"over": c_over_imp, "under": c_under_imp}
                            )
                            corners_totals.append(
                                {
                                    "line": c_line_val,
                                    "over_prob": probs.get("over"),
                                    "under_prob": probs.get("under"),
                                    "over_ml": c_over_ml,
                                    "under_ml": c_under_ml,
                                }
                            )
                    # Cards Totals (Full game) — sometimes tagged as booking cards
                    if (
                        ("card" in desc or "booking" in desc)
                        and "total" in desc
                        and "team" not in desc
                    ) and len(outcomes) >= 2:
                        k_over_imp = None
                        k_under_imp = None
                        k_over_ml = None
                        k_under_ml = None
                        k_line_val: Optional[float] = None
                        for o in outcomes:
                            oname = (
                                o.get("description") or o.get("name") or ""
                            ).lower()
                            price = o.get("price") or {}
                            american = price.get("american")
                            dec = (
                                _american_to_decimal(float(american))
                                if american is not None
                                else None
                            )
                            imp = _implied_from_decimal(dec)
                            hcap = (
                                o.get("price", {}).get("handicap")
                                or mk.get("handicap")
                                or o.get("handicap")
                            )
                            try:
                                if hcap is not None:
                                    k_line_val = float(hcap)
                            except Exception:
                                pass
                            if "over" in oname:
                                k_over_imp = imp
                                k_over_ml = (
                                    int(american)
                                    if isinstance(american, (int, float))
                                    else None
                                )
                            elif "under" in oname:
                                k_under_imp = imp
                                k_under_ml = (
                                    int(american)
                                    if isinstance(american, (int, float))
                                    else None
                                )
                        if k_over_imp is not None or k_under_imp is not None:
                            probs = _normalize_probs(
                                {"over": k_over_imp, "under": k_under_imp}
                            )
                            cards_totals.append(
                                {
                                    "line": k_line_val,
                                    "over_prob": probs.get("over"),
                                    "under_prob": probs.get("under"),
                                    "over_ml": k_over_ml,
                                    "under_ml": k_under_ml,
                                }
                            )
                    # Handicap / Asian Handicap (full game)
                    if (
                        ("handicap" in desc or "asian" in desc) and "corner" not in desc
                    ) and len(outcomes) >= 2:
                        home_ml = None
                        away_ml = None
                        h_line: Optional[float] = None
                        h_prob: Optional[float] = None
                        a_prob: Optional[float] = None
                        for o in outcomes:
                            oname = (
                                o.get("description") or o.get("name") or ""
                            ).lower()
                            price = o.get("price") or {}
                            american = price.get("american")
                            dec = (
                                _american_to_decimal(float(american))
                                if american is not None
                                else None
                            )
                            imp = _implied_from_decimal(dec)
                            hcap = (
                                o.get("price", {}).get("handicap")
                                or mk.get("handicap")
                                or o.get("handicap")
                            )
                            ln = None
                            try:
                                if hcap is not None:
                                    ln = float(hcap)
                            except Exception:
                                pass
                            name_norm = _normalize_team(oname)
                            if (
                                home_name
                                and name_norm == _normalize_team(home_name)
                                or "home" in oname
                            ):
                                h_prob = imp
                                home_ml = (
                                    int(american)
                                    if isinstance(american, (int, float))
                                    else None
                                )
                                if ln is not None:
                                    h_line = ln
                            elif (
                                away_name
                                and name_norm == _normalize_team(away_name)
                                or "away" in oname
                            ):
                                a_prob = imp
                                away_ml = (
                                    int(american)
                                    if isinstance(american, (int, float))
                                    else None
                                )
                                if ln is not None:
                                    h_line = ln if h_line is None else h_line
                        if h_prob is not None or a_prob is not None:
                            asian_handicap.append(
                                {
                                    "line": h_line,
                                    "home_prob": h_prob,
                                    "away_prob": a_prob,
                                    "home_ml": home_ml,
                                    "away_ml": away_ml,
                                }
                            )
                    # Corners Handicap
                    if ("corner" in desc and "handicap" in desc) and len(outcomes) >= 2:
                        home_ml = None
                        away_ml = None
                        c_line: Optional[float] = None
                        h_prob: Optional[float] = None
                        a_prob: Optional[float] = None
                        for o in outcomes:
                            oname = (
                                o.get("description") or o.get("name") or ""
                            ).lower()
                            price = o.get("price") or {}
                            american = price.get("american")
                            dec = (
                                _american_to_decimal(float(american))
                                if american is not None
                                else None
                            )
                            imp = _implied_from_decimal(dec)
                            hcap = (
                                o.get("price", {}).get("handicap")
                                or mk.get("handicap")
                                or o.get("handicap")
                            )
                            ln = None
                            try:
                                if hcap is not None:
                                    ln = float(hcap)
                            except Exception:
                                pass
                            name_norm = _normalize_team(oname)
                            if (
                                home_name
                                and name_norm == _normalize_team(home_name)
                                or "home" in oname
                            ):
                                h_prob = imp
                                home_ml = (
                                    int(american)
                                    if isinstance(american, (int, float))
                                    else None
                                )
                                if ln is not None:
                                    c_line = ln
                            elif (
                                away_name
                                and name_norm == _normalize_team(away_name)
                                or "away" in oname
                            ):
                                a_prob = imp
                                away_ml = (
                                    int(american)
                                    if isinstance(american, (int, float))
                                    else None
                                )
                                if ln is not None:
                                    c_line = ln if c_line is None else c_line
                        if h_prob is not None or a_prob is not None:
                            corners_handicap.append(
                                {
                                    "line": c_line,
                                    "home_prob": h_prob,
                                    "away_prob": a_prob,
                                    "home_ml": home_ml,
                                    "away_ml": away_ml,
                                }
                            )
                    # First Half Totals
                    if (
                        any(
                            k in desc
                            for k in (
                                "first half total",
                                "1st half total",
                                "1st half - total",
                            )
                        )
                        and len(outcomes) >= 2
                    ):
                        over_imp = None
                        under_imp = None
                        over_ml = None
                        under_ml = None
                        line_val: Optional[float] = None
                        for o in outcomes:
                            oname = (
                                o.get("description") or o.get("name") or ""
                            ).lower()
                            price = o.get("price") or {}
                            american = price.get("american")
                            dec = (
                                _american_to_decimal(float(american))
                                if american is not None
                                else None
                            )
                            imp = _implied_from_decimal(dec)
                            hcap = (
                                o.get("price", {}).get("handicap")
                                or mk.get("handicap")
                                or o.get("handicap")
                            )
                            try:
                                if hcap is not None:
                                    line_val = float(hcap)
                            except Exception:
                                pass
                            if "over" in oname:
                                over_imp = imp
                                over_ml = (
                                    int(american)
                                    if isinstance(american, (int, float))
                                    else None
                                )
                            elif "under" in oname:
                                under_imp = imp
                                under_ml = (
                                    int(american)
                                    if isinstance(american, (int, float))
                                    else None
                                )
                        if over_imp is not None or under_imp is not None:
                            probs = _normalize_probs(
                                {"over": over_imp, "under": under_imp}
                            )
                            fh_totals.append(
                                {
                                    "line": line_val,
                                    "over_prob": probs.get("over"),
                                    "under_prob": probs.get("under"),
                                    "over_ml": over_ml,
                                    "under_ml": under_ml,
                                }
                            )
                    # Second Half Totals
                    if (
                        any(
                            k in desc
                            for k in (
                                "second half total",
                                "2nd half total",
                                "2nd half - total",
                            )
                        )
                        and len(outcomes) >= 2
                    ):
                        over_imp = None
                        under_imp = None
                        over_ml = None
                        under_ml = None
                        line_val: Optional[float] = None
                        for o in outcomes:
                            oname = (
                                o.get("description") or o.get("name") or ""
                            ).lower()
                            price = o.get("price") or {}
                            american = price.get("american")
                            dec = (
                                _american_to_decimal(float(american))
                                if american is not None
                                else None
                            )
                            imp = _implied_from_decimal(dec)
                            hcap = (
                                o.get("price", {}).get("handicap")
                                or mk.get("handicap")
                                or o.get("handicap")
                            )
                            try:
                                if hcap is not None:
                                    line_val = float(hcap)
                            except Exception:
                                pass
                            if "over" in oname:
                                over_imp = imp
                                over_ml = (
                                    int(american)
                                    if isinstance(american, (int, float))
                                    else None
                                )
                            elif "under" in oname:
                                under_imp = imp
                                under_ml = (
                                    int(american)
                                    if isinstance(american, (int, float))
                                    else None
                                )
                        if over_imp is not None or under_imp is not None:
                            probs = _normalize_probs(
                                {"over": over_imp, "under": under_imp}
                            )
                            sh_totals.append(
                                {
                                    "line": line_val,
                                    "over_prob": probs.get("over"),
                                    "under_prob": probs.get("under"),
                                    "over_ml": over_ml,
                                    "under_ml": under_ml,
                                }
                            )
                    # Team Total Goals (for each side)
                    if ("team total" in desc or "team goals" in desc) and len(
                        outcomes
                    ) >= 2:
                        # Determine side from market description
                        side: Optional[str] = None
                        d = desc
                        hn = _normalize_team(home_name or "") or ""
                        an = _normalize_team(away_name or "") or ""
                        if hn and (hn.lower() in d or "home" in d):
                            side = "home"
                        elif an and (an.lower() in d or "away" in d):
                            side = "away"
                        # Some markets might specify team in outcomes; try fallback
                        if side is None:
                            for o in outcomes:
                                oname = (
                                    o.get("description") or o.get("name") or ""
                                ).lower()
                                if hn and hn.lower() in oname:
                                    side = "home"
                                    break
                                if an and an.lower() in oname:
                                    side = "away"
                                    break
                        if side is not None:
                            over_imp = None
                            under_imp = None
                            over_ml = None
                            under_ml = None
                            line_val: Optional[float] = None
                            for o in outcomes:
                                oname = (
                                    o.get("description") or o.get("name") or ""
                                ).lower()
                                price = o.get("price") or {}
                                american = price.get("american")
                                dec = (
                                    _american_to_decimal(float(american))
                                    if american is not None
                                    else None
                                )
                                imp = _implied_from_decimal(dec)
                                hcap = (
                                    o.get("price", {}).get("handicap")
                                    or mk.get("handicap")
                                    or o.get("handicap")
                                )
                                try:
                                    if hcap is not None:
                                        line_val = float(hcap)
                                except Exception:
                                    pass
                                if "over" in oname:
                                    over_imp = imp
                                    over_ml = (
                                        int(american)
                                        if isinstance(american, (int, float))
                                        else None
                                    )
                                elif "under" in oname:
                                    under_imp = imp
                                    under_ml = (
                                        int(american)
                                        if isinstance(american, (int, float))
                                        else None
                                    )
                            if over_imp is not None or under_imp is not None:
                                probs = _normalize_probs(
                                    {"over": over_imp, "under": under_imp}
                                )
                                team_totals.append(
                                    {
                                        "side": side,
                                        "line": line_val,
                                        "over_prob": probs.get("over"),
                                        "under_prob": probs.get("under"),
                                        "over_ml": over_ml,
                                        "under_ml": under_ml,
                                    }
                                )
                    # Team Total Corners (for each side)
                    if (
                        "corner" in desc and "team" in desc and "total" in desc
                    ) and len(outcomes) >= 2:
                        side_c: Optional[str] = None
                        d = desc
                        hn = _normalize_team(home_name or "") or ""
                        an = _normalize_team(away_name or "") or ""
                        if hn and (hn.lower() in d or "home" in d):
                            side_c = "home"
                        elif an and (an.lower() in d or "away" in d):
                            side_c = "away"
                        if side_c is None:
                            for o in outcomes:
                                oname = (
                                    o.get("description") or o.get("name") or ""
                                ).lower()
                                if hn and hn.lower() in oname:
                                    side_c = "home"
                                    break
                                if an and an.lower() in oname:
                                    side_c = "away"
                                    break
                        if side_c is not None:
                            over_imp = None
                            under_imp = None
                            over_ml = None
                            under_ml = None
                            line_val: Optional[float] = None
                            for o in outcomes:
                                oname = (
                                    o.get("description") or o.get("name") or ""
                                ).lower()
                                price = o.get("price") or {}
                                american = price.get("american")
                                dec = (
                                    _american_to_decimal(float(american))
                                    if american is not None
                                    else None
                                )
                                imp = _implied_from_decimal(dec)
                                hcap = (
                                    o.get("price", {}).get("handicap")
                                    or mk.get("handicap")
                                    or o.get("handicap")
                                )
                                try:
                                    if hcap is not None:
                                        line_val = float(hcap)
                                except Exception:
                                    pass
                                if "over" in oname:
                                    over_imp = imp
                                    over_ml = (
                                        int(american)
                                        if isinstance(american, (int, float))
                                        else None
                                    )
                                elif "under" in oname:
                                    under_imp = imp
                                    under_ml = (
                                        int(american)
                                        if isinstance(american, (int, float))
                                        else None
                                    )
                            if over_imp is not None or under_imp is not None:
                                probs = _normalize_probs(
                                    {"over": over_imp, "under": under_imp}
                                )
                                team_corners.append(
                                    {
                                        "side": side_c,
                                        "line": line_val,
                                        "over_prob": probs.get("over"),
                                        "under_prob": probs.get("under"),
                                        "over_ml": over_ml,
                                        "under_ml": under_ml,
                                    }
                                )
            out: Dict[str, Any] = {
                "home_team": home,
                "away_team": away,
                # seconds epoch for backward compatibility
                "commence_time": int(start_time_ms / 1000) if start_time_ms else None,
            }
            if h2h_probs:
                out["h2h"] = h2h_probs
                # include price representations if available
                if any(v for v in h2h_decimals.values()):
                    out["h2h_decimal"] = h2h_decimals
                if any(v for v in h2h_american.values()):
                    out["h2h_american"] = h2h_american
            if totals_markets:
                out["totals"] = totals_markets
            if fh_totals:
                out["first_half_totals"] = fh_totals
            if sh_totals:
                out["second_half_totals"] = sh_totals
            if team_totals:
                out["team_totals"] = team_totals
            if corners_totals:
                out["corners_totals"] = corners_totals
            if team_corners:
                out["team_corners"] = team_corners
            if btts:
                out["btts"] = btts
            if double_chance:
                out["double_chance"] = double_chance
            if draw_no_bet:
                out["draw_no_bet"] = draw_no_bet
            if asian_handicap:
                out["asian_handicap"] = asian_handicap
            if corners_handicap:
                out["corners_handicap"] = corners_handicap
            if cards_totals:
                out["cards_totals"] = cards_totals
            # Attach normalized start time fields for downstream filters
            if start_time_ms is not None:
                out["start_time_ms"] = start_time_ms
            if utc_date is not None:
                out["utc_date"] = utc_date
            return out

        for raw_ev in iter_events(arr):
            try:
                norm = parse_event(raw_ev)
                if norm:
                    events_out.append(norm)
            except Exception:
                continue
        # If we found events, return immediately; otherwise, try next base
        if len(events_out) > 0:
            return {"events": events_out, "provider": "bovada", "base": base}
        # capture empty attempt info and try next
        last_err = {"error": "no_events", "base": base}
    # All bases failed or returned no events
    return last_err or {"events": []}


def _fetch_bovada_multi(
    suffixes: List[str], timeout: Optional[int] = None
) -> Dict[str, Any]:
    """Try multiple Bovada URL suffixes and return the first successful events payload.
    If all fail or contain no events, return the last response (for error visibility).
    """
    last: Dict[str, Any] = {}
    for suf in suffixes:
        res = _fetch_bovada_coupon(suf, timeout=timeout)
        last = res
        evs = (res or {}).get("events")
        if isinstance(evs, list) and len(evs) > 0:
            return res
    return last or {"events": []}


def fetch_pl_odds(timeout: Optional[int] = None) -> Dict[str, Any]:
    """Premier League (England)"""
    return _fetch_bovada_multi(
        [
            "/soccer/england/premier-league",
            "/soccer/england/english-premier-league",
            "/soccer/england/epl",
        ],
        timeout=timeout,
    )


def fetch_bl1_odds(timeout: Optional[int] = None) -> Dict[str, Any]:
    """Bundesliga (Germany): Bovada slug can be '1-bundesliga' or 'bundesliga'. Try both."""
    return _fetch_bovada_multi(
        [
            "/soccer/germany/1-bundesliga",
            "/soccer/germany/bundesliga",
        ],
        timeout=timeout,
    )


def fetch_fl1_odds(timeout: Optional[int] = None) -> Dict[str, Any]:
    """Ligue 1 (France)"""
    return _fetch_bovada_multi(
        [
            "/soccer/france/ligue-1",
            "/soccer/france/ligue-1-uber-eats",
        ],
        timeout=timeout,
    )


def fetch_sa_odds(timeout: Optional[int] = None) -> Dict[str, Any]:
    """Serie A (Italy)"""
    return _fetch_bovada_multi(
        [
            "/soccer/italy/serie-a",
            "/soccer/italy/serie-a-tim",
            # Alternate season branding variants observed
            "/soccer/italy/serie-a-ea-sports",
            "/soccer/italy/serie-a-ea-sports-2025-26",
            "/soccer/italy/serie-a-ea-sports-2024-25",
        ],
        timeout=timeout,
    )


def fetch_pd_odds(timeout: Optional[int] = None) -> Dict[str, Any]:
    """La Liga (Spain)"""
    return _fetch_bovada_multi(
        [
            "/soccer/spain/la-liga",
            "/soccer/spain/laliga",
            "/soccer/spain/la-liga-ea-sports",
            "/soccer/spain/la-liga-santander",
        ],
        timeout=timeout,
    )


def fetch_eu_odds(timeout: Optional[int] = None) -> Dict[str, Any]:
    """Europe-wide soccer aggregator on Bovada.
    Useful fallback when league-specific coupon endpoints are empty or unavailable.
    """
    return _fetch_bovada_multi(
        [
            "/soccer/europe",
        ],
        timeout=timeout,
    )
