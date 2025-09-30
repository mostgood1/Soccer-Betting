"""
Betting Odds Service

Primary: Bovada coupon API (across supported leagues where available).
Fallback: The Odds API (H2H) when Bovada is unavailable for a fixture.

No synthetic odds are generated. If providers are unavailable, return empty odds
so callers can skip those matches rather than display fabricated data.
"""

import os
import requests
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import time
import logging
from dotenv import load_dotenv

from .the_odds_api_service import fetch_the_odds_api
from .bovada_odds_service import (
    fetch_pl_odds,
    fetch_bl1_odds,
    fetch_fl1_odds,
    fetch_sa_odds,
    fetch_pd_odds,
)
from .team_name_normalizer import normalize_team_name

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class BettingOddsService:
    """Service to provide betting odds for EPL matches"""

    def __init__(self):
        self.odds_api_key = os.getenv("ODDS_API_KEY")
        # Cache for odds data and indexes (per-match built structures)
        self.cache: Dict[str, Any] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        # Live odds change often—keep cache short
        self.cache_duration = int(os.getenv("ODDS_CACHE_SECONDS", "120"))
        # Per-sport latest fetch snapshot from The Odds API
        self._sport_cache: Dict[str, Dict[str, Any]] = {}
        self._sport_cache_expiry: Dict[str, datetime] = {}
        # Bovada snapshot caches per league
        self._bovada_cache: Dict[str, Dict[str, Any]] = {}
        self._bovada_cache_expiry: Dict[str, datetime] = {}
        # Supported sport keys map (covers PL, BL1, FL1, SA, PD)
        self.SPORT_KEYS = [
            "soccer_epl",
            "soccer_germany_bundesliga",
            "soccer_france_ligue_one",
            "soccer_italy_serie_a",
            "soccer_spain_la_liga",
        ]
        print("Betting Odds Service initialized (real providers)")

    # ---------- Cache management ----------
    def clear_cache(self) -> Dict[str, int]:
        sizes = {
            "entry_cache": len(self.cache),
            "sport_cache": len(self._sport_cache),
            "bovada_events_total": sum(
                len((snap or {}).get("events", []))
                for snap in self._bovada_cache.values()
            )
            if isinstance(self._bovada_cache, dict)
            else 0,
        }
        self.cache.clear()
        self.cache_expiry.clear()
        self._sport_cache.clear()
        self._sport_cache_expiry.clear()
        self._bovada_cache = {}
        self._bovada_cache_expiry = {}
        return sizes

    def prefetch(self, sport_key: Optional[str] = None) -> Dict[str, Any]:
        """Prefetch odds snapshots for one or all supported sports from The Odds API.
        Returns summary counts. No-op if ODDS_API_KEY is not set.
        """
        if not self.odds_api_key:
            return {
                "provider": "the-odds-api",
                "prefetched": 0,
                "note": "ODDS_API_KEY not set",
            }
        targets = [sport_key] if sport_key else list(self.SPORT_KEYS)
        now = datetime.now()
        ok = 0
        errs = []
        for s in targets:
            try:
                snap = fetch_the_odds_api(
                    sport_key=s,
                    markets="h2h",
                    odds_format="decimal",
                    api_key=self.odds_api_key,
                )
                if isinstance(snap, dict) and not snap.get("error"):
                    self._sport_cache[s] = snap
                    self._sport_cache_expiry[s] = now + timedelta(
                        seconds=self.cache_duration
                    )
                    ok += 1
                else:
                    errs.append(
                        {
                            s: snap.get("error")
                            if isinstance(snap, dict)
                            else "unknown error"
                        }
                    )
            except Exception as e:
                errs.append({s: str(e)})
        return {
            "provider": "the-odds-api",
            "prefetched": ok,
            "targets": targets,
            "errors": errs,
        }

    def prefetch_bovada(self) -> Dict[str, Any]:
        """Prefetch Bovada snapshots for all supported leagues.
        Includes H2H probabilities and a variety of derivative markets (totals, halves,
        team totals, BTTS, double chance, DNB, corners, cards, handicaps), where offered.
        """
        out = {}
        now = datetime.now()
        # Only keep events in the next N days for soccer
        try:
            window_days = int(os.getenv("BOVADA_WINDOW_DAYS", "14"))
            if window_days < 0:
                window_days = 14
        except Exception:
            window_days = 14
        window_end = now + timedelta(days=window_days)

        def _in_window(ev: Dict[str, Any]) -> bool:
            ts = ev.get("start_time_ms")
            if not isinstance(ts, int):
                return False
            try:
                ev_dt = datetime.fromtimestamp(ts / 1000)
                return now <= ev_dt <= window_end
            except Exception:
                return False

        def _do(key: str, fetcher):
            try:
                snap = fetcher()
                if isinstance(snap, dict) and "events" in snap:
                    # Filter events to window
                    evs = list(snap.get("events") or [])
                    filtered = [e for e in evs if _in_window(e)] if evs else []
                    # If filter eliminates all due to missing times, keep original list (avoid hiding data)
                    if not filtered and evs:
                        filtered = evs
                    snap_f = {**snap, "events": filtered}
                    self._bovada_cache[key] = snap_f
                    self._bovada_cache_expiry[key] = now + timedelta(
                        seconds=self.cache_duration
                    )
                    out[key] = len(filtered or [])
                else:
                    out[key] = 0
            except Exception as e:
                out[key] = f"error: {e}"

        _do("PL", fetch_pl_odds)
        _do("BL1", fetch_bl1_odds)
        _do("FL1", fetch_fl1_odds)
        _do("SA", fetch_sa_odds)
        _do("PD", fetch_pd_odds)
        return {"provider": "bovada", "events": out}

    def _decimal_to_american(self, decimal_odds: float) -> int:
        """Convert decimal odds to American odds (rounded to nearest integer).
        Returns 0 if decimal_odds is invalid or <= 1.0.
        """
        try:
            d = float(decimal_odds)
            if d <= 1.0:
                return 0
            # American odds: positive if underdog (decimal >= 2.0), negative if favorite
            if d >= 2.0:
                return int(round((d - 1.0) * 100))
            else:
                return int(round(-100.0 / (d - 1.0)))
        except Exception:
            return 0

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached odds are still valid"""
        if key not in self.cache_expiry:
            return False
        return datetime.now() < self.cache_expiry[key]

    def _cache_odds(self, key: str, data: Any) -> None:
        """Cache odds data with expiry"""
        self.cache[key] = data
        self.cache_expiry[key] = datetime.now() + timedelta(seconds=self.cache_duration)

    def get_match_odds(
        self,
        home_team: str,
        away_team: str,
        match_date: str = None,
        prefer_bovada_only: bool = False,
    ) -> Dict:
        """Get real betting odds for a specific match.
        Tries Bovada first (primary). If no Bovada event is found, falls back to The Odds API (H2H).
        Returns {} when providers are unavailable or match cannot be resolved.
        """
        cache_key = f"odds_{home_team}_{away_team}_{match_date or 'unknown'}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        # Normalize inputs for robust matching
        n_home = normalize_team_name(home_team) or home_team
        n_away = normalize_team_name(away_team) or away_team

        odds: Optional[Dict[str, Any]] = None

        # 1) Try Bovada first (primary across supported leagues)
        try:
            bov_ev = self._get_bovada_event(n_home, n_away)
        except Exception:
            bov_ev = None
        if bov_ev:
            try:
                odds = self._build_from_bovada(bov_ev)
            except Exception:
                odds = None

        # 2) If Bovada missing, optionally skip external fallback for speed
        if not odds and not prefer_bovada_only:
            api_odds = self._lookup_the_odds_api(n_home, n_away)
            if not api_odds and (
                ("manchester" in n_home.lower()) or ("manchester" in n_away.lower())
            ):
                # Tiny heuristic: try raw names if normalization obscured branded forms
                api_odds = self._lookup_the_odds_api(home_team, away_team)
            odds = api_odds

        # Persist minimal structure to reduce duplicate provider calls within TTL
        self._cache_odds(cache_key, odds or {})
        return odds or {}

    # ---------- Providers ----------
    def _lookup_the_odds_api(self, home: str, away: str) -> Optional[Dict[str, Any]]:
        if not self.odds_api_key:
            return None
        now = datetime.now()
        # Build or reuse per-sport snapshots
        for sport in self.SPORT_KEYS:
            fresh = self._sport_cache.get(sport)
            exp = self._sport_cache_expiry.get(sport)
            if not fresh or not exp or now >= exp:
                try:
                    snap = fetch_the_odds_api(
                        sport_key=sport,
                        markets="h2h",
                        odds_format="decimal",
                        api_key=self.odds_api_key,
                    )
                    if isinstance(snap, dict) and not snap.get("error"):
                        self._sport_cache[sport] = snap
                        self._sport_cache_expiry[sport] = now + timedelta(
                            seconds=self.cache_duration
                        )
                except Exception:
                    continue
            snap = self._sport_cache.get(sport)
            if not snap or not isinstance(snap.get("records"), list):
                continue
            # Try to match home/away team pair
            for rec in snap["records"]:
                th = normalize_team_name(rec.get("home_team") or "") or (
                    rec.get("home_team") or ""
                )
                ta = normalize_team_name(rec.get("away_team") or "") or (
                    rec.get("away_team") or ""
                )
                if not th or not ta:
                    continue
                if th.lower() == home.lower() and ta.lower() == away.lower():
                    # Compose normalized H2H odds (consensus across bookmakers is not provided here; we take first available)
                    # Prefer first bookmaker that has outcomes
                    bm = next(
                        (
                            b
                            for b in rec.get("bookmakers") or []
                            if any((m.get("outcomes") for m in b.get("markets") or []))
                        ),
                        None,
                    )
                    if not bm:
                        continue
                    mk = next(
                        (
                            m
                            for m in bm.get("markets") or []
                            if m.get("key") == "h2h" and m.get("outcomes")
                        ),
                        None,
                    )
                    if not mk:
                        continue
                    outcomes = mk.get("outcomes") or []
                    # Expected names: 'HOME', 'DRAW', 'AWAY' (case may vary)
                    prices: Dict[str, Optional[float]] = {
                        "H": None,
                        "D": None,
                        "A": None,
                    }
                    for o in outcomes:
                        name = (o.get("name") or "").upper()
                        price = o.get("price")
                        try:
                            dec = float(price) if price is not None else None
                        except Exception:
                            dec = None
                        if "DRAW" in name:
                            prices["D"] = 1.0 / dec if dec and dec > 1.0 else None
                        elif "HOME" in name:
                            prices["H"] = 1.0 / dec if dec and dec > 1.0 else None
                        elif "AWAY" in name:
                            prices["A"] = 1.0 / dec if dec and dec > 1.0 else None
                    probs = self._normalize_market_odds(
                        {"home_odds": None, "draw_odds": None, "away_odds": None},
                        override_implied=prices,
                    )

                    # Also expose decimal + American if available
                    def _first_decimal(label: str) -> Optional[float]:
                        # scan outcomes again for decimal
                        for o in outcomes:
                            name = (o.get("name") or "").upper()
                            price = o.get("price")
                            try:
                                dec = float(price) if price is not None else None
                            except Exception:
                                dec = None
                            if label == "home" and "HOME" in name:
                                return dec
                            if label == "draw" and "DRAW" in name:
                                return dec
                            if label == "away" and "AWAY" in name:
                                return dec
                        return None

                    dh = _first_decimal("home")
                    dd = _first_decimal("draw")
                    da = _first_decimal("away")
                    return {
                        "provider": "the-odds-api",
                        "timestamp": datetime.now().isoformat(),
                        "market_odds": {
                            "match_winner": {
                                "home": {
                                    "team": rec.get("home_team"),
                                    "odds": dh,
                                    "odds_american": self._decimal_to_american(dh)
                                    if dh
                                    else None,
                                    "probability": probs.get("H"),
                                },
                                "draw": {
                                    "odds": dd,
                                    "odds_american": self._decimal_to_american(dd)
                                    if dd
                                    else None,
                                    "probability": probs.get("D"),
                                },
                                "away": {
                                    "team": rec.get("away_team"),
                                    "odds": da,
                                    "odds_american": self._decimal_to_american(da)
                                    if da
                                    else None,
                                    "probability": probs.get("A"),
                                },
                            }
                        },
                        "overround": None,
                    }
        return None

    def _lookup_bovada_pl(self, home: str, away: str) -> Optional[Dict[str, Any]]:
        try:
            snap = fetch_pl_odds()
            if not isinstance(snap, dict) or "events" not in snap:
                return None
            for ev in snap["events"]:
                eh = normalize_team_name(ev.get("home_team") or "") or (
                    ev.get("home_team") or ""
                )
                ea = normalize_team_name(ev.get("away_team") or "") or (
                    ev.get("away_team") or ""
                )
                if (
                    eh
                    and ea
                    and eh.lower() == home.lower()
                    and ea.lower() == away.lower()
                ):
                    return self._build_from_bovada(ev)
        except Exception:
            return None
        return None

    # ---------- Mock removal: preserve method signatures used in tests ----------
    def _generate_realistic_odds(self, home_team: str, away_team: str) -> Dict:
        """Legacy method removed: returns empty structure to avoid synthetic data."""
        return {}

    def _identify_best_value(
        self, home_odds: float, draw_odds: float, away_odds: float
    ) -> Dict:
        """Deprecated (no synthetic value metrics). Return minimal structure."""
        return {"recommended_bet": None, "value_rating": None, "odds": None}

    def get_all_match_odds(self, fixtures: List[Dict]) -> List[Dict]:
        """Get odds for multiple matches"""
        odds_list = []

        for fixture in fixtures:
            try:
                home_team = fixture.get("home_team", "")
                away_team = fixture.get("away_team", "")
                match_date = fixture.get("date", "")

                if home_team and away_team:
                    odds = self.get_match_odds(home_team, away_team, match_date)
                    odds["fixture_id"] = fixture.get("id", "unknown")
                    odds["match_date"] = match_date
                    odds_list.append(odds)
            except Exception as e:
                logger.error(f"Error getting odds for {fixture}: {e}")
                continue

        return odds_list

    def get_betting_insights(self, fixtures_with_odds: List[Dict]) -> Dict:
        """Generate betting insights from odds data"""
        if not fixtures_with_odds:
            return {}

        total_matches = len(fixtures_with_odds)
        home_wins_expected = 0
        away_wins_expected = 0
        draws_expected = 0
        high_value_bets = []

        for match_odds in fixtures_with_odds:
            try:
                market_odds = match_odds.get("market_odds", {})
                match_winner = market_odds.get("match_winner", {})

                # Count expected outcomes based on probabilities
                home_prob = match_winner.get("home", {}).get("probability", 0)
                draw_prob = match_winner.get("draw", {}).get("probability", 0)
                away_prob = match_winner.get("away", {}).get("probability", 0)

                if home_prob > draw_prob and home_prob > away_prob:
                    home_wins_expected += 1
                elif away_prob > home_prob and away_prob > draw_prob:
                    away_wins_expected += 1
                else:
                    draws_expected += 1

                # Identify high-value bets
                betting_value = match_odds.get("betting_value", {})
                if betting_value.get("confidence", 0) > 0.8:
                    high_value_bets.append(
                        {
                            "match": match_odds.get("match", ""),
                            "recommended_bet": betting_value.get("recommended_bet", ""),
                            "value_rating": betting_value.get("value_rating", 0),
                            "confidence": betting_value.get("confidence", 0),
                        }
                    )
            except Exception as e:
                logger.error(f"Error processing betting insights: {e}")
                continue

        return {
            "total_matches_analyzed": total_matches,
            "expected_outcomes": {
                "home_wins": home_wins_expected,
                "draws": draws_expected,
                "away_wins": away_wins_expected,
            },
            "outcome_percentages": {
                "home_wins": round((home_wins_expected / total_matches) * 100, 1)
                if total_matches > 0
                else 0,
                "draws": round((draws_expected / total_matches) * 100, 1)
                if total_matches > 0
                else 0,
                "away_wins": round((away_wins_expected / total_matches) * 100, 1)
                if total_matches > 0
                else 0,
            },
            "high_value_bets": [],
            "average_home_odds": self._calculate_average_odds(
                "home", fixtures_with_odds
            ),
            "average_draw_odds": self._calculate_average_odds(
                "draw", fixtures_with_odds
            ),
            "average_away_odds": self._calculate_average_odds(
                "away", fixtures_with_odds
            ),
        }

    # ---------- Helpers ----------
    def _get_bovada_event(self, home: str, away: str) -> Optional[Dict[str, Any]]:
        """Fetch Bovada snapshots (cached) for supported leagues and return a matching event for home/away."""
        now = datetime.now()

        def _ensure_cache(key: str, fetcher) -> None:
            exp = self._bovada_cache_expiry.get(key)
            snap = self._bovada_cache.get(key)
            if not snap or not exp or now >= exp:
                try:
                    s = fetcher()
                except Exception:
                    s = None
                if isinstance(s, dict) and "events" in s:
                    self._bovada_cache[key] = s
                    self._bovada_cache_expiry[key] = now + timedelta(
                        seconds=self.cache_duration
                    )

        _ensure_cache("PL", fetch_pl_odds)
        _ensure_cache("BL1", fetch_bl1_odds)
        _ensure_cache("FL1", fetch_fl1_odds)
        _ensure_cache("SA", fetch_sa_odds)
        _ensure_cache("PD", fetch_pd_odds)
        for snap in self._bovada_cache.values():
            evs = (snap or {}).get("events") or []
            for ev in evs:
                eh = normalize_team_name(ev.get("home_team") or "") or (
                    ev.get("home_team") or ""
                )
                ea = normalize_team_name(ev.get("away_team") or "") or (
                    ev.get("away_team") or ""
                )
                if (
                    eh
                    and ea
                    and eh.lower() == home.lower()
                    and ea.lower() == away.lower()
                ):
                    return ev
        return None

    def _prob_to_decimal(self, prob: Optional[float]) -> Optional[float]:
        try:
            if prob and prob > 0:
                return round(1.0 / float(prob), 2)
        except Exception:
            return None
        return None

    def _build_from_bovada(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        """Build a full odds structure from a Bovada EPL event."""
        h2h = ev.get("h2h") or {}
        dh = self._prob_to_decimal(h2h.get("H"))
        dd = self._prob_to_decimal(h2h.get("D"))
        da = self._prob_to_decimal(h2h.get("A"))
        out = {
            "provider": "bovada",
            "timestamp": datetime.now().isoformat(),
            "market_odds": {},
            "overround": None,
        }
        if any(v is not None for v in (dh, dd, da)):
            out["market_odds"]["match_winner"] = {
                "home": {
                    "team": ev.get("home_team"),
                    "odds": dh,
                    "odds_american": self._decimal_to_american(dh) if dh else None,
                    "probability": h2h.get("H"),
                },
                "draw": {
                    "odds": dd,
                    "odds_american": self._decimal_to_american(dd) if dd else None,
                    "probability": h2h.get("D"),
                },
                "away": {
                    "team": ev.get("away_team"),
                    "odds": da,
                    "odds_american": self._decimal_to_american(da) if da else None,
                    "probability": h2h.get("A"),
                },
            }
        # Add non-H2H markets
        self._inject_bovada_markets(out, ev)
        return out

    def _merge_bovada_non_h2h(
        self, odds: Dict[str, Any], ev: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not odds:
            return self._build_from_bovada(ev)
        if "market_odds" not in odds or not isinstance(odds["market_odds"], dict):
            odds["market_odds"] = {}
        self._inject_bovada_markets(odds, ev)
        return odds

    def _inject_bovada_markets(self, base: Dict[str, Any], ev: Dict[str, Any]) -> None:
        mo = base.setdefault("market_odds", {})
        # Totals (full game)
        totals = ev.get("totals") or []
        if totals:
            rows = []
            for t in totals:
                over_prob = t.get("over_prob")
                under_prob = t.get("under_prob")
                over_dec = self._prob_to_decimal(over_prob)
                under_dec = self._prob_to_decimal(under_prob)
                rows.append(
                    {
                        "line": t.get("line"),
                        "over": {
                            "probability": over_prob,
                            "odds": over_dec,
                            "odds_american": self._decimal_to_american(over_dec)
                            if over_dec
                            else None,
                        },
                        "under": {
                            "probability": under_prob,
                            "odds": under_dec,
                            "odds_american": self._decimal_to_american(under_dec)
                            if under_dec
                            else None,
                        },
                    }
                )
            mo["totals"] = rows
        # First half totals
        fh = ev.get("first_half_totals") or []
        if fh:
            rows = []
            for t in fh:
                over_prob = t.get("over_prob")
                under_prob = t.get("under_prob")
                over_dec = self._prob_to_decimal(over_prob)
                under_dec = self._prob_to_decimal(under_prob)
                rows.append(
                    {
                        "line": t.get("line"),
                        "over": {
                            "probability": over_prob,
                            "odds": over_dec,
                            "odds_american": self._decimal_to_american(over_dec)
                            if over_dec
                            else None,
                        },
                        "under": {
                            "probability": under_prob,
                            "odds": under_dec,
                            "odds_american": self._decimal_to_american(under_dec)
                            if under_dec
                            else None,
                        },
                    }
                )
            mo["first_half_totals"] = rows
        # Second half totals
        sh = ev.get("second_half_totals") or []
        if sh:
            rows = []
            for t in sh:
                over_prob = t.get("over_prob")
                under_prob = t.get("under_prob")
                over_dec = self._prob_to_decimal(over_prob)
                under_dec = self._prob_to_decimal(under_prob)
                rows.append(
                    {
                        "line": t.get("line"),
                        "over": {
                            "probability": over_prob,
                            "odds": over_dec,
                            "odds_american": self._decimal_to_american(over_dec)
                            if over_dec
                            else None,
                        },
                        "under": {
                            "probability": under_prob,
                            "odds": under_dec,
                            "odds_american": self._decimal_to_american(under_dec)
                            if under_dec
                            else None,
                        },
                    }
                )
            mo["second_half_totals"] = rows
        # Team totals
        tt = ev.get("team_totals") or []
        if tt:
            home_rows = []
            away_rows = []
            for t in tt:
                over_prob = t.get("over_prob")
                under_prob = t.get("under_prob")
                over_dec = self._prob_to_decimal(over_prob)
                under_dec = self._prob_to_decimal(under_prob)
                row = {
                    "line": t.get("line"),
                    "over": {
                        "probability": over_prob,
                        "odds": over_dec,
                        "odds_american": self._decimal_to_american(over_dec)
                        if over_dec
                        else None,
                    },
                    "under": {
                        "probability": under_prob,
                        "odds": under_dec,
                        "odds_american": self._decimal_to_american(under_dec)
                        if under_dec
                        else None,
                    },
                }
                if t.get("side") == "home":
                    home_rows.append(row)
                elif t.get("side") == "away":
                    away_rows.append(row)
            if home_rows or away_rows:
                mo["team_totals"] = {}
                if home_rows:
                    mo["team_totals"]["home"] = home_rows
                if away_rows:
                    mo["team_totals"]["away"] = away_rows
        # Corners totals
        ct = ev.get("corners_totals") or []
        if ct:
            rows = []
            for t in ct:
                over_prob = t.get("over_prob")
                under_prob = t.get("under_prob")
                over_dec = self._prob_to_decimal(over_prob)
                under_dec = self._prob_to_decimal(under_prob)
                rows.append(
                    {
                        "line": t.get("line"),
                        "over": {
                            "probability": over_prob,
                            "odds": over_dec,
                            "odds_american": self._decimal_to_american(over_dec)
                            if over_dec
                            else None,
                        },
                        "under": {
                            "probability": under_prob,
                            "odds": under_dec,
                            "odds_american": self._decimal_to_american(under_dec)
                            if under_dec
                            else None,
                        },
                    }
                )
            mo["corners_totals"] = rows
        # Team corners
        tc = ev.get("team_corners") or []
        if tc:
            home_rows = []
            away_rows = []
            for t in tc:
                over_prob = t.get("over_prob")
                under_prob = t.get("under_prob")
                over_dec = self._prob_to_decimal(over_prob)
                under_dec = self._prob_to_decimal(under_prob)
                row = {
                    "line": t.get("line"),
                    "over": {
                        "probability": over_prob,
                        "odds": over_dec,
                        "odds_american": self._decimal_to_american(over_dec)
                        if over_dec
                        else None,
                    },
                    "under": {
                        "probability": under_prob,
                        "odds": under_dec,
                        "odds_american": self._decimal_to_american(under_dec)
                        if under_dec
                        else None,
                    },
                }
                if t.get("side") == "home":
                    home_rows.append(row)
                elif t.get("side") == "away":
                    away_rows.append(row)
            if home_rows or away_rows:
                mo["team_corners"] = {}
                if home_rows:
                    mo["team_corners"]["home"] = home_rows
                if away_rows:
                    mo["team_corners"]["away"] = away_rows
        # BTTS
        btts = ev.get("btts")
        if btts:
            mo["both_teams_to_score"] = {
                "yes": {
                    "probability": btts.get("yes_prob"),
                    "odds": self._prob_to_decimal(btts.get("yes_prob")),
                    "odds_american": self._decimal_to_american(
                        self._prob_to_decimal(btts.get("yes_prob"))
                    )
                    if self._prob_to_decimal(btts.get("yes_prob"))
                    else None,
                },
                "no": {
                    "probability": btts.get("no_prob"),
                    "odds": self._prob_to_decimal(btts.get("no_prob")),
                    "odds_american": self._decimal_to_american(
                        self._prob_to_decimal(btts.get("no_prob"))
                    )
                    if self._prob_to_decimal(btts.get("no_prob"))
                    else None,
                },
            }
        # Double Chance
        dc = ev.get("double_chance")
        if dc:

            def mk(prob):
                dec = self._prob_to_decimal(prob)
                return {
                    "probability": prob,
                    "odds": dec,
                    "odds_american": self._decimal_to_american(dec) if dec else None,
                }

            mo["double_chance"] = {
                "1X": mk((dc.get("1X") or {}).get("prob")),
                "X2": mk((dc.get("X2") or {}).get("prob")),
                "12": mk((dc.get("12") or {}).get("prob")),
            }
        # Draw No Bet
        dnb = ev.get("draw_no_bet")
        if dnb:

            def mk2(prob):
                dec = self._prob_to_decimal(prob)
                return {
                    "probability": prob,
                    "odds": dec,
                    "odds_american": self._decimal_to_american(dec) if dec else None,
                }

            mo["draw_no_bet"] = {
                "home": mk2(dnb.get("home_prob")),
                "away": mk2(dnb.get("away_prob")),
            }
        # Asian Handicap
        ah = ev.get("asian_handicap") or []
        if ah:
            rows = []
            for t in ah:
                rows.append(
                    {
                        "line": t.get("line"),
                        "home": {
                            "probability": t.get("home_prob"),
                            "odds": self._prob_to_decimal(t.get("home_prob")),
                            "odds_american": self._decimal_to_american(
                                self._prob_to_decimal(t.get("home_prob"))
                            )
                            if self._prob_to_decimal(t.get("home_prob"))
                            else None,
                        },
                        "away": {
                            "probability": t.get("away_prob"),
                            "odds": self._prob_to_decimal(t.get("away_prob")),
                            "odds_american": self._decimal_to_american(
                                self._prob_to_decimal(t.get("away_prob"))
                            )
                            if self._prob_to_decimal(t.get("away_prob"))
                            else None,
                        },
                    }
                )
            mo["asian_handicap"] = rows
        # Corners Handicap
        ch = ev.get("corners_handicap") or []
        if ch:
            rows = []
            for t in ch:
                rows.append(
                    {
                        "line": t.get("line"),
                        "home": {
                            "probability": t.get("home_prob"),
                            "odds": self._prob_to_decimal(t.get("home_prob")),
                            "odds_american": self._decimal_to_american(
                                self._prob_to_decimal(t.get("home_prob"))
                            )
                            if self._prob_to_decimal(t.get("home_prob"))
                            else None,
                        },
                        "away": {
                            "probability": t.get("away_prob"),
                            "odds": self._prob_to_decimal(t.get("away_prob")),
                            "odds_american": self._decimal_to_american(
                                self._prob_to_decimal(t.get("away_prob"))
                            )
                            if self._prob_to_decimal(t.get("away_prob"))
                            else None,
                        },
                    }
                )
            mo["corners_handicap"] = rows
        # Cards Totals
        cards = ev.get("cards_totals") or []
        if cards:
            rows = []
            for t in cards:
                rows.append(
                    {
                        "line": t.get("line"),
                        "over": {
                            "probability": t.get("over_prob"),
                            "odds": self._prob_to_decimal(t.get("over_prob")),
                            "odds_american": self._decimal_to_american(
                                self._prob_to_decimal(t.get("over_prob"))
                            )
                            if self._prob_to_decimal(t.get("over_prob"))
                            else None,
                        },
                        "under": {
                            "probability": t.get("under_prob"),
                            "odds": self._prob_to_decimal(t.get("under_prob")),
                            "odds_american": self._decimal_to_american(
                                self._prob_to_decimal(t.get("under_prob"))
                            )
                            if self._prob_to_decimal(t.get("under_prob"))
                            else None,
                        },
                    }
                )
            mo["cards_totals"] = rows

    def _normalize_market_odds(
        self,
        normalized_input: Dict[str, Any],
        override_implied: Optional[Dict[str, Optional[float]]] = None,
    ) -> Dict[str, float]:
        """Normalize market odds: consume decimal odds or implied probabilities and return normalized probs.
        If override_implied is supplied, use it directly (expects keys 'H','D','A').
        """
        if override_implied is not None:
            inv = {
                k: (v if isinstance(v, (int, float)) and v > 0 else 0.0)
                for k, v in override_implied.items()
            }
            s = sum(inv.values())
            if s <= 0:
                return {"H": 0.0, "D": 0.0, "A": 0.0}
            return {k: inv[k] / s for k in ("H", "D", "A")}
        try:
            h = (
                float(normalized_input.get("home_odds"))
                if normalized_input.get("home_odds")
                else None
            )
            d = (
                float(normalized_input.get("draw_odds"))
                if normalized_input.get("draw_odds")
                else None
            )
            a = (
                float(normalized_input.get("away_odds"))
                if normalized_input.get("away_odds")
                else None
            )
            inv = {
                "H": (1.0 / h if h and h > 1.0 else 0.0),
                "D": (1.0 / d if d and d > 1.0 else 0.0),
                "A": (1.0 / a if a and a > 1.0 else 0.0),
            }
            s = sum(inv.values())
            if s <= 0:
                return {"H": 0.0, "D": 0.0, "A": 0.0}
            return {k: inv[k] / s for k in ("H", "D", "A")}
        except Exception:
            return {"H": 0.0, "D": 0.0, "A": 0.0}

    def _calculate_average_odds(
        self, outcome: str, fixtures_with_odds: List[Dict]
    ) -> float:
        """Calculate average odds for a specific outcome"""
        odds_values = []

        for match_odds in fixtures_with_odds:
            try:
                market_odds = match_odds.get("market_odds", {})
                match_winner = market_odds.get("match_winner", {})

                if outcome in match_winner:
                    odds = match_winner[outcome].get("odds", 0)
                    if odds > 0:
                        odds_values.append(odds)
            except Exception as e:
                continue

        return round(sum(odds_values) / len(odds_values), 2) if odds_values else 0.0


# Initialize global betting odds service
betting_odds_service = BettingOddsService()


def main():
    """Test the betting odds service"""
    service = BettingOddsService()

    print("=== BETTING ODDS SERVICE TEST ===")

    # Test individual match odds
    odds = service.get_match_odds("Manchester City", "Liverpool", "2025-09-28")
    if not odds:
        print("No odds available from providers (check ODDS_API_KEY / network).")
    else:
        mw = (odds.get("market_odds") or {}).get("match_winner") or {}
        print("Provider:", odds.get("provider"))
        print("Home odds:", (mw.get("home") or {}).get("odds"))
        print("Draw odds:", (mw.get("draw") or {}).get("odds"))
        print("Away odds:", (mw.get("away") or {}).get("odds"))

    print(f"\nBetting Odds Service test complete.")


if __name__ == "__main__":
    main()
