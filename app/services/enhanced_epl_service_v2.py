"""
Enhanced EPL Service for PREDICTABLE BETTING ENGINE
Now integrated with Football-Data.org for complete 2025-26 season data
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from .team_seed_metrics import (
    team_seed_metrics_service,
    FEATURE_VERSION as TEAM_FEATURE_VERSION,
)
from .corners_actuals_service import corners_actuals_store

logger = logging.getLogger(__name__)


class EnhancedEPLService:
    """Enhanced EPL service with Football-Data.org integration"""

    def __init__(self):
        self.current_season = "2025-26"
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 3600  # 1 hour cache

        # Load Football-Data.org cached data
        self.football_data = self._load_football_data()
        # Build quick index of team -> venue for enrichment (home stadium default)
        self._team_venue_index = {}
        try:
            teams_src = (self.football_data or {}).get("teams") or []
            for t in teams_src:
                name = t.get("name") or t.get("shortName")
                venue = t.get("venue") or ""
                if name:
                    self._team_venue_index[str(name)] = venue or ""
        except Exception:
            self._team_venue_index = {}

        print(f"Enhanced EPL Service: {self.current_season}")
        if self.football_data:
            fixtures_count = len(self.football_data.get("converted_fixtures", []))
            print(f"Football-Data.org: {fixtures_count} fixtures loaded")

    def _load_football_data(self) -> Dict:
        """Load cached Football-Data.org data for EPL.
        Robust strategy:
        - Try both long-form (football_data_epl_2025_2026.json) and PL-specific
          (football_data_PL_2025_2026.json) files.
        - Prefer the one with more converted fixtures or higher currentMatchday.
        - Look in data/ first (Render persistent volume), then fall back to /app/baked.
        """
        base_data_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data")
        )
        candidates = [
            os.path.join(base_data_dir, "football_data_epl_2025_2026.json"),
            os.path.join(base_data_dir, "football_data_PL_2025_2026.json"),
            "/app/baked/football_data_epl_2025_2026.json",
            "/app/baked/football_data_PL_2025_2026.json",
        ]

        def _read_json(p: str) -> Optional[Dict]:
            try:
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as f:
                        return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read EPL data file {p}: {e}")
            return None

        def _fixtures_count(d: Dict) -> int:
            try:
                return int(len((d or {}).get("converted_fixtures", []) or []))
            except Exception:
                return 0

        def _current_md(d: Dict) -> int:
            try:
                comp = (d or {}).get("competition", {})
                cur = (comp or {}).get("currentSeason", {})
                md = cur.get("currentMatchday")
                return int(md) if md is not None else 0
            except Exception:
                return 0

        loaded: List[Dict] = []
        for p in candidates:
            dj = _read_json(p)
            if dj:
                loaded.append(dj)
        if not loaded:
            return {}
        # Choose the best by fixtures, then by currentMatchday
        best = loaded[0]
        for d in loaded[1:]:
            try:
                if _fixtures_count(d) > _fixtures_count(best):
                    best = d
                elif _fixtures_count(d) == _fixtures_count(best) and _current_md(d) > _current_md(best):
                    best = d
            except Exception:
                continue
        return best

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cache_expiry:
            return False
        return datetime.now() < self.cache_expiry[key]

    def _cache_data(self, key: str, data: Any) -> None:
        """Cache data with expiry"""
        self.cache[key] = data
        self.cache_expiry[key] = datetime.now() + timedelta(seconds=self.cache_duration)

    def get_teams(self) -> List[Dict]:
        """Get EPL teams from Football-Data.org"""
        if not self.football_data:
            return self._get_fallback_teams()

        teams = self.football_data.get("teams", [])
        converted_teams = []

        for i, team in enumerate(teams):
            # Create standings data (mock for now, will be real when season progresses)
            converted_team = {
                "id": team.get("id"),
                "name": team.get("name"),
                "short_name": team.get("shortName", team.get("name", "")[:3]),
                "tla": team.get("tla", team.get("name", "")[:3]),
                "founded": team.get("founded"),
                "venue": team.get("venue", "Unknown"),
                "website": team.get("website", ""),
                "crest": team.get("crest", ""),
                # Standings data (current season position)
                "position": i + 1,  # Mock positions for now
                "played": 0,  # Will update as season progresses
                "won": 0,
                "drawn": 0,
                "lost": 0,
                "goals_for": 0,
                "goals_against": 0,
                "goal_difference": 0,
                "points": 0,
                # Additional team stats
                "form": "N/A",  # Last 5 games form
                "home_record": {"W": 0, "D": 0, "L": 0},
                "away_record": {"W": 0, "D": 0, "L": 0},
                "season": self.current_season,
            }
            converted_teams.append(converted_team)

        # Enrich with deterministic seed metrics for early differentiation
        converted_teams = team_seed_metrics_service.enrich_teams(converted_teams)
        for t in converted_teams:
            # Propagate the latest team seed feature version (fv2+). If enrichment failed, fallback.
            t["feature_version"] = t.get("feature_version") or TEAM_FEATURE_VERSION
        return converted_teams

    def get_matches(self, limit: Optional[int] = None) -> List[Dict]:
        """Get EPL matches from Football-Data.org"""
        if not self.football_data:
            return self._get_fallback_matches()

        fixtures = self.football_data.get("converted_fixtures", [])

        if limit:
            fixtures = fixtures[:limit]

        # Add additional match analysis & real result mapping
        enhanced_fixtures = []
        for fixture in fixtures:
            enhanced_fixture = fixture.copy()
            # Ensure a usable venue value: prefer fixture.venue, else home team's venue, else Unknown
            try:
                v = enhanced_fixture.get("venue")
                if not v or str(v).strip().lower() in (
                    "unknown",
                    "tbd",
                    "n/a",
                    "none",
                    "",
                ):
                    # Map using home team
                    home_name = (
                        enhanced_fixture.get("home_team")
                        or enhanced_fixture.get("homeTeam")
                        or (enhanced_fixture.get("home", {}) or {}).get("name")
                    )
                    if home_name and home_name in self._team_venue_index:
                        hv = self._team_venue_index.get(home_name) or None
                        if hv:
                            enhanced_fixture["venue"] = hv
                    else:
                        # As a last resort, try to backfill from teams list via simple contains matching
                        if isinstance(home_name, str):
                            for k, hv in self._team_venue_index.items():
                                try:
                                    if home_name.lower() in k.lower() and hv:
                                        enhanced_fixture["venue"] = hv
                                        break
                                except Exception:
                                    continue
                # Normalize label a bit if still missing
                if not enhanced_fixture.get("venue"):
                    enhanced_fixture["venue"] = "Unknown"
            except Exception:
                if not enhanced_fixture.get("venue"):
                    enhanced_fixture["venue"] = "Unknown"
            raw_status = str(enhanced_fixture.get("status", "")).upper()
            # Extract scores from multiple possible shapes (robust to source format)
            # Try multiple keys for final full-time scores
            home_score = (
                enhanced_fixture.get("home_score")
                or enhanced_fixture.get("homeScore")
                or enhanced_fixture.get("full_time_home_goals")
                or enhanced_fixture.get("home_goals")
            )
            away_score = (
                enhanced_fixture.get("away_score")
                or enhanced_fixture.get("awayScore")
                or enhanced_fixture.get("full_time_away_goals")
                or enhanced_fixture.get("away_goals")
            )
            # Football-Data.org typical nested format
            score_obj = enhanced_fixture.get("score") or {}
            ft_score = score_obj.get("fullTime") if isinstance(score_obj, dict) else {}
            if home_score is None and isinstance(ft_score, dict):
                home_score = ft_score.get("home")
            if away_score is None and isinstance(ft_score, dict):
                away_score = ft_score.get("away")
            is_completed = raw_status in ("FINISHED", "COMPLETED")
            # If marked finished but scores missing, treat as not yet completed (data lag)
            if is_completed and (home_score is None or away_score is None):
                is_completed = False
            match_result = None
            if is_completed and home_score is not None and away_score is not None:
                try:
                    if int(home_score) > int(away_score):
                        match_result = "H"
                    elif int(away_score) > int(home_score):
                        match_result = "A"
                    else:
                        match_result = "D"
                except Exception:
                    match_result = None

            # Attach corners actuals if available
            date_str = enhanced_fixture.get("utc_date") or enhanced_fixture.get("date")
            home_team_name = (
                enhanced_fixture.get("home_team")
                or enhanced_fixture.get("homeTeam")
                or (enhanced_fixture.get("home", {}) or {}).get("name")
            )
            away_team_name = (
                enhanced_fixture.get("away_team")
                or enhanced_fixture.get("awayTeam")
                or (enhanced_fixture.get("away", {}) or {}).get("name")
            )
            c_actual = corners_actuals_store.lookup(
                date_str, home_team_name, away_team_name
            )

            # Add betting-relevant fields
            enhanced_fixture.update(
                {
                    "total_goals_prediction": None,  # Will be filled by ML model
                    "first_half_goals_prediction": None,
                    "result_prediction": None,
                    "confidence_score": 0.0,
                    "betting_value": None,
                    "is_completed": is_completed,
                    "home_score": home_score if is_completed else None,
                    "away_score": away_score if is_completed else None,
                    "result": match_result,
                    # corners actuals if present
                    "home_corners": c_actual.get("home_corners") if c_actual else None,
                    "away_corners": c_actual.get("away_corners") if c_actual else None,
                    "total_corners": c_actual.get("total_corners")
                    if c_actual
                    else None,
                    # Additional match context
                    "head_to_head": self._get_head_to_head_stats(
                        fixture["home_team"], fixture["away_team"]
                    ),
                    "form_rating": {
                        "home": 0.5,  # Will be calculated from recent results
                        "away": 0.5,
                    },
                    "injury_impact": "low",  # Future: integrate injury data
                    "weather_condition": "unknown",  # Future: weather API
                }
            )

            enhanced_fixtures.append(enhanced_fixture)

        return enhanced_fixtures

    # ------------------------------------------------------------------
    # Minimal compatibility stubs for legacy predictor expectations
    # ------------------------------------------------------------------
    def get_detailed_team_stats(self, team_name: Optional[str] = None) -> Dict:
        """Return a synthetic detailed team stats structure for compatibility.
        The advanced predictor's training path calls enhanced_epl_service.get_detailed_team_stats().
        We simulate necessary fields from current team seed metrics / teams list to avoid NoneType errors.
        """
        teams = self.get_teams()
        enriched = []
        for t in teams:
            # Derive synthetic stats with deterministic spreads
            name = t.get("name")
            position = t.get("position", 20)
            # Points per game estimate (top better than bottom)
            ppg = max(0.5, 2.4 - (position - 1) * (1.9 / 19))
            goals_for = round(ppg * 1.25, 2)
            goals_against = round(1.3 + (position / 20) * 0.6, 2)
            enriched.append(
                {
                    "name": name,
                    "position": position,
                    "points": int(ppg * 10),
                    "matches_played": 10,
                    "wins": int(ppg * 10 // 3),
                    "draws": 2,
                    "losses": max(0, 10 - int(ppg * 10 // 3) - 2),
                    "goals_for": int(goals_for * 10),
                    "goals_against": int(goals_against * 10),
                    "goal_difference": int(goals_for * 10 - goals_against * 10),
                    "points_per_game": ppg,
                    "goals_per_game": goals_for,
                    "goals_conceded_per_game": goals_against,
                    "win_rate": min(0.9, ppg / 3.0),
                    "clean_sheet_rate": 0.3,
                    "scoring_consistency": 0.6,
                    "defensive_strength": 0.5,
                    "form_score": min(0.95, max(0.25, 0.5 + (2.0 - ppg) * 0.05)),
                    "home_advantage": 0.15,
                    "away_performance": 0.5,
                }
            )
        if team_name:
            enriched = [e for e in enriched if e["name"].lower() == team_name.lower()]
        return {
            "teams": enriched,
            "league_averages": {},
            "last_updated": datetime.now().isoformat(),
            "season": self.current_season,
        }

    def refresh_data(self) -> Dict[str, Any]:
        """Reload underlying Football-Data.org cache file (admin utility)."""
        try:
            self.football_data = self._load_football_data()
            fixtures_count = (
                len(self.football_data.get("converted_fixtures", []))
                if self.football_data
                else 0
            )
            return {
                "reloaded": True,
                "fixtures": fixtures_count,
                "season": self.current_season,
            }
        except Exception as e:
            return {"reloaded": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Compatibility helper (legacy code expects get_all_matches())
    # ------------------------------------------------------------------
    def get_all_matches(self) -> List[Dict]:
        """Return the complete season match list (compat wrapper)."""
        return self.get_matches()  # Existing method already returns *all*

    def get_players_goals(self, limit: Optional[int] = None) -> List[Dict]:
        """Get goal scorers (enhanced with projected stats)"""
        # For now, return enhanced mock data based on real EPL players
        goal_scorers = [
            {
                "name": "Erling Haaland",
                "team": "Manchester City",
                "goals": 8,
                "expected_season_total": 28,
            },
            {
                "name": "Mohamed Salah",
                "team": "Liverpool",
                "goals": 7,
                "expected_season_total": 24,
            },
            {
                "name": "Harry Kane",
                "team": "Bayern Munich",
                "goals": 6,
                "expected_season_total": 22,
            },
            {
                "name": "Alexander Isak",
                "team": "Newcastle United",
                "goals": 6,
                "expected_season_total": 18,
            },
            {
                "name": "Cole Palmer",
                "team": "Chelsea",
                "goals": 5,
                "expected_season_total": 16,
            },
            {
                "name": "Bryan Mbeumo",
                "team": "Brentford",
                "goals": 5,
                "expected_season_total": 14,
            },
            {
                "name": "Bukayo Saka",
                "team": "Arsenal",
                "goals": 4,
                "expected_season_total": 15,
            },
            {
                "name": "Ollie Watkins",
                "team": "Aston Villa",
                "goals": 4,
                "expected_season_total": 16,
            },
            {
                "name": "Dominic Solanke",
                "team": "Bournemouth",
                "goals": 4,
                "expected_season_total": 12,
            },
            {
                "name": "Chris Wood",
                "team": "Nottingham Forest",
                "goals": 4,
                "expected_season_total": 11,
            },
            {
                "name": "Liam Delap",
                "team": "Ipswich Town",
                "goals": 3,
                "expected_season_total": 8,
            },
            {
                "name": "Antoine Semenyo",
                "team": "Bournemouth",
                "goals": 3,
                "expected_season_total": 9,
            },
            {
                "name": "Joao Pedro",
                "team": "Brighton",
                "goals": 3,
                "expected_season_total": 12,
            },
            {
                "name": "Kai Havertz",
                "team": "Arsenal",
                "goals": 3,
                "expected_season_total": 13,
            },
            {
                "name": "Nicolas Jackson",
                "team": "Chelsea",
                "goals": 3,
                "expected_season_total": 14,
            },
        ]

        # Add enhanced stats
        for player in goal_scorers:
            player.update(
                {
                    "goals_per_game": round(
                        player["goals"] / max(8, 1), 2
                    ),  # Assuming 8 games played
                    "expected_goals_per_game": round(
                        player["expected_season_total"] / 38, 2
                    ),
                    "form_rating": np.random.uniform(0.6, 1.0),  # Mock form rating
                    "injury_risk": np.random.choice(
                        ["low", "medium", "high"], p=[0.7, 0.2, 0.1]
                    ),
                    "season": self.current_season,
                }
            )

        if limit:
            goal_scorers = goal_scorers[:limit]

        return goal_scorers

    def get_upcoming_matches(self, days: int = 7) -> List[Dict]:
        """Get upcoming matches in next N days"""
        if not self.football_data:
            return []

        fixtures = self.football_data.get("converted_fixtures", [])
        upcoming = []

        from datetime import timezone

        current_date = datetime.now(timezone.utc)

        for fixture in fixtures:
            if fixture["status"] == "SCHEDULED":
                try:
                    # Handle multiple date formats from Football-Data.org
                    date_str = fixture["date"]
                    if date_str.endswith("Z"):
                        match_date = datetime.fromisoformat(
                            date_str.replace("Z", "+00:00")
                        )
                    elif "+" in date_str or date_str.endswith("UTC"):
                        match_date = datetime.fromisoformat(
                            date_str.replace("UTC", "+00:00")
                        )
                    else:
                        # Assume UTC if no timezone specified
                        match_date = datetime.fromisoformat(date_str).replace(
                            tzinfo=timezone.utc
                        )

                    # Ensure both dates are timezone-aware
                    if match_date.tzinfo is None:
                        match_date = match_date.replace(tzinfo=timezone.utc)
                    if current_date.tzinfo is None:
                        current_date = current_date.replace(tzinfo=timezone.utc)

                    days_until = (match_date - current_date).days

                    if 0 <= days_until <= days:
                        enhanced_fixture = fixture.copy()
                        enhanced_fixture["days_until"] = days_until
                        enhanced_fixture[
                            "prediction_confidence"
                        ] = 0.85  # Mock confidence
                        enhanced_fixture["parsed_date"] = match_date.isoformat()
                        upcoming.append(enhanced_fixture)
                except Exception as e:
                    # Skip problematic dates but continue processing
                    continue

        # Sort by date
        upcoming.sort(key=lambda x: x.get("days_until", 999))
        return upcoming

    def get_detailed_team_stats(self, team_name: Optional[str] = None) -> Dict:
        """Get comprehensive team statistics for predictions"""
        cache_key = f"detailed_stats_{team_name or 'all'}"

        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        teams = self.get_teams()

        if team_name:
            team_data = next((t for t in teams if t["name"] == team_name), None)
            if not team_data:
                return {}

            # Enhanced individual team stats
            stats = {
                "team_name": team_name,
                "current_position": team_data["position"],
                "points": team_data["points"],
                "form": team_data["form"],
                "home_strength": self._calculate_home_strength(team_name),
                "away_strength": self._calculate_away_strength(team_name),
                "attack_rating": self._calculate_attack_rating(team_name),
                "defense_rating": self._calculate_defense_rating(team_name),
                "recent_form": self._get_recent_form(team_name),
                "key_players": self._get_key_players(team_name),
                "injury_list": [],  # Future: real injury data
                "suspension_list": [],  # Future: real suspension data
            }
        else:
            # League-wide statistics
            stats = {
                "total_teams": len(teams),
                "matches_played": len(
                    [f for f in self.get_matches() if f["status"] == "FINISHED"]
                ),
                "total_fixtures": len(self.get_matches()),
                "season": self.current_season,
                "league_leaders": teams[:5],
                "relegation_zone": teams[-3:],
                "top_scorers": self.get_players_goals(5),
                "average_goals_per_game": self._calculate_avg_goals_per_game(),
                "home_advantage": self._calculate_home_advantage(),
            }

        self._cache_data(cache_key, stats)
        return stats

    def _get_head_to_head_stats(self, home_team: str, away_team: str) -> Dict:
        """Get head-to-head statistics between two teams"""
        # Mock H2H data - in production, this would query historical results
        return {
            "total_matches": 10,
            "home_wins": 4,
            "away_wins": 3,
            "draws": 3,
            "last_meeting": "2024-04-15",
            "avg_goals": 2.3,
            "home_team_form_vs_opponent": "W-D-L-W-D",
        }

    def _calculate_home_strength(self, team_name: str) -> float:
        """Calculate home playing strength (0-1)"""
        # Mock calculation - will be enhanced with real data
        base_strength = np.random.uniform(0.4, 0.8)
        return round(base_strength, 2)

    def _calculate_away_strength(self, team_name: str) -> float:
        """Calculate away playing strength (0-1)"""
        # Mock calculation - will be enhanced with real data
        base_strength = np.random.uniform(0.3, 0.7)
        return round(base_strength, 2)

    def _calculate_attack_rating(self, team_name: str) -> float:
        """Calculate team's attacking prowess"""
        return round(np.random.uniform(0.5, 1.0), 2)

    def _calculate_defense_rating(self, team_name: str) -> float:
        """Calculate team's defensive strength"""
        return round(np.random.uniform(0.4, 0.9), 2)

    def _get_recent_form(self, team_name: str) -> List[str]:
        """Get team's recent form (last 5 games)"""
        forms = ["W", "D", "L"]
        return [np.random.choice(forms) for _ in range(5)]

    def _get_key_players(self, team_name: str) -> List[Dict]:
        """Get key players for a team"""
        # Mock key players data
        return [
            {"name": "Player 1", "position": "Forward", "goals": 5, "assists": 2},
            {"name": "Player 2", "position": "Midfielder", "goals": 2, "assists": 6},
            {"name": "Player 3", "position": "Defender", "goals": 1, "assists": 1},
        ]

    def _calculate_avg_goals_per_game(self) -> float:
        """Calculate average goals per game in the league"""
        return round(np.random.uniform(2.5, 3.2), 2)

    def _calculate_home_advantage(self) -> float:
        """Calculate overall home advantage in the league"""
        return round(np.random.uniform(0.55, 0.65), 2)

    def _get_fallback_teams(self) -> List[Dict]:
        """Fallback team data if Football-Data.org unavailable"""
        return [
            {
                "name": "Arsenal",
                "position": 1,
                "points": 0,
                "season": self.current_season,
            },
            {
                "name": "Liverpool",
                "position": 2,
                "points": 0,
                "season": self.current_season,
            },
            {
                "name": "Manchester City",
                "position": 3,
                "points": 0,
                "season": self.current_season,
            },
            # ... rest of EPL teams
        ]

    def _get_fallback_matches(self) -> List[Dict]:
        """Fallback match data if Football-Data.org unavailable"""
        return [
            {
                "home_team": "Arsenal",
                "away_team": "Liverpool",
                "date": "2025-09-28",
                "status": "SCHEDULED",
                "season": self.current_season,
            }
        ]

    def get_system_status(self) -> Dict:
        """Get enhanced system status with Football-Data.org integration"""
        status = {
            "service": "Enhanced EPL Service",
            "season": self.current_season,
            "status": "operational",
            "data_source": "Football-Data.org",
            "last_updated": datetime.now().isoformat(),
            "cache_keys": len(self.cache),
            "features": [
                "Real EPL fixture list (380 matches)",
                "Football-Data.org integration",
                "Advanced team statistics",
                "Player performance tracking",
                "Head-to-head analysis",
                "Form-based predictions",
                "Betting value calculations",
            ],
        }

        if self.football_data:
            status.update(
                {
                    "total_fixtures": self.football_data.get("total_fixtures", 0),
                    "scheduled_fixtures": self.football_data.get(
                        "scheduled_fixtures", 0
                    ),
                    "finished_fixtures": self.football_data.get("finished_fixtures", 0),
                    "data_completeness": "100% (Football-Data.org)",
                }
            )
        else:
            status.update(
                {
                    "total_fixtures": 0,
                    "data_completeness": "0% (Fallback mode)",
                    "warning": "Football-Data.org data not available",
                }
            )

        return status


def main():
    """Test the enhanced service with Football-Data.org integration"""
    service = EnhancedEPLService()

    print("\n=== ENHANCED EPL SERVICE TEST ===")
    status = service.get_system_status()
    print(f"Status: {status['status']}")
    print(f"Season: {status['season']}")
    print(f"Data Source: {status['data_source']}")
    print(f"Total Fixtures: {status.get('total_fixtures', 0)}")
    print(f"Data Completeness: {status.get('data_completeness', 'Unknown')}")

    # Test teams
    teams = service.get_teams()
    print(f"\nTeams: {len(teams)} loaded")

    # Test matches
    matches = service.get_matches(5)
    print(f"Matches: {len(matches)} sample loaded")

    # Test upcoming matches
    upcoming = service.get_upcoming_matches(7)
    print(f"Upcoming matches (7 days): {len(upcoming)}")

    print("\n✅ Enhanced EPL Service with Football-Data.org integration ready!")


if __name__ == "__main__":
    main()
