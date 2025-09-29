"""
Generic Football-Data.org season cacher for multiple leagues

Supports: PL (Premier League), BL1 (Bundesliga), FL1 (Ligue 1), SA (Serie A), PD (La Liga)
Writes cache files that EnhancedFDService expects: data/football_data_{CODE}_2025_2026.json

Environment:
- FOOTBALL_DATA_API_KEY (required)
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import os
import time
import json

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore


SEASON_YEAR = "2025"  # 2025-26 season on football-data.org APIs


class FootballDataMultiService:
    def __init__(self) -> None:
        self.api_key = os.getenv("FOOTBALL_DATA_API_KEY") or os.getenv(
            "FOOTBALL_DATA_API_TOKEN"
        )
        if not self.api_key:
            raise ValueError("FOOTBALL_DATA_API_KEY not set")
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {
            "X-Auth-Token": self.api_key,
            "Content-Type": "application/json",
        }
        # 10 requests/minute -> 6s gap
        self.rate_limit_delay = 6.0
        self._last_request = 0.0

    def _sleep_rate_limit(self):
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)

    def _get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if requests is None:
            return {"error": "requests not available"}
        self._sleep_rate_limit()
        url = f"{self.base_url}/{endpoint}"
        try:
            r = requests.get(url, headers=self.headers, params=params or {}, timeout=20)
            self._last_request = time.time()
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                # back off then retry once
                time.sleep(60)
                return self._get(endpoint, params)
            return {"error": f"status {r.status_code}", "body": r.text[:400]}
        except Exception as e:  # pragma: no cover
            return {"error": str(e)}

    def _convert_fixtures(
        self, fixtures: List[Dict[str, Any]], competition_name: str
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for fx in fixtures or []:
            home = (fx.get("homeTeam") or {}).get("name")
            away = (fx.get("awayTeam") or {}).get("name")
            score = fx.get("score") or {}
            ft = score.get("fullTime") or {}
            out.append(
                {
                    "id": fx.get("id"),
                    "home_team": home,
                    "away_team": away,
                    "date": fx.get("utcDate"),
                    "utc_date": fx.get("utcDate"),
                    "status": fx.get("status"),
                    "matchday": fx.get("matchday"),
                    "season": f"{SEASON_YEAR}-{int(SEASON_YEAR)+1}",
                    "home_score": ft.get("home"),
                    "away_score": ft.get("away"),
                    "venue": fx.get("venue") or "Unknown",
                    "competition": competition_name,
                }
            )
        return out

    def cache_league(self, code: str) -> Dict[str, Any]:
        code = (code or "PL").upper()
        # 1. Competition info
        comp = self._get(f"competitions/{code}")
        if comp.get("error"):
            return {
                "success": False,
                "error": comp.get("error"),
                "body": comp.get("body"),
            }
        comp_name = comp.get("name") or code
        # 2. Teams
        teams = self._get(
            f"competitions/{code}/teams", params={"season": SEASON_YEAR}
        ).get("teams", [])
        # 3. Matches (fixtures)
        matches = self._get(
            f"competitions/{code}/matches", params={"season": SEASON_YEAR}
        ).get("matches", [])
        converted = self._convert_fixtures(matches, competition_name=comp_name)
        payload = {
            "competition": comp,
            "teams": teams,
            "fixtures": matches,
            "converted_fixtures": converted,
            "season": f"{SEASON_YEAR}-{int(SEASON_YEAR)+1}",
            "total_fixtures": len(matches),
        }
        # Write file at expected path
        base = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data")
        )
        os.makedirs(base, exist_ok=True)
        fname = f"football_data_{code}_2025_2026.json"
        path = os.path.join(base, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return {
            "success": True,
            "file": path,
            "fixtures": len(matches),
            "teams": len(teams),
        }


# module-level helper
football_data_multi = FootballDataMultiService
