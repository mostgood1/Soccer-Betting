"""
Service wrapper for FBR API (https://fbrapi.com/documentation)

Currently implements a minimal schedule backfill per league-season to
persist schedule.csv into data/fbref/{LEAGUE}/{SEASON}/ so existing
consumers can read it immediately.

Notes:
- API key is read from env FBR_API_KEY or cached in data/fbrapi/api_key.txt.
- If no key is found, we attempt to generate one via POST /generate_api_key.
- Rate limiting: FBR API suggests 1 request every 3 seconds. Our schedule
  call is a single request; keep sleeps for future expansion.
"""
from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import requests
import pandas as pd  # type: ignore


API_BASE = os.getenv("FBR_API_BASE", "https://fbrapi.com")
API_KEY_ENV = "FBR_API_KEY"
API_KEY_FILE = Path("data/fbrapi/api_key.txt")


def _season_string_for_today() -> str:
    d = datetime.utcnow()
    start_year = d.year if d.month >= 8 else d.year - 1
    return f"{start_year}-{start_year+1}"


@dataclass
class BackfillResult:
    league: str
    season: str
    schedule_rows: int = 0
    artifacts: List[str] = None
    notes: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "league": self.league,
            "season": self.season,
            "schedule_rows": self.schedule_rows,
            "artifacts": self.artifacts or [],
            "notes": self.notes or [],
        }


class FBRApiService:
    def __init__(self) -> None:
        self.base_dir = Path("data/fbref")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # --- API key helpers ---
    def _get_api_key(self) -> Optional[str]:
        key = os.getenv(API_KEY_ENV)
        if key:
            return key
        try:
            if API_KEY_FILE.exists():
                return API_KEY_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            pass
        # Best-effort generate a key
        try:
            r = requests.post(f"{API_BASE}/generate_api_key", timeout=30)
            r.raise_for_status()
            key = r.json().get("api_key")
            if key:
                API_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
                API_KEY_FILE.write_text(key, encoding="utf-8")
                return key
        except Exception:
            return None
        return None

    # --- League mapping helpers ---
    def _country_code_for_league(self, code: str) -> str:
        # Club domestic leagues country (Football Reference style three-letter codes)
        return {
            "PL": "ENG",  # England
            "BL1": "GER",  # Germany
            "FL1": "FRA",  # France
            "SA": "ITA",   # Italy
            "PD": "ESP",   # Spain
        }.get(code, "ENG")

    def _league_name_targets(self, code: str) -> List[str]:
        return {
            "PL": ["Premier League"],
            "BL1": ["Bundesliga"],
            "FL1": ["Ligue 1"],
            "SA": ["Serie A"],
            "PD": ["La Liga", "Primera División"],
        }.get(code, ["Premier League"])

    def _discover_league_id(self, code: str, api_key: str) -> Optional[int]:
        # Prefer hard-coded known id for PL to avoid extra call
        if code == "PL":
            return 9
        country = self._country_code_for_league(code)
        targets = [t.lower() for t in self._league_name_targets(code)]
        # First attempt: country-scoped
        try:
            r = requests.get(
                f"{API_BASE}/leagues",
                params={"country_code": country},
                headers={"X-API-Key": api_key},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json().get("data") or []
            for grp in data:
                for lg in grp.get("leagues", []) or []:
                    name = (lg.get("competition_name") or "").lower()
                    if any(t in name for t in targets):
                        lid = lg.get("league_id")
                        if isinstance(lid, int):
                            return lid
        except Exception:
            pass
        # Fallback: global search without country filter
        try:
            r = requests.get(
                f"{API_BASE}/leagues",
                headers={"X-API-Key": api_key},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json().get("data") or []
            for grp in data:
                for lg in grp.get("leagues", []) or []:
                    name = (lg.get("competition_name") or "").lower()
                    if any(t in name for t in targets):
                        lid = lg.get("league_id")
                        if isinstance(lid, int):
                            return lid
        except Exception:
            return None
        return None

    def _ensure_out_dir(self, league: str, season: str) -> Path:
        out = self.base_dir / league.upper() / season
        out.mkdir(parents=True, exist_ok=True)
        return out

    # --- Public: backfill schedule ---
    def backfill_schedule(self, league: str, season: Optional[str] = None) -> BackfillResult:
        season_str = season or _season_string_for_today()
        res = BackfillResult(league=league.upper(), season=season_str, artifacts=[], notes=[])

        api_key = self._get_api_key()
        if not api_key:
            res.notes.append("FBR API key unavailable; cannot fetch schedule")
            return res

        lid = self._discover_league_id(res.league, api_key)
        if not lid:
            res.notes.append(f"Failed to resolve league id for {res.league}")
            return res

        try:
            # Rate-limit friendly: single call
            r = requests.get(
                f"{API_BASE}/matches",
                params={"league_id": lid, "season_id": season_str},
                headers={"X-API-Key": api_key},
                timeout=60,
            )
            r.raise_for_status()
            payload = r.json() or {}
            records = payload.get("data") or []
            rows: List[Dict[str, Any]] = []
            for m in records:
                # League match data format
                date = m.get("date") or m.get("utcDate") or m.get("commence_time")
                time_str = m.get("time") or ""
                home = m.get("home") or m.get("home_team") or (m.get("homeTeam") or {}).get("name")
                away = m.get("away") or m.get("away_team") or (m.get("awayTeam") or {}).get("name")
                if not (date and home and away):
                    continue
                # Compose iso datetime when both date and time present
                dt_iso = date
                if isinstance(date, str) and time_str and "T" not in date:
                    try:
                        dt_iso = f"{date}T{time_str}:00Z"
                    except Exception:
                        dt_iso = date
                rows.append(
                    {
                        "date": dt_iso,
                        "home_team": home,
                        "away_team": away,
                        "league": res.league,
                        "season": season_str,
                    }
                )
            if not rows:
                res.notes.append("no schedule rows from FBR API")
                return res
            df = pd.DataFrame(rows)
            out_dir = self._ensure_out_dir(res.league, season_str)
            out_path = out_dir / "schedule.csv"
            df.to_csv(out_path, index=False)
            res.schedule_rows = int(df.shape[0])
            res.artifacts.append(str(out_path))
        except Exception as e:
            res.notes.append(f"schedule fetch failed: {e}")
        return res


fbrapi_service = FBRApiService()
