"""
Generic Football-Data.org backed league service (non-EPL variants)
Supports: Bundesliga (BL1), Ligue 1 (FL1), Serie A (SA), La Liga (PD)
Reads cached JSON files under data/football_data_{code}_2025_2026.json
Implements a subset of EnhancedEPLService_v2 API for drop-in use.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import os
import logging

from .corners_actuals_service import corners_actuals_store

logger = logging.getLogger(__name__)


class EnhancedFDService:
    def __init__(self, competition_code: str, season_label: str = "2025-26"):
        self.code = competition_code  # e.g., 'BL1', 'FL1', 'SA'
        self.current_season = season_label
        self.football_data: Dict[str, Any] = self._load_football_data()
        self._team_venue_index: Dict[str, str] = {}
        try:
            for t in (self.football_data or {}).get('teams', []) or []:
                name = t.get('name') or t.get('shortName')
                venue = t.get('venue') or ''
                if name:
                    self._team_venue_index[str(name)] = venue or ''
        except Exception:
            self._team_venue_index = {}

    def _data_path(self) -> str:
        base = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        fname = f"football_data_{self.code}_2025_2026.json"
        return os.path.normpath(os.path.join(base, fname))

    def _load_football_data(self) -> Dict[str, Any]:
        p = self._data_path()
        baked = f"/app/baked/football_data_{self.code}_2025_2026.json"
        try:
            path = p if os.path.exists(p) else baked
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load Football-Data cache for {self.code}: {e}")
        return {}

    def refresh_data(self) -> Dict[str, Any]:
        try:
            self.football_data = self._load_football_data()
            return {"reloaded": True, "fixtures": len((self.football_data or {}).get('converted_fixtures', []))}
        except Exception as e:
            return {"reloaded": False, "error": str(e)}

    def get_teams(self) -> List[Dict[str, Any]]:
        teams = (self.football_data or {}).get('teams', []) or []
        # Minimal stable shape
        out: List[Dict[str, Any]] = []
        for i, t in enumerate(teams):
            out.append({
                'id': t.get('id'),
                'name': t.get('name'),
                'short_name': t.get('shortName', t.get('name', '')[:3]),
                'tla': t.get('tla', t.get('name', '')[:3]),
                'venue': t.get('venue', 'Unknown'),
                'crest': t.get('crest', ''),
                'position': i + 1,
                'season': self.current_season,
            })
        return out

    def get_matches(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        fixtures = (self.football_data or {}).get('converted_fixtures', []) or []
        if limit:
            fixtures = fixtures[:limit]
        enhanced: List[Dict[str, Any]] = []
        for f in fixtures:
            m = dict(f)
            # Venue backfill via team index
            try:
                if not m.get('venue'):
                    home_name = m.get('home_team') or m.get('homeTeam') or (m.get('home') or {}).get('name')
                    if home_name and home_name in self._team_venue_index:
                        m['venue'] = self._team_venue_index.get(home_name) or 'Unknown'
            except Exception:
                m.setdefault('venue', 'Unknown')
            # Scores and result flag
            score_obj = m.get('score') or {}
            ft = score_obj.get('fullTime') if isinstance(score_obj, dict) else {}
            home_score = m.get('home_score') or (ft.get('home') if isinstance(ft, dict) else None)
            away_score = m.get('away_score') or (ft.get('away') if isinstance(ft, dict) else None)
            status = str(m.get('status','')).upper()
            is_completed = status in ('FINISHED','COMPLETED') and (home_score is not None and away_score is not None)
            result = None
            if is_completed:
                try:
                    if int(home_score) > int(away_score): result = 'H'
                    elif int(away_score) > int(home_score): result = 'A'
                    else: result = 'D'
                except Exception:
                    result = None
            # Attach corners actuals if any (best-effort)
            try:
                date_str = m.get('utc_date') or m.get('date')
                home = m.get('home_team') or m.get('homeTeam') or (m.get('home') or {}).get('name')
                away = m.get('away_team') or m.get('awayTeam') or (m.get('away') or {}).get('name')
                c = corners_actuals_store.lookup(date_str, home, away)
                if c:
                    if is_completed:
                        m['home_corners'] = c.get('home_corners')
                        m['away_corners'] = c.get('away_corners')
                        m['total_corners'] = c.get('total_corners')
                    else:
                        m.setdefault('home_corners', c.get('home_corners'))
                        m.setdefault('away_corners', c.get('away_corners'))
                        m.setdefault('total_corners', c.get('total_corners'))
            except Exception:
                pass
            m['is_completed'] = is_completed
            m['home_score'] = home_score if is_completed else None
            m['away_score'] = away_score if is_completed else None
            m['result'] = result
            enhanced.append(m)
        return enhanced

    def get_all_matches(self) -> List[Dict[str, Any]]:
        return self.get_matches()

    def get_system_status(self) -> Dict[str, Any]:
        try:
            return {
                'league': self.code,
                'season': self.current_season,
                'total_fixtures': len((self.football_data or {}).get('converted_fixtures', []) or []),
                'teams': len((self.football_data or {}).get('teams', []) or []),
                'last_updated': datetime.utcnow().isoformat() + 'Z'
            }
        except Exception:
            return {'league': self.code, 'season': self.current_season}

__all__ = ["EnhancedFDService"]
