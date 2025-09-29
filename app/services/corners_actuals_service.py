"""Corners actuals loader and lookup service.

Loads real match corners counts (home, away, total) from CSV/JSON files under
the project's data/ directory and provides a normalized lookup by date and teams.

Expected columns/keys:
- date: YYYY-MM-DD or ISO string with time (we'll extract the date part)
- home_team, away_team: team display names (we normalize internally)
- home_corners, away_corners (ints)
- total_corners (optional; computed as home+away if missing)

Usage:
    from .corners_actuals_service import corners_actuals_store
    rec = corners_actuals_store.lookup(date_str, home_name, away_name)
"""
from __future__ import annotations
import os
import csv
import json
from typing import Dict, Any, Iterable, Optional

from .team_name_normalizer import normalize_team_name


def _repo_root() -> str:
    # This file is at app/services; repo root is two levels up
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _only_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        # Common shapes: 'YYYY-MM-DD', 'YYYY-MM-DDTHH:MM:SSZ', ISO with TZ
        if "T" in s:
            return s.split("T", 1)[0]
        if " " in s:
            return s.split(" ", 1)[0]
        # Already a date
        return s[:10]
    except Exception:
        return None


class CornersActualsStore:
    def __init__(self) -> None:
        # key: f"{date}|{home}|{away}" (all lowercase, normalized team names)
        self._index: Dict[str, Dict[str, Any]] = {}

    def _key(self, date_str: str, home: str, away: str) -> str:
        d = _only_date(date_str) or ""
        h = normalize_team_name(home) or str(home)
        a = normalize_team_name(away) or str(away)
        return f"{d}|{h.lower()}|{a.lower()}"

    def add_record(self, rec: Dict[str, Any]) -> None:
        date_str = rec.get("date") or rec.get("match_date") or rec.get("utc_date")
        home = rec.get("home_team") or rec.get("home")
        away = rec.get("away_team") or rec.get("away")
        if not (date_str and home and away):
            return
        try:
            hc = rec.get("home_corners")
            ac = rec.get("away_corners")
            tc = rec.get("total_corners")

            # coerce to int when possible
            def _to_int(v):
                try:
                    return int(v)
                except Exception:
                    try:
                        return int(float(v))
                    except Exception:
                        return None

            hc_i = _to_int(hc)
            ac_i = _to_int(ac)
            tc_i = _to_int(tc)
            if tc_i is None and None not in (hc_i, ac_i):
                tc_i = hc_i + ac_i
            if hc_i is None and tc_i is not None and ac_i is not None:
                hc_i = tc_i - ac_i
            if ac_i is None and tc_i is not None and hc_i is not None:
                ac_i = tc_i - hc_i
            if None in (hc_i, ac_i, tc_i):
                # insufficient data to add
                return
            key = self._key(date_str, home, away)
            self._index[key] = {
                "date": _only_date(date_str),
                "home_team": normalize_team_name(home) or home,
                "away_team": normalize_team_name(away) or away,
                "home_corners": hc_i,
                "away_corners": ac_i,
                "total_corners": tc_i,
            }
        except Exception:
            return

    def load_from_json(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # allow either list or dict with key 'records'
                data = data.get("records") or data.get("matches") or []
            if isinstance(data, list):
                for rec in data:
                    if isinstance(rec, dict):
                        self.add_record(rec)
        except Exception:
            pass

    def load_from_csv(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.add_record(row)
        except Exception:
            pass

    def load_paths(self, paths: Iterable[str]) -> None:
        for p in paths:
            if not p:
                continue
            if not os.path.exists(p):
                continue
            ext = os.path.splitext(p)[1].lower()
            if ext == ".json":
                self.load_from_json(p)
            elif ext == ".csv":
                self.load_from_csv(p)

    def lookup(
        self, date_str: Optional[str], home: Optional[str], away: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if not (date_str and home and away):
            return None
        key = self._key(date_str, home, away)
        return self._index.get(key)

    def size(self) -> int:
        return len(self._index)


def _default_candidate_paths() -> Iterable[str]:
    root = _repo_root()
    data_dir = os.path.join(root, "data")
    # Most specific to most general; first existing will contribute
    candidates = [
        os.path.join(data_dir, "corners_actuals_2025_26.json"),
        os.path.join(data_dir, "corners_actuals_2025_26.csv"),
        os.path.join(data_dir, "corners_actuals_manual.json"),
        os.path.join(data_dir, "corners_actuals_manual.csv"),
    ]
    return candidates


def load_default() -> CornersActualsStore:
    store = CornersActualsStore()
    store.load_paths(_default_candidate_paths())
    return store


# Module-level singleton for convenience
corners_actuals_store: CornersActualsStore = load_default()


def reload_default_store() -> int:
    """Reload default data paths into the singleton store; returns record count."""
    global corners_actuals_store
    corners_actuals_store = load_default()
    return corners_actuals_store.size()
