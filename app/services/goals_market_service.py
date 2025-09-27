"""Goals Totals Markets Loader and Lookup Service.

Loads real market lines for match total goals Over/Under and exposes
lookup helpers keyed by (date, home, away, line).

Supported input files (any that exist will be loaded):
- data/goals_market_lines_2025_26.csv
- data/goals_market_lines_2025_26.json
- data/goals_market_lines_manual.csv
- data/goals_market_lines_manual.json

CSV schema:
date,home_team,away_team,line,over_odds,under_odds,bookmaker
2025-08-16,Tottenham,Brentford,2.5,1.90,1.90,Bet365

Notes:
- Odds are decimal odds. We convert to implied probabilities and normalize to 1 across Over/Under.
- Team names are normalized via team_name_normalizer to align with predictor and fixtures.
- Lines may vary (e.g., 2.0, 2.25, 2.5, 2.75, 3.0). Lookup prefers exact line; if not found, uses closest line.
"""
from __future__ import annotations
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Iterable, Tuple, Any

from .team_name_normalizer import normalize_team_name


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _only_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        if 'T' in s:
            return s.split('T', 1)[0]
        if ' ' in s:
            return s.split(' ', 1)[0]
        return s[:10]
    except Exception:
        return None


def _two_way_probs(over_odds: float, under_odds: float) -> Tuple[float, float, float]:
    inv_o = 1.0 / over_odds if over_odds and over_odds > 1.01 else 0.0
    inv_u = 1.0 / under_odds if under_odds and under_odds > 1.01 else 0.0
    s = inv_o + inv_u
    if s <= 0:
        return 0.0, 0.0, 0.0
    return inv_o / s, inv_u / s, (s - 1.0)


@dataclass
class GoalsMarketRecord:
    date: str
    home: str
    away: str
    line: float
    over_odds: float
    under_odds: float
    bookmaker: Optional[str]

    @property
    def key_base(self) -> str:
        return f"{self.date}|{self.home.lower()}|{self.away.lower()}|totals"


class GoalsMarketStore:
    def __init__(self) -> None:
        # index by (date|home|away|totals) -> {line: record}
        self._totals: Dict[str, Dict[float, GoalsMarketRecord]] = {}

    def _add(self, rec: GoalsMarketRecord) -> None:
        key = rec.key_base
        self._totals.setdefault(key, {})[rec.line] = rec

    def load_csv(self, path: str) -> None:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader((row for row in f if not row.lstrip().startswith('#')))
                for row in reader:
                    self._ingest_row(row)
        except Exception:
            pass

    def load_json(self, path: str) -> None:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = data.get('records') or data.get('markets') or []
            if isinstance(data, list):
                for row in data:
                    if isinstance(row, dict):
                        self._ingest_row(row)
        except Exception:
            pass

    def load_paths(self, paths: Iterable[str]) -> None:
        for p in paths:
            if not p or not os.path.exists(p):
                continue
            ext = os.path.splitext(p)[1].lower()
            if ext == '.csv':
                self.load_csv(p)
            elif ext == '.json':
                self.load_json(p)

    def _ingest_row(self, row: Dict[str, Any]) -> None:
        try:
            date = _only_date(str(row.get('date') or row.get('match_date') or row.get('utc_date')))
            home = normalize_team_name(row.get('home_team') or row.get('home'))
            away = normalize_team_name(row.get('away_team') or row.get('away'))
            line = float(row.get('line'))
            over_odds = float(row.get('over_odds'))
            under_odds = float(row.get('under_odds'))
            bookmaker = (row.get('bookmaker') or row.get('bm') or '').strip() or None
            if not (date and home and away and line and over_odds and under_odds):
                return
            rec = GoalsMarketRecord(
                date=date, home=home, away=away, line=line,
                over_odds=over_odds, under_odds=under_odds, bookmaker=bookmaker,
            )
            self._add(rec)
        except Exception:
            return

    def _closest_line(self, mapping: Dict[float, GoalsMarketRecord], target_line: float) -> Optional[GoalsMarketRecord]:
        if not mapping:
            return None
        if target_line in mapping:
            return mapping[target_line]
        best_line = min(mapping.keys(), key=lambda x: abs(x - target_line))
        return mapping.get(best_line)

    def lookup_total(self, date: str, home: str, away: str, line: float) -> Optional[Tuple[float, Optional[str], float]]:
        d = _only_date(date) or ''
        h = normalize_team_name(home) or home
        a = normalize_team_name(away) or away
        key = f"{d}|{h.lower()}|{a.lower()}|totals"
        recs = self._totals.get(key)
        if not recs:
            return None
        r = self._closest_line(recs, float(line))
        if not r:
            return None
        over_p, under_p, _ = _two_way_probs(r.over_odds, r.under_odds)
        return over_p, r.bookmaker, r.line


def _default_candidate_paths() -> Iterable[str]:
    root = _repo_root()
    data_dir = os.path.join(root, 'data')
    return [
        os.path.join(data_dir, 'goals_market_lines_2025_26.csv'),
        os.path.join(data_dir, 'goals_market_lines_2025_26.json'),
        os.path.join(data_dir, 'goals_market_lines_manual.csv'),
        os.path.join(data_dir, 'goals_market_lines_manual.json'),
    ]


def load_default() -> GoalsMarketStore:
    store = GoalsMarketStore()
    store.load_paths(_default_candidate_paths())
    return store


# Module-level singleton
goals_market_store: GoalsMarketStore = load_default()

def reload_goals_market_store() -> int:
    """Reload default goals market files. Returns number of totals entries indexed."""
    global goals_market_store
    goals_market_store = load_default()
    return sum(len(v) for v in goals_market_store._totals.values())
