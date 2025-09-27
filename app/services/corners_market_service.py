"""Corners Markets Loader and Lookup Service.

This service loads real corners market lines (Over/Under) for:
- Total corners (match)
- Team corners (home/away)

Supported input files (any that exist will be loaded):
- data/corners_market_lines_2025_26.csv
- data/corners_market_lines_2025_26.json
- data/corners_market_lines_manual.csv
- data/corners_market_lines_manual.json

CSV schema for total corners:
date,home_team,away_team,line,over_odds,under_odds,bookmaker,market_type
2025-08-16,Tottenham,Brentford,9.5,1.90,1.90,Bet365,total

CSV schema for team corners:
date,home_team,away_team,side,line,over_odds,under_odds,bookmaker,market_type
2025-08-16,Tottenham,Brentford,home,4.5,1.83,1.95,Bet365,team

Notes:
- Odds are decimal odds. We convert to implied probabilities and normalize to 1 across Over/Under (two-way market).
- Team names are normalized via team_name_normalizer.
- Lines may vary (e.g., 8.5, 9.5, 10.5). Lookup will prefer exact line; if not found, will fall back to closest line for the same fixture/market.
- Lines beginning with '#' in CSV are treated as comments and skipped.
"""
from __future__ import annotations
import csv
import json
import math
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
    """Return (over_p, under_p, overround) normalized from decimal odds."""
    inv_o = 1.0 / over_odds if over_odds and over_odds > 1.01 else 0.0
    inv_u = 1.0 / under_odds if under_odds and under_odds > 1.01 else 0.0
    s = inv_o + inv_u
    if s <= 0:
        return 0.0, 0.0, 0.0
    return inv_o / s, inv_u / s, (s - 1.0)


@dataclass
class CornersMarketRecord:
    date: str
    home: str
    away: str
    market_type: str  # 'total' or 'team'
    side: Optional[str]  # 'home' or 'away' for team market
    line: float
    over_odds: float
    under_odds: float
    bookmaker: Optional[str]

    @property
    def key_base(self) -> str:
        return f"{self.date}|{self.home.lower()}|{self.away.lower()}|{self.market_type}"


class CornersMarketStore:
    def __init__(self) -> None:
        # For totals: index by (date|home|away|total) -> {line: record}
        # For team:   index by (date|home|away|team|side) -> {line: record}
        self._totals: Dict[str, Dict[float, CornersMarketRecord]] = {}
        self._team: Dict[str, Dict[float, CornersMarketRecord]] = {}

    def _add(self, rec: CornersMarketRecord) -> None:
        if rec.market_type == 'total':
            key = rec.key_base
            self._totals.setdefault(key, {})[rec.line] = rec
        elif rec.market_type == 'team' and rec.side in ('home', 'away'):
            key = f"{rec.key_base}|{rec.side}"
            self._team.setdefault(key, {})[rec.line] = rec

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
            mkt = (row.get('market_type') or 'total').strip().lower()
            side = (row.get('side') or '').strip().lower() or None
            line = float(row.get('line'))
            over_odds = float(row.get('over_odds'))
            under_odds = float(row.get('under_odds'))
            bookmaker = (row.get('bookmaker') or row.get('bm') or '').strip() or None
            if not (date and home and away and line and over_odds and under_odds):
                return
            rec = CornersMarketRecord(
                date=date, home=home, away=away, market_type='team' if mkt.startswith('team') else 'total',
                side=side if side in ('home','away') else None,
                line=line, over_odds=over_odds, under_odds=under_odds, bookmaker=bookmaker,
            )
            self._add(rec)
        except Exception:
            return

    def _closest_line(self, mapping: Dict[float, CornersMarketRecord], target_line: float) -> Optional[CornersMarketRecord]:
        if not mapping:
            return None
        if target_line in mapping:
            return mapping[target_line]
        # fallback to closest by absolute difference
        best_line = min(mapping.keys(), key=lambda x: abs(x - target_line))
        return mapping.get(best_line)

    def lookup_total(self, date: str, home: str, away: str, line: float) -> Optional[Tuple[float, Optional[str], float]]:
        """Return (market_over_prob, bookmaker, used_line) for total corners, if available."""
        d = _only_date(date) or ''
        h = normalize_team_name(home) or home
        a = normalize_team_name(away) or away
        key = f"{d}|{h.lower()}|{a.lower()}|total"
        recs = self._totals.get(key)
        if not recs:
            return None
        r = self._closest_line(recs, float(line))
        if not r:
            return None
        over_p, under_p, _ = _two_way_probs(r.over_odds, r.under_odds)
        return over_p, r.bookmaker, r.line

    def lookup_team(self, date: str, home: str, away: str, side: str, line: float) -> Optional[Tuple[float, Optional[str], float]]:
        """Return (market_over_prob, bookmaker, used_line) for team corners, if available."""
        s = side.lower()
        if s not in ('home','away'):
            return None
        d = _only_date(date) or ''
        h = normalize_team_name(home) or home
        a = normalize_team_name(away) or away
        key = f"{d}|{h.lower()}|{a.lower()}|team|{s}"
        recs = self._team.get(key)
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
        os.path.join(data_dir, 'corners_market_lines_2025_26.csv'),
        os.path.join(data_dir, 'corners_market_lines_2025_26.json'),
        os.path.join(data_dir, 'corners_market_lines_manual.csv'),
        os.path.join(data_dir, 'corners_market_lines_manual.json'),
    ]


def load_default() -> CornersMarketStore:
    store = CornersMarketStore()
    store.load_paths(_default_candidate_paths())
    return store


# Module-level singleton
corners_market_store: CornersMarketStore = load_default()

def reload_market_store() -> Tuple[int, int]:
    """Reload default market files. Returns (#totals_indexed, #team_indexed)."""
    global corners_market_store
    corners_market_store = load_default()
    return (sum(len(v) for v in corners_market_store._totals.values()),
            sum(len(v) for v in corners_market_store._team.values()))
