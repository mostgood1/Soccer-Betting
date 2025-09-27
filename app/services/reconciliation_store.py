"""Persistent storage for match reconciliation & model performance metrics.

Stores per-match reconciliation objects and derives weekly & season-wide
performance aggregates. Uses JSON file in cache for simplicity.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime
import math
try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None


class ReconciliationStore:
    def __init__(self):
        self.path = Path("cache/reconciliation_store.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = {
            'season': '2025-26',
            'reconciliations': {},  # match_key -> reconciliation obj
            'weekly_stats': {},
            'season_stats': {},
            'trend_history': [],  # chronological list of season aggregate snapshots
            'last_updated': None
        }
        self._load()

    # ------------- File IO -------------
    def _load(self):
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding='utf-8'))
            except Exception:
                pass

    def _sanitize(self, obj):  # recursively convert to JSON-serializable primitives
        if obj is None:
            return None
        if isinstance(obj, (str, int, bool)):
            return obj
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return float(obj)
        # numpy scalar
        if _np is not None:
            if isinstance(obj, (_np.integer, )):
                return int(obj)
            if isinstance(obj, (_np.floating, )):
                v = float(obj)
                return None if (math.isnan(v) or math.isinf(v)) else v
            if isinstance(obj, _np.ndarray):  # convert small arrays to list
                return [self._sanitize(x) for x in obj.tolist()]
        if isinstance(obj, list):
            return [self._sanitize(x) for x in obj]
        if isinstance(obj, tuple):
            return [self._sanitize(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): self._sanitize(v) for k, v in obj.items()}
        # Fallback to string representation
        return str(obj)

    def _save(self):
        try:
            self.data['last_updated'] = datetime.utcnow().isoformat()
            sanitized = self._sanitize(self.data)
            self.path.write_text(json.dumps(sanitized), encoding='utf-8')
        except Exception as e:  # pragma: no cover
            # Minimal stderr logging so silent failures are visible during offline runs
            print(f"[ReconciliationStore] Save failed: {e}")

    # ------------- Helpers -------------
    @staticmethod
    def _build_key(match: Dict[str, Any]) -> str:
        mid = match.get('id') or match.get('match_id') or match.get('uuid') or 'na'
        ht = match.get('home_team') or match.get('homeTeam') or match.get('home') or 'UNK'
        at = match.get('away_team') or match.get('awayTeam') or match.get('away') or 'UNK'
        return f"{mid}_{ht}_{at}"

    # ------------- Public API -------------
    def upsert(self, match: Dict[str, Any], reconciliation: Dict[str, Any]):
        if not reconciliation or reconciliation.get('status') != 'completed':
            return
        key = self._build_key(match)
        # Attach contextual metadata for aggregation
        enriched = {
            **reconciliation,
            'game_week': match.get('game_week') or match.get('matchday') or match.get('week'),
            'home_team': match.get('home_team') or match.get('homeTeam'),
            'away_team': match.get('away_team') or match.get('awayTeam')
        }
        self.data['reconciliations'][key] = self._sanitize(enriched)
        self._save()

    def clear(self):
        self.data['reconciliations'] = {}
        self.data['weekly_stats'] = {}
        self.data['season_stats'] = {}
        self._save()

    def compute_aggregates(self):
        recs = list(self.data.get('reconciliations', {}).values())
        completed = [r for r in recs if r.get('status') == 'completed']
        if not completed:
            self.data['weekly_stats'] = {}
            self.data['season_stats'] = {'matches_analyzed': 0}
            self._save()
            return
        # Weekly grouping
        weeks: Dict[int, List[Dict[str, Any]]] = {}
        for r in completed:
            wk = r.get('game_week') or 0
            weeks.setdefault(int(wk), []).append(r)

        def _agg(list_r: List[Dict[str, Any]]):
            total = len(list_r)
            def _as_bool(v):
                if isinstance(v, bool):
                    return v
                if isinstance(v, str):
                    return v.lower() == 'true'
                return bool(v)
            res_correct = sum(1 for r in list_r if _as_bool(r['accuracy'].get('result_correct')))
            goals_w1 = sum(1 for r in list_r if _as_bool(r['accuracy'].get('goals_within_1')))
            goals_w2 = sum(1 for r in list_r if _as_bool(r['accuracy'].get('goals_within_2')))
            avg_total_diff = sum(r['accuracy'].get('total_goals_diff', 0) for r in list_r) / total
            avg_home_diff = sum(r['accuracy'].get('home_goals_diff', 0) for r in list_r) / total
            avg_away_diff = sum(r['accuracy'].get('away_goals_diff', 0) for r in list_r) / total
            # Brier score average if present
            briers = [r.get('brier_score') for r in list_r if isinstance(r.get('brier_score'), (float,int))]
            avg_brier = sum(briers)/len(briers) if briers else None
            # Sharpness: mean absolute deviation of probability distribution from uniform (H/D/A)
            sharpness_vals = []
            for r in list_r:
                pred = r.get('predicted', {})
                ph, pd, pa = pred.get('home_win_prob'), pred.get('draw_prob'), pred.get('away_win_prob')
                if None not in (ph, pd, pa):
                    # deviation from uniform 1/3 each
                    sharpness_vals.append(abs(ph-1/3)+abs(pd-1/3)+abs(pa-1/3))
            avg_sharpness = sum(sharpness_vals)/len(sharpness_vals) if sharpness_vals else None
            return {
                'matches': total,
                'result_accuracy': res_correct / total,
                'goals_within_1_accuracy': goals_w1 / total,
                'goals_within_2_accuracy': goals_w2 / total,
                'avg_total_goals_diff': round(avg_total_diff, 3),
                'avg_home_goals_diff': round(avg_home_diff, 3),
                'avg_away_goals_diff': round(avg_away_diff, 3),
                'avg_brier_score': round(avg_brier, 4) if avg_brier is not None else None,
                'prob_sharpness': round(avg_sharpness, 4) if avg_sharpness is not None else None
            }

        weekly_stats = {str(wk): _agg(rlist) for wk, rlist in weeks.items()}
        # Season aggregate across all completed
        season_stats = _agg(completed)
        season_stats['matches_analyzed'] = len(completed)
        self.data['weekly_stats'] = weekly_stats
        self.data['season_stats'] = season_stats
        # Append to trend history (keep last 200 entries max to bound file size)
        snapshot = { 'ts': datetime.utcnow().isoformat(), **season_stats }
        self.data.setdefault('trend_history', []).append(snapshot)
        if len(self.data['trend_history']) > 200:
            self.data['trend_history'] = self.data['trend_history'][-200:]
        self._save()

    def get_weekly_stats(self) -> Dict[str, Any]:
        return self.data.get('weekly_stats', {})

    def get_season_stats(self) -> Dict[str, Any]:
        return self.data.get('season_stats', {})

    def dump(self) -> Dict[str, Any]:
        return self.data


reconciliation_store = ReconciliationStore()