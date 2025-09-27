"""Baseline Prediction Store

Captures pre-retrain (baseline) predictions per game week so that later
model iterations can be compared against the stored snapshot.

Structure stored in JSON:
{
  "season": "2025-26",
  "weeks": {
     "1": {
        "timestamp": "ISO",
        "model_version": "v2.0_advanced_ml",
        "predictions": [ { match_key, prediction_payload } ]
     },
     ...
  }
}
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime


class BaselinePredictionStore:
    def __init__(self):
        self.path = Path("cache/baseline_week_predictions.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = {"season": "2025-26", "weeks": {}}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding='utf-8'))
            except Exception:
                pass

    def _save(self):
        try:
            self.path.write_text(json.dumps(self.data), encoding='utf-8')
        except Exception:
            pass

    @staticmethod
    def build_match_key(match: Dict[str, Any]) -> str:
        mid = match.get('id') or match.get('match_id') or match.get('uuid') or 'na'
        ht = match.get('home_team') or match.get('homeTeam') or match.get('home') or 'UNK'
        at = match.get('away_team') or match.get('awayTeam') or match.get('away') or 'UNK'
        return f"{mid}_{ht}_{at}"

    def store_week(self, week: int, predictions: List[Dict[str, Any]], model_version: str):
        serializable = []
        for p in predictions:
            serializable.append({
                'match_key': p.get('match_key'),
                'prediction': p.get('prediction')
            })
        self.data['weeks'][str(week)] = {
            'timestamp': datetime.utcnow().isoformat(),
            'model_version': model_version,
            'predictions': serializable
        }
        self._save()

    def get_week(self, week: int) -> Dict[str, Any]:
        return self.data['weeks'].get(str(week)) or {}

    def has_week(self, week: int) -> bool:
        return str(week) in self.data['weeks']

    def clear_week(self, week: int):
        self.data['weeks'].pop(str(week), None)
        self._save()

    def clear_all(self):
        self.data['weeks'] = {}
        self._save()


baseline_prediction_store = BaselinePredictionStore()