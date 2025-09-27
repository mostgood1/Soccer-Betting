"""Deterministic Team Seed Metrics Service

Generates pseudo-realistic differentiated early-season metrics for each team
so that feature engineering can produce diverse predictions before enough
real stats accumulate.

All values are deterministic: hashing team name -> RNG seed.

Exposed metrics (0..ranges with realistic scaling):
- attack_rating (1.1 - 2.5)
- defense_rating (0.6 - 1.8)  (lower better, but we keep direct for transforms)
- form_score (0.40 - 0.90)
- tempo_factor (0.85 - 1.15)
- finishing_quality (0.85 - 1.20)
- chance_creation (0.90 - 1.30)
- conversion_rate (0.10 - 0.22)
- suppression_factor (0.80 - 1.15) (defensive suppression of opponent chances)

Feature Versioning:
FEATURE_VERSION increments when schema/derivation changes so we can invalidate
prediction cache entries cleanly.
"""
from __future__ import annotations
import hashlib
import random
from typing import Dict, Any, List

FEATURE_VERSION = "fv2"  # Bumped due to pairing variance & engineered feature expansion

class TeamSeedMetricsService:
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _rng(self, team_name: str) -> random.Random:
        h = hashlib.sha256(team_name.strip().lower().encode()).hexdigest()
        seed = int(h[:12], 16)
        return random.Random(seed)

    def get_metrics(self, team_name: str) -> Dict[str, Any]:
        if team_name in self._cache:
            return self._cache[team_name]
        rng = self._rng(team_name)
        attack = rng.uniform(1.1, 2.5)
        defense = rng.uniform(0.6, 1.8)
        form = rng.uniform(0.40, 0.90)
        tempo = rng.uniform(0.85, 1.15)
        finishing = rng.uniform(0.85, 1.20)
        creation = rng.uniform(0.90, 1.30)
        conversion = rng.uniform(0.10, 0.22)
        suppression = rng.uniform(0.80, 1.15)
        metrics = {
            'attack_rating': round(attack, 4),
            'defense_rating': round(defense, 4),
            'form_score': round(form, 4),
            'tempo_factor': round(tempo, 4),
            'finishing_quality': round(finishing, 4),
            'chance_creation': round(creation, 4),
            'conversion_rate': round(conversion, 4),
            'suppression_factor': round(suppression, 4),
            'feature_version': FEATURE_VERSION
        }
        self._cache[team_name] = metrics
        return metrics

    def enrich_teams(self, teams: List[Dict[str, Any]]):
        for t in teams:
            name = t.get('name') or t.get('team_name')
            if not name:
                continue
            metrics = self.get_metrics(name)
            t['seed_metrics'] = metrics
        return teams

team_seed_metrics_service = TeamSeedMetricsService()
