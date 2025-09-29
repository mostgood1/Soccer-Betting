"""Expected Goals (xG) / Poisson Modeling Service
Computes per-team attack & defense strengths from historical seasons and
provides Poisson goal expectation + scoreline distribution utilities.

Methodology (simplified Dixon-Coles style without temporal decay):
- Compute league average home & away goals (lambda_home_league, lambda_away_league)
- For each team:
    attack_strength_home  = (team_home_goals / team_home_matches) / lambda_home_league
    attack_strength_away  = (team_away_goals / team_away_matches) / lambda_away_league
    defense_strength_home = (team_home_conceded / team_home_matches) / lambda_away_league
    defense_strength_away = (team_away_conceded / team_away_matches) / lambda_home_league
  (defense strengths >1 indicate weaker defense; <1 stronger)

Expected goals for a future fixture:
  lambda_home = lambda_home_league * attack_strength_home_team * defense_strength_away_opponent * HOME_ADV
  lambda_away = lambda_away_league * attack_strength_away_team * defense_strength_home_opponent

We cap & floor lambdas for stability and optionally blend with ML total goals.
"""
from __future__ import annotations
from typing import Dict, Any, Tuple
import math
from functools import lru_cache

from .enhanced_historical_data_service import enhanced_historical_data_service

HOME_ADV = 1.10  # modest home advantage multiplier
MAX_LAMBDA = 3.5
MIN_LAMBDA = 0.2


class ExpectedGoalsService:
    def __init__(self):
        self._prepared = False
        self.team_metrics: Dict[str, Dict[str, float]] = {}
        self.league_avgs: Dict[str, float] = {}
        self._prepare()

    def _prepare(self):
        try:
            matches = enhanced_historical_data_service.get_matches_for_training()
            if not matches:
                return
            # Aggregations
            agg = {}
            total_home_goals = 0
            total_away_goals = 0
            total_home_matches = 0
            total_away_matches = 0
            for m in matches:
                h = m.get("home_team")
                a = m.get("away_team")
                hg = m.get("home_goals", 0)
                ag = m.get("away_goals", 0)
                total_home_goals += hg
                total_away_goals += ag
                total_home_matches += 1
                total_away_matches += 1
                rec_h = agg.setdefault(
                    h,
                    {
                        "home_goals": 0,
                        "home_conc": 0,
                        "home_matches": 0,
                        "away_goals": 0,
                        "away_conc": 0,
                        "away_matches": 0,
                    },
                )
                rec_a = agg.setdefault(
                    a,
                    {
                        "home_goals": 0,
                        "home_conc": 0,
                        "home_matches": 0,
                        "away_goals": 0,
                        "away_conc": 0,
                        "away_matches": 0,
                    },
                )
                rec_h["home_goals"] += hg
                rec_h["home_conc"] += ag
                rec_h["home_matches"] += 1
                rec_a["away_goals"] += ag
                rec_a["away_conc"] += hg
                rec_a["away_matches"] += 1
            lambda_home_league = total_home_goals / max(total_home_matches, 1)
            lambda_away_league = total_away_goals / max(total_away_matches, 1)
            if lambda_home_league <= 0:
                lambda_home_league = 1.4
            if lambda_away_league <= 0:
                lambda_away_league = 1.2
            self.league_avgs = {
                "lambda_home_league": lambda_home_league,
                "lambda_away_league": lambda_away_league,
            }
            for team, rec in agg.items():
                hgpm = rec["home_goals"] / max(rec["home_matches"], 1)
                agpm = rec["away_goals"] / max(rec["away_matches"], 1)
                hcpm = rec["home_conc"] / max(rec["home_matches"], 1)
                acpm = rec["away_conc"] / max(rec["away_matches"], 1)
                attack_home = hgpm / lambda_home_league
                attack_away = agpm / lambda_away_league
                defense_home = hcpm / lambda_away_league
                defense_away = acpm / lambda_home_league
                self.team_metrics[team] = {
                    "attack_home": attack_home,
                    "attack_away": attack_away,
                    "defense_home": defense_home,
                    "defense_away": defense_away,
                }
            self._prepared = True
        except Exception:
            self._prepared = False

    def get_expected_goals(self, home_team: str, away_team: str) -> Dict[str, Any]:
        if not self._prepared:
            self._prepare()
        la_home = self.league_avgs.get("lambda_home_league", 1.5)
        la_away = self.league_avgs.get("lambda_away_league", 1.2)
        hm = self.team_metrics.get(home_team)
        am = self.team_metrics.get(away_team)
        # Fallback weak assumptions if team absent
        if not hm:
            hm = {
                "attack_home": 1.0,
                "attack_away": 1.0,
                "defense_home": 1.0,
                "defense_away": 1.0,
            }
        if not am:
            am = {
                "attack_home": 1.0,
                "attack_away": 1.0,
                "defense_home": 1.0,
                "defense_away": 1.0,
            }
        lambda_home = la_home * hm["attack_home"] * am["defense_away"] * HOME_ADV
        lambda_away = la_away * am["attack_away"] * hm["defense_home"]
        # Clamp
        lambda_home = min(MAX_LAMBDA, max(MIN_LAMBDA, lambda_home))
        lambda_away = min(MAX_LAMBDA, max(MIN_LAMBDA, lambda_away))
        return {
            "lambda_home": round(lambda_home, 3),
            "lambda_away": round(lambda_away, 3),
            "league_avg_home": round(la_home, 3),
            "league_avg_away": round(la_away, 3),
        }

    @staticmethod
    def _poisson(k: int, lam: float) -> float:
        try:
            return math.exp(-lam) * (lam**k) / math.factorial(k)
        except OverflowError:
            return 0.0

    @lru_cache(maxsize=2048)
    def scoreline_distribution(
        self, lambda_home: float, lambda_away: float, max_goals: int = 6
    ) -> Dict[str, Any]:
        matrix = []
        prob_total = 0.0
        for i in range(0, max_goals + 1):
            row = []
            p_i = self._poisson(i, lambda_home)
            for j in range(0, max_goals + 1):
                p_j = self._poisson(j, lambda_away)
                p = p_i * p_j
                row.append(round(p, 6))
                prob_total += p
            matrix.append(row)
        # Outcome probabilities
        p_home = sum(
            matrix[i][j]
            for i in range(len(matrix))
            for j in range(len(matrix))
            if i > j
        )
        p_draw = sum(
            matrix[i][j]
            for i in range(len(matrix))
            for j in range(len(matrix))
            if i == j
        )
        p_away = sum(
            matrix[i][j]
            for i in range(len(matrix))
            for j in range(len(matrix))
            if i < j
        )
        # Normalise small truncation loss
        scale = p_home + p_draw + p_away
        if scale > 0:
            p_home /= scale
            p_draw /= scale
            p_away /= scale
        # Over/Under 2.5
        over_25 = sum(
            matrix[i][j]
            for i in range(len(matrix))
            for j in range(len(matrix))
            if i + j >= 3
        )
        under_25 = 1 - over_25
        return {
            "matrix": matrix,
            "home_win_prob": round(p_home, 4),
            "draw_prob": round(p_draw, 4),
            "away_win_prob": round(p_away, 4),
            "over_2_5_prob": round(over_25, 4),
            "under_2_5_prob": round(under_25, 4),
            "max_goals": max_goals,
        }

    def get_full_xg(
        self, home_team: str, away_team: str, max_goals: int = 6
    ) -> Dict[str, Any]:
        eg = self.get_expected_goals(home_team, away_team)
        dist = self.scoreline_distribution(
            eg["lambda_home"], eg["lambda_away"], max_goals=max_goals
        )
        return {**eg, **dist, "home_team": home_team, "away_team": away_team}


expected_goals_service = ExpectedGoalsService()
