"""
Game Week Management Service for EPL Betting Platform
Organizes matches by game week and provides reconciliation with actual results
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
import pandas as pd
from collections import defaultdict

class GameWeekService:
    def __init__(self):
        self.current_season = "2025-26"
        # Store season start as timezone-aware UTC for consistent calculations
        self.season_start = datetime(2025, 8, 16, tzinfo=timezone.utc)
        
    def calculate_game_week(self, match_date: datetime) -> int:
        """Calculate which game week a match belongs to"""
        if isinstance(match_date, str):
            try:
                # Normalize common Football-Data.org formats
                if match_date.endswith('Z'):
                    match_date = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
                else:
                    match_date = datetime.fromisoformat(match_date)
            except Exception:
                return 1  # fallback
        # Normalize both datetimes to timezone-aware UTC
        if match_date.tzinfo is None:
            match_date = match_date.replace(tzinfo=timezone.utc)
        season_start = self.season_start
        if season_start.tzinfo is None:
            season_start = season_start.replace(tzinfo=timezone.utc)
        days_since_start = (match_date - season_start).days
        game_week = max(1, (days_since_start // 7) + 1)
        return min(game_week, 38)  # EPL has 38 game weeks
    
    def get_current_game_week(self) -> int:
        """Get the current game week based on today's date"""
        today = datetime.now()
        return self.calculate_game_week(today)
    
    def organize_matches_by_week(self, matches: List[Dict]) -> Dict[int, List[Dict]]:
        """Organize matches into game weeks"""
        weeks = defaultdict(list)
        
        for match in matches:
            # Parse match date
            match_date = match.get('utc_date') or match.get('date')
            if match_date:
                if isinstance(match_date, str):
                    try:
                        if match_date.endswith('Z'):
                            match_date = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
                        else:
                            match_date = datetime.fromisoformat(match_date)
                    except Exception:
                        continue
                if match_date.tzinfo is None:
                    match_date = match_date.replace(tzinfo=timezone.utc)
                # Prefer provided matchday (from data source) when available and valid (1-38)
                week = None
                md = match.get('matchday') or match.get('match_day') or match.get('round')
                if isinstance(md, int) and 1 <= md <= 38:
                    week = md
                else:
                    try:
                        md_int = int(str(md).strip())
                        if 1 <= md_int <= 38:
                            week = md_int
                    except Exception:
                        week = None
                if week is None:
                    week = self.calculate_game_week(match_date)
                
                # Enhanced match data
                enhanced_match = {
                    **match,
                    'game_week': week,
                    'formatted_date': match_date.strftime('%a, %b %d, %Y'),
                    'formatted_time': match_date.strftime('%I:%M %p'),
                    'is_completed': match.get('status') in ['FINISHED', 'COMPLETED'],
                    'is_live': match.get('status') == 'IN_PLAY',
                    'is_upcoming': match.get('status') in ['SCHEDULED', 'TIMED']
                }
                
                weeks[week].append(enhanced_match)
        
        # Sort matches within each week by date
        for week in weeks:
            weeks[week].sort(key=lambda m: m.get('utc_date', ''))
        
        return dict(weeks)
    
    def get_week_summary(self, week: int, matches: List[Dict]) -> Dict[str, Any]:
        """Get summary statistics for a game week"""
        week_matches = [m for m in matches if m.get('game_week') == week]
        
        completed = [m for m in week_matches if m.get('is_completed')]
        upcoming = [m for m in week_matches if m.get('is_upcoming')]
        live = [m for m in week_matches if m.get('is_live')]
        
        total_goals = sum(
            (m.get('home_score', 0) or 0) + (m.get('away_score', 0) or 0) 
            for m in completed
        )
        
        return {
            'week': week,
            'total_matches': len(week_matches),
            'completed': len(completed),
            'upcoming': len(upcoming),
            'live': len(live),
            'total_goals': total_goals,
            'avg_goals_per_match': total_goals / max(len(completed), 1),
            'completion_rate': len(completed) / max(len(week_matches), 1) * 100
        }

class ModelReconciliationService:
    """Reconcile model predictions with actual results"""
    
    def __init__(self):
        self.prediction_accuracy = {}
        self.model_performance = {}
    
    def reconcile_match_predictions(self, match: Dict, predictions: Dict) -> Dict[str, Any]:
        """Compare model predictions with actual match results"""
        if not match.get('is_completed'):
            return {
                'match_id': match.get('id'),
                'status': 'pending',
                'predictions': predictions
            }
        
        # Coerce numeric fields defensively to avoid type errors (e.g., strings)
        def _to_int(x, default=0):
            try:
                return int(x)
            except Exception:
                try:
                    return int(float(x))
                except Exception:
                    return default
        def _to_float(x, default=0.0):
            try:
                return float(x)
            except Exception:
                return default

        actual_home = _to_int(match.get('home_score', 0), 0)
        actual_away = _to_int(match.get('away_score', 0), 0)
        actual_total = actual_home + actual_away
        # Corners actuals (best-effort). Treat missing as None to avoid false reconciliation.
        hc_raw = match.get('home_corners')
        ac_raw = match.get('away_corners')
        tc_raw = match.get('total_corners')
        actual_hc = _to_int(hc_raw, 0) if hc_raw is not None else None
        actual_ac = _to_int(ac_raw, 0) if ac_raw is not None else None
        if tc_raw is not None:
            actual_tc = _to_int(tc_raw, 0)
        elif (actual_hc is not None or actual_ac is not None):
            # If we have at least one side, sum knowns (treat unknown as 0)
            actual_tc = _to_int(actual_hc or 0, 0) + _to_int(actual_ac or 0, 0)
        else:
            actual_tc = None

        predicted_home = _to_float((predictions or {}).get('home_goals', 0.0), 0.0)
        predicted_away = _to_float((predictions or {}).get('away_goals', 0.0), 0.0)
        predicted_total = predicted_home + predicted_away
        predicted_corners_total = _to_float((predictions or {}).get('total_corners', 0.0), 0.0)

        # Result prediction accuracy
        actual_result = 'H' if actual_home > actual_away else 'A' if actual_away > actual_home else 'D'
        predicted_result = (predictions or {}).get('result_prediction', 'D')

        # Extract probabilities if present in prediction (multi-class H/D/A)
        p_home = _to_float((predictions or {}).get('home_win_prob', 0.0), 0.0)
        p_away = _to_float((predictions or {}).get('away_win_prob', 0.0), 0.0)
        p_draw = _to_float((predictions or {}).get('draw_prob', 0.0), 0.0)
        prob_sum = p_home + p_away + p_draw
        if prob_sum > 0:
            p_home, p_away, p_draw = [p / prob_sum for p in (p_home, p_away, p_draw)]
        # Multi-class Brier score contribution for this match
        yh, yd, ya = (1 if actual_result == 'H' else 0), (1 if actual_result == 'D' else 0), (1 if actual_result == 'A' else 0)
        brier = ((p_home - yh)**2 + (p_draw - yd)**2 + (p_away - ya)**2) / 3.0
        # Entropy of probability distribution (normalized to [0,1] by dividing by log(3))
        import math as _math
        probs = [p_home, p_draw, p_away]
        entropy = 0.0
        for pv in probs:
            if pv > 0:
                entropy -= pv * _math.log(pv)
        max_entropy = _math.log(3)
        norm_entropy = entropy / max_entropy if max_entropy > 0 else None

        # Over/Under heuristic lines (standard)
        goals_line = 2.5
        corners_line = 9.5
        goals_pred_over = None if predicted_total is None else (predicted_total > goals_line)
        goals_actual_over = actual_total > goals_line
        # If total_corners wasn't provided in predictions, treat default 0.0 as missing (None)
        corners_pred_over = None if (predicted_corners_total is None or (predicted_corners_total == 0.0 and (predictions or {}).get('total_corners') is None)) else (predicted_corners_total > corners_line)
        corners_actual_over = None if (actual_tc is None) else (actual_tc > corners_line)

        reconciliation = {
            'match_id': match.get('id'),
            'status': 'completed',
            'actual': {
                'home_goals': actual_home,
                'away_goals': actual_away,
                'total_goals': actual_total,
                'result': actual_result,
                'home_corners': actual_hc if actual_hc is not None else None,
                'away_corners': actual_ac if actual_ac is not None else None,
                'total_corners': actual_tc if actual_tc is not None else None
            },
            'predicted': {
                'home_goals': predicted_home,
                'away_goals': predicted_away,
                'total_goals': predicted_total,
                'total_corners': predicted_corners_total if predicted_corners_total else None,
                'result': predicted_result,
                'home_win_prob': p_home,
                'draw_prob': p_draw,
                'away_win_prob': p_away
            },
            'ou': {
                'goals_line': goals_line,
                'goals_pred_over': goals_pred_over,
                'goals_actual_over': goals_actual_over,
                'corners_line': corners_line,
                'corners_pred_over': corners_pred_over,
                'corners_actual_over': corners_actual_over
            },
            'accuracy': {
                'result_correct': actual_result == predicted_result,
                'home_goals_diff': abs(actual_home - predicted_home),
                'away_goals_diff': abs(actual_away - predicted_away),
                'total_goals_diff': abs(actual_total - predicted_total),
                'goals_within_1': abs(actual_total - predicted_total) <= 1,
                'goals_within_2': abs(actual_total - predicted_total) <= 2
            },
            'confidence': (predictions or {}).get('confidence', 0.5),
            'model_edge': (predictions or {}).get('model_edge', 0),
            'brier_score': brier,
            'prob_entropy': norm_entropy
        }
        
        return reconciliation
    
    def calculate_weekly_model_performance(self, reconciliations: List[Dict]) -> Dict[str, float]:
        """Calculate model performance metrics for the week"""
        completed = [r for r in reconciliations if r['status'] == 'completed']
        
        if not completed:
            return {'insufficient_data': True}

        total_matches = len(completed)
        result_correct = sum(1 for r in completed if r['accuracy']['result_correct'])

        # Over/Under accuracies (standard lines)
        goals_ou_total = 0
        goals_ou_correct = 0
        corners_ou_total = 0
        corners_ou_correct = 0
        for r in completed:
            ou = r.get('ou') or {}
            gp = ou.get('goals_pred_over')
            ga = ou.get('goals_actual_over')
            if gp is not None and ga is not None:
                goals_ou_total += 1
                if gp == ga:
                    goals_ou_correct += 1
            cp = ou.get('corners_pred_over')
            ca = ou.get('corners_actual_over')
            if cp is not None and ca is not None:
                corners_ou_total += 1
                if cp == ca:
                    corners_ou_correct += 1

        return {
            'result_accuracy': result_correct / total_matches,
            'goals_ou_accuracy': (goals_ou_correct / goals_ou_total) if goals_ou_total else None,
            'corners_ou_accuracy': (corners_ou_correct / corners_ou_total) if corners_ou_total else None
        }

# Global instances
game_week_service = GameWeekService()
reconciliation_service = ModelReconciliationService()