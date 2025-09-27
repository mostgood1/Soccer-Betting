import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sqlalchemy.orm import Session
from typing import Dict, Any, List
import joblib
import os
import logging
from app.services.soccerdata_service import SoccerDataService

logger = logging.getLogger(__name__)


class MatchPredictor:
    """Main predictor class for soccer match predictions"""
    
    def __init__(self):
        self.models = {}
        self.feature_columns = []
        self.is_trained = False
        self.soccer_data_service = SoccerDataService()  # Add soccerdata integration
        
    def prepare_features(self, match, db: Session) -> Dict[str, Any]:
        """Prepare features for prediction based on historical data"""
        try:
            # Get team historical data
            home_team_stats = self._get_team_stats(match.home_team_id, db)
            away_team_stats = self._get_team_stats(match.away_team_id, db)
            
            # Get head-to-head statistics
            h2h_stats = self._get_head_to_head_stats(match.home_team_id, match.away_team_id, db)
            
            # Combine all features
            features = {
                **home_team_stats,
                **away_team_stats,
                **h2h_stats,
                'home_advantage': 1,  # Home team gets advantage
                'league_id': match.league_id
            }
            
            # Create feature names list
            feature_names = list(features.keys())
            
            return {
                'features': features,
                'feature_names': feature_names,
                'feature_vector': list(features.values())
            }
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            # Return default features if error occurs
            return self._get_default_features()
    
    def _get_team_stats(self, team_id: int, db: Session, last_n_matches: int = 10) -> Dict[str, float]:
        """Get team statistics from recent matches"""
        from app.models.match import Match
        
        # Get recent matches for the team
        recent_matches = db.query(Match).filter(
            ((Match.home_team_id == team_id) | (Match.away_team_id == team_id)) &
            (Match.status == "finished")
        ).order_by(Match.match_date.desc()).limit(last_n_matches).all()
        
        if not recent_matches:
            return self._get_default_team_stats()
        
        # Calculate statistics
        total_goals_for = 0
        total_goals_against = 0
        total_corners_for = 0
        total_corners_against = 0
        wins = 0
        draws = 0
        losses = 0
        
        for match in recent_matches:
            if match.home_team_id == team_id:
                # Team was playing at home
                goals_for = match.home_score or 0
                goals_against = match.away_score or 0
                corners_for = match.home_corners or 0
                corners_against = match.away_corners or 0
                
                if match.result == 'H':
                    wins += 1
                elif match.result == 'D':
                    draws += 1
                else:
                    losses += 1
            else:
                # Team was playing away
                goals_for = match.away_score or 0
                goals_against = match.home_score or 0
                corners_for = match.away_corners or 0
                corners_against = match.home_corners or 0
                
                if match.result == 'A':
                    wins += 1
                elif match.result == 'D':
                    draws += 1
                else:
                    losses += 1
            
            total_goals_for += goals_for
            total_goals_against += goals_against
            total_corners_for += corners_for
            total_corners_against += corners_against
        
        num_matches = len(recent_matches)
        
        return {
            f'team_{team_id}_avg_goals_for': total_goals_for / num_matches,
            f'team_{team_id}_avg_goals_against': total_goals_against / num_matches,
            f'team_{team_id}_avg_corners_for': total_corners_for / num_matches,
            f'team_{team_id}_avg_corners_against': total_corners_against / num_matches,
            f'team_{team_id}_win_rate': wins / num_matches,
            f'team_{team_id}_draw_rate': draws / num_matches,
            f'team_{team_id}_loss_rate': losses / num_matches
        }
    
    def _get_head_to_head_stats(self, home_team_id: int, away_team_id: int, db: Session, last_n_matches: int = 5) -> Dict[str, float]:
        """Get head-to-head statistics between two teams"""
        from app.models.match import Match
        
        # Get recent head-to-head matches
        h2h_matches = db.query(Match).filter(
            ((Match.home_team_id == home_team_id) & (Match.away_team_id == away_team_id)) |
            ((Match.home_team_id == away_team_id) & (Match.away_team_id == home_team_id)),
            Match.status == "finished"
        ).order_by(Match.match_date.desc()).limit(last_n_matches).all()
        
        if not h2h_matches:
            return {'h2h_avg_total_goals': 2.5, 'h2h_avg_total_corners': 10.0}
        
        total_goals = sum((match.total_goals or 0) for match in h2h_matches)
        total_corners = sum((match.total_corners or 0) for match in h2h_matches)
        
        return {
            'h2h_avg_total_goals': total_goals / len(h2h_matches),
            'h2h_avg_total_corners': total_corners / len(h2h_matches)
        }
    
    def _get_default_team_stats(self) -> Dict[str, float]:
        """Return default team statistics"""
        return {
            'team_avg_goals_for': 1.5,
            'team_avg_goals_against': 1.5,
            'team_avg_corners_for': 5.0,
            'team_avg_corners_against': 5.0,
            'team_win_rate': 0.33,
            'team_draw_rate': 0.33,
            'team_loss_rate': 0.34
        }
    
    def _get_default_features(self) -> Dict[str, Any]:
        """Return default features when unable to calculate"""
        return {
            'features': {
                'home_avg_goals': 1.5,
                'away_avg_goals': 1.5,
                'h2h_avg_total_goals': 2.5,
                'home_advantage': 1
            },
            'feature_names': ['home_avg_goals', 'away_avg_goals', 'h2h_avg_total_goals', 'home_advantage'],
            'feature_vector': [1.5, 1.5, 2.5, 1]
        }
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions using trained models or heuristics"""
        try:
            # If models are trained, use them; otherwise use heuristic predictions
            if self.is_trained:
                return self._predict_with_models(features)
            else:
                return self._predict_heuristic(features)
                
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return self._get_default_predictions()
    
    def _predict_heuristic(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions using heuristic methods"""
        feature_dict = features.get('features', {})
        
        # Simple heuristic predictions based on team averages
        home_goals_avg = next((v for k, v in feature_dict.items() if 'goals_for' in k and 'home' in k), 1.5)
        away_goals_avg = next((v for k, v in feature_dict.items() if 'goals_for' in k and 'away' in k), 1.5)
        
        # Total goals prediction
        total_goals_pred = home_goals_avg + away_goals_avg
        first_half_pred = total_goals_pred * 0.45  # Typically ~45% of goals in first half
        
        # Match result prediction based on goal averages
        if home_goals_avg > away_goals_avg * 1.2:
            result_pred = 'H'
            home_prob, away_prob, draw_prob = 0.5, 0.25, 0.25
        elif away_goals_avg > home_goals_avg * 1.2:
            result_pred = 'A'
            home_prob, away_prob, draw_prob = 0.25, 0.5, 0.25
        else:
            result_pred = 'D'
            home_prob, away_prob, draw_prob = 0.35, 0.35, 0.30
        
        # Corner predictions
        total_corners_pred = feature_dict.get('h2h_avg_total_corners', 10.0)
        
        return {
            'total_goals': round(total_goals_pred, 2),
            'total_goals_confidence': 0.65,
            'first_half_goals': round(first_half_pred, 2),
            'first_half_goals_confidence': 0.60,
            'match_result': result_pred,
            'match_result_confidence': 0.55,
            'home_win_prob': round(home_prob, 3),
            'away_win_prob': round(away_prob, 3),
            'draw_prob': round(draw_prob, 3),
            'total_corners': round(total_corners_pred, 1),
            'total_corners_confidence': 0.50,
            'home_corners': round(total_corners_pred * 0.52, 1),
            'away_corners': round(total_corners_pred * 0.48, 1),
            'algorithm': 'heuristic',
            'feature_importance': {k: 0.1 for k in features.get('feature_names', [])}
        }
    
    def _predict_with_models(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Use trained ML models for predictions"""
        # This would use trained scikit-learn models
        # For now, fall back to heuristic method
        return self._predict_heuristic(features)
    
    def _get_default_predictions(self) -> Dict[str, Any]:
        """Return default predictions when all else fails"""
        return {
            'total_goals': 2.5,
            'total_goals_confidence': 0.50,
            'first_half_goals': 1.0,
            'first_half_goals_confidence': 0.50,
            'match_result': 'D',
            'match_result_confidence': 0.33,
            'home_win_prob': 0.33,
            'away_win_prob': 0.33,
            'draw_prob': 0.34,
            'total_corners': 10.0,
            'total_corners_confidence': 0.50,
            'home_corners': 5.2,
            'away_corners': 4.8,
            'algorithm': 'default',
            'feature_importance': {}
        }
    
    def train_models(self, training_data: pd.DataFrame):
        """Train ML models on historical data"""
        # This method would implement actual model training
        # For initial version, we'll rely on heuristic predictions
        logger.info("Model training not yet implemented - using heuristic predictions")
        pass