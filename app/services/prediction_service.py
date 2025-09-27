from sqlalchemy.orm import Session
from app.models.prediction import Prediction
from app.models.match import Match
from app.ml.predictor import MatchPredictor
from app.ml.advanced_predictor import advanced_ml_predictor
from app.services.enhanced_epl_service import enhanced_epl_service
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, db: Session):
        self.db = db
        self.predictor = MatchPredictor()
        self.advanced_predictor = advanced_ml_predictor
        
        # Try to load pre-trained models
        if not self.advanced_predictor.is_trained:
            model_loaded = self.advanced_predictor.load_models()
            if not model_loaded:
                logger.info("No pre-trained models found. Will use heuristic predictions until models are trained.")

    def generate_prediction(self, match_id: int, model_version: str = "v2.0", use_advanced_ml: bool = True) -> Prediction:
        """Generate predictions for a match using advanced ML models or fallback methods"""
        try:
            # Get match data
            match = self.db.query(Match).filter(Match.id == match_id).first()
            if not match:
                raise ValueError(f"Match with ID {match_id} not found")

            # Check if prediction already exists
            existing_prediction = self.db.query(Prediction).filter(
                Prediction.match_id == match_id,
                Prediction.model_version == model_version
            ).first()

            if existing_prediction:
                logger.info(f"Returning existing prediction for match {match_id}")
                return existing_prediction

            # Generate predictions using advanced ML if available and requested
            if use_advanced_ml and self.advanced_predictor.is_trained:
                predictions = self._generate_advanced_ml_prediction(match)
                algorithm_used = predictions.get('algorithm', 'advanced_ml_ensemble')
            else:
                # Fallback to original predictor
                features = self.predictor.prepare_features(match, self.db)
                predictions = self.predictor.predict(features)
                algorithm_used = predictions.get('algorithm', 'heuristic')

            # Create prediction record
            prediction = Prediction(
                match_id=match_id,
                total_goals_prediction=predictions.get('total_goals'),
                total_goals_confidence=predictions.get('total_goals_confidence', 0.5),
                first_half_goals_prediction=predictions.get('first_half_goals'),
                first_half_goals_confidence=predictions.get('first_half_goals_confidence', 0.5),
                match_result_prediction=predictions.get('match_result'),
                match_result_confidence=predictions.get('match_result_confidence', 0.5),
                home_win_probability=predictions.get('home_win_probability', 0.33),
                away_win_probability=predictions.get('away_win_probability', 0.33),
                draw_probability=predictions.get('draw_probability', 0.34),
                total_corners_prediction=predictions.get('total_corners'),
                total_corners_confidence=predictions.get('total_corners_confidence', 0.5),
                home_corners_prediction=predictions.get('home_corners'),
                away_corners_prediction=predictions.get('away_corners'),
                model_version=model_version,
                algorithm_used=algorithm_used,
                features_used=predictions.get('feature_names', []),
                feature_importance=predictions.get('feature_importance', {})
            )

            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)

            logger.info(f"Generated {algorithm_used} prediction for match {match_id} with model {model_version}")
            return prediction

        except Exception as e:
            logger.error(f"Error generating prediction for match {match_id}: {str(e)}")
            self.db.rollback()
            raise

    def _generate_advanced_ml_prediction(self, match) -> Dict[str, Any]:
        """Generate prediction using advanced ML models"""
        try:
            # Extract team names from match object
            # This assumes match object has home_team and away_team attributes or relationships
            home_team = getattr(match, 'home_team_name', 'Unknown')
            away_team = getattr(match, 'away_team_name', 'Unknown')
            
            # If team names are not directly available, try to get from relationships
            if home_team == 'Unknown' and hasattr(match, 'home_team'):
                home_team = getattr(match.home_team, 'name', 'Unknown')
            if away_team == 'Unknown' and hasattr(match, 'away_team'):  
                away_team = getattr(match.away_team, 'name', 'Unknown')
            
            # Generate prediction using advanced ML
            return self.advanced_predictor.predict_match(home_team, away_team)
            
        except Exception as e:
            logger.error(f"Error in advanced ML prediction: {e}")
            # Fallback to heuristic prediction
            features = self.predictor.prepare_features(match, self.db)
            return self.predictor.predict(features)

    def generate_real_time_prediction(self, home_team: str, away_team: str, model_version: str = "v2.0") -> Dict[str, Any]:
        """Generate real-time prediction for any team matchup without requiring database match"""
        try:
            logger.info(f"Generating real-time prediction for {home_team} vs {away_team}")
            
            if self.advanced_predictor.is_trained:
                predictions = self.advanced_predictor.predict_match(home_team, away_team)
                
                # Add betting insights
                predictions['betting_tips'] = self._generate_betting_tips(predictions)
                predictions['confidence_rating'] = self._calculate_overall_confidence(predictions)
                predictions['key_factors'] = self._extract_key_factors(predictions)
                
                return predictions
            else:
                # Use enhanced EPL service for real-time predictions
                features = enhanced_epl_service.get_match_prediction_features(home_team, away_team)
                
                # Simple prediction based on current team stats
                return self._generate_simple_prediction(features, home_team, away_team)
                
        except Exception as e:
            logger.error(f"Error generating real-time prediction: {e}")
            return self._get_default_prediction(home_team, away_team)

    def _generate_betting_tips(self, predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific betting tips based on predictions"""
        tips = []
        
        # Total goals tips
        total_goals = predictions.get('total_goals', 2.5)
        over_prob = predictions.get('over_2_5_goals_probability', 0.5)
        
        if over_prob > 0.65:
            tips.append({
                'market': 'Over 2.5 Goals',
                'recommendation': 'STRONG BET',
                'confidence': over_prob,
                'reasoning': f"Expected {total_goals:.1f} goals with {over_prob:.1%} over 2.5 probability"
            })
        elif over_prob < 0.35:
            tips.append({
                'market': 'Under 2.5 Goals',
                'recommendation': 'GOOD BET',
                'confidence': 1.0 - over_prob,
                'reasoning': f"Low-scoring match expected ({total_goals:.1f} goals)"
            })
        
        # Match result tips
        home_prob = predictions.get('home_win_probability', 0.33)
        away_prob = predictions.get('away_win_probability', 0.33)
        draw_prob = predictions.get('draw_probability', 0.34)
        
        max_prob = max(home_prob, away_prob, draw_prob)
        if max_prob > 0.5:
            if max_prob == home_prob:
                tips.append({
                    'market': 'Home Win',
                    'recommendation': 'STRONG BET',
                    'confidence': home_prob,
                    'reasoning': f"Home team heavily favored ({home_prob:.1%} chance)"
                })
            elif max_prob == away_prob:
                tips.append({
                    'market': 'Away Win', 
                    'recommendation': 'STRONG BET',
                    'confidence': away_prob,
                    'reasoning': f"Away team heavily favored ({away_prob:.1%} chance)"
                })
        
        # Both teams to score
        btts_prob = predictions.get('both_teams_score_probability', 0.5)
        if btts_prob > 0.7:
            tips.append({
                'market': 'Both Teams To Score - Yes',
                'recommendation': 'GOOD BET',
                'confidence': btts_prob,
                'reasoning': f"Both teams likely to score ({btts_prob:.1%} probability)"
            })
        elif btts_prob < 0.3:
            tips.append({
                'market': 'Both Teams To Score - No',
                'recommendation': 'GOOD BET',
                'confidence': 1.0 - btts_prob,
                'reasoning': f"Clean sheet likely ({(1-btts_prob):.1%} probability)"
            })
        
        return tips

    def _calculate_overall_confidence(self, predictions: Dict[str, Any]) -> str:
        """Calculate overall confidence rating"""
        model_confidence = predictions.get('model_confidence', 0.5)
        
        if model_confidence > 0.8:
            return "VERY HIGH"
        elif model_confidence > 0.65:
            return "HIGH"
        elif model_confidence > 0.5:
            return "MEDIUM"
        else:
            return "LOW"

    def _extract_key_factors(self, predictions: Dict[str, Any]) -> List[str]:
        """Extract key factors influencing the prediction"""
        factors = []
        
        feature_importance = predictions.get('feature_importance', {})
        if feature_importance:
            # Get top 3 most important features
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for feature, importance in sorted_features:
                if 'home_goals_per_game' in feature:
                    factors.append("Home team's attacking strength")
                elif 'away_goals_per_game' in feature:
                    factors.append("Away team's attacking ability")
                elif 'form_score' in feature:
                    factors.append("Recent form and momentum")
                elif 'position_difference' in feature:
                    factors.append("League position disparity")
                elif 'home_advantage' in feature:
                    factors.append("Home advantage factor")
        
        # Add default factors if none extracted
        if not factors:
            factors = [
                "Team strength comparison", 
                "Recent performance trends",
                "Head-to-head history"
            ]
        
        return factors[:3]  # Return top 3 factors

    def _generate_simple_prediction(self, features_data: Dict, home_team: str, away_team: str) -> Dict[str, Any]:
        """Generate simple prediction when advanced ML is not available"""
        features = features_data.get('features', {})
        
        home_goals_expected = features.get('home_goals_per_game', 1.5) * features.get('home_advantage_multiplier', 1.1)
        away_goals_expected = features.get('away_goals_per_game', 1.5)
        
        total_goals = home_goals_expected + away_goals_expected
        
        return {
            'total_goals': round(total_goals, 2),
            'total_goals_confidence': 0.6,
            'first_half_goals': round(total_goals * 0.45, 2),
            'first_half_goals_confidence': 0.55,
            'home_win_probability': 0.45 if home_goals_expected > away_goals_expected else 0.35,
            'away_win_probability': 0.35 if home_goals_expected > away_goals_expected else 0.45,
            'draw_probability': 0.25,
            'match_result': 'H' if home_goals_expected > away_goals_expected * 1.1 else 'A' if away_goals_expected > home_goals_expected * 1.1 else 'D',
            'over_2_5_goals_probability': 0.65 if total_goals > 2.5 else 0.35,
            'both_teams_score_probability': 0.6,
            'algorithm': 'enhanced_heuristic',
            'model_confidence': 0.55,
            'match_context': {
                'home_team': home_team,
                'away_team': away_team,
                'data_source': 'real_epl_enhanced'
            }
        }

    def _get_default_prediction(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """Default prediction when all else fails"""
        return {
            'total_goals': 2.5,
            'total_goals_confidence': 0.5,
            'first_half_goals': 1.1,
            'first_half_goals_confidence': 0.5,
            'home_win_probability': 0.4,
            'away_win_probability': 0.35,
            'draw_probability': 0.25,
            'match_result': 'H',
            'over_2_5_goals_probability': 0.5,
            'both_teams_score_probability': 0.55,
            'algorithm': 'default_fallback',
            'model_confidence': 0.4,
            'betting_tips': [],
            'confidence_rating': 'LOW',
            'key_factors': ['Limited data available'],
            'match_context': {
                'home_team': home_team,
                'away_team': away_team,
                'data_source': 'fallback'
            }
        }

    def train_advanced_models(self, force_retrain: bool = False) -> Dict[str, Any]:
        """Train the advanced ML models"""
        try:
            if self.advanced_predictor.is_trained and not force_retrain:
                logger.info("Models already trained. Use force_retrain=True to retrain.")
                return {'status': 'already_trained', 'performance': self.advanced_predictor.get_model_performance()}
            
            logger.info("Starting advanced model training...")
            start_time = datetime.now()
            
            # Train models
            self.advanced_predictor.train_models()
            
            end_time = datetime.now()
            training_duration = (end_time - start_time).total_seconds()
            
            performance = self.advanced_predictor.get_model_performance()
            
            return {
                'status': 'training_completed',
                'training_duration_seconds': training_duration,
                'performance': performance,
                'training_completed_at': end_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error training advanced models: {e}")
            return {
                'status': 'training_failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def evaluate_prediction(self, prediction_id: int) -> Dict[str, Any]:
        """Evaluate prediction accuracy after match completion"""
        prediction = self.db.query(Prediction).filter(Prediction.id == prediction_id).first()
        if not prediction:
            raise ValueError(f"Prediction with ID {prediction_id} not found")

        match = prediction.match
        if match.status != "finished":
            raise ValueError("Cannot evaluate prediction for unfinished match")

        # Calculate accuracy scores
        accuracy_results = {}

        # Total goals accuracy
        if prediction.total_goals_prediction and match.total_goals is not None:
            goals_diff = abs(prediction.total_goals_prediction - match.total_goals)
            accuracy_results['goals_accuracy'] = max(0, 1 - (goals_diff / 10))  # Scale accuracy

        # First half goals accuracy
        if prediction.first_half_goals_prediction and match.total_goals_ht is not None:
            ht_goals_diff = abs(prediction.first_half_goals_prediction - match.total_goals_ht)
            accuracy_results['ht_goals_accuracy'] = max(0, 1 - (ht_goals_diff / 5))

        # Match result accuracy
        if prediction.match_result_prediction and match.result:
            accuracy_results['result_accuracy'] = 1.0 if prediction.match_result_prediction == match.result else 0.0

        # Overall accuracy score
        scores = list(accuracy_results.values())
        overall_accuracy = sum(scores) / len(scores) if scores else 0.0

        # Update prediction with accuracy information
        prediction.accuracy_score = overall_accuracy
        prediction.is_accurate = overall_accuracy > 0.7  # Consider accurate if > 70%

        self.db.commit()

        logger.info(f"Evaluated prediction {prediction_id} with overall accuracy {overall_accuracy}")
        return accuracy_results

    def get_prediction_statistics(self, model_version: str = None) -> Dict[str, Any]:
        """Get overall prediction accuracy statistics"""
        query = self.db.query(Prediction).filter(Prediction.is_accurate.isnot(None))
        
        if model_version:
            query = query.filter(Prediction.model_version == model_version)

        predictions = query.all()
        
        if not predictions:
            return {"message": "No evaluated predictions found"}

        total_predictions = len(predictions)
        accurate_predictions = len([p for p in predictions if p.is_accurate])
        
        avg_accuracy = sum([p.accuracy_score for p in predictions if p.accuracy_score]) / total_predictions

        return {
            "total_predictions": total_predictions,
            "accurate_predictions": accurate_predictions,
            "accuracy_rate": accurate_predictions / total_predictions,
            "average_accuracy_score": avg_accuracy,
            "model_version": model_version or "all"
        }