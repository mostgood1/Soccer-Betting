"""
Historical ML Predictor - Enhanced with 3+ Years of Data
Uses comprehensive historical data for superior predictive accuracy
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import joblib
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import asyncio
from pathlib import Path

from ..services.historical_epl_service import historical_epl_service

logger = logging.getLogger(__name__)


class HistoricalMLPredictor:
    """
    Advanced ML Predictor using 3+ years of historical EPL data
    Provides superior accuracy through comprehensive training
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.model_dir = Path("models/historical")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Training data
        self.historical_data = None
        self.feature_columns = []
        self.is_trained = False

        # Model performance tracking
        self.training_scores = {}
        self.cross_val_scores = {}

    async def load_historical_data_and_train(self):
        """Load comprehensive historical data and train models"""
        try:
            logger.info("🚀 Starting comprehensive historical ML training...")

            # Get comprehensive historical data
            all_seasons_data = (
                await historical_epl_service.get_comprehensive_historical_data()
            )

            if not all_seasons_data:
                logger.error("❌ No historical data available for training")
                return False

            # Prepare training dataset
            training_df = self._prepare_comprehensive_training_data(all_seasons_data)

            if training_df.empty:
                logger.error("❌ Failed to prepare training data")
                return False

            logger.info(
                f"📊 Training dataset: {len(training_df)} matches across {len(all_seasons_data)} seasons"
            )

            # Train all models
            success = await self._train_comprehensive_models(training_df)

            if success:
                self.is_trained = True
                logger.info("✅ Historical ML training completed successfully")
                return True
            else:
                logger.error("❌ Historical ML training failed")
                return False

        except Exception as e:
            logger.error(f"❌ Historical ML training error: {e}")
            return False

    def _prepare_comprehensive_training_data(
        self, all_seasons_data: Dict
    ) -> pd.DataFrame:
        """Prepare comprehensive training data from all seasons"""
        training_data = []

        for season, season_data in all_seasons_data.items():
            logger.info(f"Processing {season} data...")

            matches = season_data.matches
            teams_data = {team["name"]: team for team in season_data.teams}
            players_data = season_data.players

            for match in matches:
                try:
                    features = self._extract_comprehensive_features(
                        match, teams_data, players_data, season
                    )
                    if features:
                        training_data.append(features)
                except Exception as e:
                    logger.warning(f"Failed to extract features for match: {e}")

        df = pd.DataFrame(training_data)

        if not df.empty:
            # Clean and validate data
            df = self._clean_training_data(df)
            logger.info(
                f"✅ Prepared {len(df)} training samples with {len(df.columns)} features"
            )

        return df

    def _extract_comprehensive_features(
        self, match: Dict, teams_data: Dict, players_data: List[Dict], season: str
    ) -> Optional[Dict]:
        """Extract comprehensive features for ML training"""
        try:
            home_team = match["home_team"]
            away_team = match["away_team"]

            # Get team data
            home_data = teams_data.get(home_team, {})
            away_data = teams_data.get(away_team, {})

            # Calculate team strengths (goals per match ratios)
            home_strength = self._calculate_team_strength(home_data)
            away_strength = self._calculate_team_strength(away_data)

            # Get historical head-to-head (simulated for historical seasons)
            h2h_stats = self._calculate_h2h_stats(home_team, away_team, season)

            # Player impact factors
            home_top_scorer_goals = self._get_team_top_scorer_goals(
                home_team, players_data
            )
            away_top_scorer_goals = self._get_team_top_scorer_goals(
                away_team, players_data
            )

            features = {
                # Target variables
                "total_goals": match["total_goals"],
                "home_score": match["home_score"],
                "away_score": match["away_score"],
                "result": match["result"],  # home_win, away_win, draw
                "over_2_5": 1 if match["total_goals"] > 2.5 else 0,
                "btts": 1 if match["home_score"] > 0 and match["away_score"] > 0 else 0,
                # Team features
                "home_team_encoded": hash(home_team) % 1000,  # Simple encoding
                "away_team_encoded": hash(away_team) % 1000,
                "home_strength": home_strength,
                "away_strength": away_strength,
                "strength_difference": home_strength - away_strength,
                # Team statistics
                "home_goals_for": home_data.get("goals_for", 0),
                "home_goals_against": home_data.get("goals_against", 0),
                "away_goals_for": away_data.get("goals_for", 0),
                "away_goals_against": away_data.get("goals_against", 0),
                "home_wins": home_data.get("wins", 0),
                "home_draws": home_data.get("draws", 0),
                "home_losses": home_data.get("losses", 0),
                "away_wins": away_data.get("wins", 0),
                "away_draws": away_data.get("draws", 0),
                "away_losses": away_data.get("losses", 0),
                # Goal averages
                "home_avg_goals_for": home_data.get("goals_for", 0)
                / max(home_data.get("matches_played", 1), 1),
                "home_avg_goals_against": home_data.get("goals_against", 0)
                / max(home_data.get("matches_played", 1), 1),
                "away_avg_goals_for": away_data.get("goals_for", 0)
                / max(away_data.get("matches_played", 1), 1),
                "away_avg_goals_against": away_data.get("goals_against", 0)
                / max(away_data.get("matches_played", 1), 1),
                # Head-to-head features
                "h2h_home_wins": h2h_stats["home_wins"],
                "h2h_away_wins": h2h_stats["away_wins"],
                "h2h_draws": h2h_stats["draws"],
                "h2h_avg_total_goals": h2h_stats["avg_total_goals"],
                # Player impact
                "home_top_scorer_goals": home_top_scorer_goals,
                "away_top_scorer_goals": away_top_scorer_goals,
                "scorer_goals_difference": home_top_scorer_goals
                - away_top_scorer_goals,
                # Season context
                "season_encoded": self._encode_season(season),
                "matchday": match.get("matchday", 1),
                # Advanced metrics
                "home_points_per_match": home_data.get("points", 0)
                / max(home_data.get("matches_played", 1), 1),
                "away_points_per_match": away_data.get("points", 0)
                / max(away_data.get("matches_played", 1), 1),
                "goal_difference_gap": home_data.get("goal_difference", 0)
                - away_data.get("goal_difference", 0),
                # Form indicators (simulated)
                "home_recent_form": np.random.uniform(0, 1) * home_strength,
                "away_recent_form": np.random.uniform(0, 1) * away_strength,
            }

            return features

        except Exception as e:
            logger.warning(
                f"Feature extraction failed for {home_team} vs {away_team}: {e}"
            )
            return None

    def _calculate_team_strength(self, team_data: Dict) -> float:
        """Calculate team strength score"""
        if not team_data:
            return 0.5

        matches_played = max(team_data.get("matches_played", 1), 1)
        goals_for = team_data.get("goals_for", 0)
        goals_against = team_data.get("goals_against", 0)
        points = team_data.get("points", 0)

        # Normalized strength score
        attack_strength = goals_for / matches_played / 2.5  # Normalize by average goals
        defense_strength = max(0, (2.5 - (goals_against / matches_played)) / 2.5)
        points_strength = (points / matches_played) / 3.0  # Max 3 points per match

        strength = (attack_strength + defense_strength + points_strength) / 3
        return max(0, min(1, strength))  # Keep between 0 and 1

    def _calculate_h2h_stats(self, home_team: str, away_team: str, season: str) -> Dict:
        """Calculate head-to-head statistics (simulated for historical data)"""
        # Generate realistic H2H stats based on team names
        team_hash = hash(f"{home_team}_{away_team}") % 1000

        total_matches = max(3, team_hash % 10)  # 3-10 historical matches
        home_wins = max(0, (team_hash % 5))
        away_wins = max(0, ((team_hash + 123) % 5))
        draws = max(0, total_matches - home_wins - away_wins)

        # Ensure valid distribution
        if home_wins + away_wins + draws > total_matches:
            draws = max(0, total_matches - home_wins - away_wins)

        avg_goals = 2.0 + (team_hash % 20) / 10  # 2.0 - 4.0 goals per match

        return {
            "home_wins": home_wins,
            "away_wins": away_wins,
            "draws": draws,
            "total_matches": total_matches,
            "avg_total_goals": avg_goals,
        }

    def _get_team_top_scorer_goals(
        self, team_name: str, players_data: List[Dict]
    ) -> int:
        """Get top scorer goals for a team"""
        team_players = [p for p in players_data if p.get("team") == team_name]
        if not team_players:
            return 0

        max_goals = max([p.get("goals", 0) for p in team_players], default=0)
        return max_goals

    def _encode_season(self, season: str) -> int:
        """Encode season as integer"""
        season_map = {"2022-23": 1, "2023-24": 2, "2024-25": 3, "2025-26": 4}
        return season_map.get(season, 4)

    def _clean_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate training data"""
        # Remove rows with missing target variables
        df = df.dropna(subset=["total_goals", "result"])

        # Fill missing features with defaults
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)

        # Remove extreme outliers
        df = df[df["total_goals"] <= 10]  # Reasonable max goals

        # Ensure categorical targets are properly encoded
        result_mapping = {"home_win": 0, "away_win": 1, "draw": 2}
        df["result_encoded"] = df["result"].map(result_mapping).fillna(2)

        return df

    async def _train_comprehensive_models(self, training_df: pd.DataFrame) -> bool:
        """Train comprehensive ML models"""
        try:
            # Prepare features and targets
            feature_cols = [
                col
                for col in training_df.columns
                if col
                not in [
                    "total_goals",
                    "home_score",
                    "away_score",
                    "result",
                    "over_2_5",
                    "btts",
                ]
            ]

            X = training_df[feature_cols]
            self.feature_columns = feature_cols

            # Scale features
            self.scalers["main"] = StandardScaler()
            X_scaled = self.scalers["main"].fit_transform(X)

            # Train models for different targets
            models_trained = 0

            # 1. Total Goals Prediction (Regression)
            y_goals = training_df["total_goals"]
            success = await self._train_regression_models(
                X_scaled, y_goals, "total_goals"
            )
            if success:
                models_trained += 1

            # 2. Match Result Prediction (Classification)
            y_result = training_df["result_encoded"]
            success = await self._train_classification_models(
                X_scaled, y_result, "match_result"
            )
            if success:
                models_trained += 1

            # 3. Over/Under 2.5 Goals (Binary Classification)
            y_over25 = training_df["over_2_5"]
            success = await self._train_binary_models(X_scaled, y_over25, "over_2_5")
            if success:
                models_trained += 1

            # 4. Both Teams to Score (Binary Classification)
            y_btts = training_df["btts"]
            success = await self._train_binary_models(X_scaled, y_btts, "btts")
            if success:
                models_trained += 1

            # Save models
            self._save_models()

            logger.info(f"✅ Successfully trained {models_trained}/4 model categories")
            return models_trained >= 3  # Success if at least 3/4 model types trained

        except Exception as e:
            logger.error(f"❌ Model training error: {e}")
            return False

    async def _train_regression_models(
        self, X: np.ndarray, y: pd.Series, target_name: str
    ) -> bool:
        """Train regression models for continuous targets"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Random Forest Regressor
            rf_model = RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            )
            rf_model.fit(X_train, y_train)

            # Gradient Boosting Regressor
            gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gb_model.fit(X_train, y_train)

            # Evaluate models
            rf_score = rf_model.score(X_test, y_test)
            gb_score = gb_model.score(X_test, y_test)

            # Choose best model
            if rf_score >= gb_score:
                self.models[target_name] = rf_model
                best_score = rf_score
                model_type = "RandomForest"
            else:
                self.models[target_name] = gb_model
                best_score = gb_score
                model_type = "GradientBoosting"

            self.training_scores[target_name] = {
                "r2_score": best_score,
                "model_type": model_type,
                "mse": mean_squared_error(
                    y_test, self.models[target_name].predict(X_test)
                ),
            }

            logger.info(
                f"✅ {target_name} model trained: {model_type} (R² = {best_score:.3f})"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Regression training failed for {target_name}: {e}")
            return False

    async def _train_classification_models(
        self, X: np.ndarray, y: pd.Series, target_name: str
    ) -> bool:
        """Train classification models"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Random Forest Classifier
            rf_model = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            )
            rf_model.fit(X_train, y_train)

            # Logistic Regression
            lr_model = LogisticRegression(random_state=42, max_iter=1000)
            lr_model.fit(X_train, y_train)

            # Evaluate models
            rf_score = rf_model.score(X_test, y_test)
            lr_score = lr_model.score(X_test, y_test)

            # Choose best model
            if rf_score >= lr_score:
                self.models[target_name] = rf_model
                best_score = rf_score
                model_type = "RandomForest"
            else:
                self.models[target_name] = lr_model
                best_score = lr_score
                model_type = "LogisticRegression"

            self.training_scores[target_name] = {
                "accuracy": best_score,
                "model_type": model_type,
            }

            logger.info(
                f"✅ {target_name} model trained: {model_type} (Accuracy = {best_score:.3f})"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Classification training failed for {target_name}: {e}")
            return False

    async def _train_binary_models(
        self, X: np.ndarray, y: pd.Series, target_name: str
    ) -> bool:
        """Train binary classification models"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Random Forest for binary classification
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)

            score = model.score(X_test, y_test)

            self.models[target_name] = model
            self.training_scores[target_name] = {
                "accuracy": score,
                "model_type": "RandomForest",
            }

            logger.info(
                f"✅ {target_name} model trained: RandomForest (Accuracy = {score:.3f})"
            )
            return True

        except Exception as e:
            logger.error(
                f"❌ Binary classification training failed for {target_name}: {e}"
            )
            return False

    def _save_models(self):
        """Save trained models and scalers"""
        try:
            # Save models
            for model_name, model in self.models.items():
                model_file = self.model_dir / f"{model_name}_model.pkl"
                joblib.dump(model, model_file)

            # Save scalers
            for scaler_name, scaler in self.scalers.items():
                scaler_file = self.model_dir / f"{scaler_name}_scaler.pkl"
                joblib.dump(scaler, scaler_file)

            # Save feature columns and training info
            info_file = self.model_dir / "model_info.pkl"
            model_info = {
                "feature_columns": self.feature_columns,
                "training_scores": self.training_scores,
                "is_trained": self.is_trained,
                "last_training": datetime.now().isoformat(),
            }
            joblib.dump(model_info, info_file)

            logger.info(f"💾 Saved {len(self.models)} models to {self.model_dir}")

        except Exception as e:
            logger.error(f"❌ Failed to save models: {e}")

    def load_models(self) -> bool:
        """Load pre-trained models"""
        try:
            info_file = self.model_dir / "model_info.pkl"
            if not info_file.exists():
                return False

            model_info = joblib.load(info_file)
            self.feature_columns = model_info["feature_columns"]
            self.training_scores = model_info["training_scores"]
            self.is_trained = model_info.get("is_trained", False)

            # Load models
            for model_name in ["total_goals", "match_result", "over_2_5", "btts"]:
                model_file = self.model_dir / f"{model_name}_model.pkl"
                if model_file.exists():
                    self.models[model_name] = joblib.load(model_file)

            # Load scalers
            scaler_file = self.model_dir / "main_scaler.pkl"
            if scaler_file.exists():
                self.scalers["main"] = joblib.load(scaler_file)

            logger.info(f"✅ Loaded {len(self.models)} pre-trained models")
            return len(self.models) >= 3

        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            return False

    async def predict_match_comprehensive(
        self, home_team: str, away_team: str
    ) -> Dict[str, Any]:
        """Generate comprehensive match prediction using historical models"""
        try:
            if not self.is_trained:
                # Try to load pre-trained models
                if not self.load_models():
                    logger.warning(
                        "⚠️ No trained models available, using heuristic predictions"
                    )
                    return self._generate_heuristic_prediction(home_team, away_team)

            # Get current season data for feature extraction
            current_season_data = await historical_epl_service.get_season_data(
                "2025-26"
            )

            # Prepare prediction features
            features = await self._prepare_prediction_features(
                home_team, away_team, current_season_data
            )

            if not features:
                return self._generate_heuristic_prediction(home_team, away_team)

            # Generate predictions
            predictions = {}

            # Total Goals
            if "total_goals" in self.models:
                total_goals = float(self.models["total_goals"].predict([features])[0])
                predictions["total_goals"] = max(0, total_goals)
                predictions["total_goals_confidence"] = min(
                    0.9,
                    self.training_scores.get("total_goals", {}).get("r2_score", 0.5),
                )

            # Match Result
            if "match_result" in self.models:
                result_probs = self.models["match_result"].predict_proba([features])[0]
                predictions["home_win_probability"] = float(result_probs[0])
                predictions["away_win_probability"] = float(result_probs[1])
                predictions["draw_probability"] = float(result_probs[2])
                predictions["match_result"] = ["home_win", "away_win", "draw"][
                    np.argmax(result_probs)
                ]
                predictions["match_result_confidence"] = float(max(result_probs))

            # Over/Under 2.5
            if "over_2_5" in self.models:
                over25_prob = float(
                    self.models["over_2_5"].predict_proba([features])[0][1]
                )
                predictions["over_2_5_goals_probability"] = over25_prob

            # Both Teams Score
            if "btts" in self.models:
                btts_prob = float(self.models["btts"].predict_proba([features])[0][1])
                predictions["both_teams_score_probability"] = btts_prob

            # Generate betting tips
            predictions["betting_tips"] = self._generate_comprehensive_betting_tips(
                predictions
            )

            # Add model info
            predictions["prediction_method"] = "historical_ml_ensemble"
            predictions["models_used"] = list(self.models.keys())
            predictions["training_data_seasons"] = [
                "2022-23",
                "2023-24",
                "2024-25",
                "2025-26",
            ]

            return predictions

        except Exception as e:
            logger.error(f"❌ Comprehensive prediction failed: {e}")
            return self._generate_heuristic_prediction(home_team, away_team)

    async def _prepare_prediction_features(
        self, home_team: str, away_team: str, season_data
    ) -> Optional[List[float]]:
        """Prepare features for prediction"""
        try:
            # Get team data from current season
            teams_data = {team["name"]: team for team in season_data.teams}
            players_data = season_data.players

            # Create a dummy match for feature extraction
            dummy_match = {
                "home_team": home_team,
                "away_team": away_team,
                "home_score": 0,  # Will not be used for prediction
                "away_score": 0,  # Will not be used for prediction
                "total_goals": 0,  # Will not be used for prediction
                "result": "draw",  # Will not be used for prediction
                "matchday": 20,  # Mid-season
            }

            # Extract features
            feature_dict = self._extract_comprehensive_features(
                dummy_match, teams_data, players_data, "2025-26"
            )

            if not feature_dict:
                return None

            # Convert to feature vector in correct order
            features = []
            for col in self.feature_columns:
                if col in feature_dict:
                    features.append(feature_dict[col])
                else:
                    features.append(0)  # Default for missing features

            # Scale features
            if "main" in self.scalers:
                features = self.scalers["main"].transform([features])[0]

            return features.tolist()

        except Exception as e:
            logger.error(f"❌ Feature preparation failed: {e}")
            return None

    def _generate_heuristic_prediction(
        self, home_team: str, away_team: str
    ) -> Dict[str, Any]:
        """Generate heuristic prediction when ML models are not available"""
        # Simple heuristic based on team names
        team_strength_map = {
            "Manchester City": 0.9,
            "Arsenal": 0.85,
            "Liverpool": 0.85,
            "Manchester United": 0.75,
            "Chelsea": 0.75,
            "Tottenham Hotspur": 0.7,
            "Newcastle United": 0.7,
            "Brighton & Hove Albion": 0.65,
            "Aston Villa": 0.65,
            "West Ham United": 0.6,
        }

        home_strength = team_strength_map.get(home_team, 0.5)
        away_strength = team_strength_map.get(away_team, 0.5)

        # Home advantage
        home_strength += 0.1

        # Predictions
        total_goals = 2.3 + (home_strength + away_strength - 1.0)

        strength_diff = home_strength - away_strength
        if strength_diff > 0.2:
            home_win_prob = 0.6
            draw_prob = 0.25
            away_win_prob = 0.15
        elif strength_diff < -0.2:
            home_win_prob = 0.15
            draw_prob = 0.25
            away_win_prob = 0.6
        else:
            home_win_prob = 0.4
            draw_prob = 0.3
            away_win_prob = 0.3

        return {
            "total_goals": total_goals,
            "home_win_probability": home_win_prob,
            "away_win_probability": away_win_prob,
            "draw_probability": draw_prob,
            "match_result": "home_win"
            if home_win_prob > max(draw_prob, away_win_prob)
            else ("away_win" if away_win_prob > draw_prob else "draw"),
            "over_2_5_goals_probability": 0.6 if total_goals > 2.5 else 0.4,
            "both_teams_score_probability": 0.65,
            "total_goals_confidence": 0.5,
            "match_result_confidence": max(home_win_prob, draw_prob, away_win_prob),
            "betting_tips": [
                {
                    "market": "Match Result",
                    "recommendation": "Home Win" if home_win_prob > 0.5 else "Draw",
                    "reasoning": "Based on heuristic team strength",
                },
                {
                    "market": "Total Goals",
                    "recommendation": f"Over 2.5" if total_goals > 2.5 else "Under 2.5",
                    "reasoning": f"Predicted goals: {total_goals:.1f}",
                },
            ],
            "prediction_method": "heuristic_fallback",
            "models_used": [],
            "training_data_seasons": [],
        }

    def _generate_comprehensive_betting_tips(self, predictions: Dict) -> List[Dict]:
        """Generate comprehensive betting recommendations"""
        tips = []

        # Match Result tip
        result_probs = [
            predictions.get("home_win_probability", 0),
            predictions.get("away_win_probability", 0),
            predictions.get("draw_probability", 0),
        ]
        best_result_idx = np.argmax(result_probs)
        result_names = ["Home Win", "Away Win", "Draw"]
        confidence = result_probs[best_result_idx]

        if confidence > 0.5:
            tips.append(
                {
                    "market": "Match Result",
                    "recommendation": result_names[best_result_idx],
                    "reasoning": f"ML model predicts {confidence:.0%} probability",
                }
            )

        # Total Goals tip
        total_goals = predictions.get("total_goals", 2.5)
        if total_goals > 2.7:
            tips.append(
                {
                    "market": "Total Goals",
                    "recommendation": "Over 2.5",
                    "reasoning": f"Predicted {total_goals:.1f} goals",
                }
            )
        elif total_goals < 2.3:
            tips.append(
                {
                    "market": "Total Goals",
                    "recommendation": "Under 2.5",
                    "reasoning": f"Predicted {total_goals:.1f} goals",
                }
            )

        # Both Teams Score tip
        btts_prob = predictions.get("both_teams_score_probability", 0.5)
        if btts_prob > 0.6:
            tips.append(
                {
                    "market": "Both Teams Score",
                    "recommendation": "Yes",
                    "reasoning": f"{btts_prob:.0%} probability both teams score",
                }
            )
        elif btts_prob < 0.4:
            tips.append(
                {
                    "market": "Both Teams Score",
                    "recommendation": "No",
                    "reasoning": f"Only {btts_prob:.0%} probability both teams score",
                }
            )

        return tips

    def get_model_performance(self) -> Dict:
        """Get model performance information"""
        return {
            "is_trained": self.is_trained,
            "model_types": list(self.models.keys()),
            "training_accuracy": self.training_scores,
            "models_count": len(self.models),
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
        }


# Global instance
historical_ml_predictor = HistoricalMLPredictor()
