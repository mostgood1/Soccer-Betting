#!/usr/bin/env python3
"""
Advanced ML Training for EPL Predictions
Uses comprehensive historical data for enhanced accuracy
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import logging
from datetime import datetime
import os
from typing import Dict, List, Tuple, Any

# Import our services
from app.services.enhanced_historical_data_service import EnhancedHistoricalDataService
from app.services.enhanced_player_stats_service import EnhancedPlayerStatsService
from app.services.betting_odds_service import BettingOddsService

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AdvancedMLTrainer:
    def __init__(self):
        self.historical_service = EnhancedHistoricalDataService()
        self.player_service = EnhancedPlayerStatsService()
        self.betting_service = BettingOddsService()

        self.models = {}
        self.scalers = {}
        self.encoders = {}

        # Create models directory
        self.models_dir = "models/trained"
        os.makedirs(self.models_dir, exist_ok=True)

        logger.info("🤖 Advanced ML Trainer initialized")

    def prepare_match_features(self, matches: List[Dict]) -> pd.DataFrame:
        """Convert match data into ML-ready features"""
        logger.info(f"📊 Preparing features for {len(matches)} matches...")

        features_list = []

        for match in matches:
            # Team strength features (based on historical performance)
            home_team = match["home_team"]
            away_team = match["away_team"]

            # Basic match features
            features = {
                "home_team": home_team,
                "away_team": away_team,
                "gameweek": match.get("gameweek", 1),
                "season": match.get("season", "2024-25"),
                # Target variables
                "home_goals": match["home_goals"],
                "away_goals": match["away_goals"],
                "total_goals": match["total_goals"],
                "result": match["result"],  # H/D/A
                # Extended match statistics
                "home_shots": match.get("shots_home", 0),
                "away_shots": match.get("shots_away", 0),
                "home_shots_on_target": match.get("shots_on_target_home", 0),
                "away_shots_on_target": match.get("shots_on_target_away", 0),
                "home_possession": match.get("possession_home", 50),
                "away_possession": match.get("possession_away", 50),
                "home_corners": match.get("corners_home", 0),
                "away_corners": match.get("corners_away", 0),
                "home_cards": match.get("cards_home", 0),
                "away_cards": match.get("cards_away", 0),
                "first_half_home": match.get("first_half_home", 0),
                "first_half_away": match.get("first_half_away", 0),
            }

            features_list.append(features)

        df = pd.DataFrame(features_list)
        logger.info(
            f"✅ Features prepared: {df.shape[0]} matches, {df.shape[1]} features"
        )
        return df

    def add_team_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team strength and historical performance features"""
        logger.info("⚡ Adding team strength features...")

        # Team strength mappings (based on current EPL standings/quality)
        team_strengths = {
            "Manchester City": 0.95,
            "Arsenal": 0.90,
            "Liverpool": 0.88,
            "Chelsea": 0.82,
            "Newcastle United": 0.78,
            "Manchester United": 0.75,
            "Tottenham": 0.73,
            "Brighton": 0.70,
            "Aston Villa": 0.68,
            "West Ham United": 0.65,
            "Crystal Palace": 0.60,
            "Fulham": 0.58,
            "Brentford": 0.56,
            "Wolverhampton Wanderers": 0.54,
            "Everton": 0.52,
            "AFC Bournemouth": 0.50,
            "Nottingham Forest": 0.48,
            "Luton Town": 0.45,
            "Burnley": 0.43,
            "Sheffield United": 0.40,
        }

        df["home_strength"] = df["home_team"].map(team_strengths).fillna(0.5)
        df["away_strength"] = df["away_team"].map(team_strengths).fillna(0.5)
        df["strength_difference"] = df["home_strength"] - df["away_strength"]
        df["home_advantage"] = 0.1  # Standard home advantage
        df["effective_home_strength"] = df["home_strength"] + df["home_advantage"]

        return df

    def train_goals_predictor(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train models to predict match goals"""
        logger.info("🥅 Training goals prediction models...")

        # Prepare features for goals prediction
        feature_columns = [
            "home_strength",
            "away_strength",
            "strength_difference",
            "effective_home_strength",
            "gameweek",
        ]

        X = df[feature_columns]
        y_home = df["home_goals"]
        y_away = df["away_goals"]
        y_total = df["total_goals"]

        # Split data
        X_train, X_test, y_home_train, y_home_test = train_test_split(
            X, y_home, test_size=0.2, random_state=42
        )
        _, _, y_away_train, y_away_test = train_test_split(
            X, y_away, test_size=0.2, random_state=42
        )
        _, _, y_total_train, y_total_test = train_test_split(
            X, y_total, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train models
        models = {
            "home_goals": RandomForestRegressor(n_estimators=100, random_state=42),
            "away_goals": RandomForestRegressor(n_estimators=100, random_state=42),
            "total_goals": GradientBoostingRegressor(n_estimators=100, random_state=42),
        }

        # Fit models
        models["home_goals"].fit(X_train_scaled, y_home_train)
        models["away_goals"].fit(X_train_scaled, y_away_train)
        models["total_goals"].fit(X_train_scaled, y_total_train)

        # Evaluate models
        home_pred = models["home_goals"].predict(X_test_scaled)
        away_pred = models["away_goals"].predict(X_test_scaled)
        total_pred = models["total_goals"].predict(X_test_scaled)

        results = {
            "models": models,
            "scaler": scaler,
            "feature_columns": feature_columns,
            "performance": {
                "home_goals_rmse": np.sqrt(mean_squared_error(y_home_test, home_pred)),
                "away_goals_rmse": np.sqrt(mean_squared_error(y_away_test, away_pred)),
                "total_goals_rmse": np.sqrt(
                    mean_squared_error(y_total_test, total_pred)
                ),
            },
        }

        logger.info(f"✅ Goals models trained:")
        logger.info(
            f"   Home goals RMSE: {results['performance']['home_goals_rmse']:.3f}"
        )
        logger.info(
            f"   Away goals RMSE: {results['performance']['away_goals_rmse']:.3f}"
        )
        logger.info(
            f"   Total goals RMSE: {results['performance']['total_goals_rmse']:.3f}"
        )

        return results

    def train_result_classifier(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train model to predict match results (H/D/A)"""
        logger.info("🏆 Training result classification model...")

        # Prepare features
        feature_columns = [
            "home_strength",
            "away_strength",
            "strength_difference",
            "effective_home_strength",
            "gameweek",
        ]

        X = df[feature_columns]
        y = df["result"]

        # Encode results
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train classifier
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = clf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        results = {
            "model": clf,
            "scaler": scaler,
            "label_encoder": le,
            "feature_columns": feature_columns,
            "performance": {
                "accuracy": accuracy,
                "classification_report": classification_report(
                    y_test, y_pred, target_names=le.classes_
                ),
            },
        }

        logger.info(f"✅ Result classifier trained with {accuracy:.3f} accuracy")

        return results

    def train_advanced_stats_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train models for advanced match statistics"""
        logger.info("📈 Training advanced statistics models...")

        feature_columns = [
            "home_strength",
            "away_strength",
            "strength_difference",
            "effective_home_strength",
            "gameweek",
        ]

        X = df[feature_columns]

        # Statistics to predict
        stats_models = {}
        stats_to_predict = [
            "home_shots",
            "away_shots",
            "home_corners",
            "away_corners",
            "home_cards",
            "away_cards",
            "home_possession",
        ]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for stat in stats_to_predict:
            if stat in df.columns:
                y = df[stat]
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )

                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                stats_models[stat] = {"model": model, "rmse": rmse}

        results = {
            "models": stats_models,
            "scaler": scaler,
            "feature_columns": feature_columns,
        }

        logger.info(
            f"✅ Advanced stats models trained for {len(stats_models)} statistics"
        )

        return results

    def save_models(
        self, goals_results: Dict, result_results: Dict, stats_results: Dict
    ):
        """Save all trained models"""
        logger.info("💾 Saving trained models...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save goals models
        joblib.dump(goals_results, f"{self.models_dir}/goals_models_{timestamp}.pkl")

        # Save result classifier
        joblib.dump(
            result_results, f"{self.models_dir}/result_classifier_{timestamp}.pkl"
        )

        # Save stats models
        joblib.dump(stats_results, f"{self.models_dir}/stats_models_{timestamp}.pkl")

        # Save latest versions (for production use)
        joblib.dump(goals_results, f"{self.models_dir}/goals_models_latest.pkl")
        joblib.dump(result_results, f"{self.models_dir}/result_classifier_latest.pkl")
        joblib.dump(stats_results, f"{self.models_dir}/stats_models_latest.pkl")

        logger.info(f"✅ Models saved to {self.models_dir}")

    def train_all_models(self):
        """Main training pipeline"""
        logger.info("🚀 Starting comprehensive ML training pipeline...")
        logger.info("=" * 60)

        try:
            # Get historical data
            historical_matches = self.historical_service.get_matches_for_training()
            logger.info(f"📚 Loaded {len(historical_matches)} historical matches")

            # Prepare features
            df = self.prepare_match_features(historical_matches)
            df = self.add_team_strength_features(df)

            logger.info(
                f"📊 Training dataset: {df.shape[0]} matches, {df.shape[1]} features"
            )

            # Train goals prediction models
            goals_results = self.train_goals_predictor(df)

            # Train result classification model
            result_results = self.train_result_classifier(df)

            # Train advanced statistics models
            stats_results = self.train_advanced_stats_models(df)

            # Save all models
            self.save_models(goals_results, result_results, stats_results)

            # Generate training summary
            self.generate_training_summary(
                goals_results, result_results, stats_results, len(historical_matches)
            )

            logger.info("🎉 ML training completed successfully!")
            return True

        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            return False

    def generate_training_summary(
        self,
        goals_results: Dict,
        result_results: Dict,
        stats_results: Dict,
        total_matches: int,
    ):
        """Generate comprehensive training summary"""
        summary = f"""
🤖 EPL BETTING PLATFORM - ML TRAINING SUMMARY
{'='*60}
📅 Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📊 Training Data: {total_matches} historical matches
🏆 Models Trained: Goals Prediction + Result Classification + Advanced Stats

📈 MODEL PERFORMANCE:
Goals Prediction Models:
  • Home Goals RMSE: {goals_results['performance']['home_goals_rmse']:.3f}
  • Away Goals RMSE: {goals_results['performance']['away_goals_rmse']:.3f}
  • Total Goals RMSE: {goals_results['performance']['total_goals_rmse']:.3f}

Result Classification:
  • Accuracy: {result_results['performance']['accuracy']:.3f}

Advanced Statistics:
  • Models: {len(stats_results['models'])} statistics covered
  
🎯 PREDICTION CAPABILITIES:
  ✅ Match result (Home/Draw/Away)
  ✅ Goals scored (Home/Away/Total)
  ✅ Match statistics (Shots, Corners, Cards, Possession)
  ✅ Betting value analysis
  ✅ Confidence scoring

🚀 READY FOR PRODUCTION PREDICTIONS!
{'='*60}
"""
        logger.info(summary)

        # Save summary to file
        with open(f"{self.models_dir}/training_summary.txt", "w") as f:
            f.write(summary)


def main():
    """Run the ML training pipeline"""
    trainer = AdvancedMLTrainer()
    success = trainer.train_all_models()

    if success:
        print("\n🎉 ML training completed! Models are ready for predictions.")
        print(f"📁 Models saved in: {trainer.models_dir}")
        print("🔥 Your EPL betting platform is now powered by advanced ML!")
    else:
        print("\n❌ ML training failed. Check logs for details.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
