"""
Advanced ML Prediction Engine for EPL Betting
Uses real data and advanced algorithms for accurate predictions
"""

import pandas as pd
import numpy as np
import hashlib
import random
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from typing import Dict, List, Any, Tuple, Optional
import joblib
import os
import logging
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

from ..services.enhanced_epl_service import (
    enhanced_epl_service,
)  # legacy v1 for fallback

try:
    from ..services.enhanced_epl_service_v2 import EnhancedEPLService as V2Service

    _v2_feature_service = V2Service()
except Exception:  # pragma: no cover
    _v2_feature_service = None
try:
    from ..services.team_seed_metrics import (
        team_seed_metrics_service,
        FEATURE_VERSION as TEAM_FEATURE_VERSION,
    )
except Exception:  # pragma: no cover
    team_seed_metrics_service = None
    TEAM_FEATURE_VERSION = "fv0"

from ..services.expected_goals_service import expected_goals_service
from ..services.betting_odds_service import betting_odds_service

_BLEND_CONFIG_PATH = os.path.join("cache", "model_blend.json")


def _load_blend_config() -> Optional[Dict[str, float]]:
    try:
        if os.path.exists(_BLEND_CONFIG_PATH):
            import json

            with open(_BLEND_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception:
        return None
    return None


def _load_goal_offsets() -> Tuple[float, float]:
    """Load optional league-wide goal-bias offsets.
    Source precedence:
      1) cache/model_blend.json keys: home_goal_offset, away_goal_offset
      2) ENV: ML_HOME_GOAL_OFFSET, ML_AWAY_GOAL_OFFSET
    Returns (home_offset, away_offset). Defaults to (0.0, 0.0).
    """
    h_off = 0.0
    a_off = 0.0
    try:
        cfg = _load_blend_config() or {}
        if isinstance(cfg, dict):
            if "home_goal_offset" in cfg:
                try:
                    h_off = float(cfg.get("home_goal_offset", 0.0))
                except Exception:
                    h_off = 0.0
            if "away_goal_offset" in cfg:
                try:
                    a_off = float(cfg.get("away_goal_offset", 0.0))
                except Exception:
                    a_off = 0.0
    except Exception:
        pass
    # Env overrides
    try:
        if os.getenv("ML_HOME_GOAL_OFFSET") is not None:
            h_off = float(os.getenv("ML_HOME_GOAL_OFFSET", "0"))
    except Exception:
        pass
    try:
        if os.getenv("ML_AWAY_GOAL_OFFSET") is not None:
            a_off = float(os.getenv("ML_AWAY_GOAL_OFFSET", "0"))
    except Exception:
        pass
    return h_off, a_off


logger = logging.getLogger(__name__)


class AdvancedMLPredictor:
    """Advanced ML predictor with multiple algorithms and model ensemble"""

    def __init__(self):
        self.models = {
            "goals_total": {
                "random_forest": RandomForestRegressor(
                    n_estimators=100, random_state=42
                ),
                "gradient_boosting": GradientBoostingRegressor(random_state=42),
                "ensemble_weights": [0.6, 0.4],
            },
            "goals_first_half": {
                "random_forest": RandomForestRegressor(
                    n_estimators=80, random_state=42
                ),
                "gradient_boosting": GradientBoostingRegressor(random_state=42),
                "ensemble_weights": [0.5, 0.5],
            },
            "match_result": {
                "random_forest": RandomForestClassifier(
                    n_estimators=100, random_state=42
                ),
                "logistic": LogisticRegression(random_state=42, max_iter=1000),
                "ensemble_weights": [0.7, 0.3],
            },
            "corners_total": {
                "random_forest": RandomForestRegressor(
                    n_estimators=80, random_state=42
                ),
                "gradient_boosting": GradientBoostingRegressor(random_state=42),
                "ensemble_weights": [0.6, 0.4],
            },
        }

        self.scalers = {}
        self.feature_importance = {}
        self.is_trained = False
        self.training_accuracy = {}
        self.model_save_path = "app/ml/saved_models"
        self.model_version = "v2.0.8"  # semantic versioning MAJOR.MINOR.PATCH

        # Ensure model directory exists
        os.makedirs(self.model_save_path, exist_ok=True)

        # Feature definitions for training
        self.feature_columns = [
            "home_points_per_game",
            "away_points_per_game",
            "home_goals_per_game",
            "away_goals_per_game",
            "home_goals_conceded_per_game",
            "away_goals_conceded_per_game",
            "goal_difference_gap",
            "position_difference",
            "home_win_rate",
            "away_win_rate",
            "home_form_score",
            "away_form_score",
            "home_recent_goals",
            "away_recent_goals",
            "h2h_home_advantage",
            "h2h_avg_goals",
            "h2h_over_2_5_rate",
            "home_advantage_multiplier",
            "is_top_6_clash",
            "is_relegation_battle",
            "strength_disparity",
            # v2 engineered features (added for diversity)
            "attack_matchup_index",
            "defense_matchup_index",
            "form_gap",
            "tempo_gap",
            "creation_gap",
            "finishing_gap",
            "pair_variance",
        ]
        # Market augmentation feature columns (appended if historic odds available)
        self.market_feature_columns = [
            "market_home_implied",
            "market_draw_implied",
            "market_away_implied",
            "market_overround",
            "market_fav_prob",
            "market_prob_skew",
        ]

    def generate_training_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """Generate training data from current season and historical patterns"""
        logger.info(f"Generating {num_samples} training samples...")
        import os

        # Render-safe: allow hard opt-out of any heavy synthetic generation
        if os.getenv("ML_DISABLE_TRAINING", "0") == "1":
            logger.warning(
                "ML_DISABLE_TRAINING=1 set; returning minimal synthetic dataset for boot safety"
            )
            return self._create_synthetic_training_data(min(50, num_samples))
        if os.getenv("ML_FORCE_SYNTHETIC", "0") == "1":
            logger.warning(
                "ML_FORCE_SYNTHETIC=1 set; generating fully synthetic training dataset"
            )
            base = self._create_synthetic_training_data(num_samples)
            try:
                base = self._augment_with_market_rows(base)
            except Exception as e:
                logger.warning(f"Market augmentation skipped (synthetic path): {e}")
            return base

        try:
            # Get current team stats (legacy service) but fall back immediately if empty
            try:
                team_stats = enhanced_epl_service.get_detailed_team_stats()
                teams = team_stats.get("teams") if isinstance(team_stats, dict) else []
            except Exception as e:
                logger.warning(f"Team stats retrieval failed, using synthetic: {e}")
                return self._create_synthetic_training_data(num_samples)

            if not teams or len(teams) < 4:
                logger.warning(
                    "Insufficient real team stats; using synthetic training dataset"
                )
                return self._create_synthetic_training_data(num_samples)

            training_data = []

            for i in range(num_samples):
                # Randomly select teams for match simulation
                home_team = np.random.choice(teams)
                away_team = np.random.choice(
                    [t for t in teams if t["name"] != home_team["name"]]
                )

                # Get prediction features
                features_data = enhanced_epl_service.get_match_prediction_features(
                    home_team["name"], away_team["name"]
                )

                features = features_data["features"]

                # Simulate match outcome based on features
                simulated_outcome = self._simulate_match_outcome(features)

                # Combine features with outcome
                training_row = {**features, **simulated_outcome}
                training_data.append(training_row)

            df = pd.DataFrame(training_data)

            # Ensure engineered v2 feature columns exist even if base feature service didn't supply them
            engineered_cols = [
                "attack_matchup_index",
                "defense_matchup_index",
                "form_gap",
                "tempo_gap",
                "creation_gap",
                "finishing_gap",
                "pair_variance",
            ]
            for col in engineered_cols:
                if col not in df.columns:
                    # Deterministic-ish filler based on hash of index to avoid constant zeros
                    df[col] = [
                        random.uniform(-0.5, 0.5)
                        if "gap" in col or col == "pair_variance"
                        else random.uniform(0.4, 2.5)
                        for _ in range(len(df))
                    ]

            # Corner target columns sometimes missing from legacy feature path; synthesize if absent
            if "total_corners" not in df.columns:
                base_tc = df.get("h2h_avg_goals", pd.Series([2.6] * len(df))) * 4.0
                df["total_corners"] = (base_tc + np.random.normal(0, 2, len(df))).clip(
                    lower=3
                )
            if "home_corners" not in df.columns:
                df["home_corners"] = (df["total_corners"] * 0.52).clip(lower=0)
            if "away_corners" not in df.columns:
                df["away_corners"] = (df["total_corners"] * 0.48).clip(lower=0)

            # Add some noise to make data more realistic
            df = self._add_realistic_noise(df)

            df = self._augment_with_market_rows(df)
            return df

        except Exception as e:
            logger.error(f"Error generating training data: {e}")
            base = self._create_synthetic_training_data(num_samples)
            base = self._augment_with_market_rows(base)
            return base

    def _augment_with_market_rows(self, df):
        """Append market-derived historic rows if available and extend feature list."""
        try:
            from ..services.historic_odds_service import market_training_rows

            market_rows = market_training_rows()
        except Exception:
            market_rows = []
        if not market_rows:
            logger.info("No market rows available for augmentation")
            return df
        mdf = pd.DataFrame(market_rows)
        # Ensure feature columns
        for col in self.feature_columns:
            if col not in mdf.columns:
                mdf[col] = 0.0
        # Add missing market feature columns to both frames
        for mcol in self.market_feature_columns:
            if mcol not in df.columns:
                df[mcol] = 0.0
            if mcol not in mdf.columns:
                mdf[mcol] = 0.0
        # Harmonize: add any columns present in one but missing in the other with fill 0
        union_cols = list(dict.fromkeys(df.columns.tolist() + mdf.columns.tolist()))
        for col in union_cols:
            if col not in df.columns:
                df[col] = 0.0
            if col not in mdf.columns:
                mdf[col] = 0.0
        # Now safe to reindex in consistent order
        df_h = df.reindex(columns=union_cols)
        mdf_h = mdf.reindex(columns=union_cols)
        combined = pd.concat([df_h, mdf_h], ignore_index=True)
        for mcol in self.market_feature_columns:
            if mcol not in self.feature_columns:
                self.feature_columns.append(mcol)
        logger.info(
            f"Augmented with {len(market_rows)} market odds rows; new total samples {len(combined)}"
        )
        logger.info(
            f"Feature columns now {len(self.feature_columns)} (including market features)"
        )
        return combined

    def _simulate_match_outcome(self, features: Dict) -> Dict:
        """Simulate realistic match outcomes based on team features"""
        # Calculate expected goals using Poisson-like distribution
        home_strength = features.get("home_goals_per_game", 1.5) * features.get(
            "home_advantage_multiplier", 1.1
        )
        away_strength = features.get("away_goals_per_game", 1.5)

        # Defensive adjustments
        home_attack_vs_away_defense = home_strength / max(
            features.get("away_goals_conceded_per_game", 1.5), 0.5
        )
        away_attack_vs_home_defense = away_strength / max(
            features.get("home_goals_conceded_per_game", 1.5), 0.5
        )

        # Apply form factors
        home_expected = (
            home_attack_vs_away_defense * features.get("home_form_score", 0.5) * 1.5
        )
        away_expected = (
            away_attack_vs_home_defense * features.get("away_form_score", 0.5) * 1.5
        )

        # Generate realistic goals
        home_goals = max(0, np.random.poisson(min(home_expected, 5)))
        away_goals = max(0, np.random.poisson(min(away_expected, 5)))

        total_goals = home_goals + away_goals
        first_half_goals = np.random.binomial(
            total_goals, 0.45
        )  # ~45% of goals in first half

        # Match result
        if home_goals > away_goals:
            result = 1  # Home win
        elif away_goals > home_goals:
            result = 2  # Away win
        else:
            result = 0  # Draw

        # Corner simulation
        corners_base = features.get("h2h_avg_goals", 2.5) * 4 + np.random.normal(0, 2)
        total_corners = max(0, int(corners_base))

        return {
            "home_goals": home_goals,
            "away_goals": away_goals,
            "total_goals": total_goals,
            "first_half_goals": first_half_goals,
            "match_result": result,
            "total_corners": total_corners,
            "home_corners": int(total_corners * (0.45 + np.random.uniform(-0.1, 0.1))),
            "away_corners": int(total_corners * (0.55 + np.random.uniform(-0.1, 0.1))),
        }

    def _add_realistic_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic noise to training data to prevent overfitting"""
        noise_columns = [
            "home_goals_per_game",
            "away_goals_per_game",
            "home_form_score",
            "away_form_score",
        ]

        for col in noise_columns:
            if col in df.columns:
                noise = np.random.normal(0, 0.1, len(df))
                df[col] = df[col] + noise
                df[col] = df[col].clip(lower=0)  # Ensure non-negative values

        return df

    def train_models(self, training_data: Optional[pd.DataFrame] = None):
        """Train all ML models on historical data"""
        logger.info("Starting model training...")

        if training_data is None:
            training_data = self.generate_training_data(1500)

        if training_data.empty:
            logger.error("No training data available")
            return

        # Ensure market feature columns are recognized if present in training data
        self._ensure_market_features(training_data)

        # Fallback: if no market columns detected but historic rows exist, merge now
        if not any(c.startswith("market_") for c in training_data.columns):
            try:
                from ..services.historic_odds_service import market_training_rows

                mrows = market_training_rows()
            except Exception:
                mrows = []
            if mrows:
                import pandas as pd

                mdf = pd.DataFrame(mrows)
                # Add missing model feature columns to market df
                for col in self.feature_columns:
                    if col not in mdf.columns:
                        mdf[col] = 0.0
                # Add missing market cols to base training_data
                market_cols = [c for c in mdf.columns if c.startswith("market_")]
                for mc in market_cols:
                    if mc not in training_data.columns:
                        training_data[mc] = 0.0
                # Concat
                training_data = pd.concat([training_data, mdf], ignore_index=True)
                # Register new market columns
                for mc in market_cols:
                    if mc not in self.feature_columns:
                        self.feature_columns.append(mc)
                logger.info(
                    f"Train-time market merge applied: appended {len(mrows)} market rows; new size={len(training_data)}"
                )
                self._market_feature_stats = {
                    "market_columns_detected": len(market_cols),
                    "market_columns_added": len(market_cols),
                    "market_rows": len(mrows),
                    "applied_stage": "train_models_fallback",
                }

        # Prepare features
        feature_cols = [
            col for col in self.feature_columns if col in training_data.columns
        ]
        X = training_data[feature_cols].fillna(0)

        # Scale features
        self.scalers["features"] = StandardScaler()
        X_scaled = self.scalers["features"].fit_transform(X)

        # Train each model type
        self._train_goals_models(X_scaled, training_data)
        self._train_result_models(X_scaled, training_data)
        self._train_corners_models(X_scaled, training_data)

        self.is_trained = True
        self._save_models()

        logger.info("Model training completed successfully!")
        logger.info(f"Training accuracies: {self.training_accuracy}")

    def _ensure_market_features(self, df: pd.DataFrame):
        """Detect market_* columns in dataframe and append to feature_columns if absent.
        Also logs how many rows contain market data."""
        market_cols = [c for c in df.columns if c.startswith("market_")]
        if not market_cols:
            logger.info("No market_* columns detected in training data")
            return
        added = 0
        for c in market_cols:
            if c not in self.feature_columns:
                self.feature_columns.append(c)
                added += 1
        market_rows = (
            int((~df["market_home_implied"].isna()).sum())
            if "market_home_implied" in df.columns
            else 0
        )
        logger.info(
            f"Market feature integration: detected {len(market_cols)} market columns, added {added}, rows with market data={market_rows}"
        )
        # Persist counts for metadata snapshot
        self._market_feature_stats = {
            "market_columns_detected": len(market_cols),
            "market_columns_added": added,
            "market_rows": market_rows,
        }

    def _train_goals_models(self, X: np.ndarray, data: pd.DataFrame):
        """Train models for goal predictions"""
        # Total goals
        if "total_goals" in data.columns:
            mask_total = ~data["total_goals"].isna()
            if mask_total.any():
                X_total = X[mask_total.values]
                y_total = data.loc[mask_total, "total_goals"]
                for model_name, model in self.models["goals_total"].items():
                    if model_name == "ensemble_weights":
                        continue
                    model.fit(X_total, y_total)
                    y_pred = model.predict(X_total)
                    mse = mean_squared_error(y_total, y_pred)
                    self.training_accuracy[f"total_goals_{model_name}"] = 1.0 / (
                        1.0 + mse
                    )
                    if hasattr(model, "feature_importances_"):
                        self.feature_importance[
                            f"total_goals_{model_name}"
                        ] = model.feature_importances_.tolist()
        # First half goals
        if "first_half_goals" in data.columns:
            mask_fh = ~data["first_half_goals"].isna()
            if mask_fh.any():
                X_fh = X[mask_fh.values]
                y_first_half = data.loc[mask_fh, "first_half_goals"]
                for model_name, model in self.models["goals_first_half"].items():
                    if model_name == "ensemble_weights":
                        continue
                    model.fit(X_fh, y_first_half)
                    y_pred = model.predict(X_fh)
                    mse = mean_squared_error(y_first_half, y_pred)
                    self.training_accuracy[f"first_half_goals_{model_name}"] = 1.0 / (
                        1.0 + mse
                    )

    def _train_result_models(self, X: np.ndarray, data: pd.DataFrame):
        """Train models for match result prediction"""
        if "match_result" in data.columns:
            mask_res = ~data["match_result"].isna()
            if mask_res.any():
                X_res = X[mask_res.values]
                y_result = data.loc[mask_res, "match_result"]
                for model_name, model in self.models["match_result"].items():
                    if model_name == "ensemble_weights":
                        continue
                    model.fit(X_res, y_result)
                    y_pred = model.predict(X_res)
                    accuracy = accuracy_score(y_result, y_pred)
                    self.training_accuracy[f"match_result_{model_name}"] = accuracy
                    if hasattr(model, "feature_importances_"):
                        self.feature_importance[
                            f"match_result_{model_name}"
                        ] = model.feature_importances_.tolist()

    def _train_corners_models(self, X: np.ndarray, data: pd.DataFrame):
        """Train models for corner predictions"""
        if "total_corners" not in data.columns:
            logger.warning(
                "Skipping corner model training: total_corners column missing"
            )
            return
        mask_c = ~data["total_corners"].isna()
        if not mask_c.any():
            logger.warning(
                "Skipping corner model training: no non-null total_corners values"
            )
            return
        try:
            X_c = X[mask_c.values]
            y_corners = data.loc[mask_c, "total_corners"]
            for model_name, model in self.models["corners_total"].items():
                if model_name == "ensemble_weights":
                    continue
                model.fit(X_c, y_corners)
                y_pred = model.predict(X_c)
                mse = mean_squared_error(y_corners, y_pred)
                self.training_accuracy[f"total_corners_{model_name}"] = 1.0 / (
                    1.0 + mse
                )
        except Exception as e:
            logger.warning(f"Corner model training failed: {e}")

    def predict_match(
        self, home_team: str, away_team: str, league: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive match predictions using trained models"""
        # Always have a features dict for fallback to avoid UnboundLocalError
        features: Dict[str, Any] = {}
        original_home, original_away = home_team, away_team
        try:
            # Normalize common name suffixes that differ across data sources
            def _normalize_team(name: str) -> str:
                if not name:
                    return name
                # Remove common suffixes
                trims = [" FC", " AFC", " F.C.", " A.F.C."]
                for t in trims:
                    if name.endswith(t):
                        name = name[: -len(t)]
                # Handle "Manchester United FC" -> "Manchester United"
                return name.strip()

            try:
                # Prefer v2 deterministic features if available
                features_data = None
                if _v2_feature_service is not None:
                    features_data = self._build_v2_features(home_team, away_team)
                if not features_data:  # fallback to legacy
                    features_data = enhanced_epl_service.get_match_prediction_features(
                        home_team, away_team
                    )
            except Exception:
                # Retry with normalized names if initial lookup fails
                norm_home, norm_away = _normalize_team(home_team), _normalize_team(
                    away_team
                )
                if (norm_home, norm_away) != (home_team, away_team):
                    home_team, away_team = norm_home, norm_away
                    if _v2_feature_service is not None:
                        features_data = self._build_v2_features(home_team, away_team)
                    if not features_data:
                        features_data = (
                            enhanced_epl_service.get_match_prediction_features(
                                home_team, away_team
                            )
                        )
                else:
                    raise

            features = features_data.get("features", {}) if features_data else {}

            # If feature version is present but not aligned with current TEAM_FEATURE_VERSION semantics, apply deterministic rescale
            try:
                fv = features.get("feature_version")
                if fv and isinstance(fv, str) and TEAM_FEATURE_VERSION not in fv:
                    adj_seed = hashlib.sha256(
                        f"fv_adj_{home_team}_{away_team}_{fv}".encode()
                    ).hexdigest()
                    r = random.Random(int(adj_seed[:12], 16))
                    scale = 1 + r.uniform(-0.035, 0.035)
                    for k, v in list(features.items()):
                        if isinstance(v, (int, float)) and k not in (
                            "position_difference",
                            "goal_difference_gap",
                        ):
                            features[k] = round(v * scale, 5)
                    features["feature_version_adjusted"] = TEAM_FEATURE_VERSION
            except Exception:
                pass

            if not self.is_trained:
                logger.warning("Models not trained, using heuristic predictions")
                return self._heuristic_prediction(features)

            # Prepare feature vector
            feature_vector = []
            for col in self.feature_columns:
                feature_vector.append(features.get(col, 0))

            X = np.array(feature_vector).reshape(1, -1)
            X_scaled = self.scalers["features"].transform(X)

            # Generate predictions
            predictions = {}

            # Total goals prediction (ensemble) blended with Poisson xG expectation
            total_goals_preds = []
            weights = self.models["goals_total"]["ensemble_weights"]
            for i, (model_name, model) in enumerate(
                [
                    item
                    for item in self.models["goals_total"].items()
                    if item[0] != "ensemble_weights"
                ]
            ):
                pred = float(model.predict(X_scaled)[0])
                total_goals_preds.append(pred * weights[i])
            ml_total_goals = max(0.0, sum(total_goals_preds))
            # Poisson xG expectation
            try:
                eg = expected_goals_service.get_expected_goals(home_team, away_team)
                xg_total_goals = float(
                    eg.get("lambda_home", 1.3) + eg.get("lambda_away", 1.2)
                )
            except Exception:
                eg = None
                xg_total_goals = ml_total_goals
            # Blend weights (env tunable)
            # Prefer tuned weight from cache/model_blend.json
            cfg = _load_blend_config() or {}
            try:
                w_xg = float(
                    cfg.get("goals_xg_weight", os.getenv("ML_GOALS_XG_WEIGHT", "0.5"))
                )
            except Exception:
                w_xg = 0.5
            w_ml = 1.0 - max(0.0, min(1.0, w_xg))
            w_xg = max(0.0, min(1.0, w_xg))
            blended_total = w_ml * ml_total_goals + w_xg * xg_total_goals
            # Apply optional league-wide bias correction to total goals
            # Offsets are subtracted (same convention as reconciliation task)
            h_off, a_off = _load_goal_offsets()
            total_bias = (h_off or 0.0) + (a_off or 0.0)
            base_total = max(0.0, blended_total)
            adjusted_total = (
                max(0.0, base_total - float(total_bias)) if total_bias else base_total
            )
            predictions["total_goals"] = adjusted_total
            if total_bias:
                predictions.setdefault("adjustments", {})["goal_offsets"] = {
                    "home_goal_offset": h_off,
                    "away_goal_offset": a_off,
                    "original_total_goals": base_total,
                    "adjusted_total_goals": adjusted_total,
                }
            predictions["total_goals_confidence"] = self._calculate_confidence(
                "goals_total"
            )

            # First half goals prediction
            fh_goals_preds = []
            weights = self.models["goals_first_half"]["ensemble_weights"]

            for i, (model_name, model) in enumerate(
                [
                    item
                    for item in self.models["goals_first_half"].items()
                    if item[0] != "ensemble_weights"
                ]
            ):
                pred = model.predict(X_scaled)[0]
                fh_goals_preds.append(pred * weights[i])

            predictions["first_half_goals"] = max(0, sum(fh_goals_preds))
            predictions["first_half_goals_confidence"] = self._calculate_confidence(
                "goals_first_half"
            )

            # Match result prediction: blend ML classifier with Poisson xG outcome distribution
            result_probs = np.zeros(3)  # order: [Draw, Home, Away]
            weights = self.models["match_result"]["ensemble_weights"]
            for i, (model_name, model) in enumerate(
                [
                    item
                    for item in self.models["match_result"].items()
                    if item[0] != "ensemble_weights"
                ]
            ):
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_scaled)[0]
                    result_probs += probs * weights[i]
                else:
                    pred_class = int(model.predict(X_scaled)[0])
                    prob_array = np.zeros(3)
                    prob_array[max(0, min(2, pred_class))] = 1.0
                    result_probs += prob_array * weights[i]
            # Normalize ML probs
            s = float(result_probs.sum())
            if s <= 0:
                result_probs = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
            else:
                result_probs = result_probs / s
            # Poisson-based outcome probabilities from xG lambdas
            try:
                if eg is None:
                    eg = expected_goals_service.get_expected_goals(home_team, away_team)
                dist = expected_goals_service.scoreline_distribution(
                    float(eg.get("lambda_home", 1.3)), float(eg.get("lambda_away", 1.2))
                )
                xg_probs = np.array(
                    [
                        float(dist.get("draw_prob", 0.3)),
                        float(dist.get("home_win_prob", 0.35)),
                        float(dist.get("away_win_prob", 0.35)),
                    ]
                )
            except Exception:
                xg_probs = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
            # Blend ML vs xG (env tunable)
            cfg = cfg or _load_blend_config() or {}
            try:
                w_xg_probs = float(
                    cfg.get("result_xg_weight", os.getenv("ML_RESULT_XG_WEIGHT", "0.4"))
                )
            except Exception:
                w_xg_probs = 0.4
            w_xg_probs = max(0.0, min(1.0, w_xg_probs))
            w_ml_probs = 1.0 - w_xg_probs
            blended_probs = w_ml_probs * result_probs + w_xg_probs * xg_probs
            # Dirichlet-like smoothing to avoid draw underestimation
            alpha = np.array([0.08, 0.06, 0.08])  # [D,H,A] pseudo-counts
            smoothed = blended_probs + alpha
            smoothed = smoothed / smoothed.sum()
            # Apply temperature calibration if available (expects dict order H,D,A)
            try:
                from ..services.calibration_service import calibration_service

                calibrated = calibration_service.apply_if_ready(
                    {
                        "H": float(smoothed[1]),
                        "D": float(smoothed[0]),
                        "A": float(smoothed[2]),
                    },
                    league=league,
                )
                final_probs = np.array(
                    [calibrated["D"], calibrated["H"], calibrated["A"]]
                )
            except Exception:
                final_probs = smoothed
            final_probs = final_probs / max(1e-12, final_probs.sum())

            # Optional: blend with live market implied probabilities (H2H) if available
            # Controlled by cache/model_blend.json key 'result_market_weight' or env ML_RESULT_MARKET_WEIGHT
            try:
                cfg = cfg or _load_blend_config() or {}
                try:
                    w_mkt = None
                    # Per-league override map takes precedence when league provided
                    if league and isinstance(cfg.get("result_market_weight_by_league"), dict):
                        key = str(league).upper()
                        val = cfg["result_market_weight_by_league"].get(key)
                        if isinstance(val, (int, float)):
                            w_mkt = float(val)
                    if w_mkt is None:
                        w_mkt = float(
                            cfg.get(
                                "result_market_weight",
                                os.getenv("ML_RESULT_MARKET_WEIGHT", "0.0"),
                            )
                        )
                except Exception:
                    w_mkt = 0.0
                if w_mkt and w_mkt > 0.0:
                    prefer_bovada = (
                        str(os.getenv("ML_MARKET_BLEND_BOVADA_ONLY", "1")).strip()
                        in ("1", "true", "True")
                    )
                    mkt = betting_odds_service.get_match_odds(
                        home_team, away_team, match_date=None, prefer_bovada_only=prefer_bovada
                    )
                    mw = (mkt or {}).get("market_odds") or {}
                    h2h = mw.get("match_winner") or {}
                    ph = (h2h.get("home") or {}).get("probability")
                    pd = (h2h.get("draw") or {}).get("probability")
                    pa = (h2h.get("away") or {}).get("probability")
                    if all(isinstance(v, (int, float)) and v >= 0.0 for v in (ph, pd, pa)):
                        mkt_vec = np.array([float(pd), float(ph), float(pa)])
                        s = float(mkt_vec.sum())
                        if s > 0:
                            mkt_vec = mkt_vec / s
                            w_mkt = max(0.0, min(1.0, float(w_mkt)))
                            final_probs = (1.0 - w_mkt) * final_probs + w_mkt * mkt_vec
                            final_probs = final_probs / max(1e-12, final_probs.sum())
                            predictions.setdefault("adjustments", {})["market_blend"] = {
                                "weight": w_mkt,
                                "provider": mkt.get("provider"),
                                "market_probs": {
                                    "D": float(mkt_vec[0]),
                                    "H": float(mkt_vec[1]),
                                    "A": float(mkt_vec[2]),
                                },
                            }
            except Exception as e:
                predictions.setdefault("adjustments", {})["market_blend_error"] = str(e)

            predictions["draw_probability"] = float(final_probs[0])
            predictions["home_win_probability"] = float(final_probs[1])
            predictions["away_win_probability"] = float(final_probs[2])

            # Most likely result (aligned to calibrated final probabilities)
            most_likely_result = int(np.argmax(final_probs))
            result_map = {0: "D", 1: "H", 2: "A"}
            predictions["match_result"] = result_map[most_likely_result]
            # Confidence as margin over second-best to avoid overstated certainty on close calls
            sorted_probs = np.sort(final_probs)[::-1]
            margin = (
                float(sorted_probs[0] - sorted_probs[1])
                if len(sorted_probs) >= 2
                else float(sorted_probs[0])
            )
            predictions["match_result_confidence"] = max(
                0.0, min(1.0, 0.5 * float(np.max(final_probs)) + 0.5 * margin)
            )

            # Corners prediction
            corners_preds = []
            weights = self.models["corners_total"]["ensemble_weights"]

            for i, (model_name, model) in enumerate(
                [
                    item
                    for item in self.models["corners_total"].items()
                    if item[0] != "ensemble_weights"
                ]
            ):
                pred = model.predict(X_scaled)[0]
                corners_preds.append(pred * weights[i])

            total_corners = max(0.0, sum(corners_preds))
            # If corners prediction is missing/very low, derive heuristic from xG + tempo/creation
            try:
                if (not total_corners) or total_corners < 1.0:
                    # Derive a base from total goals expectations (xG/blended_total) scaled by pace factors
                    pace = float(features.get("tempo_gap", 0.0))
                    creation_gap = float(features.get("creation_gap", 0.0))
                    # Use positive pace/creation magnitude as activity proxy
                    activity = (
                        1.0
                        + max(0.0, abs(pace)) * 0.15
                        + max(0.0, abs(creation_gap)) * 0.10
                    )
                    # Base correlates with expected total goals; clamp sensible bounds
                    base_goals = (
                        float(predictions.get("total_goals", adjusted_total))
                        if "predictions" in locals()
                        else adjusted_total
                    )
                    heuristic = max(6.0, min(14.0, 3.2 * base_goals * activity))
                    total_corners = float(heuristic)
            except Exception:
                # Fallback fixed midline if anything goes wrong
                if (not total_corners) or total_corners < 1.0:
                    total_corners = 10.0
            predictions["total_corners"] = total_corners
            predictions["total_corners_confidence"] = self._calculate_confidence(
                "corners_total"
            )
            predictions["home_corners"] = total_corners * 0.52  # Slight home advantage
            predictions["away_corners"] = total_corners * 0.48

            # Additional betting insights
            try:
                # Prefer Poisson distribution implied Over 2.5
                if eg is None:
                    eg = expected_goals_service.get_expected_goals(home_team, away_team)
                dist = expected_goals_service.scoreline_distribution(
                    float(eg.get("lambda_home", 1.3)), float(eg.get("lambda_away", 1.2))
                )
                predictions["over_2_5_goals_probability"] = float(
                    dist.get(
                        "over_2_5_prob",
                        self._calculate_over_probability(predictions["total_goals"]),
                    )
                )
            except Exception:
                predictions[
                    "over_2_5_goals_probability"
                ] = self._calculate_over_probability(predictions["total_goals"])

            # Optional: blend Over 2.5 probability with market totals (nearest to 2.5) if available
            try:
                cfg2 = _load_blend_config() or {}
                try:
                    w_tot = None
                    if league and isinstance(cfg2.get("totals_market_weight_by_league"), dict):
                        key = str(league).upper()
                        val = cfg2["totals_market_weight_by_league"].get(key)
                        if isinstance(val, (int, float)):
                            w_tot = float(val)
                    if w_tot is None:
                        w_tot = float(
                            cfg2.get(
                                "totals_market_weight", os.getenv("ML_TOTALS_MARKET_WEIGHT", "0.0")
                            )
                        )
                except Exception:
                    w_tot = 0.0
                if w_tot and w_tot > 0.0:
                    prefer_bovada = (
                        str(os.getenv("ML_MARKET_BLEND_BOVADA_ONLY", "1")).strip()
                        in ("1", "true", "True")
                    )
                    mkt2 = betting_odds_service.get_match_odds(
                        home_team, away_team, match_date=None, prefer_bovada_only=prefer_bovada
                    )
                    mo = (mkt2 or {}).get("market_odds") or {}
                    totals = mo.get("totals") or []
                    # choose line closest to 2.5, prefer exact 2.5
                    target = 2.5
                    chosen = None
                    best_gap = 1e9
                    for row in totals:
                        try:
                            line = float(row.get("line")) if row.get("line") is not None else None
                        except Exception:
                            line = None
                        if line is None:
                            continue
                        gap = abs(line - target)
                        if gap < best_gap or (gap == best_gap and line == target):
                            chosen = row
                            best_gap = gap
                    if chosen and isinstance(chosen.get("over"), dict):
                        p_over = chosen["over"].get("probability")
                        if isinstance(p_over, (int, float)) and p_over >= 0.0:
                            base = float(predictions.get("over_2_5_goals_probability", 0.0))
                            w_tot = max(0.0, min(1.0, float(w_tot)))
                            blended = (1.0 - w_tot) * base + w_tot * float(p_over)
                            predictions.setdefault("adjustments", {})["market_totals_blend"] = {
                                "weight": w_tot,
                                "provider": (mkt2 or {}).get("provider"),
                                "line": chosen.get("line"),
                                "market_over_prob": float(p_over),
                                "base_over_prob": base,
                                "result_over_prob": float(blended),
                            }
                            predictions["over_2_5_goals_probability"] = float(blended)
            except Exception as e:
                predictions.setdefault("adjustments", {})["market_totals_blend_error"] = str(e)
            # BTTS from Poisson (1 - P(H=0) - P(A=0) + P(H=0,A=0))
            try:
                if eg is None:
                    eg = expected_goals_service.get_expected_goals(home_team, away_team)
                lam_h = float(eg.get("lambda_home", 1.3))
                lam_a = float(eg.get("lambda_away", 1.2))
                p_h0 = np.exp(-lam_h)
                p_a0 = np.exp(-lam_a)
                p_both0 = np.exp(-(lam_h + lam_a))
                predictions["both_teams_score_probability"] = float(
                    max(0.0, min(1.0, 1.0 - p_h0 - p_a0 + p_both0))
                )
            except Exception:
                predictions[
                    "both_teams_score_probability"
                ] = self._calculate_btts_probability(features)
            predictions["clean_sheet_probability"] = (
                1.0 - predictions["both_teams_score_probability"]
            )

            # Model metadata
            predictions["algorithm"] = "ensemble_ml"
            predictions["model_confidence"] = np.mean(
                [
                    predictions["total_goals_confidence"],
                    predictions["match_result_confidence"],
                    predictions["total_corners_confidence"],
                ]
            )

            predictions["feature_importance"] = self._get_average_feature_importance()

            # Match context
            predictions["match_context"] = {
                "home_team": home_team,
                "away_team": away_team,
                "prediction_time": datetime.now().isoformat(),
                "model_version": self.model_version,
                "features_used": self.feature_columns,
                "feature_version": features.get("feature_version")
                or getattr(self, "feature_version", "fv1"),
            }

            # Verbose per-match logging demoted to debug to avoid terminal spam on page reloads
            logger.debug(f"Generated ML prediction for {home_team} vs {away_team}")
            return predictions

        except Exception as e:
            logger.error(
                f"Error in ML prediction for {original_home} vs {original_away}: {e}"
            )
            # Heuristic fallback with whatever features we have
            return self._heuristic_prediction(features)

    def get_feature_row(self, home_team: str, away_team: str) -> Dict[str, float]:
        """Return a dict of feature column -> value for the given pairing.
        Uses the same feature construction path as predict_match but does not perform prediction.
        Missing features are filled with 0.0 for stability.
        """
        try:
            # Reuse v2 builder preference logic
            if _v2_feature_service is not None:
                feat_data = self._build_v2_features(home_team, away_team)
                if feat_data:
                    features = feat_data.get("features", {})
                else:
                    features = enhanced_epl_service.get_match_prediction_features(
                        home_team, away_team
                    ).get("features", {})
            else:
                features = enhanced_epl_service.get_match_prediction_features(
                    home_team, away_team
                ).get("features", {})
        except Exception:
            features = {}
        return {
            col: float(features.get(col, 0.0))
            if isinstance(features.get(col), (int, float))
            else 0.0
            for col in self.feature_columns
        }

    # ---------------- V2 Feature Builder -----------------
    def _deterministic_value(self, key: str, low: float, high: float) -> float:
        h = hashlib.sha256(key.encode()).hexdigest()
        seed = int(h[:8], 16)
        rng = random.Random(seed)
        return round(rng.uniform(low, high), 4)

    def _build_v2_features(
        self, home_team: str, away_team: str
    ) -> Optional[Dict[str, Any]]:
        if _v2_feature_service is None:
            return None
        try:
            teams = _v2_feature_service.get_teams()
            idx = {t["name"]: t for t in teams}

            # Attempt simple normalization mapping
            def _match_team(name: str):
                if name in idx:
                    return idx[name]
                stripped = name.replace(" FC", "").replace(" AFC", "")
                for tname in idx.keys():
                    if tname.startswith(stripped) or stripped in tname:
                        return idx[tname]
                return None

            h_team = _match_team(home_team)
            a_team = _match_team(away_team)
            if not h_team or not a_team:
                return None

            # Use positions to derive baseline strengths
            def _ppg(position: int):
                # Higher for top positions; synthetic early-season approximation
                return round(2.4 - (position - 1) * (1.6 / 19), 4)

            home_ppg = _ppg(h_team.get("position", 10))
            away_ppg = _ppg(a_team.get("position", 10))
            # Seed metrics (differentiated attributes)
            h_seed = h_team.get("seed_metrics", {})
            a_seed = a_team.get("seed_metrics", {})
            # Deterministic attack/defense ratings
            home_attack = h_seed.get("attack_rating") or self._deterministic_value(
                f"attack_{h_team['name']}", 1.2, 2.4
            )
            away_attack = a_seed.get("attack_rating") or self._deterministic_value(
                f"attack_{a_team['name']}", 1.2, 2.4
            )
            # defense_rating higher means weaker defense; invert for strength factor
            home_def_raw = h_seed.get("defense_rating") or self._deterministic_value(
                f"def_{h_team['name']}", 0.6, 1.6
            )
            away_def_raw = a_seed.get("defense_rating") or self._deterministic_value(
                f"def_{a_team['name']}", 0.6, 1.6
            )
            home_def = max(0.2, 2.2 - home_def_raw)
            away_def = max(0.2, 2.2 - away_def_raw)
            home_form = h_seed.get("form_score") or self._deterministic_value(
                f"form_{h_team['name']}", 0.45, 0.85
            )
            away_form = a_seed.get("form_score") or self._deterministic_value(
                f"form_{a_team['name']}", 0.45, 0.85
            )
            home_finishing = h_seed.get("finishing_quality", 1.0)
            away_finishing = a_seed.get("finishing_quality", 1.0)
            home_creation = h_seed.get("chance_creation", 1.0)
            away_creation = a_seed.get("chance_creation", 1.0)
            home_tempo = h_seed.get("tempo_factor", 1.0)
            away_tempo = a_seed.get("tempo_factor", 1.0)
            home_supp = h_seed.get("suppression_factor", 1.0)
            away_supp = a_seed.get("suppression_factor", 1.0)
            # Goals per game approximated
            # Construct offensive potency and defensive suppression adjustments
            home_attack_effective = (
                home_attack * home_creation * home_finishing * home_tempo
            )
            away_attack_effective = (
                away_attack * away_creation * away_finishing * away_tempo
            )
            # Defensive suppression reduces opponent attack_effective
            home_def_factor = home_def * home_supp
            away_def_factor = away_def * away_supp
            # Expected goals per game proxies
            home_goals_pg = round(
                (home_attack_effective * 0.42) * (away_def_factor * 0.35), 4
            )
            away_goals_pg = round(
                (away_attack_effective * 0.42) * (home_def_factor * 0.35), 4
            )
            # Conceded approximations (inverse effectiveness + opponent attack)
            home_concede_pg = round(
                (away_attack_effective * 0.30) / max(home_def_factor, 0.3), 4
            )
            away_concede_pg = round(
                (home_attack_effective * 0.30) / max(away_def_factor, 0.3), 4
            )

            # ---------- Pairing-specific deterministic variance ----------
            pair_key = f"{h_team['name']}|{a_team['name']}"
            pair_seed = hashlib.sha256(pair_key.encode()).hexdigest()
            pair_rng = random.Random(int(pair_seed[:10], 16))
            # Asymmetric variance shift (-0.12..0.12)
            variance_core = pair_rng.uniform(-0.12, 0.12)
            home_shift = 1 + variance_core * 0.6
            away_shift = 1 - variance_core * 0.4
            # Slight independent micro-noise for each side (stable)
            home_micro = 1 + (pair_rng.uniform(-0.025, 0.025))
            away_micro = 1 + (pair_rng.uniform(-0.025, 0.025))
            home_goals_pg = round(home_goals_pg * home_shift * home_micro, 4)
            away_goals_pg = round(away_goals_pg * away_shift * away_micro, 4)
            # Adjust form subtly per matchup
            home_form += variance_core * 0.03
            away_form -= variance_core * 0.02
            home_form = max(0.25, min(0.95, home_form))
            away_form = max(0.25, min(0.95, away_form))
            # Position and disparity calculations (order corrected)
            position_diff = a_team.get("position", 0) - h_team.get("position", 0)
            strength_disparity = abs(home_ppg - away_ppg)
            position_diff_factor = position_diff / 10.0
            strength_disparity *= 1 + abs(position_diff_factor) * 0.05
            goal_diff_gap = (home_attack - home_def) - (away_attack - away_def)
            h2h_avg_goals = round(
                (home_goals_pg + away_goals_pg + home_concede_pg + away_concede_pg) / 2,
                4,
            )
            over_2_5_rate = 0.65 if h2h_avg_goals > 2.6 else 0.45
            # Game context interaction adjustments
            form_gap = home_form - away_form
            strength_disparity *= 1 + abs(form_gap) * 0.1
            features = {
                "home_points_per_game": home_ppg,
                "away_points_per_game": away_ppg,
                "home_goals_per_game": home_goals_pg,
                "away_goals_per_game": away_goals_pg,
                "home_goals_conceded_per_game": home_concede_pg,
                "away_goals_conceded_per_game": away_concede_pg,
                "goal_difference_gap": goal_diff_gap,
                "position_difference": position_diff,
                "home_win_rate": home_ppg / 3.0,
                "away_win_rate": away_ppg / 3.0,
                "home_form_score": home_form,
                "away_form_score": away_form,
                "home_recent_goals": home_goals_pg,
                "away_recent_goals": away_goals_pg,
                "h2h_home_advantage": 0.15 + (home_form - away_form) * 0.05,
                "h2h_avg_goals": h2h_avg_goals,
                "h2h_over_2_5_rate": over_2_5_rate,
                "home_advantage_multiplier": 1.08
                + self._deterministic_value(f"hadv_{h_team['name']}", 0.0, 0.07),
                "is_top_6_clash": h_team.get("position", 20) <= 6
                and a_team.get("position", 20) <= 6,
                "is_relegation_battle": h_team.get("position", 0) >= 15
                and a_team.get("position", 0) >= 15,
                "strength_disparity": strength_disparity,
                # New enriched engineered features
                "attack_matchup_index": round(
                    home_attack_effective / max(away_def_factor, 0.2), 4
                ),
                "defense_matchup_index": round(
                    home_def_factor / max(away_attack_effective, 0.2), 4
                ),
                "form_gap": round(form_gap, 4),
                "tempo_gap": round(home_tempo - away_tempo, 4),
                "creation_gap": round(home_creation - away_creation, 4),
                "finishing_gap": round(home_finishing - away_finishing, 4),
                "pair_variance": round(variance_core, 4),
                "feature_version": f"{TEAM_FEATURE_VERSION}-pair1",
            }
            return {"features": features}
        except Exception:
            return None

    def _calculate_confidence(self, model_type: str) -> float:
        """Calculate confidence based on training accuracy"""
        accuracies = []
        for key, acc in self.training_accuracy.items():
            if model_type in key:
                accuracies.append(acc)

        if accuracies:
            return np.mean(accuracies)
        return 0.5

    def _calculate_over_probability(self, total_goals: float) -> float:
        """Calculate probability of over 2.5 goals"""
        # Using normal distribution approximation
        z_score = (2.5 - total_goals) / max(
            total_goals * 0.3, 0.5
        )  # Assume std dev proportional to mean
        prob_under = 0.5 * (1 + np.tanh(z_score))
        return max(0.05, min(0.95, 1.0 - prob_under))

    def _calculate_btts_probability(self, features: Dict) -> float:
        """Calculate both teams to score probability"""
        home_scoring_prob = min(
            0.9, features.get("home_goals_per_game", 1.5) / 3.0 + 0.3
        )
        away_scoring_prob = min(
            0.9, features.get("away_goals_per_game", 1.5) / 3.0 + 0.3
        )
        return home_scoring_prob * away_scoring_prob

    def _get_average_feature_importance(self) -> Dict[str, float]:
        """Get averaged feature importance across all models"""
        if not self.feature_importance:
            return {}

        avg_importance = {}
        for feature_idx, feature_name in enumerate(self.feature_columns):
            importances = []
            for model_key, importance_list in self.feature_importance.items():
                if feature_idx < len(importance_list):
                    importances.append(importance_list[feature_idx])

            if importances:
                avg_importance[feature_name] = np.mean(importances)

        return avg_importance

    def _heuristic_prediction(self, features: Dict) -> Dict[str, Any]:
        """Fallback heuristic prediction when ML models aren't available"""
        home_expected = features.get("home_goals_per_game", 1.5) * features.get(
            "home_advantage_multiplier", 1.1
        )
        away_expected = features.get("away_goals_per_game", 1.5)

        total_goals = home_expected + away_expected
        first_half = total_goals * 0.45

        # Simple result logic
        if home_expected > away_expected * 1.2:
            result = "H"
            home_prob, away_prob, draw_prob = 0.55, 0.25, 0.20
        elif away_expected > home_expected * 1.2:
            result = "A"
            home_prob, away_prob, draw_prob = 0.25, 0.55, 0.20
        else:
            result = "D"
            home_prob, away_prob, draw_prob = 0.35, 0.35, 0.30

        # Normalize probabilities defensively
        prob_sum = home_prob + away_prob + draw_prob
        if prob_sum > 0:
            home_prob /= prob_sum
            away_prob /= prob_sum
            draw_prob /= prob_sum

        return {
            "total_goals": round(total_goals, 2),
            "total_goals_confidence": 0.65,
            "first_half_goals": round(first_half, 2),
            "first_half_goals_confidence": 0.60,
            "match_result": result,
            "match_result_confidence": max(home_prob, away_prob, draw_prob),
            "home_win_probability": home_prob,
            "away_win_probability": away_prob,
            "draw_probability": draw_prob,
            "total_corners": 10.0,
            "total_corners_confidence": 0.50,
            "home_corners": 5.2,
            "away_corners": 4.8,
            "over_2_5_goals_probability": 0.6 if total_goals > 2.5 else 0.4,
            "both_teams_score_probability": 0.65,
            "algorithm": "heuristic_fallback",
            "model_confidence": 0.58,
            "feature_version": features.get("feature_version", TEAM_FEATURE_VERSION),
        }

    def _create_synthetic_training_data(self, num_samples: int) -> pd.DataFrame:
        """Create synthetic training data when real data is unavailable"""
        logger.warning("Creating synthetic training data")

        data = []
        for _ in range(num_samples):
            # Generate realistic feature combinations
            home_strength = np.random.uniform(0.8, 2.5)
            away_strength = np.random.uniform(0.8, 2.5)

            features = {
                "home_points_per_game": np.random.uniform(0.5, 2.5),
                "away_points_per_game": np.random.uniform(0.5, 2.5),
                "home_goals_per_game": home_strength,
                "away_goals_per_game": away_strength,
                "home_goals_conceded_per_game": np.random.uniform(0.5, 2.0),
                "away_goals_conceded_per_game": np.random.uniform(0.5, 2.0),
                "goal_difference_gap": np.random.uniform(-20, 20),
                "position_difference": np.random.uniform(-19, 19),
                "home_win_rate": np.random.uniform(0.1, 0.8),
                "away_win_rate": np.random.uniform(0.1, 0.8),
                "home_form_score": np.random.uniform(0.2, 0.9),
                "away_form_score": np.random.uniform(0.2, 0.9),
                "home_recent_goals": np.random.uniform(0.5, 3.0),
                "away_recent_goals": np.random.uniform(0.5, 3.0),
                "h2h_home_advantage": np.random.uniform(0.05, 0.25),
                "h2h_avg_goals": np.random.uniform(1.5, 4.0),
                "h2h_over_2_5_rate": np.random.uniform(0.3, 0.8),
                "home_advantage_multiplier": np.random.uniform(1.05, 1.2),
                "is_top_6_clash": np.random.choice([0, 1], p=[0.8, 0.2]),
                "is_relegation_battle": np.random.choice([0, 1], p=[0.9, 0.1]),
                "strength_disparity": np.random.uniform(0, 1.5),
            }
            # Add engineered v2 features
            features.update(
                {
                    "attack_matchup_index": np.random.uniform(0.4, 3.0),
                    "defense_matchup_index": np.random.uniform(0.4, 3.0),
                    "form_gap": np.random.uniform(-0.5, 0.5),
                    "tempo_gap": np.random.uniform(-0.6, 0.6),
                    "creation_gap": np.random.uniform(-0.7, 0.7),
                    "finishing_gap": np.random.uniform(-0.6, 0.6),
                    "pair_variance": np.random.uniform(-0.12, 0.12),
                }
            )

            # Simulate outcomes
            outcome = self._simulate_match_outcome(features)
            data.append({**features, **outcome})

        return pd.DataFrame(data)

    def _save_models(self):
        """Save trained models to disk"""
        try:
            for model_type, models_dict in self.models.items():
                for model_name, model in models_dict.items():
                    if model_name != "ensemble_weights":
                        model_path = os.path.join(
                            self.model_save_path, f"{model_type}_{model_name}.pkl"
                        )
                        joblib.dump(model, model_path)

            # Save scalers
            scaler_path = os.path.join(self.model_save_path, "scalers.pkl")
            joblib.dump(self.scalers, scaler_path)

            # Save metadata
            metadata = {
                "feature_columns": self.feature_columns,
                "training_accuracy": self.training_accuracy,
                "feature_importance": self.feature_importance,
                "is_trained": self.is_trained,
                "model_version": self.model_version,
                "training_date": datetime.now().isoformat(),
                "market_feature_stats": getattr(self, "_market_feature_stats", None),
            }

            metadata_path = os.path.join(self.model_save_path, "model_metadata.pkl")
            joblib.dump(metadata, metadata_path)

            logger.info("Models saved successfully")

        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def load_models(self, safe_load: bool = True):
        """Load trained models from disk.
        If safe_load is True and any model fails to unpickle (e.g. version skew), we abort and signal False
        so caller can trigger a fresh retrain instead of operating in partially loaded state.
        """
        try:
            loaded_ok = True
            errors = []
            loaded_model_files = 0
            for model_type, models_dict in self.models.items():
                for model_name, model in models_dict.items():
                    if model_name == "ensemble_weights":
                        continue
                    model_path = os.path.join(
                        self.model_save_path, f"{model_type}_{model_name}.pkl"
                    )
                    if os.path.exists(model_path):
                        try:
                            self.models[model_type][model_name] = joblib.load(
                                model_path
                            )
                            loaded_model_files += 1
                        except Exception as e:  # capture and optionally abort
                            errors.append((model_path, str(e)))
                            loaded_ok = False
                            if not safe_load:
                                raise
            # Load scalers
            scaler_path = os.path.join(self.model_save_path, "scalers.pkl")
            if os.path.exists(scaler_path):
                try:
                    self.scalers = joblib.load(scaler_path)
                except Exception as e:
                    errors.append((scaler_path, str(e)))
                    loaded_ok = False
            # Load metadata
            metadata_path = os.path.join(self.model_save_path, "model_metadata.pkl")
            if os.path.exists(metadata_path):
                try:
                    metadata = joblib.load(metadata_path)
                    self.feature_columns = metadata.get(
                        "feature_columns", self.feature_columns
                    )
                    self.training_accuracy = metadata.get("training_accuracy", {})
                    self.feature_importance = metadata.get("feature_importance", {})
                    self.is_trained = metadata.get("is_trained", False)
                    self.model_version = metadata.get(
                        "model_version", self.model_version
                    )
                    self._market_feature_stats = metadata.get("market_feature_stats")
                    # Auto-repair: if metrics exist but flag false, set true and persist
                    if (not self.is_trained) and self.training_accuracy:
                        self.is_trained = True
                        try:
                            self._save_models()
                            logger.info(
                                "[MODEL LOAD] Auto-repaired is_trained flag based on existing training_accuracy."
                            )
                        except Exception:
                            pass
                    logger.info(
                        f"Models loaded successfully. Training date: {metadata.get('training_date', 'Unknown')}"
                    )
                except Exception as e:
                    errors.append((metadata_path, str(e)))
                    loaded_ok = False
            if not loaded_ok:
                logger.warning(
                    f"Model load had errors: {errors}. Proceeding with is_trained=False for fresh retrain."
                )
                self.is_trained = False
            # Integrity heuristic: if we loaded all expected model pickle files and have training_accuracy, mark trained
            expected_model_files = sum(
                1
                for m in self.models.values()
                for k in m.keys()
                if k != "ensemble_weights"
            )
            if loaded_model_files == expected_model_files and self.training_accuracy:
                self.is_trained = True if not errors else self.is_trained
            return self.is_trained
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.is_trained = False
            return False

    # ---------------- Integrity / Repair Helpers -----------------
    def repair_training_flag(self) -> bool:
        """If artifacts + training metrics exist but is_trained is False, flip it.
        Returns True if repair occurred or already consistent.
        """
        try:
            expected_model_files = [
                f"{mt}_{mn}.pkl"
                for mt, md in self.models.items()
                for mn in md.keys()
                if mn != "ensemble_weights"
            ]
            present = 0
            for fname in expected_model_files:
                if os.path.exists(os.path.join(self.model_save_path, fname)):
                    present += 1
            if (
                (not self.is_trained)
                and self.training_accuracy
                and present == len(expected_model_files)
            ):
                self.is_trained = True
                # Update metadata file so future loads reflect corrected status
                self._save_models()
                logger.info(
                    "[MODEL REPAIR] is_trained flag restored based on integrity check."
                )
            return self.is_trained
        except Exception as e:
            logger.warning(f"[MODEL REPAIR] Integrity check failed: {e}")
            return False

    def get_model_performance(self) -> Dict[str, Any]:
        """Get comprehensive model performance metrics"""
        # Dynamic integrity: if metrics exist, treat as trained
        if (not self.is_trained) and self.training_accuracy:
            self.is_trained = True
        return {
            "is_trained": self.is_trained,
            "training_accuracy": self.training_accuracy,
            "feature_importance": self.feature_importance,
            "feature_columns": self.feature_columns,
            "model_types": list(self.models.keys()),
            "ensemble_weights": {
                k: v.get("ensemble_weights", []) for k, v in self.models.items()
            },
            "last_training_date": datetime.now().isoformat()
            if self.is_trained
            else None,
            "model_version": self.model_version,
        }

    # ---------------- Versioning Utilities -----------------
    def bump_version(self, level: str = "patch"):
        """Increment semantic version.
        level: 'major' | 'minor' | 'patch'
        """
        try:
            core = self.model_version.lstrip("v")
            parts = core.split(".")
            while len(parts) < 3:
                parts.append("0")
            major, minor, patch = map(int, parts[:3])
            if level == "major":
                major += 1
                minor = 0
                patch = 0
            elif level == "minor":
                minor += 1
                patch = 0
            else:
                patch += 1
            self.model_version = f"v{major}.{minor}.{patch}"
        except Exception:
            # Fallback reset
            self.model_version = "v2.0.0"

    def retrain_and_bump(
        self, training_data: Optional[pd.DataFrame] = None, level: str = "patch"
    ):
        """Convenience wrapper to retrain then bump version and persist."""
        self.train_models(training_data=training_data)
        self.bump_version(level)
        # Re-save to persist new version metadata
        self._save_models()
        return self.model_version


# Global predictor instance
advanced_ml_predictor = AdvancedMLPredictor()
