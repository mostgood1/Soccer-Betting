"""
Training Data Pipeline
Builds ML-ready training datasets from historical matches by projecting
the current feature schema for each historical pairing and attaching targets.
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import logging

from .historical_epl_service import historical_epl_service
from ..ml.advanced_predictor import advanced_ml_predictor

logger = logging.getLogger(__name__)


def _result_to_code(label: str) -> int:
    """Map textual result to numeric code expected by classifier.
    0=Draw, 1=Home win, 2=Away win
    """
    if not label:
        return 0
    s = str(label).lower()
    if s in ("home_win", "h", "home", "1"):
        return 1
    if s in ("away_win", "a", "away", "2"):
        return 2
    return 0


def build_historical_training_dataset(limit: Optional[int] = None) -> pd.DataFrame:
    """Construct a DataFrame of feature rows + targets from historical matches.
    - Features: current predictor's feature_columns populated via get_feature_row()
    - Targets: total_goals, first_half_goals (approx), match_result (coded),
               total_corners (heuristic), home/away_corners (split)

    Args:
      limit: optional max number of historical matches to include.

    Returns:
      DataFrame ready for advanced_ml_predictor.train_models(training_data=...)
    """
    try:
        hist_df = historical_epl_service.get_ml_training_data()
        if hist_df is None or hist_df.empty:
            logger.warning(
                "Historical dataset is empty; falling back to predictor's internal generator"
            )
            return pd.DataFrame()
        if limit is not None and limit > 0:
            hist_df = hist_df.head(int(limit))

        rows: List[Dict[str, Any]] = []
        feat_cols = list(getattr(advanced_ml_predictor, "feature_columns", []))
        # Safety: ensure list
        if not isinstance(feat_cols, list):
            feat_cols = []

        for _, r in hist_df.iterrows():
            home = r.get("home_team")
            away = r.get("away_team")
            if not home or not away:
                continue
            # Fetch features using the same path as live predictions
            features_map = advanced_ml_predictor.get_feature_row(str(home), str(away))
            # Ensure full feature vector keys present
            for fc in feat_cols:
                if fc not in features_map:
                    features_map[fc] = 0.0

            # Targets
            home_score = float(r.get("home_score", 0) or 0)
            away_score = float(r.get("away_score", 0) or 0)
            total_goals = float(
                r.get("total_goals", home_score + away_score)
                or (home_score + away_score)
            )
            # First-half goals not present in this dataset; approximate ~45%
            first_half_goals = float(
                np.clip(total_goals * 0.45 + np.random.normal(0, 0.2), 0, None)
            )
            # Corner totals heuristic from goals + mild noise, clamped to sensible range
            tc_base = 3.4 * total_goals + np.random.normal(0, 1.5)
            total_corners = float(np.clip(tc_base, 4.0, 18.0))
            home_corners = float(total_corners * 0.52)
            away_corners = float(total_corners * 0.48)
            match_result_code = _result_to_code(r.get("result"))

            row = {
                **features_map,
                "home_goals": home_score,
                "away_goals": away_score,
                "total_goals": total_goals,
                "first_half_goals": first_half_goals,
                "match_result": match_result_code,
                "total_corners": total_corners,
                "home_corners": home_corners,
                "away_corners": away_corners,
                # Optional market placeholders to maintain stable schema if present
                "market_home_implied": 0.0,
                "market_draw_implied": 0.0,
                "market_away_implied": 0.0,
                "market_overround": 0.0,
                "market_fav_prob": 0.0,
                "market_prob_skew": 0.0,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        # Ensure columns exist for all current feature columns + targets expected by trainer
        for fc in feat_cols:
            if fc not in df.columns:
                df[fc] = 0.0
        logger.info(
            f"Built historical training dataset with {len(df)} rows and {len(df.columns)} columns"
        )
        return df
    except Exception as e:
        logger.error(f"Failed to build historical training dataset: {e}")
        return pd.DataFrame()
