from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime


class PredictionBase(BaseModel):
    match_id: int
    total_goals_prediction: Optional[float] = None
    first_half_goals_prediction: Optional[float] = None
    match_result_prediction: Optional[str] = None
    total_corners_prediction: Optional[float] = None


class PredictionCreate(PredictionBase):
    prediction_model_version: str = "v1.0"
    algorithm_used: str = "heuristic"


class PredictionResponse(PredictionBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    total_goals_confidence: Optional[float] = None
    first_half_goals_confidence: Optional[float] = None
    match_result_confidence: Optional[float] = None
    home_win_probability: Optional[float] = None
    away_win_probability: Optional[float] = None
    draw_probability: Optional[float] = None
    total_corners_confidence: Optional[float] = None
    home_corners_prediction: Optional[float] = None
    away_corners_prediction: Optional[float] = None
    prediction_model_version: str
    algorithm_used: str
    features_used: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, Any]] = None
    is_accurate: Optional[bool] = None
    accuracy_score: Optional[float] = None
    prediction_date: datetime
    created_at: datetime
    updated_at: Optional[datetime] = None
