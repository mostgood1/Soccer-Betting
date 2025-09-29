from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    ForeignKey,
    Float,
    Text,
    JSON,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)

    # Prediction types and values
    total_goals_prediction = Column(Float)  # Predicted total goals
    total_goals_confidence = Column(Float)  # Confidence score 0-1

    first_half_goals_prediction = Column(Float)  # Predicted 1H goals
    first_half_goals_confidence = Column(Float)

    match_result_prediction = Column(String)  # 'H', 'A', 'D'
    match_result_confidence = Column(Float)
    home_win_probability = Column(Float)
    away_win_probability = Column(Float)
    draw_probability = Column(Float)

    # Corner predictions
    total_corners_prediction = Column(Float)
    total_corners_confidence = Column(Float)
    home_corners_prediction = Column(Float)
    away_corners_prediction = Column(Float)

    # Model information
    model_version = Column(String, nullable=False)
    algorithm_used = Column(
        String, nullable=False
    )  # 'random_forest', 'neural_network', etc.

    # Feature importance and model details
    features_used = Column(JSON)  # Store which features were used
    feature_importance = Column(JSON)  # Store feature importance scores

    # Accuracy tracking
    is_accurate = Column(Boolean)  # Set after match completion
    accuracy_score = Column(Float)  # Overall prediction accuracy

    # Timestamps
    prediction_date = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    # match = relationship("Match", back_populates="predictions")

    def __repr__(self):
        return f"<Prediction(match_id={self.match_id}, total_goals={self.total_goals_prediction}, result={self.match_result_prediction})>"
