from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from app.core.database import get_db
from app.models.prediction import Prediction
from app.schemas.prediction import PredictionCreate, PredictionResponse
from app.services.prediction_service import PredictionService

router = APIRouter()


@router.get("/", response_model=List[PredictionResponse])
async def get_predictions(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    match_id: Optional[int] = None,
    model_version: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get predictions with optional filtering"""
    query = db.query(Prediction)
    
    if match_id:
        query = query.filter(Prediction.match_id == match_id)
    if model_version:
        query = query.filter(Prediction.model_version == model_version)
    
    predictions = query.offset(skip).limit(limit).all()
    return predictions


@router.get("/{prediction_id}", response_model=PredictionResponse)
async def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    """Get a specific prediction by ID"""
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction


@router.post("/generate/{match_id}", response_model=PredictionResponse)
async def generate_prediction(
    match_id: int,
    model_version: str = "v1.0",
    db: Session = Depends(get_db)
):
    """Generate a new prediction for a match"""
    # Check if match exists
    from app.models.match import Match
    match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    # Generate prediction using the service
    prediction_service = PredictionService(db)
    prediction = prediction_service.generate_prediction(match_id, model_version)
    
    return prediction


@router.get("/match/{match_id}", response_model=List[PredictionResponse])
async def get_match_predictions(match_id: int, db: Session = Depends(get_db)):
    """Get all predictions for a specific match"""
    predictions = db.query(Prediction).filter(Prediction.match_id == match_id).all()
    return predictions


@router.post("/evaluate/{prediction_id}")
async def evaluate_prediction(prediction_id: int, db: Session = Depends(get_db)):
    """Evaluate prediction accuracy after match completion"""
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    prediction_service = PredictionService(db)
    accuracy_results = prediction_service.evaluate_prediction(prediction_id)
    
    return {
        "prediction_id": prediction_id,
        "accuracy_results": accuracy_results,
        "message": "Prediction accuracy evaluated successfully"
    }