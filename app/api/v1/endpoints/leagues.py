from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from app.core.database import get_db
from app.models.league import League

router = APIRouter()


@router.get("/", response_model=List[dict])
async def get_leagues(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    country: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get leagues with optional filtering"""
    query = db.query(League)
    
    if country:
        query = query.filter(League.country == country)
    
    leagues = query.offset(skip).limit(limit).all()
    return [{"id": league.id, "name": league.name, "country": league.country} for league in leagues]


@router.get("/{league_id}")
async def get_league(league_id: int, db: Session = Depends(get_db)):
    """Get a specific league by ID"""
    league = db.query(League).filter(League.id == league_id).first()
    if not league:
        raise HTTPException(status_code=404, detail="League not found")
    return {"id": league.id, "name": league.name, "country": league.country}