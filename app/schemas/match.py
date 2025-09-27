from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime


class MatchBase(BaseModel):
    match_date: datetime
    home_team_id: int
    away_team_id: int
    league_id: int
    status: Optional[str] = "scheduled"
    venue: Optional[str] = None
    referee: Optional[str] = None
    weather_conditions: Optional[str] = None


class MatchCreate(MatchBase):
    api_id: Optional[str] = None


class MatchUpdate(BaseModel):
    match_date: Optional[datetime] = None
    status: Optional[str] = None
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    home_score_ht: Optional[int] = None
    away_score_ht: Optional[int] = None
    total_goals: Optional[int] = None
    total_goals_ht: Optional[int] = None
    total_corners: Optional[int] = None
    home_corners: Optional[int] = None
    away_corners: Optional[int] = None
    attendance: Optional[int] = None
    venue: Optional[str] = None
    referee: Optional[str] = None
    weather_conditions: Optional[str] = None


class MatchResponse(MatchBase):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    home_score_ht: Optional[int] = None
    away_score_ht: Optional[int] = None
    total_goals: Optional[int] = None
    total_goals_ht: Optional[int] = None
    total_corners: Optional[int] = None
    home_corners: Optional[int] = None
    away_corners: Optional[int] = None
    attendance: Optional[int] = None
    api_id: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None