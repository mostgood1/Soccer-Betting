from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime, date
from app.services.soccerdata_service import SoccerDataService

router = APIRouter()

# Initialize SoccerData service for development
soccer_data = SoccerDataService()


@router.get("/", response_model=List[dict])
async def get_matches(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    team: Optional[str] = None,
    status: Optional[str] = None,
):
    """Get matches with optional filtering - uses SoccerData service"""
    try:
        # Get matches from SoccerData service
        matches = soccer_data.get_epl_matches()

        # Apply filters
        if team:
            matches = [
                match
                for match in matches
                if team.lower() in match.get("home_team", "").lower()
                or team.lower() in match.get("away_team", "").lower()
            ]

        if status:
            matches = [match for match in matches if match.get("status") == status]

        # Apply pagination
        matches = matches[skip : skip + limit]

        # Format response
        return [
            {
                "id": idx + skip + 1,
                "match_id": match.get("id", f"match_{idx}"),
                "date": match.get("date"),
                "home_team": match.get("home_team"),
                "away_team": match.get("away_team"),
                "home_score": match.get("home_score"),
                "away_score": match.get("away_score"),
                "total_goals": match.get(
                    "total_goals",
                    (match.get("home_score", 0) + match.get("away_score", 0))
                    if match.get("home_score") is not None
                    else None,
                ),
                "first_half_goals": match.get("first_half_goals"),
                "status": match.get("status"),
                "result": get_match_result(match),
            }
            for idx, match in enumerate(matches)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching matches: {str(e)}")


def get_match_result(match: dict) -> Optional[str]:
    """Determine match result (Win/Draw/Loss from home team perspective)"""
    home_score = match.get("home_score")
    away_score = match.get("away_score")

    if home_score is None or away_score is None:
        return None

    if home_score > away_score:
        return "Win"
    elif home_score < away_score:
        return "Loss"
    else:
        return "Draw"


@router.get("/{match_id}")
async def get_match(match_id: int):
    """Get a specific match by ID"""
    try:
        matches = soccer_data.get_epl_matches()
        if 1 <= match_id <= len(matches):
            match = matches[match_id - 1]

            return {
                "id": match_id,
                "match_id": match.get("id", f"match_{match_id}"),
                "date": match.get("date"),
                "home_team": match.get("home_team"),
                "away_team": match.get("away_team"),
                "home_score": match.get("home_score"),
                "away_score": match.get("away_score"),
                "total_goals": match.get(
                    "total_goals",
                    (match.get("home_score", 0) + match.get("away_score", 0))
                    if match.get("home_score") is not None
                    else None,
                ),
                "first_half_goals": match.get("first_half_goals"),
                "status": match.get("status"),
                "result": get_match_result(match),
            }
        else:
            raise HTTPException(status_code=404, detail="Match not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching match: {str(e)}")


@router.get("/upcoming/")
async def get_upcoming_matches(limit: int = Query(10, ge=1, le=50)):
    """Get upcoming matches"""
    try:
        matches = soccer_data.get_epl_matches()
        upcoming = [match for match in matches if match.get("status") == "scheduled"]
        upcoming = upcoming[:limit]

        return [
            {
                "id": idx + 1,
                "match_id": match.get("id", f"upcoming_{idx}"),
                "date": match.get("date"),
                "home_team": match.get("home_team"),
                "away_team": match.get("away_team"),
                "status": match.get("status"),
            }
            for idx, match in enumerate(upcoming)
        ]
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching upcoming matches: {str(e)}"
        )
