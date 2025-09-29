from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from app.services.soccerdata_service import SoccerDataService

router = APIRouter()

# Initialize SoccerData service for development
soccer_data = SoccerDataService()


@router.get("/", response_model=List[dict])
async def get_teams(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    league: Optional[str] = None,
):
    """Get teams with optional filtering - uses SoccerData service for development"""
    try:
        # Get teams from SoccerData service
        teams = soccer_data.get_epl_teams()

        # Filter by league if specified
        if league:
            teams = [
                team
                for team in teams
                if league.lower() in team.get("league", "").lower()
            ]

        # Apply pagination
        total_teams = len(teams)
        teams = teams[skip : skip + limit]

        # Return with additional metadata
        return [
            {
                "id": idx + skip + 1,
                "name": team["name"],
                "league": team.get("league", "Premier League"),
                "country": team.get("country", "England"),
            }
            for idx, team in enumerate(teams)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching teams: {str(e)}")


@router.get("/{team_id}")
async def get_team(team_id: int):
    """Get a specific team by ID"""
    try:
        teams = soccer_data.get_epl_teams()
        if 1 <= team_id <= len(teams):
            team = teams[team_id - 1]

            # Get additional team statistics
            team_stats = soccer_data.get_team_statistics()
            stats = next((s for s in team_stats if s["name"] == team["name"]), {})

            return {
                "id": team_id,
                "name": team["name"],
                "league": team.get("league", "Premier League"),
                "country": team.get("country", "England"),
                "stats": stats,
            }
        else:
            raise HTTPException(status_code=404, detail="Team not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching team: {str(e)}")


@router.get("/{team_id}/form")
async def get_team_form(team_id: int):
    """Get team form (recent match results)"""
    try:
        teams = soccer_data.get_epl_teams()
        if 1 <= team_id <= len(teams):
            team = teams[team_id - 1]
            form = soccer_data.calculate_team_form(team["name"])

            return {"team_id": team_id, "team_name": team["name"], "form": form}
        else:
            raise HTTPException(status_code=404, detail="Team not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching team form: {str(e)}"
        )
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    return {"id": team.id, "name": team.name, "short_name": team.short_name}
