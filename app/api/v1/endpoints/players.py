from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from app.services.soccerdata_service import SoccerDataService

router = APIRouter()

# Initialize SoccerData service for development
soccer_data = SoccerDataService()


@router.get("/", response_model=List[dict])
async def get_players(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    team: Optional[str] = None,
    position: Optional[str] = None,
):
    """Get players with optional filtering - uses SoccerData service"""
    try:
        # Get players from SoccerData service
        players = soccer_data.get_player_statistics()
        
        # Apply filters
        if team:
            players = [
                player for player in players
                if team.lower() in player.get("team", "").lower()
            ]
        
        if position:
            players = [
                player for player in players
                if position.lower() in player.get("position", "").lower()
            ]
        
        # Apply pagination
        players = players[skip:skip + limit]
        
        # Format response with betting-relevant stats
        return [
            {
                "id": idx + skip + 1,
                "name": player.get("name"),
                "team": player.get("team"),
                "position": player.get("position"),
                "goals": player.get("goals", 0),
                "assists": player.get("assists", 0),
                "shots": player.get("shots", 0),
                "shots_on_goal": player.get("shots_on_goal", 0),
                "saves": player.get("saves", 0),
                "matches_played": player.get("matches_played", 0),
                "goals_per_match": round(player.get("goals", 0) / max(player.get("matches_played", 1), 1), 2),
                "shots_per_match": round(player.get("shots", 0) / max(player.get("matches_played", 1), 1), 2)
            }
            for idx, player in enumerate(players)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching players: {str(e)}")


@router.get("/{player_id}")
async def get_player(player_id: int):
    """Get a specific player by ID"""
    try:
        players = soccer_data.get_player_statistics()
        if 1 <= player_id <= len(players):
            player = players[player_id - 1]
            
            return {
                "id": player_id,
                "name": player.get("name"),
                "team": player.get("team"),
                "position": player.get("position"),
                "stats": {
                    "goals": player.get("goals", 0),
                    "assists": player.get("assists", 0),
                    "shots": player.get("shots", 0),
                    "shots_on_goal": player.get("shots_on_goal", 0),
                    "saves": player.get("saves", 0),
                    "matches_played": player.get("matches_played", 0),
                    "goals_per_match": round(player.get("goals", 0) / max(player.get("matches_played", 1), 1), 2),
                    "assists_per_match": round(player.get("assists", 0) / max(player.get("matches_played", 1), 1), 2),
                    "shots_per_match": round(player.get("shots", 0) / max(player.get("matches_played", 1), 1), 2),
                    "shot_accuracy": round(
                        (player.get("shots_on_goal", 0) / max(player.get("shots", 1), 1)) * 100, 1
                    ) if player.get("shots", 0) > 0 else 0
                }
            }
        else:
            raise HTTPException(status_code=404, detail="Player not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching player: {str(e)}")


@router.get("/top-scorers/")
async def get_top_scorers(limit: int = Query(10, ge=1, le=50)):
    """Get top goal scorers"""
    try:
        players = soccer_data.get_player_statistics()
        # Sort by goals in descending order
        top_scorers = sorted(players, key=lambda x: x.get("goals", 0), reverse=True)
        top_scorers = top_scorers[:limit]
        
        return [
            {
                "rank": idx + 1,
                "name": player.get("name"),
                "team": player.get("team"),
                "goals": player.get("goals", 0),
                "matches_played": player.get("matches_played", 0),
                "goals_per_match": round(player.get("goals", 0) / max(player.get("matches_played", 1), 1), 2)
            }
            for idx, player in enumerate(top_scorers)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching top scorers: {str(e)}")


@router.get("/goalkeepers/")
async def get_goalkeepers(limit: int = Query(20, ge=1, le=50)):
    """Get goalkeeper statistics"""
    try:
        players = soccer_data.get_player_statistics()
        goalkeepers = [player for player in players if player.get("position") == "Goalkeeper"]
        # Sort by saves in descending order
        goalkeepers = sorted(goalkeepers, key=lambda x: x.get("saves", 0), reverse=True)
        goalkeepers = goalkeepers[:limit]
        
        return [
            {
                "name": player.get("name"),
                "team": player.get("team"),
                "saves": player.get("saves", 0),
                "matches_played": player.get("matches_played", 0),
                "saves_per_match": round(player.get("saves", 0) / max(player.get("matches_played", 1), 1), 2)
            }
            for player in goalkeepers
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching goalkeepers: {str(e)}")