from fastapi import APIRouter
from app.api.v1.endpoints import matches, predictions, teams, players, leagues

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(leagues.router, prefix="/leagues", tags=["leagues"])
api_router.include_router(teams.router, prefix="/teams", tags=["teams"])
api_router.include_router(players.router, prefix="/players", tags=["players"])
api_router.include_router(matches.router, prefix="/matches", tags=["matches"])
api_router.include_router(
    predictions.router, prefix="/predictions", tags=["predictions"]
)
