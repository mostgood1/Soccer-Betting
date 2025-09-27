# Import all models to ensure they are registered with SQLAlchemy
from app.core.database import Base

# Import models - import order matters due to relationships
from app.models.league import League  # noqa
from app.models.team import Team  # noqa  
from app.models.player import Player  # noqa
from app.models.match import Match, MatchPlayerStats  # noqa
from app.models.prediction import Prediction  # noqa
from app.models.bet_market import BetMarket  # noqa

# This ensures all models are imported and registered