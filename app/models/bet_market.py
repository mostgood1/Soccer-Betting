from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, Text, JSON
from sqlalchemy.sql import func
from app.core.database import Base


class BetMarket(Base):
    __tablename__ = "bet_markets"

    id = Column(Integer, primary_key=True, index=True)
    
    # Market identification
    market_name = Column(String, nullable=False, index=True)  # 'total_goals', '1h_goals', 'match_result', etc.
    market_type = Column(String, nullable=False)  # 'over_under', 'exact', 'outcome'
    
    # Market parameters
    line_value = Column(Float)  # e.g., 2.5 for Over/Under 2.5 goals
    description = Column(Text)
    
    # Market status
    is_active = Column(Boolean, default=True)
    
    # Market metadata
    priority = Column(Integer, default=0)  # For ordering markets
    category = Column(String, default='standard')  # 'standard', 'player_props', 'team_props'
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<BetMarket(name='{self.market_name}', type='{self.market_type}', line={self.line_value})>"


# Supported betting markets configuration
SUPPORTED_MARKETS = {
    'total_goals': {
        'name': 'Total Goals',
        'type': 'over_under',
        'common_lines': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        'description': 'Total goals scored in the match'
    },
    'first_half_goals': {
        'name': 'First Half Goals', 
        'type': 'over_under',
        'common_lines': [0.5, 1.5, 2.5],
        'description': 'Total goals scored in the first half'
    },
    'match_result': {
        'name': 'Match Result',
        'type': 'outcome',
        'options': ['home_win', 'away_win', 'draw'],
        'description': 'Final result of the match'
    },
    'total_corners': {
        'name': 'Total Corners',
        'type': 'over_under', 
        'common_lines': [8.5, 9.5, 10.5, 11.5, 12.5],
        'description': 'Total corner kicks in the match'
    },
    'team_corners': {
        'name': 'Team Corners',
        'type': 'over_under',
        'common_lines': [4.5, 5.5, 6.5],
        'description': 'Corner kicks for a specific team'
    },
    'player_goals': {
        'name': 'Player Goals',
        'type': 'over_under',
        'common_lines': [0.5, 1.5],
        'description': 'Goals scored by a specific player'
    },
    'player_assists': {
        'name': 'Player Assists',
        'type': 'over_under', 
        'common_lines': [0.5],
        'description': 'Assists by a specific player'
    },
    'player_shots': {
        'name': 'Player Shots',
        'type': 'over_under',
        'common_lines': [1.5, 2.5, 3.5],
        'description': 'Total shots by a specific player'
    },
    'player_shots_on_goal': {
        'name': 'Player Shots on Goal',
        'type': 'over_under',
        'common_lines': [0.5, 1.5, 2.5],
        'description': 'Shots on goal by a specific player'
    },
    'goalie_saves': {
        'name': 'Goalkeeper Saves',
        'type': 'over_under',
        'common_lines': [2.5, 3.5, 4.5],
        'description': 'Saves made by the goalkeeper'
    }
}