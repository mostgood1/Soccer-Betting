from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Float, Date
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    position = Column(String, nullable=False)  # GK, DEF, MID, FWD
    jersey_number = Column(Integer)
    birth_date = Column(Date)
    nationality = Column(String)
    height = Column(Integer)  # in cm
    weight = Column(Integer)  # in kg
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    api_id = Column(String, unique=True, index=True)  # External API reference
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Season statistics
    season_goals = Column(Integer, default=0)
    season_assists = Column(Integer, default=0)
    season_appearances = Column(Integer, default=0)
    season_minutes_played = Column(Integer, default=0)
    season_shots = Column(Integer, default=0)
    season_shots_on_goal = Column(Integer, default=0)
    season_saves = Column(Integer, default=0)  # For goalkeepers
    season_yellow_cards = Column(Integer, default=0)
    season_red_cards = Column(Integer, default=0)

    # Relationships
    # team = relationship("Team", back_populates="players")
    # match_stats = relationship("MatchPlayerStats", back_populates="player")

    def __repr__(self):
        return f"<Player(name='{self.name}', position='{self.position}', team='{self.team.name if self.team else 'N/A'}')>"