from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    ForeignKey,
    Float,
    Date,
    Text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, index=True)
    match_date = Column(DateTime, nullable=False, index=True)
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    league_id = Column(Integer, ForeignKey("leagues.id"), nullable=False)

    # Match status
    status = Column(
        String, default="scheduled"
    )  # scheduled, live, finished, postponed, cancelled

    # Full time scores
    home_score = Column(Integer)
    away_score = Column(Integer)

    # Half time scores
    home_score_ht = Column(Integer)
    away_score_ht = Column(Integer)

    # Match statistics
    total_goals = Column(Integer, default=0)
    total_goals_ht = Column(Integer, default=0)  # First half goals
    total_corners = Column(Integer, default=0)
    home_corners = Column(Integer, default=0)
    away_corners = Column(Integer, default=0)

    # Additional match data
    attendance = Column(Integer)
    referee = Column(String)
    venue = Column(String)
    weather_conditions = Column(String)

    # API reference
    api_id = Column(String, unique=True, index=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    # home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    # away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    # league = relationship("League", back_populates="matches")
    # player_stats = relationship("MatchPlayerStats", back_populates="match")
    # predictions = relationship("Prediction", back_populates="match")

    @property
    def result(self):
        """Get match result: 'H' for home win, 'A' for away win, 'D' for draw"""
        if self.home_score is None or self.away_score is None:
            return None
        if self.home_score > self.away_score:
            return "H"
        elif self.away_score > self.home_score:
            return "A"
        else:
            return "D"

    def __repr__(self):
        return f"<Match({self.home_team.name if self.home_team else 'TBD'} vs {self.away_team.name if self.away_team else 'TBD'} on {self.match_date})>"


class MatchPlayerStats(Base):
    __tablename__ = "match_player_stats"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)

    # Player performance in this match
    goals = Column(Integer, default=0)
    assists = Column(Integer, default=0)
    shots = Column(Integer, default=0)
    shots_on_goal = Column(Integer, default=0)
    saves = Column(Integer, default=0)  # For goalkeepers
    minutes_played = Column(Integer, default=0)
    yellow_cards = Column(Integer, default=0)
    red_cards = Column(Integer, default=0)

    # Starting lineup
    is_starter = Column(Boolean, default=False)
    position_played = Column(String)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    # match = relationship("Match", back_populates="player_stats")
    # player = relationship("Player", back_populates="match_stats")

    def __repr__(self):
        return f"<MatchPlayerStats(player='{self.player.name if self.player else 'TBD'}', match='{self.match.id}')>"
