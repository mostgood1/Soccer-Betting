from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    short_name = Column(String, nullable=False)
    logo_url = Column(String)
    founded_year = Column(Integer)
    venue_name = Column(String)
    venue_capacity = Column(Integer)
    league_id = Column(Integer, ForeignKey("leagues.id"), nullable=False)
    api_id = Column(String, unique=True, index=True)  # External API reference
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    # league = relationship("League", back_populates="teams")
    # home_matches = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    # away_matches = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")
    # players = relationship("Player", back_populates="team")

    def __repr__(self):
        return f"<Team(name='{self.name}', short_name='{self.short_name}')>"
