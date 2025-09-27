"""
Data synchronization service to populate database with soccerdata
"""
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.models.league import League
from app.models.team import Team
from app.models.player import Player
from app.models.match import Match, MatchPlayerStats
from app.services.soccerdata_service import SoccerDataService
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class DataSyncService:
    """Service to sync soccerdata with our database"""
    
    def __init__(self):
        self.soccer_data_service = SoccerDataService()
        self.db = SessionLocal()
    
    def sync_epl_teams(self, season: str = "2024-25") -> int:
        """Sync EPL teams from soccerdata to database"""
        try:
            # Get or create EPL league
            epl_league = self.db.query(League).filter(
                League.name == "Premier League",
                League.season == season
            ).first()
            
            if not epl_league:
                epl_league = League(
                    name="Premier League",
                    country="England", 
                    season=season,
                    is_active=True
                )
                self.db.add(epl_league)
                self.db.commit()
                self.db.refresh(epl_league)
            
            # Get teams from soccerdata
            teams_df = self.soccer_data_service.get_epl_teams(season)
            
            if teams_df.empty:
                logger.warning("No team data retrieved from soccerdata")
                return 0
            
            teams_synced = 0
            
            # Process each team
            for _, team_row in teams_df.iterrows():
                team_name = team_row.get('Squad', team_row.name if hasattr(team_row, 'name') else str(team_row))
                
                # Check if team already exists
                existing_team = self.db.query(Team).filter(
                    Team.name == team_name,
                    Team.league_id == epl_league.id
                ).first()
                
                if not existing_team:
                    # Create new team
                    team = Team(
                        name=team_name,
                        short_name=team_name[:3].upper(),
                        league_id=epl_league.id,
                        is_active=True
                    )
                    self.db.add(team)
                    teams_synced += 1
            
            self.db.commit()
            logger.info(f"Synced {teams_synced} EPL teams for season {season}")
            return teams_synced
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error syncing EPL teams: {str(e)}")
            return 0
    
    def sync_epl_matches(self, season: str = "2024-25") -> int:
        """Sync EPL matches from soccerdata to database"""
        try:
            # Get EPL league
            epl_league = self.db.query(League).filter(
                League.name == "Premier League",
                League.season == season
            ).first()
            
            if not epl_league:
                logger.error("EPL league not found. Run sync_epl_teams first.")
                return 0
            
            # Get matches from soccerdata
            matches_df = self.soccer_data_service.get_epl_matches(season)
            
            if matches_df.empty:
                logger.warning("No match data retrieved from soccerdata")
                return 0
            
            matches_synced = 0
            
            # Process each match
            for _, match_row in matches_df.iterrows():
                try:
                    # Extract match data
                    home_team_name = match_row.get('Home', '')
                    away_team_name = match_row.get('Away', '')
                    match_date = match_row.get('Date', datetime.now())
                    
                    # Convert date if it's a string
                    if isinstance(match_date, str):
                        match_date = pd.to_datetime(match_date)
                    
                    # Get team IDs
                    home_team = self.db.query(Team).filter(
                        Team.name == home_team_name,
                        Team.league_id == epl_league.id
                    ).first()
                    
                    away_team = self.db.query(Team).filter(
                        Team.name == away_team_name,
                        Team.league_id == epl_league.id
                    ).first()
                    
                    if not home_team or not away_team:
                        logger.warning(f"Teams not found: {home_team_name} vs {away_team_name}")
                        continue
                    
                    # Check if match already exists
                    existing_match = self.db.query(Match).filter(
                        Match.home_team_id == home_team.id,
                        Match.away_team_id == away_team.id,
                        Match.match_date == match_date,
                        Match.league_id == epl_league.id
                    ).first()
                    
                    if existing_match:
                        # Update existing match with scores if available
                        if 'Score' in match_row and pd.notna(match_row['Score']):
                            score_str = str(match_row['Score'])
                            if '–' in score_str or '-' in score_str:
                                home_score, away_score = score_str.replace('–', '-').split('-')
                                existing_match.home_score = int(home_score)
                                existing_match.away_score = int(away_score)
                                existing_match.total_goals = int(home_score) + int(away_score)
                                existing_match.status = "finished"
                        continue
                    
                    # Create new match
                    match = Match(
                        match_date=match_date,
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        league_id=epl_league.id,
                        status="scheduled"
                    )
                    
                    # Add scores if match is finished
                    if 'Score' in match_row and pd.notna(match_row['Score']):
                        score_str = str(match_row['Score'])
                        if '–' in score_str or '-' in score_str:
                            home_score, away_score = score_str.replace('–', '-').split('-')
                            match.home_score = int(home_score)
                            match.away_score = int(away_score)
                            match.total_goals = int(home_score) + int(away_score)
                            match.status = "finished"
                    
                    self.db.add(match)
                    matches_synced += 1
                    
                except Exception as e:
                    logger.error(f"Error processing match {match_row}: {str(e)}")
                    continue
            
            self.db.commit()
            logger.info(f"Synced {matches_synced} EPL matches for season {season}")
            return matches_synced
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error syncing EPL matches: {str(e)}")
            return 0
    
    def sync_player_stats(self, season: str = "2024-25") -> int:
        """Sync player statistics from soccerdata"""
        try:
            # Get player stats
            player_stats_df = self.soccer_data_service.get_player_statistics(season, "standard")
            
            if player_stats_df.empty:
                logger.warning("No player statistics retrieved from soccerdata")
                return 0
            
            players_synced = 0
            
            # Process each player
            for _, player_row in player_stats_df.iterrows():
                try:
                    player_name = player_row.get('Player', '')
                    team_name = player_row.get('Squad', '')
                    position = player_row.get('Pos', 'Unknown')
                    
                    # Find team
                    team = self.db.query(Team).filter(Team.name == team_name).first()
                    if not team:
                        continue
                    
                    # Check if player exists
                    existing_player = self.db.query(Player).filter(
                        Player.name == player_name,
                        Player.team_id == team.id
                    ).first()
                    
                    if not existing_player:
                        # Create new player
                        player = Player(
                            name=player_name,
                            position=position.split('-')[0] if '-' in position else position,
                            team_id=team.id,
                            is_active=True
                        )
                        self.db.add(player)
                        players_synced += 1
                    else:
                        # Update existing player stats
                        existing_player.season_goals = int(player_row.get('Gls', 0))
                        existing_player.season_assists = int(player_row.get('Ast', 0))
                        existing_player.season_appearances = int(player_row.get('MP', 0))
                        existing_player.season_minutes_played = int(player_row.get('Min', 0))
                        
                except Exception as e:
                    logger.error(f"Error processing player {player_row}: {str(e)}")
                    continue
            
            self.db.commit()
            logger.info(f"Synced {players_synced} new players for season {season}")
            return players_synced
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error syncing player stats: {str(e)}")
            return 0
    
    def full_sync(self, season: str = "2024-25") -> Dict[str, int]:
        """Perform full data synchronization"""
        logger.info(f"Starting full data sync for season {season}")
        
        results = {
            "teams": self.sync_epl_teams(season),
            "matches": self.sync_epl_matches(season),
            "players": self.sync_player_stats(season)
        }
        
        logger.info(f"Full sync completed: {results}")
        return results
    
    def __del__(self):
        """Close database connection"""
        if hasattr(self, 'db'):
            self.db.close()


if __name__ == "__main__":
    # Test the sync service
    print("🔄 Starting EPL data synchronization...")
    
    sync_service = DataSyncService()
    results = sync_service.full_sync()
    
    print("✅ Synchronization completed!")
    print(f"   Teams synced: {results['teams']}")
    print(f"   Matches synced: {results['matches']}")
    print(f"   Players synced: {results['players']}")