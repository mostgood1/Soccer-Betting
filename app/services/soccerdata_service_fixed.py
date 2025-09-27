"""
Fixed Soccer data service for fetching real EPL data using soccerdata library.
Based on actual soccerdata API testing.
"""
import soccerdata as sd
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SoccerDataService:
    def __init__(self):
        """Initialize soccerdata connections."""
        self.fbref = sd.FBref()
        
    def get_available_leagues(self) -> List[str]:
        """Get available leagues from soccerdata."""
        try:
            leagues = self.fbref.read_leagues()
            return leagues.tolist() if not leagues.empty else []
        except Exception as e:
            logger.error(f"Error fetching leagues: {e}")
            return ["Premier League"]  # Fallback
    
    def get_epl_teams(self) -> List[Dict]:
        """
        Get EPL teams from schedule data.
        
        Returns:
            List of team dictionaries with name and basic info
        """
        try:
            # Get schedule which contains team information
            schedule = self.fbref.read_schedule()
            
            if schedule.empty:
                return []
            
            # Extract unique teams
            teams = set()
            if 'home_team' in schedule.columns:
                teams.update(schedule['home_team'].dropna().unique())
            if 'away_team' in schedule.columns:
                teams.update(schedule['away_team'].dropna().unique())
            
            team_list = []
            for team_name in sorted(teams):
                team_list.append({
                    "name": str(team_name),
                    "league": "Premier League",
                    "country": "England"
                })
            
            return team_list
            
        except Exception as e:
            logger.error(f"Error fetching EPL teams: {e}")
            return []

    def get_epl_matches(self) -> List[Dict]:
        """
        Get EPL match data from schedule.
        
        Returns:
            List of match dictionaries
        """
        try:
            schedule = self.fbref.read_schedule()
            
            if schedule.empty:
                return []
            
            matches = []
            for idx, match in schedule.iterrows():
                match_dict = {
                    "date": match.get('date'),
                    "home_team": match.get('home_team'),
                    "away_team": match.get('away_team'),
                    "home_score": match.get('home_score'),
                    "away_score": match.get('away_score'),
                    "status": "completed" if pd.notna(match.get('home_score')) else "scheduled"
                }
                matches.append(match_dict)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error fetching EPL matches: {e}")
            return []
    
    def get_player_statistics(self, stat_type: str = "standard") -> List[Dict]:
        """
        Get player statistics.
        
        Args:
            stat_type: Type of stats to fetch
            
        Returns:
            List of player stat dictionaries
        """
        try:
            player_stats = self.fbref.read_player_season_stats(stat_type=stat_type)
            
            if player_stats.empty:
                return []
            
            players = []
            for idx, player in player_stats.iterrows():
                player_dict = {
                    "name": idx if isinstance(idx, str) else str(idx),
                    "team": player.get('team', 'Unknown'),
                    "stats": player.to_dict()
                }
                players.append(player_dict)
            
            return players
            
        except Exception as e:
            logger.error(f"Error fetching player statistics: {e}")
            return []
    
    def get_team_statistics(self, stat_type: str = "standard") -> List[Dict]:
        """
        Get team statistics.
        
        Args:
            stat_type: Type of stats to fetch
            
        Returns:
            List of team stat dictionaries
        """
        try:
            team_stats = self.fbref.read_team_season_stats(stat_type=stat_type)
            
            if team_stats.empty:
                return []
            
            teams = []
            for idx, team in team_stats.iterrows():
                team_dict = {
                    "name": idx if isinstance(idx, str) else str(idx),
                    "league": "Premier League",
                    "stats": team.to_dict()
                }
                teams.append(team_dict)
            
            return teams
            
        except Exception as e:
            logger.error(f"Error fetching team statistics: {e}")
            return []
    
    def calculate_team_form(self, team_name: str, matches: List[Dict]) -> Dict:
        """
        Calculate team form based on recent matches.
        
        Args:
            team_name: Name of the team
            matches: List of match data
            
        Returns:
            Dictionary with form statistics
        """
        team_matches = []
        
        for match in matches:
            if match.get('home_team') == team_name or match.get('away_team') == team_name:
                # Only include completed matches
                if match.get('status') == 'completed':
                    team_matches.append(match)
        
        # Sort by date (most recent first)
        team_matches.sort(key=lambda x: x.get('date', ''), reverse=True)
        recent_matches = team_matches[:5]  # Last 5 matches
        
        wins = draws = losses = 0
        goals_for = goals_against = 0
        
        for match in recent_matches:
            is_home = match.get('home_team') == team_name
            
            if is_home:
                team_score = match.get('home_score', 0)
                opponent_score = match.get('away_score', 0)
            else:
                team_score = match.get('away_score', 0)
                opponent_score = match.get('home_score', 0)
            
            goals_for += team_score or 0
            goals_against += opponent_score or 0
            
            if team_score > opponent_score:
                wins += 1
            elif team_score == opponent_score:
                draws += 1
            else:
                losses += 1
        
        return {
            "matches_played": len(recent_matches),
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "form_points": wins * 3 + draws  # Standard points system
        }
    
    def test_connection(self) -> Dict:
        """Test soccerdata connection and return available data summary."""
        result = {
            "status": "unknown",
            "leagues_available": False,
            "schedule_available": False,
            "teams_count": 0,
            "matches_count": 0,
            "error": None
        }
        
        try:
            # Test leagues
            try:
                leagues = self.fbref.read_leagues()
                result["leagues_available"] = not leagues.empty
            except:
                pass
            
            # Test schedule
            try:
                schedule = self.fbref.read_schedule()
                result["schedule_available"] = not schedule.empty
                result["matches_count"] = len(schedule) if not schedule.empty else 0
                
                # Count teams
                if not schedule.empty:
                    teams = set()
                    if 'home_team' in schedule.columns:
                        teams.update(schedule['home_team'].dropna().unique())
                    if 'away_team' in schedule.columns:
                        teams.update(schedule['away_team'].dropna().unique())
                    result["teams_count"] = len(teams)
                    
            except Exception as e:
                result["error"] = str(e)
            
            result["status"] = "connected" if result["schedule_available"] else "limited"
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        
        return result