"""
Simplified Soccer data service with fallback mock data for EPL betting platform.
"""
import soccerdata as sd
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging
import random

logger = logging.getLogger(__name__)

class SoccerDataService:
    def __init__(self):
        """Initialize soccerdata connections with fallback data."""
        try:
            self.fbref = sd.FBref()
            self.connected = True
        except Exception as e:
            logger.error(f"Failed to initialize soccerdata: {e}")
            self.connected = False
    
    def get_epl_teams(self) -> List[Dict]:
        """
        Get EPL teams - fallback to known EPL teams if API fails.
        
        Returns:
            List of team dictionaries
        """
        # Try real data first
        if self.connected:
            try:
                # Try different approaches to get data
                schedule = self.fbref.read_schedule()
                if not schedule.empty:
                    teams = set()
                    if 'home_team' in schedule.columns:
                        teams.update(schedule['home_team'].dropna().unique())
                    if 'away_team' in schedule.columns:
                        teams.update(schedule['away_team'].dropna().unique())
                    
                    if teams:
                        return [{"name": str(team), "league": "Premier League"} for team in sorted(teams)]
            except Exception as e:
                logger.warning(f"Real data unavailable, using fallback: {e}")
        
        # Fallback to current EPL teams (2024-25 season)
        epl_teams = [
            "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton & Hove Albion",
            "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich Town",
            "Leicester City", "Liverpool", "Manchester City", "Manchester United", 
            "Newcastle United", "Nottingham Forest", "Southampton", "Tottenham Hotspur", 
            "West Ham United", "Wolverhampton Wanderers"
        ]
        
        return [{"name": team, "league": "Premier League", "country": "England"} for team in epl_teams]

    def get_epl_matches(self, num_matches: int = 50) -> List[Dict]:
        """
        Get EPL matches - fallback to mock data if API fails.
        
        Returns:
            List of match dictionaries
        """
        # Try real data first
        if self.connected:
            try:
                schedule = self.fbref.read_schedule()
                if not schedule.empty:
                    matches = []
                    for idx, match in schedule.iterrows():
                        match_dict = {
                            "id": f"match_{idx}",
                            "date": str(match.get('date', datetime.now().date())),
                            "home_team": str(match.get('home_team', 'Team A')),
                            "away_team": str(match.get('away_team', 'Team B')),
                            "home_score": match.get('home_score'),
                            "away_score": match.get('away_score'),
                            "status": "completed" if pd.notna(match.get('home_score')) else "scheduled"
                        }
                        matches.append(match_dict)
                    
                    if matches:
                        return matches[:num_matches]
            except Exception as e:
                logger.warning(f"Real match data unavailable, using mock data: {e}")
        
        # Generate mock match data for testing
        teams = [team["name"] for team in self.get_epl_teams()]
        matches = []
        
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(num_matches):
            home_team = random.choice(teams)
            away_team = random.choice([t for t in teams if t != home_team])
            
            match_date = base_date + timedelta(days=random.randint(0, 60))
            
            # 70% of matches are completed, 30% are upcoming
            is_completed = random.random() < 0.7
            
            match_dict = {
                "id": f"mock_match_{i+1}",
                "date": match_date.strftime("%Y-%m-%d"),
                "home_team": home_team,
                "away_team": away_team,
                "status": "completed" if is_completed else "scheduled"
            }
            
            if is_completed:
                match_dict["home_score"] = random.randint(0, 4)
                match_dict["away_score"] = random.randint(0, 4)
                match_dict["total_goals"] = match_dict["home_score"] + match_dict["away_score"]
                match_dict["first_half_goals"] = random.randint(0, match_dict["total_goals"])
            
            matches.append(match_dict)
        
        return matches

    def get_player_statistics(self) -> List[Dict]:
        """Get player statistics with mock data fallback."""
        teams = [team["name"] for team in self.get_epl_teams()]
        
        # Mock player data for betting platform
        players = []
        for team in teams:
            # Generate 5-8 players per team
            for i in range(random.randint(5, 8)):
                position = random.choice(["Forward", "Midfielder", "Defender", "Goalkeeper"])
                
                player = {
                    "name": f"{team} Player {i+1}",
                    "team": team,
                    "position": position,
                    "goals": random.randint(0, 15) if position != "Goalkeeper" else 0,
                    "assists": random.randint(0, 10) if position != "Goalkeeper" else random.randint(0, 2),
                    "shots": random.randint(5, 50),
                    "shots_on_goal": random.randint(2, 25),
                    "saves": random.randint(20, 80) if position == "Goalkeeper" else 0,
                    "matches_played": random.randint(5, 20)
                }
                players.append(player)
        
        return players

    def get_team_statistics(self) -> List[Dict]:
        """Get team statistics with mock data."""
        teams = self.get_epl_teams()
        team_stats = []
        
        for team in teams:
            stats = {
                "name": team["name"],
                "league": team["league"],
                "matches_played": random.randint(15, 25),
                "wins": random.randint(5, 15),
                "draws": random.randint(2, 8),
                "losses": random.randint(2, 10),
                "goals_for": random.randint(20, 50),
                "goals_against": random.randint(15, 40),
                "total_corners": random.randint(80, 150),
                "avg_corners_per_match": random.uniform(4.0, 8.0),
                "total_shots": random.randint(200, 400),
                "shots_on_goal": random.randint(100, 200),
                "form": random.choice(["Excellent", "Good", "Average", "Poor"])
            }
            
            stats["points"] = stats["wins"] * 3 + stats["draws"]
            stats["goal_difference"] = stats["goals_for"] - stats["goals_against"]
            
            team_stats.append(stats)
        
        return team_stats

    def calculate_team_form(self, team_name: str, num_matches: int = 5) -> Dict:
        """Calculate team form from recent matches."""
        matches = self.get_epl_matches()
        team_matches = []
        
        for match in matches:
            if match.get('home_team') == team_name or match.get('away_team') == team_name:
                if match.get('status') == 'completed':
                    team_matches.append(match)
        
        # Sort by date and take recent matches
        team_matches.sort(key=lambda x: x.get('date', ''), reverse=True)
        recent_matches = team_matches[:num_matches]
        
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
            
            goals_for += team_score
            goals_against += opponent_score
            
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
            "form_points": wins * 3 + draws,
            "form_string": "W" * wins + "D" * draws + "L" * losses
        }

    def test_connection(self) -> Dict:
        """Test service functionality."""
        teams = self.get_epl_teams()
        matches = self.get_epl_matches(10)
        
        return {
            "status": "connected" if self.connected else "mock_mode",
            "teams_available": len(teams) > 0,
            "matches_available": len(matches) > 0,
            "teams_count": len(teams),
            "sample_teams": [team["name"] for team in teams[:5]],
            "mock_data": not self.connected
        }