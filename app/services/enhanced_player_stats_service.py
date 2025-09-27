"""
Enhanced Player Statistics Service
Provides comprehensive player stats including goals, assists, cards, shots, etc.
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class EnhancedPlayerStatsService:
    """Service for comprehensive player statistics"""
    
    def __init__(self):
        self.season = "2025-26"
        self.cache = {}
        # REAL_DATA_ONLY mode disables synthetic generation
        self.real_only = os.getenv('REAL_DATA_ONLY', '1') == '1'
        if self.real_only:
            self.is_mock = False
            self.is_available = False
            self.players_data = []
            print(f"Enhanced Player Stats Service (REAL_DATA_ONLY): no player stats provider configured; returning empty data")
        else:
            # NOTE: In non-real-only mode, this service generates synthetic player stats
            self.is_mock = True
            self.is_available = True
            self.players_data = self._generate_comprehensive_player_stats()
            print(f"Enhanced Player Stats Service initialized for {self.season} (mock stats)")
    
    def _generate_comprehensive_player_stats(self) -> List[Dict]:
        """Generate comprehensive player statistics for EPL 2025-26"""
        
        # Real EPL players with realistic stats for current season
        players_base_data = [
            # Manchester City
            {'name': 'Erling Haaland', 'team': 'Manchester City', 'position': 'Forward', 'age': 25},
            {'name': 'Phil Foden', 'team': 'Manchester City', 'position': 'Midfielder', 'age': 24},
            {'name': 'Kevin De Bruyne', 'team': 'Manchester City', 'position': 'Midfielder', 'age': 32},
            {'name': 'Rodri', 'team': 'Manchester City', 'position': 'Midfielder', 'age': 28},
            
            # Arsenal
            {'name': 'Bukayo Saka', 'team': 'Arsenal', 'position': 'Forward', 'age': 23},
            {'name': 'Martin Ødegaard', 'team': 'Arsenal', 'position': 'Midfielder', 'age': 26},
            {'name': 'Kai Havertz', 'team': 'Arsenal', 'position': 'Forward', 'age': 26},
            {'name': 'William Saliba', 'team': 'Arsenal', 'position': 'Defender', 'age': 24},
            
            # Liverpool
            {'name': 'Mohamed Salah', 'team': 'Liverpool', 'position': 'Forward', 'age': 33},
            {'name': 'Sadio Mané', 'team': 'Liverpool', 'position': 'Forward', 'age': 33},
            {'name': 'Virgil van Dijk', 'team': 'Liverpool', 'position': 'Defender', 'age': 34},
            {'name': 'Alisson Becker', 'team': 'Liverpool', 'position': 'Goalkeeper', 'age': 32},
            
            # Chelsea  
            {'name': 'Cole Palmer', 'team': 'Chelsea', 'position': 'Forward', 'age': 23},
            {'name': 'Nicolas Jackson', 'team': 'Chelsea', 'position': 'Forward', 'age': 24},
            {'name': 'Enzo Fernández', 'team': 'Chelsea', 'position': 'Midfielder', 'age': 24},
            {'name': 'Thiago Silva', 'team': 'Chelsea', 'position': 'Defender', 'age': 41},
            
            # Newcastle United
            {'name': 'Alexander Isak', 'team': 'Newcastle United', 'position': 'Forward', 'age': 26},
            {'name': 'Bruno Guimarães', 'team': 'Newcastle United', 'position': 'Midfielder', 'age': 27},
            {'name': 'Anthony Gordon', 'team': 'Newcastle United', 'position': 'Forward', 'age': 24},
            
            # Tottenham
            {'name': 'Son Heung-min', 'team': 'Tottenham', 'position': 'Forward', 'age': 33},
            {'name': 'James Maddison', 'team': 'Tottenham', 'position': 'Midfielder', 'age': 28},
            {'name': 'Dejan Kulusevski', 'team': 'Tottenham', 'position': 'Forward', 'age': 25},
            
            # Manchester United
            {'name': 'Marcus Rashford', 'team': 'Manchester United', 'position': 'Forward', 'age': 28},
            {'name': 'Bruno Fernandes', 'team': 'Manchester United', 'position': 'Midfielder', 'age': 31},
            {'name': 'Casemiro', 'team': 'Manchester United', 'position': 'Midfielder', 'age': 33},
            
            # Aston Villa
            {'name': 'Ollie Watkins', 'team': 'Aston Villa', 'position': 'Forward', 'age': 29},
            {'name': 'John McGinn', 'team': 'Aston Villa', 'position': 'Midfielder', 'age': 30},
            {'name': 'Emiliano Martínez', 'team': 'Aston Villa', 'position': 'Goalkeeper', 'age': 32},
            
            # Brighton
            {'name': 'João Pedro', 'team': 'Brighton', 'position': 'Forward', 'age': 24},
            {'name': 'Kaoru Mitoma', 'team': 'Brighton', 'position': 'Forward', 'age': 28},
            
            # West Ham
            {'name': 'Jarrod Bowen', 'team': 'West Ham United', 'position': 'Forward', 'age': 28},
            {'name': 'Lucas Paquetá', 'team': 'West Ham United', 'position': 'Midfielder', 'age': 28},
            
            # Brentford  
            {'name': 'Ivan Toney', 'team': 'Brentford', 'position': 'Forward', 'age': 29},
            {'name': 'Bryan Mbeumo', 'team': 'Brentford', 'position': 'Forward', 'age': 26},
            
            # Other key players
            {'name': 'Dominic Solanke', 'team': 'AFC Bournemouth', 'position': 'Forward', 'age': 28},
            {'name': 'Chris Wood', 'team': 'Nottingham Forest', 'position': 'Forward', 'age': 33},
            {'name': 'Matheus Cunha', 'team': 'Wolverhampton Wanderers', 'position': 'Forward', 'age': 26},
            {'name': 'Eberechi Eze', 'team': 'Crystal Palace', 'position': 'Forward', 'age': 27}
        ]
        
        enhanced_players = []
        
        for player in players_base_data:
            # Generate realistic stats based on position and team strength
            enhanced_stats = self._generate_player_season_stats(player)
            enhanced_players.append(enhanced_stats)
        
        # Sort by goals scored
        enhanced_players.sort(key=lambda x: x['goals'], reverse=True)
        
        return enhanced_players
    
    def _generate_player_season_stats(self, player: Dict) -> Dict:
        """Generate realistic season statistics for a player"""
        position = player['position']
        age = player['age']
        
        # Base stats influenced by position and age
        if position == 'Forward':
            goals = max(0, int(np.random.normal(12, 8)))  # Forwards score more
            assists = max(0, int(np.random.normal(4, 3)))
            shots = max(goals * 4, int(np.random.normal(80, 30)))
            shots_on_target = int(shots * np.random.uniform(0.3, 0.5))
            
        elif position == 'Midfielder':
            goals = max(0, int(np.random.normal(5, 4)))
            assists = max(0, int(np.random.normal(8, 5)))  # Midfielders assist more
            shots = max(goals * 3, int(np.random.normal(50, 25)))
            shots_on_target = int(shots * np.random.uniform(0.25, 0.4))
            
        elif position == 'Defender':
            goals = max(0, int(np.random.normal(2, 2)))
            assists = max(0, int(np.random.normal(3, 2)))
            shots = max(goals * 2, int(np.random.normal(20, 15)))
            shots_on_target = int(shots * np.random.uniform(0.2, 0.35))
            
        else:  # Goalkeeper
            goals = 0
            assists = max(0, int(np.random.normal(1, 1)))
            shots = 0
            shots_on_target = 0
        
        # Age factor (peak performance around 26-28)
        age_factor = 1.0
        if age < 22:
            age_factor = 0.85  # Young players still developing
        elif age > 32:
            age_factor = 0.9   # Older players slightly declining
        
        # Apply age factor
        goals = int(goals * age_factor)
        assists = int(assists * age_factor)
        
        # Additional stats
        games_played = max(5, int(np.random.normal(25, 8)))  # Games this season
        minutes_played = games_played * int(np.random.normal(75, 20))
        
        # Disciplinary stats
        yellow_cards = max(0, int(np.random.normal(3, 2)))
        red_cards = max(0, int(np.random.poisson(0.1)))
        
        # Passing stats (higher for midfielders and defenders)
        if position in ['Midfielder', 'Defender']:
            passes_attempted = int(np.random.normal(1500, 600))
            pass_accuracy = np.random.uniform(0.82, 0.95)
        else:
            passes_attempted = int(np.random.normal(800, 400))
            pass_accuracy = np.random.uniform(0.75, 0.88)
        
        passes_completed = int(passes_attempted * pass_accuracy)
        
        # Special stats for goalkeepers
        goalkeeper_stats = {}
        if position == 'Goalkeeper':
            goalkeeper_stats = {
                'saves': int(np.random.normal(60, 25)),
                'clean_sheets': max(0, int(np.random.normal(8, 4))),
                'goals_conceded': max(0, int(np.random.normal(20, 10))),
                'penalty_saves': max(0, int(np.random.poisson(1))),
                'save_percentage': round(np.random.uniform(0.65, 0.80), 3)
            }
        
        # Performance ratings
        average_rating = round(np.random.uniform(6.2, 8.5), 2)
        form_rating = round(np.random.uniform(0.6, 1.0), 2)
        
        return {
            'name': player['name'],
            'team': player['team'],
            'position': player['position'],
            'age': age,
            'season': self.season,
            
            # Core stats
            'games_played': games_played,
            'minutes_played': minutes_played,
            'goals': goals,
            'assists': assists,
            'shots': shots,
            'shots_on_target': shots_on_target,
            
            # Disciplinary
            'yellow_cards': yellow_cards,
            'red_cards': red_cards,
            
            # Passing
            'passes_attempted': passes_attempted,
            'passes_completed': passes_completed,
            'pass_accuracy': round(pass_accuracy, 3),
            
            # Performance metrics
            'average_rating': average_rating,
            'form_rating': form_rating,
            'goals_per_game': round(goals / max(games_played, 1), 2),
            'assists_per_game': round(assists / max(games_played, 1), 2),
            'minutes_per_goal': int(minutes_played / max(goals, 1)) if goals > 0 else 0,
            'minutes_per_assist': int(minutes_played / max(assists, 1)) if assists > 0 else 0,
            'shot_accuracy': round(shots_on_target / max(shots, 1), 3) if shots > 0 else 0,
            
            # Goalkeeper specific stats
            **goalkeeper_stats,
            
            # Market value (mock)
            'market_value_millions': max(5, int(np.random.normal(30, 20))),
            'injury_status': np.random.choice(['fit', 'minor_knock', 'injured'], p=[0.85, 0.12, 0.03]),
            'contract_expires': f"202{np.random.randint(6, 9)}"
        }
    
    def get_top_scorers(self, limit: int = 20) -> List[Dict]:
        """Get top goal scorers"""
        return sorted(self.players_data, key=lambda x: x['goals'], reverse=True)[:limit]
    
    def get_top_assisters(self, limit: int = 20) -> List[Dict]:
        """Get top assist providers"""
        return sorted(self.players_data, key=lambda x: x['assists'], reverse=True)[:limit]
    
    def get_team_players(self, team_name: str) -> List[Dict]:
        """Get all players from a specific team"""
        if not getattr(self, 'is_available', False):
            return []
        return [p for p in self.players_data if p['team'] == team_name]
    
    def get_players_by_position(self, position: str) -> List[Dict]:
        """Get players by position"""
        return [p for p in self.players_data if p['position'] == position]
    
    def get_top_performers(self, metric: str = 'average_rating', limit: int = 15) -> List[Dict]:
        """Get top performers by any metric"""
        if metric not in ['average_rating', 'goals', 'assists', 'form_rating', 'pass_accuracy']:
            metric = 'average_rating'
        
        return sorted(self.players_data, key=lambda x: x.get(metric, 0), reverse=True)[:limit]
    
    def get_disciplinary_stats(self) -> Dict:
        """Get disciplinary statistics across the league"""
        total_yellows = sum(p['yellow_cards'] for p in self.players_data)
        total_reds = sum(p['red_cards'] for p in self.players_data)
        
        most_cards = sorted(
            self.players_data, 
            key=lambda x: x['yellow_cards'] + (x['red_cards'] * 2), 
            reverse=True
        )[:10]
        
        return {
            'total_yellow_cards': total_yellows,
            'total_red_cards': total_reds,
            'average_yellows_per_player': round(total_yellows / len(self.players_data), 2),
            'most_carded_players': most_cards,
            'season': self.season
        }
    
    def get_goalkeeper_stats(self) -> List[Dict]:
        """Get goalkeeper-specific statistics"""
        goalkeepers = [p for p in self.players_data if p['position'] == 'Goalkeeper']
        
        # Sort by save percentage
        return sorted(goalkeepers, key=lambda x: x.get('save_percentage', 0), reverse=True)
    
    def get_player_stats(self, player_name: str) -> Optional[Dict]:
        """Get detailed stats for a specific player"""
        for player in self.players_data:
            if player['name'].lower() == player_name.lower():
                return player
        return None
    
    def get_stats_summary(self) -> Dict:
        """Get overall statistics summary"""
        total_players = len(self.players_data)
        total_goals = sum(p['goals'] for p in self.players_data) if self.players_data else 0
        total_assists = sum(p['assists'] for p in self.players_data) if self.players_data else 0
        
        return {
            'total_players': total_players,
            'total_goals': total_goals,
            'total_assists': total_assists,
            'average_goals_per_player': round(total_goals / total_players, 2) if total_players else 0,
            'average_assists_per_player': round(total_assists / total_players, 2) if total_players else 0,
            'top_scorer': self.get_top_scorers(1)[0] if self.players_data else None,
            'top_assister': self.get_top_assisters(1)[0] if self.players_data else None,
            'season': self.season,
            'is_mock': getattr(self, 'is_mock', False),
            'is_available': getattr(self, 'is_available', False)
        }


# Initialize global enhanced player stats service
enhanced_player_stats_service = EnhancedPlayerStatsService()


def main():
    """Test the enhanced player stats service"""
    service = EnhancedPlayerStatsService()
    
    print("=== ENHANCED PLAYER STATS SERVICE TEST ===")
    
    # Test top scorers
    top_scorers = service.get_top_scorers(5)
    print(f"Top 5 scorers:")
    for i, player in enumerate(top_scorers, 1):
        print(f"{i}. {player['name']} ({player['team']}) - {player['goals']} goals")
    
    # Test summary stats
    summary = service.get_stats_summary()
    print(f"\nTotal goals across league: {summary['total_goals']}")
    print(f"Average goals per player: {summary['average_goals_per_player']}")
    
    print(f"\n✅ Enhanced Player Stats Service ready!")


if __name__ == "__main__":
    main()