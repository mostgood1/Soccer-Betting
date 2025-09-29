"""
Enhanced EPL Service for PREDICTABLE BETTING ENGINE
Integrates real EPL data with advanced ML features for 2025-2026 season
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

# Optional dependency: `premier_league` is not guaranteed in all envs (e.g., Render)
# Provide defensive stubs so this module can import and gracefully fall back.
try:
    from premier_league import RankingTable, MatchStatistics, PlayerSeasonLeaders  # type: ignore
    _PREMIER_LEAGUE_AVAILABLE = True
except Exception:  # pragma: no cover
    _PREMIER_LEAGUE_AVAILABLE = False

    class RankingTable:  # minimal stub
        def __init__(self, *args, **kwargs) -> None:
            pass

        # enhanced_epl_service.get_detailed_team_stats() calls get_ranking()
        def get_ranking(self):
            return None  # returning None triggers the built-in fallback path

    class MatchStatistics:  # minimal stub (not used directly here)
        def __init__(self, *args, **kwargs) -> None:
            pass

    class PlayerSeasonLeaders:  # minimal stub used by get_top_scorer_predictions()
        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_goal_leaders(self):
            try:
                import pandas as pd  # local import to avoid global dependency if pandas missing
                return pd.DataFrame([])
            except Exception:
                return None

logger = logging.getLogger(__name__)


class EnhancedEPLService:
    """Enhanced EPL service with advanced predictive analytics"""
    
    def __init__(self):
        self.current_season = "2025-26"
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 3600  # 1 hour cache
        # Throttle noisy warnings for team stats issues
        self._team_stats_warned_once = False
        
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cache_expiry:
            return False
        return datetime.now() < self.cache_expiry[key]
    
    def _cache_data(self, key: str, data: Any) -> None:
        """Cache data with expiry"""
        self.cache[key] = data
        self.cache_expiry[key] = datetime.now() + timedelta(seconds=self.cache_duration)

    def get_detailed_team_stats(self, team_name: Optional[str] = None) -> Dict:
        """Get comprehensive team statistics for predictions"""
        cache_key = f"detailed_stats_{team_name or 'all'}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            ranking_table = RankingTable()
            teams_data = None
            try:
                teams_data = ranking_table.get_ranking()
            except Exception as inner_e:
                logger.warning(f"RankingTable.get_ranking failed: {inner_e}")
            # Coerce to DataFrame if possible; bail out to fallback if we can't
            if teams_data is None:
                fallback = self._get_fallback_team_stats()
                # cache fallback to avoid repeated fetch attempts during the cache window
                self._cache_data(cache_key, fallback)
                return fallback
            if not isinstance(teams_data, pd.DataFrame):
                try:
                    teams_data = pd.DataFrame(teams_data)
                except Exception:
                    fallback = self._get_fallback_team_stats()
                    self._cache_data(cache_key, fallback)
                    return fallback
            if getattr(teams_data, 'empty', True):
                fallback = self._get_fallback_team_stats()
                self._cache_data(cache_key, fallback)
                return fallback
            
            # Enhanced statistics calculation
            enhanced_stats = []
            
            # Normalize column names (case-insensitive) to handle variants like 'GF', 'GA', 'Pts', etc.
            def get_val(s: pd.Series, candidates, default=None):
                if isinstance(candidates, (list, tuple)):
                    for k in candidates:
                        if k in s:
                            v = s.get(k)
                            if v is not None and v == v:  # exclude NaN
                                return v
                        # try case-insensitive
                        for col in s.index:
                            try:
                                if str(col).strip().lower() == str(k).strip().lower():
                                    v = s.get(col)
                                    if v is not None and v == v:
                                        return v
                            except Exception:
                                continue
                else:
                    if candidates in s:
                        return s.get(candidates, default)
                return default

            for _, team in teams_data.iterrows():
                # Pull safe basics with generous fallbacks
                name = get_val(team, ['Team', 'Club', 'Name', 'team', 'club', 'name'], 'Unknown')
                played = get_val(team, ['Played', 'P', 'Matches', 'MP', 'played'], 0) or 0
                won = get_val(team, ['Won', 'W', 'won'], 0) or 0
                drawn = get_val(team, ['Drawn', 'D', 'drawn'], 0) or 0
                lost = get_val(team, ['Lost', 'L', 'lost'], 0) or 0
                points = get_val(team, ['Points', 'Pts', 'points', 'pts'], 0) or 0
                gf = get_val(team, ['Goals For', 'GF', 'For', 'GoalsFor', 'goals_for'], 0) or 0
                ga = get_val(team, ['Goals Against', 'GA', 'Against', 'GoalsAgainst', 'goals_against'], 0) or 0
                gd = get_val(team, ['Goal Difference', 'GD', 'Diff', 'GoalDiff', 'goal_difference'], None)
                position = get_val(team, ['Position', 'Pos', 'position', 'pos'], 0) or 0
                # numeric coercions
                try:
                    played = int(played)
                except Exception:
                    played = 0
                for key_name in ('won','drawn','lost','points','gf','ga','position'):
                    try:
                        locals()[key_name] = int(locals()[key_name])
                    except Exception:
                        locals()[key_name] = int(float(locals()[key_name])) if str(locals()[key_name]).replace('.','',1).isdigit() else 0
                try:
                    gd = int(gd) if gd is not None else (gf - ga)
                except Exception:
                    gd = gf - ga
                team_stats = {
                    'name': str(name) if name is not None else 'Unknown',
                    'position': position,
                    'points': points,
                    'matches_played': played,
                    'wins': won,
                    'draws': drawn,
                    'losses': lost,
                    'goals_for': gf,
                    'goals_against': ga,
                    'goal_difference': gd,
                    
                    # Advanced metrics for predictions
                    'points_per_game': (float(points) / max(int(played), 1)) if isinstance(points, (int, float)) else 0.0,
                    'goals_per_game': (float(gf) / max(int(played), 1)) if isinstance(gf, (int, float)) else 0.0,
                    'goals_conceded_per_game': (float(ga) / max(int(played), 1)) if isinstance(ga, (int, float)) else 0.0,
                    'win_rate': (float(won) / max(int(played), 1)) if isinstance(won, (int, float)) else 0.0,
                    'clean_sheet_rate': self._calculate_clean_sheets(team),
                    'scoring_consistency': self._calculate_scoring_consistency(team),
                    'defensive_strength': self._calculate_defensive_strength(team),
                    'form_score': self._calculate_form_score(team),
                    'home_advantage': self._calculate_home_advantage(team),
                    'away_performance': self._calculate_away_performance(team)
                }
                
                enhanced_stats.append(team_stats)
            
            result = {
                'teams': enhanced_stats,
                'league_averages': self._calculate_league_averages(enhanced_stats),
                'last_updated': datetime.now().isoformat(),
                'season': self.current_season
            }
            
            if team_name:
                # Filter for specific team
                try:
                    result['teams'] = [t for t in enhanced_stats if str(t.get('name','')).lower() == str(team_name).lower()]
                except Exception:
                    result['teams'] = []
            
            self._cache_data(cache_key, result)
            return result
            
        except Exception as e:
            # Reduce log noise: warn once, debug thereafter
            if not self._team_stats_warned_once:
                logger.warning(f"Error getting detailed team stats (using fallback once): {e}")
                self._team_stats_warned_once = True
            else:
                logger.debug(f"Error getting detailed team stats (fallback): {e}")
            fb = self._get_fallback_team_stats()
            self._cache_data(cache_key, fb)
            return fb

    def get_match_prediction_features(self, home_team: str, away_team: str) -> Dict:
        """Extract comprehensive features for match prediction"""
        try:
            # Get team stats
            team_stats = self.get_detailed_team_stats() or {}
            teams_list = team_stats.get('teams', []) if isinstance(team_stats, dict) else []
            home_stats = next((t for t in teams_list if str(t.get('name','')).lower() == home_team.lower()), None)
            away_stats = next((t for t in teams_list if str(t.get('name','')).lower() == away_team.lower()), None)
            
            if not home_stats or not away_stats:
                return self._get_fallback_prediction_features()
            
            # Head-to-head data
            h2h_stats = self.get_head_to_head_stats(home_team, away_team)
            
            # Recent form (last 5 matches)
            home_form = self.get_recent_form(home_team, 5)
            away_form = self.get_recent_form(away_team, 5)
            
            # Match prediction features
            features = {
                # Basic team strength
                'home_points_per_game': home_stats['points_per_game'],
                'away_points_per_game': away_stats['points_per_game'],
                'home_goals_per_game': home_stats['goals_per_game'],
                'away_goals_per_game': away_stats['goals_per_game'],
                'home_goals_conceded_per_game': home_stats['goals_conceded_per_game'],
                'away_goals_conceded_per_game': away_stats['goals_conceded_per_game'],
                
                # Advanced metrics
                'goal_difference_gap': home_stats['goal_difference'] - away_stats['goal_difference'],
                'position_difference': away_stats['position'] - home_stats['position'],  # Negative favors home
                'home_win_rate': home_stats['win_rate'],
                'away_win_rate': away_stats['win_rate'],
                
                # Form and momentum
                'home_form_score': home_form['form_score'],
                'away_form_score': away_form['form_score'],
                'home_recent_goals': home_form['recent_goals_avg'],
                'away_recent_goals': away_form['recent_goals_avg'],
                
                # Head-to-head
                'h2h_home_advantage': h2h_stats['home_advantage'],
                'h2h_avg_goals': h2h_stats['avg_total_goals'],
                'h2h_over_2_5_rate': h2h_stats['over_2_5_rate'],
                
                # Home advantage factor
                'home_advantage_multiplier': 1.1 + (home_stats['home_advantage'] * 0.1),
                
                # Match context
                'is_top_6_clash': home_stats['position'] <= 6 and away_stats['position'] <= 6,
                'is_relegation_battle': home_stats['position'] >= 15 and away_stats['position'] >= 15,
                'strength_disparity': abs(home_stats['points_per_game'] - away_stats['points_per_game'])
            }
            
            return {
                'features': features,
                'feature_names': list(features.keys()),
                'feature_vector': list(features.values()),
                'home_team_stats': home_stats,
                'away_team_stats': away_stats,
                'match_context': {
                    'home_team': home_team,
                    'away_team': away_team,
                    'prediction_date': datetime.now().isoformat(),
                    'season': self.current_season
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting match features for {home_team} vs {away_team}: {e}")
            return self._get_fallback_prediction_features()

    def get_head_to_head_stats(self, home_team: str, away_team: str, last_n: int = 5) -> Dict:
        """Get head-to-head statistics between teams"""
        try:
            match_stats = MatchStatistics()
            
            # This is a simplified version - in production, you'd query historical H2H data
            # For now, we'll use team averages as proxy
            team_stats = self.get_detailed_team_stats() or {}
            teams_list = team_stats.get('teams', []) if isinstance(team_stats, dict) else []
            home_stats = next((t for t in teams_list if str(t.get('name','')).lower() == home_team.lower()), None)
            away_stats = next((t for t in teams_list if str(t.get('name','')).lower() == away_team.lower()), None)
            
            if not home_stats or not away_stats:
                return {
                    'home_advantage': 0.15,
                    'avg_total_goals': 2.7,
                    'over_2_5_rate': 0.52,
                    'matches_analyzed': 0
                }
            
            # Calculate based on team tendencies
            avg_goals = (home_stats['goals_per_game'] + away_stats['goals_per_game'] + 
                        home_stats['goals_conceded_per_game'] + away_stats['goals_conceded_per_game']) / 2
            
            return {
                'home_advantage': 0.15 + (home_stats['home_advantage'] * 0.1),
                'avg_total_goals': avg_goals,
                'over_2_5_rate': 0.6 if avg_goals > 2.5 else 0.4,
                'matches_analyzed': 5  # Placeholder
            }
            
        except Exception as e:
            logger.error(f"Error getting H2H stats: {e}")
            return {
                'home_advantage': 0.15,
                'avg_total_goals': 2.7,
                'over_2_5_rate': 0.52,
                'matches_analyzed': 0
            }

    def get_recent_form(self, team_name: str, last_n_matches: int = 5) -> Dict:
        """Get recent form statistics for a team"""
        try:
            # In production, this would query recent match results
            # For now, we'll calculate based on current season stats
            team_stats = self.get_detailed_team_stats(team_name) or {}
            teams_list = team_stats.get('teams', []) if isinstance(team_stats, dict) else []
            
            if not teams_list:
                return {
                    'form_score': 0.5,
                    'recent_goals_avg': 1.5,
                    'recent_wins': 2,
                    'recent_points': 6
                }
            
            team = teams_list[0]
            
            # Simulate form based on current performance
            form_score = (team['points_per_game'] / 3.0)  # Convert to 0-1 scale
            form_score = min(max(form_score, 0.0), 1.0)
            
            return {
                'form_score': form_score,
                'recent_goals_avg': team['goals_per_game'],
                'recent_wins': int(team['win_rate'] * last_n_matches),
                'recent_points': team['points_per_game'] * last_n_matches
            }
            
        except Exception as e:
            logger.error(f"Error getting recent form for {team_name}: {e}")
            return {
                'form_score': 0.5,
                'recent_goals_avg': 1.5,
                'recent_wins': 2,
                'recent_points': 6
            }

    def get_betting_odds_data(self) -> Dict:
        """Get current betting market data and predictions"""
        try:
            team_stats = self.get_detailed_team_stats() or {}
            teams_list = team_stats.get('teams', []) if isinstance(team_stats, dict) else []
            top_teams = sorted(teams_list, key=lambda x: x.get('points', 0), reverse=True)[:6]
            
            return {
                'title_odds': [
                    {
                        'team': team['name'],
                        'odds': self._calculate_title_odds(team, top_teams),
                        'probability': self._calculate_title_probability(team, top_teams),
                        'form_trend': 'up' if team['form_score'] > 0.6 else 'down' if team['form_score'] < 0.4 else 'stable'
                    }
                    for team in top_teams
                ],
                'top_scorer_odds': self.get_top_scorer_predictions(),
                'relegation_odds': self.get_relegation_predictions(),
                'market_insights': {
                    'highest_scoring_team': max(teams_list, key=lambda x: x.get('goals_per_game', 0))['name'] if teams_list else 'N/A',
                    'best_defense': min(teams_list, key=lambda x: x.get('goals_conceded_per_game', 999))['name'] if teams_list else 'N/A',
                    'most_consistent': max(teams_list, key=lambda x: x.get('scoring_consistency', 0))['name'] if teams_list else 'N/A',
                    'league_avg_goals': (team_stats.get('league_averages', {}) or {}).get('goals_per_game', 2.7) if isinstance(team_stats, dict) else 2.7,
                    'season_prediction': self._generate_season_prediction(teams_list)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting betting odds data: {e}")
            return self._get_fallback_betting_data()

    def _calculate_clean_sheets(self, team: pd.Series) -> float:
        """Calculate clean sheet rate (simplified)"""
        return max(0.3 - (team.get('Goals Against', 20) / max(team.get('Played', 1), 1)) * 0.1, 0.0)
    
    def _calculate_scoring_consistency(self, team: pd.Series) -> float:
        """Calculate scoring consistency score"""
        goals_per_game = team.get('Goals For', 0) / max(team.get('Played', 1), 1)
        return min(goals_per_game / 3.0, 1.0)  # Normalize to 0-1
    
    def _calculate_defensive_strength(self, team: pd.Series) -> float:
        """Calculate defensive strength score"""
        goals_conceded_per_game = team.get('Goals Against', 30) / max(team.get('Played', 1), 1)
        return max(1.0 - (goals_conceded_per_game / 3.0), 0.0)
    
    def _calculate_form_score(self, team: pd.Series) -> float:
        """Calculate current form score"""
        points_per_game = team.get('Points', 0) / max(team.get('Played', 1), 1)
        return min(points_per_game / 3.0, 1.0)
    
    def _calculate_home_advantage(self, team: pd.Series) -> float:
        """Calculate home advantage factor"""
        # Simplified - in production would analyze home vs away performance
        return 0.1 + (team.get('Won', 0) / max(team.get('Played', 1), 1)) * 0.1
    
    def _calculate_away_performance(self, team: pd.Series) -> float:
        """Calculate away performance factor"""
        # Simplified calculation
        return team.get('Won', 0) / max(team.get('Played', 1), 1)

    def _calculate_league_averages(self, teams_data: List[Dict]) -> Dict:
        """Calculate league-wide averages"""
        if not teams_data:
            return {
                'goals_per_game': 2.7,
                'points_per_game': 1.5,
                'win_rate': 0.33
            }
        
        return {
            'goals_per_game': np.mean([t['goals_per_game'] for t in teams_data]),
            'points_per_game': np.mean([t['points_per_game'] for t in teams_data]),
            'win_rate': np.mean([t['win_rate'] for t in teams_data]),
            'goals_conceded_per_game': np.mean([t['goals_conceded_per_game'] for t in teams_data])
        }

    def _calculate_title_odds(self, team: Dict, top_teams: List[Dict]) -> float:
        """Calculate title odds based on current performance"""
        position_factor = max(7 - team['position'], 1)
        points_factor = team['points_per_game']
        form_factor = team['form_score']
        
        raw_odds = 1.5 + (position_factor * 0.5) + (points_factor * 0.3) + (form_factor * 0.2)
        return round(raw_odds, 2)
    
    def _calculate_title_probability(self, team: Dict, top_teams: List[Dict]) -> float:
        """Calculate title probability percentage"""
        if team['position'] > 6:
            return 0.0
        
        position_weight = max(7 - team['position'], 1) / 6.0
        form_weight = team['form_score']
        points_weight = team['points_per_game'] / 3.0
        
        probability = (position_weight * 0.4 + form_weight * 0.3 + points_weight * 0.3) * 100
        return round(min(probability, 95.0), 1)

    def get_top_scorer_predictions(self) -> List[Dict]:
        """Get top scorer predictions with odds"""
        try:
            player_leaders = PlayerSeasonLeaders('goals')  # Specify stat type
            goal_scorers = player_leaders.get_goal_leaders()
            
            if goal_scorers is None or goal_scorers.empty:
                return self._get_fallback_top_scorers()
            
            top_scorers = []
            for _, player in goal_scorers.head(10).iterrows():
                player_data = {
                    'player': player.get('Player', 'Unknown'),
                    'team': player.get('Team', 'Unknown'),
                    'goals': player.get('Goals', 0),
                    'odds': self._calculate_top_scorer_odds(player.get('Goals', 0)),
                    'probability': self._calculate_scoring_probability(player.get('Goals', 0))
                }
                top_scorers.append(player_data)
            
            return top_scorers
            
        except Exception as e:
            logger.error(f"Error getting top scorer predictions: {e}")
            return self._get_fallback_top_scorers()

    def get_relegation_predictions(self) -> List[Dict]:
        """Get relegation battle predictions"""
        try:
            team_stats = self.get_detailed_team_stats() or {}
            teams_list = team_stats.get('teams', []) if isinstance(team_stats, dict) else []
            bottom_teams = sorted(teams_list, key=lambda x: x.get('points', 0))[:8]
            
            relegation_candidates = []
            for team in bottom_teams:
                relegation_risk = self._calculate_relegation_risk(team)
                relegation_candidates.append({
                    'team': team['name'],
                    'position': team['position'],
                    'points': team['points'],
                    'relegation_odds': self._calculate_relegation_odds(relegation_risk),
                    'relegation_probability': relegation_risk * 100,
                    'risk_level': 'High' if relegation_risk > 0.6 else 'Medium' if relegation_risk > 0.3 else 'Low'
                })
            
            return relegation_candidates
            
        except Exception as e:
            logger.error(f"Error getting relegation predictions: {e}")
            return []

    def _calculate_relegation_risk(self, team: Dict) -> float:
        """Calculate relegation risk based on current performance"""
        position_factor = max(team['position'] - 17, 0) / 3.0  # Teams below 17th
        points_factor = max(40 - team['points'], 0) / 40.0      # Distance from safety
        form_factor = 1.0 - team['form_score']                 # Poor form increases risk
        
        risk = (position_factor * 0.5 + points_factor * 0.3 + form_factor * 0.2)
        return min(max(risk, 0.0), 1.0)

    def _generate_season_prediction(self, teams: List[Dict]) -> Dict:
        """Generate overall season predictions"""
        if not teams:
            return {
                'predicted_champion': 'Liverpool',
                'champion_probability': 35.0,
                'predicted_final_points': 88,
                'top_4_prediction': ['Liverpool', 'Arsenal', 'Manchester City', 'Chelsea'],
                'relegation_prediction': ['Sheffield United', 'Burnley', 'Luton Town'],
                'season_goals_prediction': 1000
            }
            
        top_team = max(teams, key=lambda x: x['points'])
        
        return {
            'predicted_champion': top_team['name'],
            'champion_probability': self._calculate_title_probability(top_team, teams[:6]),
            'predicted_final_points': int(top_team['points'] + (top_team['points_per_game'] * (38 - top_team['matches_played']))),
            'top_4_prediction': [team['name'] for team in sorted(teams, key=lambda x: x['points'], reverse=True)[:4]],
            'relegation_prediction': [team['name'] for team in sorted(teams, key=lambda x: x['points'])[:3]],
            'season_goals_prediction': sum(team['goals_for'] for team in teams) + 
                                     int(sum(team['goals_per_game'] for team in teams) * 
                                     max(1, np.mean([38 - team['matches_played'] for team in teams])))
        }

    # Fallback methods for error handling
    def _get_fallback_team_stats(self) -> Dict:
        """Fallback team statistics"""
        synthetic_names = [
            'Arsenal','Aston Villa','Bournemouth','Brentford','Brighton','Chelsea','Crystal Palace',
            'Everton','Fulham','Ipswich Town','Leicester City','Liverpool','Manchester City','Manchester United',
            'Newcastle United','Nottingham Forest','Southampton','Tottenham Hotspur','West Ham United','Wolverhampton'
        ]
        teams = []
        for idx, name in enumerate(synthetic_names, start=1):
            ppg = max(0.5, 2.4 - (idx-1)*(1.9/19))
            goals_for = round(ppg * 1.3, 2)
            goals_against = round(1.4 + (idx/25), 2)
            teams.append({
                'name': name,
                'position': idx,
                'points': int(ppg*10),
                'matches_played': 10,
                'wins': int(ppg*10//3),
                'draws': 2,
                'losses': max(0, 10 - int(ppg*10//3) - 2),
                'goals_for': int(goals_for*10),
                'goals_against': int(goals_against*10),
                'goal_difference': int(goals_for*10 - goals_against*10),
                'points_per_game': ppg,
                'goals_per_game': goals_for,
                'goals_conceded_per_game': goals_against,
                'win_rate': min(0.9, ppg/3.0),
                'form_score': min(0.95, max(0.25, 0.5 + (2.0-ppg)*0.05)),
                'home_advantage': 0.15,
                'goal_difference_gap': goals_for - goals_against,
            })
        return {
            'teams': teams,
            'league_averages': {'goals_per_game': 2.7, 'points_per_game': 1.5, 'win_rate': 0.33},
            'last_updated': datetime.now().isoformat(),
            'season': self.current_season,
            'data_source': 'fallback_synthetic'
        }

    def _get_fallback_prediction_features(self) -> Dict:
        """Fallback prediction features"""
        return {
            'features': {
                'home_points_per_game': 1.5,
                'away_points_per_game': 1.5,
                'home_goals_per_game': 1.5,
                'away_goals_per_game': 1.5,
                'goal_difference_gap': 0,
                'home_advantage_multiplier': 1.1
            },
            'feature_names': ['home_points_per_game', 'away_points_per_game', 'home_goals_per_game', 
                             'away_goals_per_game', 'goal_difference_gap', 'home_advantage_multiplier'],
            'feature_vector': [1.5, 1.5, 1.5, 1.5, 0, 1.1],
            'match_context': {'data_source': 'fallback'}
        }

    def _get_fallback_betting_data(self) -> Dict:
        """Fallback betting data"""
        return {
            'title_odds': [],
            'top_scorer_odds': [],
            'relegation_odds': [],
            'market_insights': {
                'data_source': 'fallback',
                'league_avg_goals': 2.7
            }
        }

    def _get_fallback_top_scorers(self) -> List[Dict]:
        """Fallback top scorer data"""
        return [
            {'player': 'Erling Haaland', 'team': 'Manchester City', 'goals': 27, 'odds': 1.5, 'probability': 35.0},
            {'player': 'Mohamed Salah', 'team': 'Liverpool', 'goals': 22, 'odds': 2.1, 'probability': 28.0},
            {'player': 'Cole Palmer', 'team': 'Chelsea', 'goals': 18, 'odds': 3.2, 'probability': 18.0}
        ]

    def _calculate_top_scorer_odds(self, current_goals: int) -> float:
        """Calculate top scorer odds based on current goals"""
        if current_goals >= 25:
            return round(1.2 + np.random.uniform(0, 0.5), 2)
        elif current_goals >= 20:
            return round(1.8 + np.random.uniform(0, 0.8), 2)
        elif current_goals >= 15:
            return round(2.5 + np.random.uniform(0, 1.5), 2)
        else:
            return round(4.0 + np.random.uniform(0, 6.0), 2)

    def _calculate_scoring_probability(self, current_goals: int) -> float:
        """Calculate probability of winning top scorer"""
        total_goals_projection = current_goals * (38 / max(25, 1))  # Assume 25 games played
        
        if total_goals_projection >= 30:
            return round(40.0 + np.random.uniform(0, 15), 1)
        elif total_goals_projection >= 25:
            return round(25.0 + np.random.uniform(0, 15), 1)
        elif total_goals_projection >= 20:
            return round(15.0 + np.random.uniform(0, 10), 1)
        else:
            return round(5.0 + np.random.uniform(0, 10), 1)

    def _calculate_relegation_odds(self, risk_factor: float) -> float:
        """Calculate relegation odds based on risk factor"""
        base_odds = 2.0 + (1.0 - risk_factor) * 8.0
        return round(base_odds, 2)


# Global service instance
enhanced_epl_service = EnhancedEPLService()