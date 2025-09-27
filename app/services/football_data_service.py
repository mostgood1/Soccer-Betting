"""
Football-Data.org API Integration Service
Provides complete EPL fixture list and match data for 2025-26 season
"""

import os
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FootballDataService:
    """Service to integrate with Football-Data.org API"""
    
    def __init__(self):
        self.api_key = os.getenv('FOOTBALL_DATA_API_KEY')
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {
            'X-Auth-Token': self.api_key,
            'Content-Type': 'application/json'
        }
        self.rate_limit_delay = 6  # 10 calls per minute = 6 seconds between calls
        self.last_request_time = 0
        
        if not self.api_key:
            raise ValueError("FOOTBALL_DATA_API_KEY not found in environment variables")
        
        # EPL Competition ID in Football-Data.org
        self.epl_competition_id = "PL"  # Premier League
        self.season = "2025"  # Season year (2025-26 season)
        
        print(f"Football-Data.org Service initialized for {self.season}-{int(self.season)+1} season")
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make rate-limited API request"""
        # Respect rate limit
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            print(f"Rate limiting: sleeping {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print("Rate limit exceeded, waiting 60 seconds...")
                time.sleep(60)
                return self._make_request(endpoint, params)  # Retry
            else:
                print(f"API Error {response.status_code}: {response.text}")
                return {}
                
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return {}
    
    def get_competition_info(self) -> Dict:
        """Get Premier League competition information"""
        endpoint = f"competitions/{self.epl_competition_id}"
        return self._make_request(endpoint)
    
    def get_teams(self) -> List[Dict]:
        """Get all EPL teams for current season"""
        endpoint = f"competitions/{self.epl_competition_id}/teams"
        params = {"season": self.season}
        
        data = self._make_request(endpoint, params)
        teams = data.get('teams', [])
        
        print(f"Retrieved {len(teams)} EPL teams for {self.season}-{int(self.season)+1} season")
        return teams
    
    def get_fixtures(self, status: str = None) -> List[Dict]:
        """
        Get EPL fixtures for current season
        status: 'SCHEDULED', 'FINISHED', 'LIVE', or None for all
        """
        endpoint = f"competitions/{self.epl_competition_id}/matches"
        params = {"season": self.season}
        
        if status:
            params["status"] = status
            
        data = self._make_request(endpoint, params)
        fixtures = data.get('matches', [])
        
        print(f"Retrieved {len(fixtures)} EPL fixtures for {self.season}-{int(self.season)+1} season")
        return fixtures
    
    def get_standings(self) -> Dict:
        """Get current EPL standings"""
        endpoint = f"competitions/{self.epl_competition_id}/standings"
        params = {"season": self.season}
        
        return self._make_request(endpoint, params)
    
    def get_match_details(self, match_id: int) -> Dict:
        """Get detailed information for a specific match"""
        endpoint = f"matches/{match_id}"
        return self._make_request(endpoint)
    
    def convert_to_epl_format(self, fixtures: List[Dict]) -> List[Dict]:
        """Convert Football-Data.org format to our EPL service format"""
        converted_fixtures = []
        
        for fixture in fixtures:
            # Extract teams
            home_team = fixture.get('homeTeam', {}).get('name', 'Unknown')
            away_team = fixture.get('awayTeam', {}).get('name', 'Unknown')
            
            # Extract match info
            match_date = fixture.get('utcDate', '')
            status = fixture.get('status', 'SCHEDULED')
            matchday = fixture.get('matchday', 0)
            
            # Extract score if match is finished
            score = fixture.get('score', {})
            full_time = score.get('fullTime', {})
            home_goals = full_time.get('home')
            away_goals = full_time.get('away')
            
            converted_match = {
                'id': fixture.get('id'),
                'home_team': home_team,
                'away_team': away_team,
                'date': match_date,
                'status': status,
                'matchday': matchday,
                'season': f"{self.season}-{int(self.season)+1}",
                'home_goals': home_goals,
                'away_goals': away_goals,
                'venue': fixture.get('venue', 'Unknown'),
                'referee': fixture.get('referees', [{}])[0].get('name', 'Unknown') if fixture.get('referees') else 'Unknown',
                'competition': 'Premier League'
            }
            
            converted_fixtures.append(converted_match)
        
        return converted_fixtures
    
    def get_complete_season_data(self) -> Dict:
        """Get comprehensive season data including teams, fixtures, and standings"""
        print(f"\n=== Fetching Complete EPL {self.season}-{int(self.season)+1} Season Data ===")
        
        # Get competition info
        print("1. Getting competition information...")
        competition = self.get_competition_info()
        
        # Get teams
        print("2. Getting teams...")
        teams = self.get_teams()
        
        # Get all fixtures
        print("3. Getting all fixtures...")
        all_fixtures = self.get_fixtures()
        
        # Get current standings
        print("4. Getting current standings...")
        standings = self.get_standings()
        
        # Convert fixtures to our format
        print("5. Converting to EPL service format...")
        converted_fixtures = self.convert_to_epl_format(all_fixtures)
        
        return {
            'competition': competition,
            'teams': teams,
            'fixtures': all_fixtures,  # Raw format
            'converted_fixtures': converted_fixtures,  # Our format
            'standings': standings,
            'season': f"{self.season}-{int(self.season)+1}",
            'total_fixtures': len(all_fixtures),
            'scheduled_fixtures': len([f for f in all_fixtures if f.get('status') == 'SCHEDULED']),
            'finished_fixtures': len([f for f in all_fixtures if f.get('status') == 'FINISHED'])
        }
    
    def cache_season_data(self, filename: str = None) -> str:
        """Cache complete season data to file"""
        if not filename:
            filename = f"football_data_epl_{self.season}_{int(self.season)+1}.json"
        
        filepath = os.path.join(os.path.dirname(__file__), '..', '..', 'data', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        season_data = self.get_complete_season_data()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(season_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nSeason data cached to: {filepath}")
        print(f"Total fixtures: {season_data['total_fixtures']}")
        print(f"Scheduled: {season_data['scheduled_fixtures']}")
        print(f"Finished: {season_data['finished_fixtures']}")
        
        return filepath

def main():
    """Test the Football-Data.org integration"""
    try:
        service = FootballDataService()
        
        # Get complete season data and cache it
        cached_file = service.cache_season_data()
        
        print(f"\n✅ Football-Data.org integration successful!")
        print(f"✅ Complete EPL 2025-26 season data cached")
        print(f"✅ File: {cached_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()