#!/usr/bin/env python3
"""
Quick API Test - Verify Real Data Access
"""

import requests
import json

def test_api_endpoints():
    """Test key API endpoints"""
    base_url = "http://127.0.0.1:8000"
    
    print("🧪 TESTING EPL BETTING API ENDPOINTS")
    print("=" * 50)
    
    endpoints = [
        ("/api/teams", "Teams Data"),
        ("/api/players/top-scorers?limit=5", "Top Scorers"),
        ("/api/matches?limit=5", "Matches"),
        ("/health", "Health Check")
    ]
    
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                if endpoint == "/api/teams":
                    teams = data.get('teams', [])
                    print(f"✅ {name}: {len(teams)} teams loaded")
                    if teams:
                        print(f"   Sample: {teams[0].get('name', 'Unknown')}")
                
                elif endpoint == "/api/players/top-scorers?limit=5":
                    players = data.get('top_scorers', [])
                    print(f"✅ {name}: {len(players)} top scorers")
                    if players:
                        top = players[0]
                        print(f"   Leading scorer: {top.get('name', 'Unknown')} ({top.get('goals', 0)} goals)")
                
                elif endpoint == "/api/matches?limit=5":
                    matches = data.get('matches', [])
                    print(f"✅ {name}: {len(matches)} matches loaded")
                    if matches:
                        sample = matches[0]
                        print(f"   Sample: {sample.get('home_team', 'Unknown')} vs {sample.get('away_team', 'Unknown')}")
                
                elif endpoint == "/health":
                    status = data.get('status', 'unknown')
                    print(f"✅ {name}: Status = {status}")
                    
            else:
                print(f"❌ {name}: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ {name}: Connection failed - {str(e)}")
    
    print("\n" + "=" * 50)
    print("🎯 API Test Complete")

if __name__ == "__main__":
    test_api_endpoints()