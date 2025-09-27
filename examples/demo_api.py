"""
Example usage of the Soccer Betting Platform API
"""
import requests
import json
from datetime import datetime


def demo_api():
    """Demonstrate the API functionality"""
    base_url = "http://127.0.0.1:8000"
    
    print("🏈 Soccer Betting Platform Demo")
    print("=" * 40)
    
    try:
        # Test root endpoint
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ API is running successfully!")
            print(f"   Response: {response.json()}")
        else:
            print("❌ API connection failed")
            return
            
        print()
        
        # Test health check
        response = requests.get(f"{base_url}/health")
        print(f"🏥 Health Check: {response.json()}")
        print()
        
        # Test API endpoints
        endpoints = [
            ("/api/v1/leagues/", "Leagues"),
            ("/api/v1/teams/", "Teams"),
            ("/api/v1/players/", "Players"),
            ("/api/v1/matches/", "Matches"),
            ("/api/v1/predictions/", "Predictions")
        ]
        
        for endpoint, name in endpoints:
            response = requests.get(f"{base_url}{endpoint}")
            if response.status_code == 200:
                print(f"✅ {name}: {len(response.json())} items found")
            else:
                print(f"❌ {name}: Error {response.status_code}")
        
        print()
        print("🎯 Key Features Available:")
        print("   • Total Goals Prediction")
        print("   • First Half Goals Prediction")  
        print("   • Match Result Prediction (Win/Loss/Draw)")
        print("   • Player Statistics (Goals, Assists)")
        print("   • Goalkeeper Saves Tracking")
        print("   • Corner Statistics (Total & Team Corners)")
        print("   • Player Shots & Shots on Goal")
        print("   • Machine Learning Models for Predictions")
        print("   • RESTful API for Data Access")
        print()
        
        print("📊 Supported Betting Markets:")
        from app.models.bet_market import SUPPORTED_MARKETS
        for market_key, market_info in SUPPORTED_MARKETS.items():
            print(f"   • {market_info['name']}")
        print()
        
        print("🌐 API Documentation:")
        print(f"   • Interactive Docs: {base_url}/docs")
        print(f"   • ReDoc: {base_url}/redoc")
        print()
        
        print("🚀 Next Steps:")
        print("   1. Set up PostgreSQL database connection")
        print("   2. Run database migrations: alembic upgrade head")
        print("   3. Seed initial data: python -m scripts.seed_data")
        print("   4. Connect to EPL data APIs")
        print("   5. Train ML models with historical data")
        print("   6. Start making predictions!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running.")
        print("   Run: uvicorn app.main:app --reload")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    demo_api()