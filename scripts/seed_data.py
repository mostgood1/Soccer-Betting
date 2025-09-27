"""
Data seeding script for EPL teams and basic setup
"""
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.models.league import League
from app.models.team import Team


def seed_epl_data():
    """Seed EPL teams and league data"""
    db = SessionLocal()
    
    try:
        # Create EPL league
        epl = League(
            name="Premier League",
            country="England",
            season="2024-25",
            api_id="39",  # Common API ID for EPL
            is_active=True
        )
        db.add(epl)
        db.commit()
        db.refresh(epl)
        
        # EPL Teams for 2024-25 season
        epl_teams = [
            {"name": "Arsenal", "short_name": "ARS", "api_id": "42"},
            {"name": "Aston Villa", "short_name": "AVL", "api_id": "66"},
            {"name": "Bournemouth", "short_name": "BOU", "api_id": "35"},
            {"name": "Brentford", "short_name": "BRE", "api_id": "55"},
            {"name": "Brighton & Hove Albion", "short_name": "BHA", "api_id": "51"},
            {"name": "Chelsea", "short_name": "CHE", "api_id": "49"},
            {"name": "Crystal Palace", "short_name": "CRY", "api_id": "52"},
            {"name": "Everton", "short_name": "EVE", "api_id": "45"},
            {"name": "Fulham", "short_name": "FUL", "api_id": "36"},
            {"name": "Ipswich Town", "short_name": "IPS", "api_id": "40"},
            {"name": "Leicester City", "short_name": "LEI", "api_id": "46"},
            {"name": "Liverpool", "short_name": "LIV", "api_id": "40"},
            {"name": "Manchester City", "short_name": "MCI", "api_id": "50"},
            {"name": "Manchester United", "short_name": "MUN", "api_id": "33"},
            {"name": "Newcastle United", "short_name": "NEW", "api_id": "34"},
            {"name": "Nottingham Forest", "short_name": "NFO", "api_id": "65"},
            {"name": "Southampton", "short_name": "SOU", "api_id": "41"},
            {"name": "Tottenham Hotspur", "short_name": "TOT", "api_id": "47"},
            {"name": "West Ham United", "short_name": "WHU", "api_id": "48"},
            {"name": "Wolverhampton Wanderers", "short_name": "WOL", "api_id": "39"}
        ]
        
        # Create team records
        for team_data in epl_teams:
            team = Team(
                name=team_data["name"],
                short_name=team_data["short_name"],
                league_id=epl.id,
                api_id=team_data["api_id"],
                is_active=True
            )
            db.add(team)
        
        db.commit()
        print(f"Successfully seeded EPL data: 1 league and {len(epl_teams)} teams")
        
    except Exception as e:
        db.rollback()
        print(f"Error seeding data: {str(e)}")
    finally:
        db.close()


def seed_bet_markets():
    """Seed betting market definitions"""
    from app.models.bet_market import BetMarket, SUPPORTED_MARKETS
    
    db = SessionLocal()
    
    try:
        for market_key, market_info in SUPPORTED_MARKETS.items():
            market = BetMarket(
                market_name=market_key,
                market_type=market_info['type'],
                description=market_info['description'],
                is_active=True,
                category='standard'
            )
            db.add(market)
        
        db.commit()
        print(f"Successfully seeded {len(SUPPORTED_MARKETS)} betting markets")
        
    except Exception as e:
        db.rollback()
        print(f"Error seeding betting markets: {str(e)}")
    finally:
        db.close()


if __name__ == "__main__":
    print("Seeding database with initial data...")
    seed_epl_data()
    seed_bet_markets()
    print("Database seeding completed!")