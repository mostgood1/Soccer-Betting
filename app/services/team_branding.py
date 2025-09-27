"""Static team branding: primary/secondary colors and crest overrides when needed.
This supports frontend game cards with consistent visuals.
"""
from typing import Dict, Any

BRANDING: Dict[str, Dict[str, Any]] = {
    # Big 6
    "Arsenal FC": {"primary": "#EF0107", "secondary": "#FFFFFF"},
    "Chelsea FC": {"primary": "#034694", "secondary": "#FFFFFF"},
    "Liverpool FC": {"primary": "#C8102E", "secondary": "#00B2A9"},
    "Manchester City FC": {"primary": "#6CABDD", "secondary": "#1C2C5B"},
    "Manchester United FC": {"primary": "#DA291C", "secondary": "#FBE122"},
    "Tottenham Hotspur FC": {"primary": "#132257", "secondary": "#FFFFFF"},
    # London clubs
    "West Ham United FC": {"primary": "#7A263A", "secondary": "#1BB1E7"},
    "Crystal Palace FC": {"primary": "#1B458F", "secondary": "#C4122E"},
    "Fulham FC": {"primary": "#000000", "secondary": "#FFFFFF"},
    "Brentford FC": {"primary": "#D20000", "secondary": "#FFFFFF"},
    # Others
    "Aston Villa FC": {"primary": "#670E36", "secondary": "#95BFE5"},
    "Newcastle United FC": {"primary": "#241F20", "secondary": "#FFFFFF"},
    "Brighton & Hove Albion FC": {"primary": "#0057B8", "secondary": "#FFFFFF"},
    "Wolverhampton Wanderers FC": {"primary": "#FDB913", "secondary": "#231F20"},
    "Everton FC": {"primary": "#003399", "secondary": "#FFFFFF"},
    "AFC Bournemouth": {"primary": "#D00000", "secondary": "#000000"},
    "Nottingham Forest FC": {"primary": "#DD0000", "secondary": "#FFFFFF"},
    "Burnley FC": {"primary": "#6C1D45", "secondary": "#99D6EA"},
    # Promoted/alternate in dataset
    "Leeds United FC": {"primary": "#1D428A", "secondary": "#FFFFFF"},
    "Sunderland AFC": {"primary": "#E31B23", "secondary": "#FFFFFF"},
}

def get_team_branding() -> Dict[str, Dict[str, Any]]:
    return BRANDING
