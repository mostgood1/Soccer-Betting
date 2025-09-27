from __future__ import annotations
from typing import Dict

from .enhanced_epl_service_v2 import EnhancedEPLService
from .enhanced_fd_service import EnhancedFDService

SUPPORTED: Dict[str, str] = {
    'PL': 'Premier League',  # England
    'BL1': 'Bundesliga',     # Germany
    'FL1': 'Ligue 1',        # France
    'SA': 'Serie A',         # Italy
    'PD': 'La Liga',         # Spain
}

ALIASES: Dict[str, str] = {
    'epl': 'PL', 'premierleague': 'PL', 'premier-league': 'PL', 'england': 'PL',
    'bundesliga': 'BL1', 'germany': 'BL1',
    'ligue1': 'FL1', 'france': 'FL1', 'league1': 'FL1', 'l1': 'FL1',
    'seriea': 'SA', 'italy': 'SA',
    'laliga': 'PD', 'la-liga': 'PD', 'spain': 'PD', 'primera': 'PD', 'pd': 'PD',
}

_INSTANCES: Dict[str, object] = {}


def normalize_league_code(league: str | None) -> str:
    if not league:
        return 'PL'
    lk = str(league).strip()
    if lk.upper() in SUPPORTED:
        return lk.upper()
    key = lk.lower()
    if key in ALIASES:
        return ALIASES[key]
    # default EPL
    return 'PL'


def get_service(league: str | None):
    code = normalize_league_code(league)
    if code in _INSTANCES:
        return _INSTANCES[code]
    if code == 'PL':
        svc = EnhancedEPLService()
    else:
        svc = EnhancedFDService(code)
    _INSTANCES[code] = svc
    return svc


def list_supported():
    return [{'code': k, 'name': v} for k, v in SUPPORTED.items()]
