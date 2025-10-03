"""Shared team name normalization utilities.
Centralizes canonical mapping so odds, predictions, and UI use consistent names.
"""
from __future__ import annotations
from typing import Dict, Optional
import os
import json
import unicodedata

NORMALIZATION_MAP: Dict[str, str] = {
    "manchester united": "Manchester United",
    "man utd": "Manchester United",
    "man united": "Manchester United",
    "manchester city": "Manchester City",
    "man city": "Manchester City",
    "tottenham hotspur": "Tottenham",
    "spurs": "Tottenham",
    "arsenal": "Arsenal",
    "chelsea": "Chelsea",
    "liverpool fc": "Liverpool",
    "liverpool": "Liverpool",
    "everton": "Everton",
    "newcastle united": "Newcastle",
    "newcastle": "Newcastle",
    "west ham united": "West Ham",
    "west ham": "West Ham",
    "wolverhampton wanderers": "Wolves",
    "wolves": "Wolves",
    "brighton & hove albion": "Brighton",
    "brighton and hove albion": "Brighton",
    "brighton": "Brighton",
    "afc bournemouth": "Bournemouth",
    "bournemouth": "Bournemouth",
    "crystal palace": "Crystal Palace",
    "nottingham forest": "Nottm Forest",
    "nottm forest": "Nottm Forest",
    "forest": "Nottm Forest",
    # Handle apostrophe variant from some datasets
    "nott'm forest": "Nottm Forest",
    "sheffield united": "Sheffield Utd",
    "sheffield utd": "Sheffield Utd",
    "sheff utd": "Sheffield Utd",
    "aston villa": "Aston Villa",
    "brentford": "Brentford",
    "fulham": "Fulham",
    "burnley": "Burnley",
    "luton town": "Luton",
    "luton": "Luton",
    "west bromwich albion": "West Brom",
    "west brom": "West Brom",
    # Newly promoted / potential variants
    "ipswich town": "Ipswich",
    "ipswich": "Ipswich",
    "leicester city": "Leicester",
    "leicester": "Leicester",
    # Common short forms
    "leeds": "Leeds United",
    # Bundesliga common forms
    "fc bayern munchen": "Bayern Munich",
    "bayern munchen": "Bayern Munich",
    "bayern münchen": "Bayern Munich",
    "bayern munich": "Bayern Munich",
    "borussia dortmund": "Borussia Dortmund",
    "borussia monchengladbach": "Borussia Mönchengladbach",
    "borussia mönchengladbach": "Borussia Mönchengladbach",
    "monchengladbach": "Borussia Mönchengladbach",
    "1 fc koln": "1. FC Köln",
    "fc koln": "1. FC Köln",
    "koln": "1. FC Köln",
    "koln fc": "1. FC Köln",
    "eintracht frankfurt": "Eintracht Frankfurt",
    "rb leipzig": "RB Leipzig",
    "bayer 04 leverkusen": "Bayer 04 Leverkusen",
    "bayer leverkusen": "Bayer 04 Leverkusen",
    "tsg 1899 hoffenheim": "TSG 1899 Hoffenheim",
    "vfb stuttgart": "VfB Stuttgart",
    "vfl wolfsburg": "VfL Wolfsburg",
    "vfl bochum 1848": "VfL Bochum 1848",
    "1 fsv mainz 05": "1. FSV Mainz 05",
    "mainz 05": "1. FSV Mainz 05",
    "1 fc heidenheim 1846": "1. FC Heidenheim 1846",
    "1 fc union berlin": "1. FC Union Berlin",
    "union berlin": "1. FC Union Berlin",
    "sc freiburg": "SC Freiburg",
    "fc augsburg": "FC Augsburg",
    "sv werder bremen": "SV Werder Bremen",
    # La Liga
    "fc barcelona": "FC Barcelona",
    "barcelona": "FC Barcelona",
    "real madrid": "Real Madrid CF",
    "real madrid cf": "Real Madrid CF",
    "atletico madrid": "Atlético de Madrid",
    "atlético de madrid": "Atlético de Madrid",
    "athletic bilbao": "Athletic Club",
    "athletic club": "Athletic Club",
    "real sociedad": "Real Sociedad",
    "real betis": "Real Betis",
    "real betis balompie": "Real Betis",
    "sevilla": "Sevilla FC",
    "sevilla fc": "Sevilla FC",
    "valencia": "Valencia CF",
    "valencia cf": "Valencia CF",
    "getafe cf": "Getafe",
    "lorient": "Lorient",
    # Ligue 1
    "psg": "Paris Saint-Germain",
    "paris saint-germain": "Paris Saint-Germain",
    "olympique lyonnais": "Olympique Lyonnais",
    "lyon": "Olympique Lyonnais",
    "olympique de marseille": "Olympique de Marseille",
    "marseille": "Olympique de Marseille",
    "as monaco": "AS Monaco",
    "monaco": "AS Monaco",
    "lille": "Lille OSC",
    "lille osc": "Lille OSC",
    "stade rennais": "Stade Rennais FC",
    "rennes": "Stade Rennais FC",
    "paris fc": "Paris FC",
    "fc lorient": "Lorient",
    # Serie A
    "inter": "Inter",
    "inter milan": "Inter",
    "internazionale": "Inter",
    "internazionale milano": "Inter",
    "fc internazionale milano": "Inter",
    "ac milan": "AC Milan",
    "milan": "AC Milan",
    "juventus": "Juventus",
    "juventus fc": "Juventus",
    "napoli": "SSC Napoli",
    "ssc napoli": "SSC Napoli",
    "as roma": "AS Roma",
    "roma": "AS Roma",
    "lazio": "Lazio",
    "ss lazio": "Lazio",
    "torino": "Torino",
    "torino fc": "Torino",
    "fiorentina": "Fiorentina",
    "acf fiorentina": "Fiorentina",
    "bologna": "Bologna",
    "bologna fc": "Bologna",
    "sassuolo": "Sassuolo",
    "us sassuolo": "Sassuolo",
    "us sassuolo calcio": "Sassuolo",
    "hellas verona": "Hellas Verona",
    "verona": "Hellas Verona",
    "udinese": "Udinese",
    "udinese calcio": "Udinese",
    "cagliari": "Cagliari",
    "cagliari calcio": "Cagliari",
    "genoa": "Genoa",
    "genoa cfc": "Genoa",
    "pisa": "Pisa",
    "ac pisa": "Pisa",
    "como": "Como",
    "como 1907": "Como",
    "parma": "Parma",
    "parma calcio": "Parma",
    "cremonese": "Cremonese",
    "us cremonese": "Cremonese",
    "atalanta": "Atalanta",
    "atalanta bc": "Atalanta",
}

SUFFIXES = (" afc", " fc", " a.f.c.", " f.c.")

# Common club prefixes/suffixes and artifacts across leagues to ignore for matching
_STOP_TOKENS = {
    # generic
    "fc",
    "afc",
    "cf",
    "cfc",
    "sc",
    "sd",
    "cd",
    # italy-specific
    "calcio",
    "ss",
    "us",
    "acf",
    "asd",
    # spain/france variants
    "sa",
    "sad",
    "ud",
    "ac",
    # spain-specific common prefixes
    "rcd",  # Real Club Deportivo (e.g., RCD Espanyol)
    "ca",   # Club Atlético (e.g., CA Osasuna)
    "deportivo",  # Deportivo Alavés -> Alavés
}


def _strip_diacritics(s: str) -> str:
    try:
        return (
            unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        )
    except Exception:
        return s


def normalize_team_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return name
    raw = name.strip()
    key = raw.lower()
    key = _strip_diacritics(key)
    cleaned = key
    for suf in SUFFIXES:
        cleaned = cleaned.replace(suf, "")
    cleaned = cleaned.replace("&", "and")
    # Normalize apostrophes and quotes sometimes appearing in sources (e.g., Nott'm Forest)
    cleaned = cleaned.replace("'", "").replace("’", "")
    if cleaned in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[cleaned]
    if key in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[key]
    # Also drop apostrophes from tokenization/title-casing path
    tokens = []
    for t in raw.replace(".", " ").split():
        tl = _strip_diacritics(t.lower()).replace("'", "").replace("’", "")
        # skip common stop tokens (club designations) and pure numeric tokens (years like 1909/1913)
        if tl in _STOP_TOKENS:
            continue
        if tl.isdigit():
            continue
        tokens.append(t)
    if not tokens:
        return raw
    candidate = " ".join(tokens)
    # Apply diacritic strip on candidate too
    candidate = _strip_diacritics(candidate)
    return " ".join(
        w[0].upper() + w[1:] if len(w) > 1 else w.upper() for w in candidate.split()
    )


def canonical_pair(home: str, away: str) -> str:
    h = normalize_team_name(home) or home
    a = normalize_team_name(away) or away
    return f"{h}|{a}".lower()


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_external_aliases() -> None:
    """Merge alias maps from data/team_aliases*.json into NORMALIZATION_MAP.

    Files checked (if exist):
      - data/team_aliases.json (global)
      - data/team_aliases_PL.json, _BL1.json, _FL1.json, _SA.json, _PD.json
    Keys are case-insensitive; we store lowercase stripped keys without diacritics.
    """
    root = _repo_root()
    data_dir = os.path.join(root, "data")
    files = [
        os.path.join(data_dir, "team_aliases.json"),
        os.path.join(data_dir, "team_aliases_PL.json"),
        os.path.join(data_dir, "team_aliases_BL1.json"),
        os.path.join(data_dir, "team_aliases_FL1.json"),
        os.path.join(data_dir, "team_aliases_SA.json"),
        os.path.join(data_dir, "team_aliases_PD.json"),
    ]
    for fp in files:
        try:
            if not os.path.exists(fp):
                continue
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    if not isinstance(k, str) or not isinstance(v, str):
                        continue
                    key = _strip_diacritics(k.strip().lower())
                    NORMALIZATION_MAP[key] = v.strip()
        except Exception:
            continue


# Load external alias files at import time (best-effort)
try:
    _load_external_aliases()
except Exception:
    pass


__all__ = ["normalize_team_name", "NORMALIZATION_MAP", "canonical_pair"]
