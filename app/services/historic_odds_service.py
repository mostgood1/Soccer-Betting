"""Historic Odds Ingestion & Normalization Service.

This module ingests locally stored historical odds CSV files (e.g. from
https://www.football-data.co.uk/ which publishes season-by-season
Premier League (and other league) historical results with bookmaker odds).

It parses bookmaker odds (e.g., Bet365, Pinnacle, etc.), computes implied
probabilities with overround adjustment, and stores a canonical structure
for later model calibration & edge analysis.

Usage expectations:
1. User downloads CSV(s) (e.g. E0, EPL seasons) into data/odds/
2. Run offline task command:  python -m app.offline.tasks ingest-odds --path data/odds
3. Result stored in cache/historic_odds.json

We avoid any web-scraping here; only local file parsing.
"""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
import json
import math

# --- Team Normalization (duplicated minimal subset; consider centralizing later) ---
NORMALIZATION_MAP = {
    'manchester united': 'Manchester United',
    'man utd': 'Manchester United',
    'man united': 'Manchester United',
    'manchester city': 'Manchester City',
    'man city': 'Manchester City',
    'tottenham hotspur': 'Tottenham',
    'spurs': 'Tottenham',
    'wolverhampton wanderers': 'Wolves',
    'wolves': 'Wolves',
    'brighton & hove albion': 'Brighton',
    'afc bournemouth': 'Bournemouth',
    'nottingham forest': 'Nottm Forest',
    'nottm forest': 'Nottm Forest',
    'forest': 'Nottm Forest',
    'sheffield united': 'Sheffield Utd',
    'sheffield utd': 'Sheffield Utd',
    'sheff utd': 'Sheffield Utd',
    'west ham united': 'West Ham',
    'west ham': 'West Ham',
    'newcastle united': 'Newcastle',
}

def normalize_team(name: str | None) -> str | None:
    if not name:
        return name
    raw = name.strip()
    key = raw.lower()
    cleaned = key.replace(' afc', '').replace(' fc', '').replace('&', 'and')
    if cleaned in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[cleaned]
    if key in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[key]
    # Title-case fallback
    return ' '.join(w.capitalize() for w in raw.split())

# --- Parsing Helpers ---
DATE_FORMATS = ["%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"]

BOOKMAKER_PREFIXES = [
    "B365",  # Bet365
    "PS",    # Pinnacle (PSH/PSD/PSA often appear as closing prices)
    "WH",    # William Hill
    "VC",    # BetVictor
    "LB",    # Ladbrokes
    "IW",    # Interwetten
    "BW",    # Bwin
    "SJ",    # Stan James
    "GB",    # Gamebookers
    "BS",    # Blue Square (historic)
]

def parse_date(val: str) -> str | None:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(val.strip(), fmt).strftime('%Y-%m-%d')
        except Exception:
            continue
    return None

def extract_bookmaker_odds(row: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    bm_data: Dict[str, Dict[str, float]] = {}
    for prefix in BOOKMAKER_PREFIXES:
        h = row.get(f"{prefix}H")
        d = row.get(f"{prefix}D")
        a = row.get(f"{prefix}A")
        if not h or not d or not a:
            continue
        try:
            oh, od, oa = float(h), float(d), float(a)
            # Basic sanity filter
            if min(oh, od, oa) <= 1.01 or max(oh, od, oa) > 200:
                continue
            bm_data[prefix] = { 'H': oh, 'D': od, 'A': oa }
        except Exception:
            continue
    return bm_data

def implied_probabilities(odds_triplet: Dict[str, float]) -> Tuple[Dict[str,float], float]:
    # odds_triplet keys: H,D,A ; values decimal odds
    inv = {k: 1.0/v if v > 0 else 0.0 for k,v in odds_triplet.items()}
    overround = sum(inv.values())
    if overround <= 0:
        return {k: 0.0 for k in odds_triplet}, 0.0
    norm = {k: inv[k]/overround for k in inv}
    # overround expressed as (overround -1)
    return norm, overround - 1.0

def parse_csv_file(path: Path, season_hint: str | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with path.open('r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                date_raw = r.get('Date') or r.get('date')
                date_std = parse_date(date_raw) if date_raw else None
                h_raw = r.get('HomeTeam') or r.get('home_team')
                a_raw = r.get('AwayTeam') or r.get('away_team')
                home = normalize_team(h_raw)
                away = normalize_team(a_raw)
                if not (date_std and home and away):
                    continue
                bookmakers = extract_bookmaker_odds(r)
                if not bookmakers:
                    continue
                # Extract outcomes if present
                try:
                    home_goals = int(r.get('FTHG')) if r.get('FTHG') not in (None, '') else None
                except Exception:
                    home_goals = None
                try:
                    away_goals = int(r.get('FTAG')) if r.get('FTAG') not in (None, '') else None
                except Exception:
                    away_goals = None
                ftr = (r.get('FTR') or '').strip().upper()  # 'H','D','A'
                if ftr not in ('H','D','A'):
                    ftr = None
                entry = {
                    'date': date_std,
                    'home_team': home,
                    'away_team': away,
                    'season': season_hint,
                    'bookmakers': {},
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    'result_code': ftr,  # market style result
                }
                # Compute implied probabilities & overround per bookmaker
                for bm, odds in bookmakers.items():
                    probs, overround = implied_probabilities(odds)
                    entry['bookmakers'][bm] = {
                        'odds': odds,
                        'implied_probabilities': probs,
                        'overround': round(overround, 4)
                    }
                # Totals (Over/Under 2.5) parsing per bookmaker if present
                # Football-Data CSVs commonly use columns like "B365>2.5" and "B365<2.5".
                for bm_prefix in BOOKMAKER_PREFIXES:
                    over_key = f"{bm_prefix}>2.5"
                    under_key = f"{bm_prefix}<2.5"
                    o_val = r.get(over_key)
                    u_val = r.get(under_key)
                    if not o_val or not u_val:
                        continue
                    try:
                        o_odds = float(o_val)
                        u_odds = float(u_val)
                        if min(o_odds, u_odds) <= 1.01 or max(o_odds, u_odds) > 200:
                            continue
                        # implied probabilities (not normalized to 1 necessarily)
                        inv_o = 1.0 / o_odds
                        inv_u = 1.0 / u_odds
                        ou_overround = inv_o + inv_u
                        if ou_overround <= 0:
                            continue
                        # Normalize to sum to 1 across Over/Under
                        over_p = inv_o / ou_overround
                        under_p = inv_u / ou_overround
                        # ensure bookmaker container exists
                        if bm_prefix not in entry['bookmakers']:
                            entry['bookmakers'][bm_prefix] = {}
                        entry['bookmakers'][bm_prefix].setdefault('totals_2_5', {})
                        entry['bookmakers'][bm_prefix]['totals_2_5'] = {
                            'line': 2.5,
                            'over_odds': o_odds,
                            'under_odds': u_odds,
                            'over_implied': round(over_p, 6),
                            'under_implied': round(under_p, 6),
                            'overround': round(ou_overround - 1.0, 6)
                        }
                    except Exception:
                        continue
                # Compute consensus (median) implied probabilities & overround if any odds available
                if entry['bookmakers']:
                    from statistics import median
                    prob_samples = {'H': [], 'D': [], 'A': []}
                    over_samples = []
                    for bm, data in entry['bookmakers'].items():
                        ip = data.get('implied_probabilities', {})
                        for k in ('H','D','A'):
                            v = ip.get(k)
                            if isinstance(v,(int,float)) and v>0:
                                prob_samples[k].append(float(v))
                        ov = data.get('overround')
                        if isinstance(ov,(int,float)):
                            over_samples.append(float(ov))
                    if any(prob_samples[k] for k in prob_samples):
                        entry['consensus_implied'] = {k: (median(prob_samples[k]) if prob_samples[k] else None) for k in prob_samples}
                        entry['consensus_overround'] = (median(over_samples) if over_samples else None)
                    # Consensus for totals 2.5 if present across bookmakers
                    tot_over_samples: List[float] = []
                    tot_under_samples: List[float] = []
                    tot_ovr_overrounds: List[float] = []
                    for bm_key, bm_data in entry['bookmakers'].items():
                        t = bm_data.get('totals_2_5')
                        if not isinstance(t, dict):
                            continue
                        ov = t.get('over_implied'); un = t.get('under_implied')
                        oor = t.get('overround')
                        if isinstance(ov, (int,float)):
                            tot_over_samples.append(float(ov))
                        if isinstance(un, (int,float)):
                            tot_under_samples.append(float(un))
                        if isinstance(oor, (int,float)):
                            tot_ovr_overrounds.append(float(oor))
                    if tot_over_samples and tot_under_samples:
                        entry['consensus_totals_2_5'] = {
                            'line': 2.5,
                            'over_implied': round(median(tot_over_samples), 6),
                            'under_implied': round(median(tot_under_samples), 6),
                            'overround': round(median(tot_ovr_overrounds), 6) if tot_ovr_overrounds else None
                        }
                rows.append(entry)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[historic_odds] Failed parsing {path}: {e}")
    return rows

def ingest_path(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {'error': f'Path not found: {path}'}
    files: List[Path] = []
    if p.is_dir():
        for fp in p.iterdir():
            if fp.suffix.lower() == '.csv':
                files.append(fp)
    else:
        if p.suffix.lower() == '.csv':
            files.append(p)
    aggregated: List[Dict[str, Any]] = []
    for f in files:
        # Extract season hint from filename (heuristic)
        # e.g. E0 2024-2025 or 2024-25; fallback None
        season_hint = None
        parts = f.stem.replace('-', ' ').replace('_',' ').split()
        for token in parts:
            if token.isdigit() and len(token) == 4:
                # treat as starting year
                try:
                    yr = int(token)
                    season_hint = f"{yr}-{(yr+1)%100:02d}"
                except Exception:
                    pass
        parsed = parse_csv_file(f, season_hint=season_hint)
        aggregated.extend(parsed)
    # Build index for fast lookup later by (date, home, away)
    index = {}
    for e in aggregated:
        key = f"{e['date']}_{e['home_team']}_{e['away_team']}"
        index[key] = e
    # Basic stats
    total_bm = sum(len(e['bookmakers']) for e in aggregated)
    avg_bm_per_match = total_bm / max(len(aggregated), 1)
    overround_samples = []
    for e in aggregated:
        for bm, data in e['bookmakers'].items():
            overround_samples.append(data['overround'])
    overround_mean = round(sum(overround_samples)/len(overround_samples),4) if overround_samples else None
    out = {
        'matches': len(aggregated),
        'avg_bookmakers_per_match': round(avg_bm_per_match,2),
        'mean_overround': overround_mean,
        'store_index_size': len(index),
    }
    # Persist
    cache_path = Path('cache/historic_odds.json')
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'summary': out,
        'records': aggregated,
    }
    try:
        cache_path.write_text(json.dumps(payload), encoding='utf-8')
    except Exception as e:
        out['persist_error'] = str(e)
    return out

# ---------------------------------------------------------------------------
# Training Data Extraction from Historic Odds
# ---------------------------------------------------------------------------
def market_training_rows(cache_path: str = 'cache/historic_odds.json') -> List[Dict[str, Any]]:
    """Produce training-compatible rows from historic odds store.

    Each row maps consensus implied probabilities to outcome targets.
    Missing outcomes (no goals) are skipped.
    Constructs minimal feature set comprised of market_* features; other
    model feature columns can be filled later with zeros when merged.
    """
    path = Path(cache_path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return []
    records = data.get('records') or []
    rows: List[Dict[str, Any]] = []
    for rec in records:
        hg = rec.get('home_goals')
        ag = rec.get('away_goals')
        res = rec.get('result_code')
        cons = rec.get('consensus_implied') or {}
        if hg is None or ag is None or res not in ('H','D','A'):
            continue
        # Encode result to numeric 1=Home,0=Draw,2=Away (align with predictor usage)
        if res == 'H':
            match_result = 1
        elif res == 'A':
            match_result = 2
        else:
            match_result = 0
        h_prob = cons.get('H'); d_prob = cons.get('D'); a_prob = cons.get('A')
        if not all(isinstance(v,(int,float)) for v in (h_prob, d_prob, a_prob)):
            continue
        over = rec.get('consensus_overround')
        # Derived market shape features
        fav_prob = max(h_prob, a_prob)
        prob_skew = abs(h_prob - a_prob)
        total_goals = hg + ag
        first_half_goals = round(total_goals * 0.45)  # heuristic if no half data
        rows.append({
            'home_goals': hg,
            'away_goals': ag,
            'total_goals': total_goals,
            'first_half_goals': first_half_goals,
            'match_result': match_result,
            'market_home_implied': h_prob,
            'market_draw_implied': d_prob,
            'market_away_implied': a_prob,
            'market_overround': over if isinstance(over,(int,float)) else 0.0,
            'market_fav_prob': fav_prob,
            'market_prob_skew': prob_skew,
        })
    return rows
