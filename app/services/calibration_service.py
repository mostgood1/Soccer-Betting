"""Probability calibration via temperature scaling.

Fits a single temperature parameter T by minimizing negative log-likelihood
on completed matches up to a given game week. Applies to model class
probabilities (H/D/A) to improve calibration.

We avoid external optimizers; use a lightweight grid search over T in [0.5, 3.0].
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import os
import json
from pathlib import Path
import math

from .team_name_normalizer import normalize_team_name
from .game_week_service import game_week_service
from .enhanced_epl_service_v2 import EnhancedEPLService
from ..ml.advanced_predictor import advanced_ml_predictor


CALIBRATION_PATH = Path("cache/calibration.json")
HIST_ODDS_PATH = Path("cache/historic_odds.json")


def _softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps) or 1.0
    return [x / s for x in exps]


def _to_logits_from_probs(p: List[float], eps: float = 1e-12) -> List[float]:
    # Use log(p) as pseudo-logits; safe-guard with eps
    return [math.log(max(min(x, 1 - eps), eps)) for x in p]


def apply_temperature(probs: Dict[str, float], T: float) -> Dict[str, float]:
    # Order keys H,D,A for consistency
    p = [probs.get('H', 0.0), probs.get('D', 0.0), probs.get('A', 0.0)]
    logits = _to_logits_from_probs(p)
    scaled = [x / max(T, 1e-6) for x in logits]
    out = _softmax(scaled)
    return {'H': out[0], 'D': out[1], 'A': out[2]}


class CalibrationService:
    def __init__(self):
        self.temperature: Optional[float] = None
        self.meta: Dict[str, Any] = {}
        # New: per-league temperatures (e.g., {'PL': 1.12, 'BL1': 0.96, ...})
        self.temperatures_by_league: Dict[str, float] = {}
        self._load()

    def _load(self):
        try:
            if CALIBRATION_PATH.exists():
                data = json.loads(CALIBRATION_PATH.read_text(encoding='utf-8'))
                self.temperature = data.get('temperature')
                self.meta = data
                # Back-compat: load by_league mapping if present
                m = data.get('by_league')
                if isinstance(m, dict):
                    # normalise keys to upper
                    self.temperatures_by_league = {str(k).upper(): float(v) for k, v in m.items() if isinstance(v, (int, float))}
        except Exception:
            self.temperature = None
            self.meta = {}
            self.temperatures_by_league = {}

    def _save(self):
        payload = {**self.meta, 'temperature': self.temperature, 'by_league': self.temperatures_by_league}
        CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        CALIBRATION_PATH.write_text(json.dumps(payload), encoding='utf-8')

    def status(self) -> Dict[str, Any]:
        return {'temperature': self.temperature, 'by_league': self.temperatures_by_league, 'meta': self.meta}

    def is_ready(self, league: Optional[str] = None) -> bool:
        if league:
            t = self.temperatures_by_league.get(str(league).upper())
            return isinstance(t, (int, float)) and t > 0
        return isinstance(self.temperature, (int, float)) and self.temperature > 0

    def calibrate_up_to_week(self, up_to_week: int, league: Optional[str] = None) -> Dict[str, Any]:
        # Build dataset: for completed matches up to week, collect model probs and actual labels
        # Resolve league service
        if league:
            try:
                from .league_manager import get_service as _get_service
                service = _get_service(league)
            except Exception:
                service = EnhancedEPLService()
        else:
            service = EnhancedEPLService()
        # Ensure models are loaded so we don't fall back to heuristics in a fresh process
        try:
            if not advanced_ml_predictor.is_trained:
                advanced_ml_predictor.load_models()
        except Exception:
            pass
        matches = service.get_all_matches()
        weeks = game_week_service.organize_matches_by_week(matches)
        X: List[Tuple[float, float, float]] = []  # model probs H,D,A
        y: List[int] = []  # 0:H,1:D,2:A
        M: List[Optional[Tuple[float, float, float]]] = []  # market probs if available

    # Load historic market consensus index (date|home|away)
        market_idx: Dict[str, Dict[str, Any]] = {}
        try:
            if HIST_ODDS_PATH.exists():
                data = json.loads(HIST_ODDS_PATH.read_text(encoding='utf-8'))
                pref_env = os.getenv('PREFERRED_BOOKMAKERS', 'bet365,draftkings,fanduel,bovada')
                preferred_keys = {k.strip().lower() for k in pref_env.split(',') if k.strip()}
                for rec in data.get('records', []):
                    ht = normalize_team_name(rec.get('home_team')) or rec.get('home_team')
                    at = normalize_team_name(rec.get('away_team')) or rec.get('away_team')
                    if not (ht and at):
                        continue
                    # Prefer Bet365 ('B365' in historic csvs) if configured
                    bookmakers = rec.get('bookmakers') or {}
                    if 'bet365' in preferred_keys:
                        b365 = bookmakers.get('B365')
                        if isinstance(b365, dict):
                            ip = b365.get('implied_probabilities')
                            ov = b365.get('overround')
                            if isinstance(ip, dict) and all(k in ip for k in ('H','D','A')):
                                rec = {**rec, 'preferred_implied': ip, 'preferred_overround': ov if isinstance(ov,(int,float)) else None}
                    key_basic = f"{ht}|{at}".lower()
                    market_idx[key_basic] = rec
                    date = rec.get('date')
                    if date:
                        market_idx[f"{date}|{key_basic}"] = rec
        except Exception:
            market_idx = {}
        for wk in range(1, up_to_week + 1):
            for m in weeks.get(wk, []):
                hs = m.get('home_score') if 'home_score' in m else m.get('homeScore')
                as_ = m.get('away_score') if 'away_score' in m else m.get('awayScore')
                status_completed = m.get('status') in ['FINISHED', 'COMPLETED'] or m.get('is_completed')
                if hs is None or as_ is None or not status_completed:
                    continue
                home_raw = m.get('home_team') or m.get('homeTeam')
                away_raw = m.get('away_team') or m.get('awayTeam')
                home = normalize_team_name(home_raw) or home_raw
                away = normalize_team_name(away_raw) or away_raw
                try:
                    raw = advanced_ml_predictor.predict_match(home, away, league=str(league).upper() if league else None)
                    if not raw:
                        continue
                    p_home = float(raw.get('home_win_probability') or raw.get('home_prob') or 0.34)
                    p_away = float(raw.get('away_win_probability') or raw.get('away_prob') or 0.33)
                    p_draw = float(raw.get('draw_probability') or max(0.0, 1 - (p_home + p_away)))
                    s = p_home + p_draw + p_away
                    if s <= 0:
                        continue
                    p_home, p_draw, p_away = [p / s for p in (p_home, p_draw, p_away)]
                    X.append((p_home, p_draw, p_away))
                    # Attach market probs if available from historic index
                    date = (m.get('utc_date') or m.get('date') or '').split('T')[0]
                    mk = None
                    if date:
                        mk = market_idx.get(f"{date}|{home.lower()}|{away.lower()}")
                    if mk is None:
                        mk = market_idx.get(f"{home}|{away}".lower())
                    if mk:
                        cons = (mk.get('preferred_implied') or mk.get('consensus_implied') or {})
                        mh = cons.get('H'); md = cons.get('D'); ma = cons.get('A')
                        if all(isinstance(v,(int,float)) for v in (mh,md,ma)):
                            tot = mh + md + ma
                            if tot and tot > 0:
                                M.append((mh/tot, md/tot, ma/tot))
                            else:
                                M.append((mh, md, ma))
                        else:
                            M.append(None)
                    else:
                        M.append(None)
                    if hs > as_:
                        y.append(0)
                    elif as_ > hs:
                        y.append(2)
                    else:
                        y.append(1)
                except Exception:
                    continue
        if not X:
            self.temperature = None
            self.meta = {'trained_on': {'matches': 0, 'weeks': list(range(1, up_to_week+1))}}
            self._save()
            return {'trained': False, 'reason': 'no data'}

        # Grid search for T to minimize NLL
        def nll_for_T(T: float) -> float:
            eps = 1e-12
            loss = 0.0
            reg_w = float(os.getenv('CALIB_MARKET_REG', '0.1'))
            for idx, ((ph, pd, pa), cls) in enumerate(zip(X, y)):
                cal = apply_temperature({'H': ph, 'D': pd, 'A': pa}, T)
                p = [cal['H'], cal['D'], cal['A']][cls]
                p = max(min(p, 1 - eps), eps)
                loss += -math.log(p)
                # Market-informed regularization: encourage calibrated probs to be near market consensus (if available)
                if reg_w > 0 and idx < len(M) and M[idx] is not None:
                    mh, md, ma = M[idx] or (None, None, None)
                    if None not in (mh, md, ma):
                        # simple MSE between distributions
                        loss += reg_w * ((cal['H'] - mh)**2 + (cal['D'] - md)**2 + (cal['A'] - ma)**2) / 3.0
            return loss / len(X)

        best_T = 1.0
        best_loss = float('inf')
        # Coarse grid
        for T in [x / 20.0 for x in range(10, 61)]:  # 0.5 to 3.0 step 0.05
            loss = nll_for_T(T)
            if loss < best_loss:
                best_loss = loss; best_T = T
        # Fine-tune near best
        start = max(0.5, best_T - 0.2); end = min(3.0, best_T + 0.2)
        step = 0.01
        t = start
        while t <= end + 1e-9:
            loss = nll_for_T(t)
            if loss < best_loss:
                best_loss = loss; best_T = t
            t += step

        # Save either globally or under league key
        if league:
            self.temperatures_by_league[str(league).upper()] = round(best_T, 4)
        else:
            self.temperature = round(best_T, 4)
        self.meta = {
            'trained_on': {'matches': len(X), 'weeks': list(range(1, up_to_week+1))},
            'market_regularization': {
                'weight': float(os.getenv('CALIB_MARKET_REG', '0.1')),
                'matches_with_market': sum(1 for m in M if m is not None)
            }
        }
        self._save()
        return {
            'trained': True,
            'temperature': (self.temperatures_by_league.get(str(league).upper()) if league else self.temperature),
            'matches': len(X),
            'league': (str(league).upper() if league else None)
        }

    def apply_if_ready(self, probs: Dict[str, float], league: Optional[str] = None) -> Dict[str, float]:
        if league and self.is_ready(league):
            t = self.temperatures_by_league.get(str(league).upper()) or 1.0
            return apply_temperature(probs, t)
        if not self.is_ready():
            return probs
        return apply_temperature(probs, self.temperature or 1.0)


calibration_service = CalibrationService()

__all__ = [
    'CalibrationService', 'calibration_service', 'apply_temperature'
]
