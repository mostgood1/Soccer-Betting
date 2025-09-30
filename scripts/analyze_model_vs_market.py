import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from statistics import mean
import math

# Faster startup: avoid retraining; allow on-demand predictions
os.environ.setdefault("ML_SKIP_STARTUP_TRAIN", "1")
os.environ.setdefault("ALLOW_ON_DEMAND_PREDICTIONS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.league_manager import list_supported as list_supported_leagues, get_service as get_league_service
from app.services.game_week_service import game_week_service
from app.services.team_name_normalizer import normalize_team_name
from app.ml.advanced_predictor import advanced_ml_predictor

HIST_ODDS_PATH = Path("cache/historic_odds.json")


def _build_market_index() -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    if not HIST_ODDS_PATH.exists():
        return idx
    try:
        data = json.loads(HIST_ODDS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return idx
    for rec in data.get("records", []) or []:
        ht = normalize_team_name(rec.get("home_team")) or rec.get("home_team")
        at = normalize_team_name(rec.get("away_team")) or rec.get("away_team")
        date = (rec.get("date") or "").split("T")[0]
        if not (ht and at):
            continue
        key_basic = f"{ht}|{at}".lower()
        idx[key_basic] = rec
        if date:
            idx[f"{date}|{key_basic}"] = rec
    return idx


def _market_probs(rec: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    if not rec:
        return None
    # Prefer Bet365 implied if available; else consensus
    bms = rec.get("bookmakers") or {}
    b365 = bms.get("B365") if isinstance(bms, dict) else None
    if isinstance(b365, dict):
        ip = b365.get("implied_probabilities") or {}
        mh, md, ma = ip.get("H"), ip.get("D"), ip.get("A")
        if all(isinstance(x, (int, float)) for x in (mh, md, ma)):
            s = (mh or 0) + (md or 0) + (ma or 0)
            if s > 0:
                return (mh / s, md / s, ma / s)
            return (mh, md, ma)
    cons = rec.get("consensus_implied") or {}
    mh, md, ma = cons.get("H"), cons.get("D"), cons.get("A")
    if all(isinstance(x, (int, float)) for x in (mh, md, ma)):
        s = (mh or 0) + (md or 0) + (ma or 0)
        if s > 0:
            return (mh / s, md / s, ma / s)
        return (mh, md, ma)
    return None


def _nll(probs: Tuple[float, float, float], cls: int) -> float:
    eps = 1e-12
    p = max(min([probs[0], probs[1], probs[2]][cls], 1 - eps), eps)
    return -math.log(p)


def _brier(probs: Tuple[float, float, float], cls: int) -> float:
    yh, yd, ya = (1 if cls == 0 else 0, 1 if cls == 1 else 0, 1 if cls == 2 else 0)
    ph, pd, pa = probs
    return ((ph - yh) ** 2 + (pd - yd) ** 2 + (pa - ya) ** 2) / 3.0


def _top1_acc(probs: Tuple[float, float, float], cls: int) -> int:
    return 1 if int(max(range(3), key=lambda i: probs[i])) == cls else 0


def main():
    up_to_week = int(os.getenv("HIST_UP_TO_WEEK", "8"))
    leagues = list_supported_leagues()
    market_idx = _build_market_index()
    # Ensure models are ready
    try:
        if not advanced_ml_predictor.is_trained:
            advanced_ml_predictor.load_models()
    except Exception:
        pass

    rows = []
    matched = 0
    for lg in leagues:
        code = (lg.get("code") if isinstance(lg, dict) else str(lg)).upper()
        svc = get_league_service(code)
        matches = svc.get_all_matches() if hasattr(svc, "get_all_matches") else svc.get_matches()
        weeks = game_week_service.organize_matches_by_week(matches)
        for wk in range(1, up_to_week + 1):
            for m in weeks.get(wk, []):
                hs = m.get("home_score") if "home_score" in m else m.get("homeScore")
                as_ = m.get("away_score") if "away_score" in m else m.get("awayScore")
                status_completed = m.get("status") in ["FINISHED", "COMPLETED"] or m.get("is_completed")
                if hs is None or as_ is None or not status_completed:
                    continue
                home_raw = m.get("home_team") or m.get("homeTeam")
                away_raw = m.get("away_team") or m.get("awayTeam")
                home = normalize_team_name(home_raw) or home_raw
                away = normalize_team_name(away_raw) or away_raw
                # Pull market
                date = (m.get("utc_date") or m.get("date") or "").split("T")[0]
                rec = market_idx.get(f"{date}|{home.lower()}|{away.lower()}") or market_idx.get(
                    f"{home}|{away}".lower()
                )
                mp = _market_probs(rec) if rec else None
                if not mp:
                    continue
                # Model
                raw = advanced_ml_predictor.predict_match(home, away, league=code)
                if not raw:
                    continue
                ph = float(raw.get("home_win_probability", 0.34))
                pa = float(raw.get("away_win_probability", 0.33))
                pd = float(raw.get("draw_probability", max(0.0, 1 - (ph + pa))))
                s = ph + pd + pa
                if s <= 0:
                    continue
                ph, pd, pa = (ph / s, pd / s, pa / s)
                # Actual class
                cls = 0 if hs > as_ else 2 if as_ > hs else 1
                rows.append({
                    "league": code,
                    "week": wk,
                    "model": (ph, pd, pa),
                    "market": mp,
                    "cls": cls,
                })
                matched += 1

    if not rows:
        print(json.dumps({"error": "no matched market records"}, indent=2))
        return

    def agg(vals: List[float]) -> Optional[float]:
        return round(mean(vals), 4) if vals else None

    model_nll = [
        _nll(r["model"], r["cls"]) for r in rows
    ]
    market_nll = [
        _nll(r["market"], r["cls"]) for r in rows
    ]
    model_brier = [
        _brier(r["model"], r["cls"]) for r in rows
    ]
    market_brier = [
        _brier(r["market"], r["cls"]) for r in rows
    ]
    model_acc = [
        _top1_acc(r["model"], r["cls"]) for r in rows
    ]
    market_acc = [
        _top1_acc(r["market"], r["cls"]) for r in rows
    ]

    # Simple correlation between model and market Home-win probability
    try:
        import statistics as _stats

        mh = [r["model"][0] for r in rows]
        kh = [r["market"][0] for r in rows]
        # Pearson r
        mean_m = mean(mh)
        mean_k = mean(kh)
        num = sum((a - mean_m) * (b - mean_k) for a, b in zip(mh, kh))
        den = math.sqrt(sum((a - mean_m) ** 2 for a in mh) * sum((b - mean_k) ** 2 for b in kh))
        corr_home = num / den if den > 0 else None
    except Exception:
        corr_home = None

    # Output summary
    out = {
        "matched_completed_with_market": matched,
        "model": {
            "avg_nll": agg(model_nll),
            "avg_brier": agg(model_brier),
            "top1_accuracy": agg(model_acc),
        },
        "market": {
            "avg_nll": agg(market_nll),
            "avg_brier": agg(market_brier),
            "top1_accuracy": agg(market_acc),
        },
        "delta": {
            "avg_nll": round((mean(market_nll) - mean(model_nll)), 4) if model_nll and market_nll else None,
            "avg_brier": round((mean(market_brier) - mean(model_brier)), 4) if model_brier and market_brier else None,
            "top1_accuracy": round((mean(model_acc) - mean(market_acc)), 4) if model_acc and market_acc else None,
        },
        "correlation": {
            "pearson_model_vs_market_home_prob": None if corr_home is None else round(corr_home, 4)
        },
        "notes": "NLL/Brier lower is better; delta>0 means model better on that metric."
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
