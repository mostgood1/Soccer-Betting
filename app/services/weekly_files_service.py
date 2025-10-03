"""Weekly files service

Generates per-league, per-week CSV artifacts and a JSON bundle that the frontend can consume directly:
 - data/weekly/{LEAGUE}/week_{W}_odds.csv
 - data/weekly/{LEAGUE}/week_{W}_predictions.csv
 - data/weekly/{LEAGUE}/week_{W}_results.csv
 - data/weekly/{LEAGUE}/week_{W}_bundle.json

Odds are sourced from CSV historics (h2h_{LEAGUE}.csv) via a recent-window consensus index to avoid live provider calls.
Predictions are computed via the current ML predictor and snapshotted (one-time write for determinism).
Results are extracted from the football-data converted fixtures caches.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .league_manager import get_service as get_league_service
from .game_week_service import game_week_service
from .team_name_normalizer import normalize_team_name
from ..ml.advanced_predictor import advanced_ml_predictor


BASE_DIR = Path("data/weekly")
BASE_DIR.mkdir(parents=True, exist_ok=True)


def _league_dir(league: str) -> Path:
    code = (league or "PL").upper()
    p = BASE_DIR / code
    p.mkdir(parents=True, exist_ok=True)
    return p


def _weekly_paths(league: str, week: int) -> Dict[str, Path]:
    base = _league_dir(league)
    return {
        "odds": base / f"week_{int(week)}_odds.csv",
        "predictions": base / f"week_{int(week)}_predictions.csv",
        "results": base / f"week_{int(week)}_results.csv",
        "bundle": base / f"week_{int(week)}_bundle.json",
    }


def _week_matches(league: str, week: int) -> List[Dict[str, Any]]:
    try:
        svc = get_league_service(league)
    except Exception:
        svc = get_league_service("PL")
    matches = (
        svc.get_all_matches() if hasattr(svc, "get_all_matches") else []
    )
    weeks = game_week_service.organize_matches_by_week(matches)
    return weeks.get(int(week), [])


def _match_date_str(m: Dict[str, Any]) -> str:
    d = m.get("utc_date") or m.get("date") or ""
    if isinstance(d, str) and "T" in d:
        return d.split("T")[0]
    return d or ""


def write_weekly_odds(league: str, week: int, recent_days: int = 365) -> Dict[str, Any]:
    """Write simplified H2H odds per match into weekly odds CSV using CSV historics.

    Columns: league,week,date,home_team,away_team,dec_H,dec_D,dec_A,imp_H,imp_D,imp_A,source
    """
    from .odds_csv_store import load_h2h_index_from_csv

    lg = (league or "PL").upper()
    mlist = _week_matches(lg, int(week))
    idx = load_h2h_index_from_csv(lg, days=max(1, int(recent_days)))
    paths = _weekly_paths(lg, int(week))
    header = [
        "league",
        "week",
        "date",
        "home_team",
        "away_team",
        "dec_H",
        "dec_D",
        "dec_A",
        "imp_H",
        "imp_D",
        "imp_A",
        "source",
    ]
    rows = 0
    with paths["odds"].open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for m in mlist:
            home = (
                m.get("home_team")
                or m.get("homeTeam")
                or (m.get("home") or {}).get("name")
            )
            away = (
                m.get("away_team")
                or m.get("awayTeam")
                or (m.get("away") or {}).get("name")
            )
            if not (home and away):
                continue
            hn = normalize_team_name(home) or home
            an = normalize_team_name(away) or away
            date = _match_date_str(m)
            key1 = f"{date}|{hn.lower()}|{an.lower()}"
            key2 = f"{hn.lower()}|{an.lower()}"
            rec = idx.get(key1) or idx.get(key2) or {}
            cons = rec.get("consensus_implied") or {}
            prefs = rec.get("preferred_decimals") or {}
            
            def _dec(tag: str) -> Optional[float]:
                v = prefs.get(tag)
                if isinstance(v, (int, float)) and v > 1:
                    return float(v)
                p = cons.get(tag)
                try:
                    if isinstance(p, (int, float)) and p > 0:
                        return round(1.0 / float(p), 2)
                except Exception:
                    return None
                return None
            dec_h = _dec("H")
            dec_d = _dec("D")
            dec_a = _dec("A")
            imp_h = cons.get("H")
            imp_d = cons.get("D")
            imp_a = cons.get("A")
            src = "csv-historic" if rec else "missing"
            w.writerow([
                lg,
                int(week),
                date,
                hn,
                an,
                dec_h if dec_h is not None else "",
                dec_d if dec_d is not None else "",
                dec_a if dec_a is not None else "",
                round(float(imp_h), 6) if isinstance(imp_h, (int, float)) else "",
                round(float(imp_d), 6) if isinstance(imp_d, (int, float)) else "",
                round(float(imp_a), 6) if isinstance(imp_a, (int, float)) else "",
                src,
            ])
            rows += 1
    return {"path": str(paths["odds"]), "rows": rows}


def write_weekly_predictions(league: str, week: int) -> Dict[str, Any]:
    """Compute model predictions for each match of the week and snapshot to CSV.

    Columns: league,week,date,home_team,away_team,pH,pD,pA,total_goals,pick,confidence,model_version
    """
    lg = (league or "PL").upper()
    mlist = _week_matches(lg, int(week))
    # Ensure model loaded
    try:
        if not advanced_ml_predictor.is_trained:
            advanced_ml_predictor.load_models()
    except Exception:
        pass
    paths = _weekly_paths(lg, int(week))
    header = [
        "league",
        "week",
        "date",
        "home_team",
        "away_team",
        "pH",
        "pD",
        "pA",
        "total_goals",
        "pick",
        "confidence",
        "model_version",
    ]
    rows = 0
    with paths["predictions"].open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for m in mlist:
            home = (
                m.get("home_team")
                or m.get("homeTeam")
                or (m.get("home") or {}).get("name")
            )
            away = (
                m.get("away_team")
                or m.get("awayTeam")
                or (m.get("away") or {}).get("name")
            )
            if not (home and away):
                continue
            hn = normalize_team_name(home) or home
            an = normalize_team_name(away) or away
            date = _match_date_str(m)
            try:
                pred = advanced_ml_predictor.predict_match(hn, an)
            except Exception:
                pred = {}
            pH = pred.get("home_win_probability")
            pD = pred.get("draw_probability")
            pA = pred.get("away_win_probability")
            total = pred.get("total_goals")
            pick = pred.get("match_result")
            conf = pred.get("match_result_confidence")
            mv = advanced_ml_predictor.model_version
            w.writerow([
                lg,
                int(week),
                date,
                hn,
                an,
                round(float(pH), 6) if isinstance(pH, (int, float)) else "",
                round(float(pD), 6) if isinstance(pD, (int, float)) else "",
                round(float(pA), 6) if isinstance(pA, (int, float)) else "",
                round(float(total), 3) if isinstance(total, (int, float)) else "",
                pick or "",
                round(float(conf), 3) if isinstance(conf, (int, float)) else "",
                str(mv),
            ])
            rows += 1
    return {"path": str(paths["predictions"]), "rows": rows}


def write_weekly_results(league: str, week: int) -> Dict[str, Any]:
    """Write completed match results for the league & week into a CSV.

    Columns: league,week,date,home_team,away_team,home_score,away_score,result,status
    """
    lg = (league or "PL").upper()
    # Map to football-data cache file
    file_map = {
        "PL": "football_data_PL_2025_2026.json",
        "BL1": "football_data_BL1_2025_2026.json",
        "FL1": "football_data_FL1_2025_2026.json",
        "SA": "football_data_SA_2025_2026.json",
        "PD": "football_data_PD_2025_2026.json",
    }
    path = Path("data") / file_map.get(lg, "")
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    fixtures = payload.get("converted_fixtures") or []
    # Filter week fixtures
    wk = int(week)
    subset = [
        m
        for m in fixtures
        if (m.get("matchday") == wk) or (m.get("game_week") == wk)
    ]
    paths = _weekly_paths(lg, wk)
    header = [
        "league",
        "week",
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "result",
        "status",
    ]
    rows = 0
    with paths["results"].open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for m in subset:
            hs = m.get("home_score") or m.get("homeScore")
            as_ = m.get("away_score") or m.get("awayScore")
            date = _match_date_str(m)
            home = (
                m.get("home_team")
                or m.get("homeTeam")
                or (m.get("home") or {}).get("name")
            )
            away = (
                m.get("away_team")
                or m.get("awayTeam")
                or (m.get("away") or {}).get("name")
            )
            if not (home and away):
                continue
            res = ""
            if isinstance(hs, (int, float)) and isinstance(as_, (int, float)):
                if hs > as_:
                    res = "H"
                elif as_ > hs:
                    res = "A"
                else:
                    res = "D"
            status = m.get("status") or ("COMPLETED" if res else (m.get("stage") or ""))
            w.writerow([
                lg,
                wk,
                date,
                normalize_team_name(home) or home,
                normalize_team_name(away) or away,
                hs if hs is not None else "",
                as_ if as_ is not None else "",
                res,
                status,
            ])
            rows += 1
    return {"path": str(paths["results"]), "rows": rows}


def build_weekly_bundle(league: str, week: int) -> Dict[str, Any]:
    """Read generated CSVs (if present) and emit a JSON bundle for the frontend."""
    import csv as _csv

    lg = (league or "PL").upper()
    paths = _weekly_paths(lg, int(week))
    out: Dict[str, Any] = {"league": lg, "week": int(week), "odds": [], "predictions": [], "results": []}
    # Odds
    if paths["odds"].exists():
        with paths["odds"].open("r", newline="", encoding="utf-8") as f:
            r = _csv.DictReader(f)
            out["odds"] = list(r)
    # Predictions
    if paths["predictions"].exists():
        with paths["predictions"].open("r", newline="", encoding="utf-8") as f:
            r = _csv.DictReader(f)
            out["predictions"] = list(r)
    # Results
    if paths["results"].exists():
        with paths["results"].open("r", newline="", encoding="utf-8") as f:
            r = _csv.DictReader(f)
            out["results"] = list(r)
    # Persist bundle JSON
    paths["bundle"].write_text(json.dumps(out, indent=2), encoding="utf-8")
    return {"path": str(paths["bundle"]), "counts": {k: len(v) for k, v in out.items() if isinstance(v, list)}}


def write_all_weekly(league: str, week: int, include: Optional[List[str]] = None) -> Dict[str, Any]:
    """Convenience: write selected artifacts and the bundle.

    include: subset of ["odds","predictions","results"]; default = all
    """
    parts = include or ["odds", "predictions", "results"]
    res: Dict[str, Any] = {"league": (league or "PL").upper(), "week": int(week), "written": {}}
    if "odds" in parts:
        res["written"]["odds"] = write_weekly_odds(league, week)
    if "predictions" in parts:
        res["written"]["predictions"] = write_weekly_predictions(league, week)
    if "results" in parts:
        res["written"]["results"] = write_weekly_results(league, week)
    res["bundle"] = build_weekly_bundle(league, week)
    return res
