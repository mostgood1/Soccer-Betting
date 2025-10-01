"""Offline operational tasks (retrain, rebuild predictions) so we can progress
when the ASGI server cannot stay resident in this execution environment.

Usage (from project root):
  python -m app.offline.tasks retrain --level patch
  python -m app.offline.tasks rebuild
  python -m app.offline.tasks summary --week 1
"""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set

from ..ml.advanced_predictor import advanced_ml_predictor
from ..services.team_name_normalizer import normalize_team_name  # unified normalizer

# Attempt to load .env variables (e.g., FOOTBALL_DATA_API_TOKEN) if present
try:
    from dotenv import load_dotenv  # python-dotenv is in requirements

    load_dotenv()
except Exception:
    pass
from ..services.enhanced_epl_service_v2 import EnhancedEPLService
from ..services.league_manager import (
    get_service as get_league_service,
    normalize_league_code,
)
from ..services.corners_actuals_service import corners_actuals_store
from ..services.corners_market_service import corners_market_store
from ..services.goals_market_service import goals_market_store
from ..services.corners_calibration_service import corners_calibration_service
from ..services.expected_goals_service import expected_goals_service
from ..services.game_week_service import game_week_service, reconciliation_service
from ..services.reconciliation_store import reconciliation_store
from ..services.historic_odds_service import ingest_path as ingest_odds_path
from ..services.football_data_odds_service import fetch_week_odds
from ..services.bovada_odds_service import (
    fetch_pl_odds,
    fetch_bl1_odds,
    fetch_fl1_odds,
    fetch_sa_odds,
    fetch_pd_odds,
)
from ..services.betting_odds_service import BettingOddsService
from ..services.calibration_service import calibration_service
from ..services.odds_api_corners_service import fetch_corners_totals_from_odds_api
from ..services.odds_csv_store import load_h2h_index_from_csv
from ..services.corners_market_service import reload_market_store

try:
    from ..services.historic_odds_downloader import download_epl_history
except Exception:
    download_epl_history = None  # type: ignore
try:
    from ..services.odds_api_service import fetch_h2h_odds
except Exception:
    fetch_h2h_odds = None  # type: ignore
from ..services.week_snapshot_service import WeekSnapshotService

OUTPUT_DIR = Path("cache/offline_runs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_prediction_cache: Dict[str, Dict[str, Any]] = {}

# Path to the server's persisted prediction cache so we can reuse identical
# model probabilities that power the card "Model" section. This prevents
# confusing mismatches between the card and the compare panels.
SERVER_PRED_CACHE_PATH = Path(
    os.getenv("PREDICTIONS_CACHE_PATH", "data/predictions_cache.json")
)


def _load_server_prediction_cache() -> Dict[str, Dict[str, Any]]:
    try:
        if SERVER_PRED_CACHE_PATH.exists():
            data = json.loads(SERVER_PRED_CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data  # keyed by f"{match_id}_{home}_{away}" using RAW team names
        # Backward-compat: also check legacy path under cache/
        legacy = Path("cache/predictions_cache.json")
        if legacy.exists():
            data = json.loads(legacy.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


# Helper: map league code to Bovada fetcher
def _get_bovada_fetcher(league_code: str):
    code = normalize_league_code(league_code)
    return {
        "PL": fetch_pl_odds,
        "BL1": fetch_bl1_odds,
        "FL1": fetch_fl1_odds,
        "SA": fetch_sa_odds,
        "PD": fetch_pd_odds,
    }.get(code)


# ---------------------------------------------------------------------------
# Daily update orchestrator
# ---------------------------------------------------------------------------
def daily_update(
    retrain_level: str = "patch",
    capture_closing: bool = True,
    generate_snapshot: bool = True,
    include_bovada_corners: bool = True,
    include_odds_api_corners: bool = True,
    odds_api_regions: str = "eu,uk,us",
    odds_api_bookmakers: Optional[
        str
    ] = "pinnacle,bet365,williamhill,unibet,betfair_ex",
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"steps": [], "errors": []}

    def step(name: str, payload: Dict[str, Any]):
        summary["steps"].append({"name": name, **payload})

    # 1) Refresh schedule/times
    try:
        epl = EnhancedEPLService()
        ref = epl.refresh_data()
        step("refresh_data", ref)
    except Exception as e:
        summary["errors"].append({"refresh_data": str(e)})

    # Determine current and previous week
    try:
        matches = EnhancedEPLService().get_all_matches()
        weeks_map = game_week_service.organize_matches_by_week(matches)
        current_week = game_week_service.get_current_game_week()
        prev_week = max(1, current_week - 1)
        summary["current_week"] = current_week
        summary["previous_week"] = prev_week
        step(
            "week_detection",
            {
                "current_week": current_week,
                "previous_week": prev_week,
                "weeks_found": len(weeks_map),
            },
        )
    except Exception as e:
        current_week = 1
        prev_week = 1
        summary["errors"].append({"week_detection": str(e)})

    # 2) Fetch final scores for prev and current week and reconcile
    token_present = bool(os.getenv("FOOTBALL_DATA_API_TOKEN"))
    for w in [prev_week, current_week]:
        fetch_info = {"week": w, "token_present": token_present}
        try:
            res = fetch_scores(
                "PL", 2025, w, auto_reconcile=True, force_repredict=False
            )
            fetch_info.update(res)
        except Exception as e:
            fetch_info["error"] = str(e)
        step("fetch_scores_and_reconcile", fetch_info)

    # 3) Update corners markets from Odds API (if key present) for all supported leagues
    if include_odds_api_corners and os.getenv("ODDS_API_KEY"):
        try:
            from ..services.odds_api_corners_service import (
                fetch_corners_totals_from_odds_api as _fetch_corners,
            )

            # Map league codes to The Odds API sport keys
            sport_map = {
                "PL": "soccer_epl",
                "BL1": "soccer_germany_bundesliga",
                "FL1": "soccer_france_ligue_one",
                "SA": "soccer_italy_serie_a",
                "PD": "soccer_spain_la_liga",
            }
            out_path = Path("data/corners_market_lines_manual.json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Load existing once to merge across leagues
            existing: List[Dict[str, Any]] = []
            try:
                if out_path.exists():
                    existing_payload = json.loads(out_path.read_text(encoding="utf-8"))
                    if isinstance(existing_payload, dict) and isinstance(
                        existing_payload.get("records"), list
                    ):
                        existing = existing_payload["records"]
            except Exception:
                existing = []

            def _k(r: Dict[str, Any]):
                return (
                    str(r.get("date"))[:19],
                    (r.get("home_team") or r.get("home") or "").lower(),
                    (r.get("away_team") or r.get("away") or "").lower(),
                    float(r.get("line")) if r.get("line") is not None else None,
                    (r.get("bookmaker") or "").lower(),
                    (r.get("market_type") or "total").lower(),
                    (r.get("side") or "").lower(),
                )

            seen = set(_k(r) for r in existing if isinstance(r, dict))
            merged = list(existing)
            added_total = 0
            # Iterate leagues and append records
            for code, sport_key in sport_map.items():
                try:
                    res = _fetch_corners(
                        sport_key=sport_key,
                        regions=odds_api_regions,
                        historical=False,
                        markets="alternate_totals_corners",
                        bookmakers=odds_api_bookmakers,
                    )
                    new_recs = res.get("records") or []
                    for r in new_recs:
                        try:
                            key = _k(r)
                            if key not in seen:
                                merged.append(r)
                                seen.add(key)
                                added_total += 1
                        except Exception:
                            merged.append(r)
                            added_total += 1
                except Exception:
                    # Continue with other leagues
                    continue
            out_payload = {
                "records": merged,
                "source": "oddsapi",
                "added_now": added_total,
            }
            out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
            totals, team = reload_market_store()
            step(
                "update_corners_oddsapi",
                {
                    "added_now": added_total,
                    "total_records": len(merged),
                    "reloaded": {"totals": totals, "team": team},
                },
            )
        except Exception as e:
            summary["errors"].append({"update_corners_oddsapi": str(e)})
    else:
        step(
            "update_corners_oddsapi",
            {"skipped": True, "reason": "no ODDS_API_KEY or disabled"},
        )

    # 4) Bovada backfill (corners totals + team corners)
    if include_bovada_corners:
        try:
            # Reuse the CLI function body without spawning a new process
            from ..services.bovada_odds_service import (
                fetch_pl_odds as _fetch_bov,
                _american_to_decimal as _ml_to_dec,
            )

            payload = _fetch_bov() if _fetch_bov is not None else {"events": []}
            events = payload.get("events") or []
            rows: List[Dict[str, Any]] = []
            for ev in events:
                date = ev.get("commence_time")
                home = normalize_team_name(ev.get("home_team")) or ev.get("home_team")
                away = normalize_team_name(ev.get("away_team")) or ev.get("away_team")
                for t in ev.get("corners_totals") or []:
                    try:
                        line = (
                            float(t.get("line")) if t.get("line") is not None else None
                        )
                    except Exception:
                        line = None
                    if not home or not away or not date or line is None:
                        continue
                    op = t.get("over_prob")
                    up = t.get("under_prob")
                    oml = t.get("over_ml")
                    uml = t.get("under_ml")
                    over_odds = (
                        _ml_to_dec(float(oml))
                        if oml is not None
                        else (
                            1.0 / float(op)
                            if isinstance(op, (int, float)) and float(op) > 0
                            else None
                        )
                    )
                    under_odds = (
                        _ml_to_dec(float(uml))
                        if uml is not None
                        else (
                            1.0 / float(up)
                            if isinstance(up, (int, float)) and float(up) > 0
                            else None
                        )
                    )
                    if over_odds and under_odds:
                        rows.append(
                            {
                                "date": date,
                                "home_team": home,
                                "away_team": away,
                                "line": float(line),
                                "over_odds": float(round(over_odds, 4)),
                                "under_odds": float(round(under_odds, 4)),
                                "bookmaker": "Bovada",
                                "market_type": "total",
                            }
                        )
                for t in ev.get("team_corners") or []:
                    try:
                        line = (
                            float(t.get("line")) if t.get("line") is not None else None
                        )
                    except Exception:
                        line = None
                    side = (t.get("side") or "").lower()
                    if side not in ("home", "away"):
                        continue
                    if not home or not away or not date or line is None:
                        continue
                    op = t.get("over_prob")
                    up = t.get("under_prob")
                    oml = t.get("over_ml")
                    uml = t.get("under_ml")
                    over_odds = (
                        _ml_to_dec(float(oml))
                        if oml is not None
                        else (
                            1.0 / float(op)
                            if isinstance(op, (int, float)) and float(op) > 0
                            else None
                        )
                    )
                    under_odds = (
                        _ml_to_dec(float(uml))
                        if uml is not None
                        else (
                            1.0 / float(up)
                            if isinstance(up, (int, float)) and float(up) > 0
                            else None
                        )
                    )
                    if over_odds and under_odds:
                        rows.append(
                            {
                                "date": date,
                                "home_team": home,
                                "away_team": away,
                                "side": side,
                                "line": float(line),
                                "over_odds": float(round(over_odds, 4)),
                                "under_odds": float(round(under_odds, 4)),
                                "bookmaker": "Bovada",
                                "market_type": "team",
                            }
                        )
            out_path = Path("data/corners_market_lines_manual.json")
            existing: List[Dict[str, Any]] = []
            try:
                if out_path.exists():
                    prev = json.loads(out_path.read_text(encoding="utf-8"))
                    if isinstance(prev, dict) and isinstance(prev.get("records"), list):
                        existing = prev["records"]
            except Exception:
                existing = []

            def _k2(r: Dict[str, Any]):
                return (
                    str(r.get("date"))[:19],
                    (r.get("home_team") or r.get("home") or "").lower(),
                    (r.get("away_team") or r.get("away") or "").lower(),
                    float(r.get("line")) if r.get("line") is not None else None,
                    (r.get("bookmaker") or "").lower(),
                    (r.get("market_type") or "total").lower(),
                    (r.get("side") or "").lower(),
                )

            seen = set(_k2(r) for r in existing if isinstance(r, dict))
            merged = list(existing)
            added_now = 0
            for r in rows:
                try:
                    key = _k2(r)
                    if key not in seen:
                        merged.append(r)
                        seen.add(key)
                        added_now += 1
                except Exception:
                    merged.append(r)
                    added_now += 1
            out_path.write_text(
                json.dumps(
                    {"records": merged, "source": "bovada", "added_now": added_now},
                    indent=2,
                ),
                encoding="utf-8",
            )
            totals, team = reload_market_store()
            step(
                "bovada_corners_backfill",
                {
                    "added_now": added_now,
                    "total_records": len(merged),
                    "reloaded": {"totals": totals, "team": team},
                },
            )
        except Exception as e:
            summary["errors"].append({"bovada_corners_backfill": str(e)})
    else:
        step("bovada_corners_backfill", {"skipped": True})

    # 5) Calibrate and retrain
    try:
        cal = calibration_service.calibrate_up_to_week(prev_week)
        step("calibrate", {"up_to_week": prev_week, "result": cal})
    except Exception as e:
        summary["errors"].append({"calibrate": str(e)})

    # Optional: corners calibration if service present
    try:
        c_status = corners_calibration_service.status()
        step("corners_calibration_status", c_status or {"status": "n/a"})
    except Exception:
        pass

    # Retrain and rebuild predictions
    retrain_info = {}
    try:
        new_version = advanced_ml_predictor.retrain_and_bump(level=retrain_level)
        retrain_info["model_version"] = new_version
        advanced_ml_predictor.load_models()
        perf = advanced_ml_predictor.get_model_performance()
        retrain_info["performance"] = perf
        step("retrain", retrain_info)
    except Exception as e:
        summary["errors"].append({"retrain": str(e)})
    try:
        rebuild_stats = rebuild_predictions()
        # persist full server prediction cache so API can reuse without recompute
        try:
            SERVER_PRED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            SERVER_PRED_CACHE_PATH.write_text(
                json.dumps(_prediction_cache), encoding="utf-8"
            )
        except Exception:
            pass
        step("rebuild_predictions", rebuild_stats)
    except Exception as e:
        summary["errors"].append({"rebuild_predictions": str(e)})

    # 6) Capture closing snapshot for current week
    if capture_closing:
        try:
            ws = WeekSnapshotService()
            closing = ws.capture_closing(current_week, force=True)
            step(
                "capture_closing",
                {"week": current_week, "status": closing.get("status")},
            )
        except Exception as e:
            summary["errors"].append({"capture_closing": str(e)})

    # 7) Generate odds snapshot for current week
    if generate_snapshot:
        try:
            edge = 0.05
            payload = {
                "week": current_week,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "edge_threshold": edge,
                "markets": {},
            }
            payload["markets"]["match_odds"] = compare_week_odds(
                current_week, edge_threshold=edge
            )
            payload["markets"]["totals_full"] = compare_week_totals(
                current_week, line=2.5, edge_threshold=edge
            )
            payload["markets"]["totals_first_half"] = compare_week_first_half_totals(
                current_week, line=1.0, edge_threshold=edge
            )
            payload["markets"]["totals_second_half"] = compare_week_second_half_totals(
                current_week, line=1.0, edge_threshold=edge
            )
            payload["markets"]["corners_totals"] = compare_week_corners_totals(
                current_week, line=9.5, edge_threshold=edge
            )
            payload["markets"]["team_goals_home"] = compare_week_team_goals_totals(
                current_week, side="home", line=1.5, edge_threshold=edge
            )
            payload["markets"]["team_goals_away"] = compare_week_team_goals_totals(
                current_week, side="away", line=1.5, edge_threshold=edge
            )
            payload["markets"]["team_corners_home"] = compare_week_team_corners_totals(
                current_week, side="home", line=4.5, edge_threshold=edge
            )
            payload["markets"]["team_corners_away"] = compare_week_team_corners_totals(
                current_week, side="away", line=4.5, edge_threshold=edge
            )
            out_dir = Path("data/odds_snapshots")
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            out_path = out_dir / f"week_{current_week}_odds_snapshot_{ts}.json"
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            step(
                "odds_snapshot",
                {"saved": str(out_path), "markets": list(payload["markets"].keys())},
            )
        except Exception as e:
            summary["errors"].append({"odds_snapshot": str(e)})

    summary["completed_at"] = datetime.utcnow().isoformat()
    return summary


# ---------------------------------------------------------------------------
"""Removed local NORMALIZATION_MAP in favor of shared service (team_name_normalizer)."""


def _normalize_prediction(
    raw: Dict[str, Any], home_team: str, away_team: str
) -> Dict[str, Any]:
    # Mirrors logic from main._build_normalized_prediction (keep minimal)
    total_goals = raw.get("total_goals") or raw.get("expected_total_goals") or 2.6
    p_home = float(raw.get("home_win_probability") or raw.get("home_prob") or 0.34)
    p_away = float(raw.get("away_win_probability") or raw.get("away_prob") or 0.33)
    p_draw = float(raw.get("draw_probability") or max(0.0, 1 - (p_home + p_away)))
    prob_sum = p_home + p_away + p_draw
    if prob_sum <= 0:
        p_home, p_away, p_draw = 0.34, 0.33, 0.33
        prob_sum = 1.0
    p_home, p_away, p_draw = [p / prob_sum for p in (p_home, p_away, p_draw)]
    raw_lambda_home = p_home + 0.5 * p_draw
    raw_lambda_away = p_away + 0.5 * p_draw
    lam_sum = raw_lambda_home + raw_lambda_away
    if lam_sum <= 0:
        raw_lambda_home = raw_lambda_away = 1
        lam_sum = 2
    lambda_home = total_goals * (raw_lambda_home / lam_sum)
    lambda_away = total_goals - lambda_home
    try:
        xg = expected_goals_service.get_expected_goals(home_team, away_team)
        lambda_home = 0.6 * lambda_home + 0.4 * xg["lambda_home"]
        lambda_away = 0.6 * lambda_away + 0.4 * xg["lambda_away"]
    except Exception:
        xg = None
    home_goals = round(lambda_home, 2)
    away_goals = round(lambda_away, 2)
    result_pred = "H"
    if p_away > p_home and p_away > p_draw:
        result_pred = "A"
    elif p_draw >= p_home and p_draw >= p_away:
        result_pred = "D"
    return {
        "home_goals": home_goals,
        "away_goals": away_goals,
        "confidence": raw.get(
            "model_confidence", raw.get("match_result_confidence", 0.55)
        ),
        "draw_prob": p_draw,
        "away_win_prob": p_away,
        "home_win_prob": p_home,
        "total_goals": total_goals,
        # pass-throughs when available from predictor for downstream markets
        "first_half_goals": raw.get("first_half_goals"),
        "total_corners": raw.get("total_corners"),
        "home_corners": raw.get("home_corners"),
        "away_corners": raw.get("away_corners"),
        "over_2_5_goals_probability": raw.get("over_2_5_goals_probability"),
        "result_prediction": result_pred,
        "algorithm": raw.get("algorithm", "ensemble_ml"),
        "feature_version": raw.get("match_context", {}).get("feature_version"),
    }


def _ensure_loaded():
    # Attempt to load persisted models if not already trained
    if not advanced_ml_predictor.is_trained:
        try:
            advanced_ml_predictor.load_models()
        except Exception:
            pass


def rebuild_predictions() -> Dict[str, Any]:
    _ensure_loaded()
    service = EnhancedEPLService()
    matches = service.get_all_matches()
    built = 0
    failed = 0
    # Default to EPL for this legacy sweep
    from ..services.league_manager import normalize_league_code as _norm

    code = _norm("PL")
    for m in matches:
        home_raw = m.get("home_team") or m.get("homeTeam")
        away_raw = m.get("away_team") or m.get("awayTeam")
        home = normalize_team_name(home_raw)
        away = normalize_team_name(away_raw)
        if not home or not away:
            failed += 1
            continue
        try:
            # Use league-aware features when available
            raw = advanced_ml_predictor.predict_match(home, away, league=code)
            if not raw:
                failed += 1
                continue
            norm = _normalize_prediction(raw, home, away)
            key = f"{m.get('id') or m.get('match_id')}_{home}_{away}"
            _prediction_cache[key] = norm
            built += 1
        except Exception:
            failed += 1
    summary = {
        "total_matches": len(matches),
        "predictions_built": built,
        "failed": failed,
        "cache_size": len(_prediction_cache),
    }
    return summary


def retrain(level: str) -> Dict[str, Any]:
    new_version = advanced_ml_predictor.retrain_and_bump(level=level)
    # After retrain, predictor is trained in-memory; also reload to validate persistence
    advanced_ml_predictor.load_models()
    perf = advanced_ml_predictor.get_model_performance()
    return {"model_version": new_version, "performance": perf}


def summarize_week(week: int) -> Dict[str, Any]:
    # Derive week by filtering cache keys using service's organizer
    service = EnhancedEPLService()
    matches = service.get_all_matches()
    from ..services.game_week_service import game_week_service

    weeks = game_week_service.organize_matches_by_week(matches)
    wk_matches = weeks.get(week, [])
    rows = []
    for m in wk_matches:
        home = m.get("home_team") or m.get("homeTeam")
        away = m.get("away_team") or m.get("awayTeam")
        key = f"{m.get('id') or m.get('match_id')}_{home}_{away}"
        pred = _prediction_cache.get(key)
        if pred:
            rows.append(pred)
    import statistics as stats

    def _metric(arr, func, default=None):
        return func(arr) if arr else default

    home_goals = [r["home_goals"] for r in rows]
    away_goals = [r["away_goals"] for r in rows]
    total_goals = [r["total_goals"] for r in rows]
    return {
        "week": week,
        "matches_with_predictions": len(rows),
        "home_goals_mean": _metric(home_goals, lambda a: round(sum(a) / len(a), 3)),
        "home_goals_stdev": _metric(home_goals, lambda a: round(stats.pstdev(a), 3)),
        "away_goals_mean": _metric(away_goals, lambda a: round(sum(a) / len(a), 3)),
        "away_goals_stdev": _metric(away_goals, lambda a: round(stats.pstdev(a), 3)),
        "total_goals_mean": _metric(total_goals, lambda a: round(sum(a) / len(a), 3)),
        "total_goals_stdev": _metric(total_goals, lambda a: round(stats.pstdev(a), 3)),
        "feature_versions": list({r.get("feature_version") for r in rows}),
    }


def reconcile_week(
    week: int,
    manual_path: str | None = None,
    force_repredict: bool = False,
    apply_goal_adjust: bool = False,
) -> Dict[str, Any]:
    """Reconcile predictions vs actual scores for a given game week.
    If no completed matches with scores are found and a manual file path is provided,
    load that file to inject scores. Manual file format (list of objects):
      {
        "match_id": 12345, "home_team": "Team A", "away_team": "Team B",
        "home_score": 2, "away_score": 1
      }
    """
    service = EnhancedEPLService()
    matches = service.get_all_matches()
    weeks = game_week_service.organize_matches_by_week(matches)
    # This variant operates on EPL data; use PL for league-aware predictor features
    from ..services.league_manager import normalize_league_code as _norm

    code = _norm("PL")
    wk_matches = weeks.get(week, [])
    # Build local map for quick match id lookup for manual injection
    match_index = {}
    for m in wk_matches:
        key = (
            m.get("id") or m.get("match_id"),
            normalize_team_name(m.get("home_team") or m.get("homeTeam")),
            normalize_team_name(m.get("away_team") or m.get("awayTeam")),
        )
        match_index[key] = m
    # Manual results ingest
    manual_loaded = []
    unmatched_manual: List[Dict[str, Any]] = []
    if manual_path:
        mp = Path(manual_path)
        if mp.exists():
            try:
                manual_loaded = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:
                manual_loaded = []
            # Inject manual scores into week matches (find by id or by team names)
            for row in manual_loaded:
                mid = row.get("match_id")
                ht = normalize_team_name(row.get("home_team"))
                at = normalize_team_name(row.get("away_team"))
                target = None
                if mid is not None:
                    for (rid, rht, rat), mv in match_index.items():
                        if rid == mid:
                            target = mv
                            break
                if not target:
                    for (rid, rht, rat), mv in match_index.items():
                        if rht == ht and rat == at:
                            target = mv
                            break
                if target:
                    target["home_score"] = row.get("home_score")
                    target["away_score"] = row.get("away_score")
                    target["status"] = "COMPLETED"
                    target["is_completed"] = True
                else:
                    unmatched_manual.append(row)
    # Ensure predictions exist
    if not _prediction_cache:
        rebuild_predictions()
    reconciled = 0
    considered = 0
    # Lazy compute goal adjustment offsets if requested (using existing reconciliations prior to this week run)
    offsets = None
    if apply_goal_adjust:
        # Derive bias from existing reconciled matches (excluding current week if present)
        existing = [
            r
            for r in reconciliation_store.dump().get("reconciliations", {}).values()
            if r.get("status") == "completed" and r.get("game_week") != week
        ]
        if existing:
            sh = [
                r["predicted"]["home_goals"] - r["actual"]["home_goals"]
                for r in existing
            ]
            sa = [
                r["predicted"]["away_goals"] - r["actual"]["away_goals"]
                for r in existing
            ]
            mean_h = sum(sh) / len(sh) if sh else 0
            mean_a = sum(sa) / len(sa) if sa else 0
            offsets = {"home_offset": mean_h, "away_offset": mean_a}
        else:
            offsets = {"home_offset": 0.0, "away_offset": 0.0}

    for m in wk_matches:
        home = normalize_team_name(m.get("home_team") or m.get("homeTeam"))
        away = normalize_team_name(m.get("away_team") or m.get("awayTeam"))
        # IMPORTANT: don't use `or` because 0 is a valid score and is falsy.
        hs = m.get("home_score") if "home_score" in m else m.get("homeScore")
        as_ = m.get("away_score") if "away_score" in m else m.get("awayScore")
        status_completed = m.get("status") in ["FINISHED", "COMPLETED"] or m.get(
            "is_completed"
        )
        if hs is None or as_ is None or not status_completed:
            continue
        key = f"{m.get('id') or m.get('match_id')}_{home}_{away}"
        pred = _prediction_cache.get(key)
        need_repredict = force_repredict or not pred
        if need_repredict:
            try:
                raw = advanced_ml_predictor.predict_match(home, away, league=code)
                if raw:
                    pred = _normalize_prediction(raw, home, away)
                    _prediction_cache[key] = pred
            except Exception:
                pred = None
        if not pred:
            continue
        # Build match record for reconciliation service
        match_record = dict(m)
        match_record["home_score"] = hs
        match_record["away_score"] = as_
        match_record["game_week"] = week
        # Optionally apply goal adjustment BEFORE reconciliation for fairness in metrics
        if apply_goal_adjust and offsets:
            adj_home = max(0.0, pred["home_goals"] - offsets["home_offset"])
            adj_away = max(0.0, pred["away_goals"] - offsets["away_offset"])
            adjusted_pred = dict(pred)
            adjusted_pred["home_goals_adjusted"] = round(adj_home, 2)
            adjusted_pred["away_goals_adjusted"] = round(adj_away, 2)
            # Keep original but we reconcile on adjusted values for accuracy fairness
            reconcile_input = dict(adjusted_pred)
            reconcile_input["home_goals"] = adjusted_pred["home_goals_adjusted"]
            reconcile_input["away_goals"] = adjusted_pred["away_goals_adjusted"]
            rec = reconciliation_service.reconcile_match_predictions(
                match_record, reconcile_input
            )
            rec["adjustment_applied"] = True
            rec["original_prediction"] = pred
            rec["applied_offsets"] = offsets
        else:
            rec = reconciliation_service.reconcile_match_predictions(match_record, pred)
        reconciliation_store.upsert(match_record, rec)
        reconciled += 1
        considered += 1
    reconciliation_store.compute_aggregates()
    store_dump = reconciliation_store.dump()
    return {
        "week": week,
        "matches_reconciled": reconciled,
        "weekly_stats": store_dump.get("weekly_stats", {}).get(str(week)),
        "season_stats": store_dump.get("season_stats"),
        "manual_results_loaded": len(manual_loaded) if manual_loaded else 0,
        "unmatched_manual_rows": unmatched_manual,
        "force_repredict": force_repredict,
        "goal_adjust_applied": apply_goal_adjust,
        "applied_offsets": offsets if apply_goal_adjust else None,
    }


def fetch_scores(
    competition: str,
    season: int,
    week: int,
    auto_reconcile: bool = False,
    force_repredict: bool = False,
) -> Dict[str, Any]:
    """Fetch final scores from Football-Data.org API.
    Requires environment variable FOOTBALL_DATA_API_TOKEN.
    Saves a manual results file for the specified week.
    """
    token = os.getenv("FOOTBALL_DATA_API_TOKEN")
    if not token:
        return {"error": "FOOTBALL_DATA_API_TOKEN not set in environment"}
    import requests

    # Football-Data.org season format e.g. 2025 => 2025 or 2025 for 2025-26 (API uses YYYY)
    season_str = str(season)
    base_url = f"https://api.football-data.org/v4/competitions/{competition}/matches"
    params = {"season": season_str, "matchday": week}
    headers = {"X-Auth-Token": token}
    try:
        resp = requests.get(base_url, headers=headers, params=params, timeout=15)
        status_code = resp.status_code
        if status_code != 200:
            return {"error": f"API status {status_code}", "body": resp.text[:500]}
        data = resp.json()
        matches = data.get("matches", [])
        manual_rows = []
        for m in matches:
            score = m.get("score", {})
            full = score.get("fullTime", {})
            home_score = full.get("home")
            away_score = full.get("away")
            status = m.get("status")
            # Only include finished matches with numeric scores
            if (
                home_score is None
                or away_score is None
                or status not in ("FINISHED", "IN_PLAY", "TIMED")
            ):  # FINISHED primary, but keep others if partial
                continue
            manual_rows.append(
                {
                    "match_id": m.get("id"),
                    "home_team": m.get("homeTeam", {}).get("name"),
                    "away_team": m.get("awayTeam", {}).get("name"),
                    "home_score": home_score,
                    "away_score": away_score,
                    "status": status,
                }
            )
        out_dir = Path("data")
        out_dir.mkdir(parents=True, exist_ok=True)
        manual_path = out_dir / f"manual_results_week{week}.json"
        manual_path.write_text(json.dumps(manual_rows, indent=2), encoding="utf-8")
        result = {
            "fetched": len(matches),
            "usable_finished": len(manual_rows),
            "manual_file": str(manual_path),
        }
        if auto_reconcile and manual_rows:
            rec = reconcile_week(
                week, manual_path=str(manual_path), force_repredict=force_repredict
            )
            result["reconciliation"] = rec
        return result
    except Exception as e:
        return {"error": str(e)}


def diagnostics(
    week: int | None = None, calibration: bool = False, goal_adjust: bool = False
) -> Dict[str, Any]:
    """Compute extended diagnostics over reconciled matches.
    - Signed goal errors (home/away/total) & MAE
    - Optional probability calibration bins (max-class probability vs empirical accuracy)
    - Optional goal adjustment suggestion (offsets to remove mean bias)
    """
    store = reconciliation_store.dump()
    recs = list(store.get("reconciliations", {}).values())
    if week is not None:
        recs = [r for r in recs if str(r.get("game_week")) == str(week)]
    completed = [r for r in recs if r.get("status") == "completed"]
    if not completed:
        return {"matches": 0, "error": "No completed reconciliations in scope"}
    signed_home: List[float] = []
    signed_away: List[float] = []
    signed_total: List[float] = []
    calibration_rows = []
    for r in completed:
        a = r.get("actual", {})
        p = r.get("predicted", {})
        ah, aa = a.get("home_goals", 0), a.get("away_goals", 0)
        ph, pa = p.get("home_goals", 0), p.get("away_goals", 0)
        signed_home.append(ph - ah)
        signed_away.append(pa - aa)
        signed_total.append((ph + pa) - (ah + aa))
        if calibration:
            phw, pdw, paw = (
                p.get("home_win_prob"),
                p.get("draw_prob"),
                p.get("away_win_prob"),
            )
            if None not in (phw, pdw, paw):
                max_prob = max(phw, pdw, paw)
                pred_cls = (
                    "H" if phw >= pdw and phw >= paw else ("D" if pdw >= paw else "A")
                )
                actual_cls = a.get("result") or (
                    "H" if ah > aa else "A" if aa > ah else "D"
                )
                calibration_rows.append(
                    {"p": max_prob, "correct": int(pred_cls == actual_cls)}
                )

    def _m(arr: List[float]):
        return round(sum(arr) / len(arr), 4) if arr else None

    def _mae(arr: List[float]):
        return round(sum(abs(x) for x in arr) / len(arr), 4) if arr else None

    diag: Dict[str, Any] = {
        "matches": len(completed),
        "scope_week": week,
        "signed_error_home_mean": _m(signed_home),
        "signed_error_away_mean": _m(signed_away),
        "signed_error_total_mean": _m(signed_total),
        "mae_home": _mae(signed_home),
        "mae_away": _mae(signed_away),
        "mae_total": _mae(signed_total),
        "home_over_bias": _m(signed_home),
        "away_over_bias": _m(signed_away),
        "total_over_bias": _m(signed_total),
    }
    if calibration and calibration_rows:
        # Build bins of width 0.1 from 0.0 to 1.0
        bins = []
        for lo in [i / 10 for i in range(0, 10)]:
            hi = lo + 0.1
            bucket = [
                r
                for r in calibration_rows
                if lo <= r["p"] < hi or (hi >= 1.0 and r["p"] == 1.0)
            ]
            if not bucket:
                continue
            mean_p = sum(r["p"] for r in bucket) / len(bucket)
            acc = sum(r["correct"] for r in bucket) / len(bucket)
            bins.append(
                {
                    "range": f"{lo:.1f}-{hi:.1f}",
                    "count": len(bucket),
                    "mean_prob": round(mean_p, 4),
                    "empirical_accuracy": round(acc, 4),
                    "calibration_gap": round(acc - mean_p, 4),
                }
            )
        diag["calibration_bins"] = bins
    if goal_adjust:
        # Propose subtracting mean signed error (cannot go below zero at floor in application layer)
        diag["suggested_goal_adjustment"] = {
            "home_offset": -_m(signed_home) if signed_home else 0,
            "away_offset": -_m(signed_away) if signed_away else 0,
            "total_offset": -_m(signed_total) if signed_total else 0,
            "method": "subtract_mean_signed_error",
        }
    return diag


# ---------------------------------------------------------------------------
# Walk-forward: predict -> reconcile -> calibrate -> retrain -> next week
# ---------------------------------------------------------------------------
def walkforward_train(
    start_week: int,
    end_week: int,
    fetch_scores_first: bool = True,
    retrain_level: str = "patch",
    apply_goal_adjust: bool = False,
) -> Dict[str, Any]:
    """Run a week-by-week pipeline:
    For each week in [start_week, end_week]:
      1) Ensure predictions exist (rebuild if empty)
      2) Optionally fetch final scores from API into data/manual_results_week{w}.json
      3) Reconcile that week (using manual file if available)
      4) Calibrate model probabilities up to this week
      5) Retrain models and bump version (level)
      6) Rebuild predictions for all matches with the new model
        weeks = game_week_service.organize_matches_by_week(matches)
        # Normalize league code for predictor calls in this function
        from ..services.league_manager import normalize_league_code as _norm
        code = _norm(league)
    Returns a summary with per-week reconciliation stats and model version after each step.
    """
    if start_week > end_week:
        start_week, end_week = end_week, start_week
    # Step 0: ensure we can predict
    if not _prediction_cache:
        rebuild_predictions()
    weeks_summary: List[Dict[str, Any]] = []
    model_versions: List[str] = []
    for w in range(start_week, end_week + 1):
        step: Dict[str, Any] = {"week": w}
        # 1) Optionally fetch scores (writes manual file) then reconcile
        manual_path = None
        if fetch_scores_first:
            fetched = fetch_scores("PL", 2025, w, auto_reconcile=False)
            step["fetch_scores"] = fetched
            if isinstance(fetched, dict) and fetched.get("manual_file"):
                manual_path = fetched["manual_file"]
        else:
            p = Path(f"data/manual_results_week{w}.json")
            if p.exists():
                manual_path = str(p)
        rec = reconcile_week(
            w,
            manual_path=manual_path,
            force_repredict=False,
            apply_goal_adjust=apply_goal_adjust,
        )
        step["reconcile"] = rec
        # 2) Calibrate up to this week (uses completed matches only)
        cal = calibration_service.calibrate_up_to_week(w)
        step["calibration"] = {
            "up_to_week": w,
            "result": cal,
            "status": calibration_service.status(),
        }
        # 3) Retrain & bump model version
        ver = advanced_ml_predictor.retrain_and_bump(level=retrain_level)
        model_versions.append(ver)
        step["retrain"] = {
            "model_version": ver,
            "performance": advanced_ml_predictor.get_model_performance(),
        }
        # 4) Rebuild predictions to refresh cache with new model
        step["rebuild"] = rebuild_predictions()
        weeks_summary.append(step)
    # Persist summary
    summary = {
        "range": {"start_week": start_week, "end_week": end_week},
        "model_versions": model_versions,
        "steps": weeks_summary,
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }
    path = _write_output(f"walkforward_w{start_week}-w{end_week}", summary)
    return {"saved": path, **summary}


# ---------------------------------------------------------------------------
# Week Odds vs Model vs Actual Outcome Comparison
# ---------------------------------------------------------------------------
def _load_historic_odds_index(
    cache_path: str = "cache/historic_odds.json",
) -> Dict[str, Dict[str, Any]]:
    """Build an index (date_home_away and home_away) to historic consensus implied probs.
    Returns mapping to a record containing consensus implied probs and overround.
    """
    idx: Dict[str, Dict[str, Any]] = {}
    # Preferred bookmakers (by key/title) environment control; default to Bovada/DraftKings/FanDuel/Bet365
    pref_env = os.getenv("PREFERRED_BOOKMAKERS", "bet365,draftkings,fanduel,bovada")
    preferred_keys: Set[str] = {
        k.strip().lower() for k in pref_env.split(",") if k.strip()
    }
    try:
        p = Path(cache_path)
        if not p.exists():
            return idx
        data = json.loads(p.read_text(encoding="utf-8"))
        for rec in data.get("records", []):
            ht = normalize_team_name(rec.get("home_team")) or rec.get("home_team")
            at = normalize_team_name(rec.get("away_team")) or rec.get("away_team")
            cons = rec.get("consensus_implied") or {}
            if not (ht and at and cons):
                continue
            # Prefer Bet365 from historic CSVs if available (maps to key 'B365') when user preference includes bet365
            bookmakers = rec.get("bookmakers") or {}
            preferred_implied: Optional[Dict[str, Any]] = None
            preferred_over: Optional[float] = None
            # Football-Data CSV uses 'B365' for Bet365
            if "bet365" in preferred_keys:
                b365 = bookmakers.get("B365")
                if isinstance(b365, dict):
                    ip = b365.get("implied_probabilities")
                    ov = b365.get("overround")
                    if isinstance(ip, dict) and all(k in ip for k in ("H", "D", "A")):
                        preferred_implied = ip
                        preferred_over = ov if isinstance(ov, (int, float)) else None
            # Attach preferred into the record for downstream consumers
            if preferred_implied is not None:
                rec = {
                    **rec,
                    "preferred_implied": preferred_implied,
                    "preferred_overround": preferred_over,
                }
            key_basic = f"{ht}|{at}".lower()
            idx[key_basic] = rec
            date = rec.get("date")
            if date:
                idx[f"{date}|{key_basic}"] = rec
    except Exception:
        return idx
    return idx


def compare_week_odds(
    week: int,
    edge_threshold: float = 0.05,
    league: Optional[str] = None,
    use_live: Optional[bool] = None,
    prob_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compare (historic) market consensus odds vs model vs actual outcomes for a game week.

    Produces per-match rows with:
      - model probabilities (H/D/A)
      - market consensus implied probabilities (H/D/A)
      - actual result (H/D/A) if available
      - model_pick / market_pick and correctness
      - edge for model_pick (model_prob - market_prob)
      - basic scoring metrics aggregated (log loss, brier, accuracy)
    """
    # Ensure predictions (local cache for on-demand); also load the server's
    # persisted cache so we can align with what the UI card shows.
    if not _prediction_cache:
        rebuild_predictions()
    server_pred_cache = _load_server_prediction_cache()
    code = normalize_league_code(league)
    service = get_league_service(code)
    matches = (
        service.get_all_matches()
        if hasattr(service, "get_all_matches")
        else EnhancedEPLService().get_all_matches()
    )
    weeks = game_week_service.organize_matches_by_week(matches)
    wk_matches = weeks.get(week, [])
    historic_idx = _load_historic_odds_index()
    # Augment with recent CSV historics when present (last 60 days)
    try:
        csv_idx = load_h2h_index_from_csv(
            code or "PL",
            days=60,
            preferred_bookmakers=[
                s.strip()
                for s in (
                    os.getenv(
                        "PREFERRED_BOOKMAKERS", "bet365,draftkings,fanduel,bovada"
                    )
                ).split(",")
                if s.strip()
            ],
        )
        # Merge: prefer historic (older, curated) first; fill missing keys with CSV
        for k, v in (csv_idx or {}).items():
            if k not in historic_idx:
                historic_idx[k] = v
    except Exception:
        pass
    # Build live odds index from The Odds API (real provider) if available
    live_idx: Dict[str, Dict[str, Any]] = {}
    live_error: Optional[str] = None
    # Allow disabling live fetch for speed via parameter or env var
    allow_live: bool = False
    try:
        if use_live is not None:
            allow_live = bool(use_live)
        else:
            allow_live = str(os.getenv("ODDS_COMPARE_USE_LIVE", "0")).strip() in (
                "1",
                "true",
                "True",
            )
    except Exception:
        allow_live = False
    try:
        if allow_live and fetch_h2h_odds is not None:
            # Default to UK/EU/US to maximize bookmaker coverage (can override via env)
            regions = os.getenv("ODDS_API_REGIONS", "uk,eu,us")
            # Map league to Odds API sport key
            sport_map = {
                "PL": "soccer_epl",
                "BL1": "soccer_germany_bundesliga",
                "FL1": "soccer_france_ligue_one",
                "SA": "soccer_italy_serie_a",
                "PD": "soccer_spain_la_liga",
            }
            sport_key = sport_map.get(code, "soccer_epl")
            live_payload = fetch_h2h_odds(
                sport_key=sport_key, regions=regions, markets="h2h"
            )
            if isinstance(live_payload, dict) and live_payload.get("error"):
                live_error = str(live_payload.get("error"))
            events = live_payload.get("events") or []
            pref_env = os.getenv(
                "PREFERRED_BOOKMAKERS", "bet365,draftkings,fanduel,bovada"
            )
            preferred_keys: Set[str] = {
                k.strip().lower() for k in pref_env.split(",") if k.strip()
            }
            for ev in events:
                h = normalize_team_name(ev.get("home_team"))
                a = normalize_team_name(ev.get("away_team"))
                if not (h and a):
                    continue
                # gather bookmaker implied probs and overrounds
                bms = ev.get("bookmakers") or []

                # Optionally filter to preferred bookmakers if present; only keep those with valid h2h implied probabilities
                def has_probs(bm: Dict[str, Any]) -> bool:
                    ip = (bm.get("h2h") or {}).get("implied_probabilities") or {}
                    return all(
                        isinstance(ip.get(k), (int, float)) and ip.get(k) > 0
                        for k in ("H", "D", "A")
                    )

                preferred_present = [
                    bm
                    for bm in bms
                    if (
                        (bm.get("key") or "").lower() in preferred_keys
                        or (bm.get("title") or "").lower() in preferred_keys
                    )
                ]
                filtered_valid = [bm for bm in preferred_present if has_probs(bm)]
                any_valid = [bm for bm in bms if has_probs(bm)]
                if filtered_valid:
                    use_bms = filtered_valid
                elif any_valid:
                    use_bms = any_valid
                else:
                    use_bms = []
                probs_acc = {"H": [], "D": [], "A": []}
                over_acc: List[float] = []
                used_keys: List[str] = []
                for bm in use_bms:
                    h2h = bm.get("h2h") or {}
                    ip = h2h.get("implied_probabilities") or {}
                    ov = h2h.get("overround")
                    for k in ("H", "D", "A"):
                        v = ip.get(k)
                        if isinstance(v, (int, float)) and v > 0:
                            probs_acc[k].append(float(v))
                    if isinstance(ov, (int, float)):
                        over_acc.append(float(ov))
                    if bm.get("key"):
                        used_keys.append(str(bm.get("key")).lower())

                # derive simple consensus (median if available, else average)
                def _cons(vals: List[float]):
                    if not vals:
                        return None
                    try:
                        import statistics as stats

                        return float(stats.median(vals))
                    except Exception:
                        return float(sum(vals) / len(vals))

                consensus = {k: _cons(vs) for k, vs in probs_acc.items()}
                # normalize to sum 1 if all present
                if all(consensus.get(k) is not None for k in ("H", "D", "A")):
                    s = sum(float(consensus[k]) for k in ("H", "D", "A"))
                    if s and s > 0:
                        consensus = {
                            k: float(consensus[k]) / s for k in ("H", "D", "A")
                        }
                overround = _cons(over_acc)
                key_basic = f"{h}|{a}".lower()
                # Only store when we have at least one probability sample; otherwise skip to avoid false positives
                if any(vs for vs in probs_acc.values()):
                    live_idx[key_basic] = {
                        "consensus_implied": consensus,
                        "consensus_overround": overround,
                        "source": (
                            "live_preferred" if filtered_valid else "live_odds_api"
                        ),
                        "commence_time": ev.get("commence_time"),
                    }
                    if used_keys:
                        live_idx[key_basic]["bookmakers_used"] = used_keys
                if used_keys:
                    # add date-index key only if record created
                    pass
                # add date-index key
                ct = ev.get("commence_time")
                if isinstance(ct, str) and "T" in ct:
                    date_only = ct.split("T")[0]
                    if key_basic in live_idx:
                        live_idx[f"{date_only}|{key_basic}"] = live_idx[key_basic]
    except Exception as e:
        # If live fetch fails, proceed without it but record error for diagnostics
        live_idx = {}
        try:
            live_error = str(e)
        except Exception:
            live_error = "unknown"
    # Backup: Bovada odds (best-effort) to fill missing matches
    try:
        fetcher = _get_bovada_fetcher(code)
        if fetcher is not None:
            bova = fetcher() if fetcher is not None else {"events": []}
            b_events = bova.get("events") or []
            for ev in b_events:
                h = normalize_team_name(ev.get("home_team"))
                a = normalize_team_name(ev.get("away_team"))
                if not (h and a):
                    continue
                key_basic = f"{h}|{a}".lower()
                # do not overwrite if already present from primary live source
                if key_basic in live_idx:
                    continue
                h2h = ev.get("h2h") or {}
                if all(isinstance(h2h.get(k), (int, float)) for k in ("H", "D", "A")):
                    tot = h2h["H"] + h2h["D"] + h2h["A"]
                    if tot and tot > 0:
                        probs = {k: float(h2h[k]) / tot for k in ("H", "D", "A")}
                    else:
                        probs = h2h
                    live_idx[key_basic] = {
                        "consensus_implied": probs,
                        "consensus_overround": None,
                        "source": "live_bovada",
                        "commence_time": ev.get("commence_time"),
                    }
    except Exception:
        # ignore bovada errors silently as it's a backup
        pass
    # Fallback provider (mock). Only used if explicitly enabled via env.
    fallback_odds_service = BettingOddsService()
    allow_fallback = os.getenv("ALLOW_MARKET_FALLBACK", "0") == "1"
    rows: List[Dict[str, Any]] = []
    model_log_losses: List[float] = []
    market_log_losses: List[float] = []
    model_brier: List[float] = []
    market_brier: List[float] = []
    model_correct = 0
    market_correct = 0
    total_evaluable = 0
    eps = 1e-12

    for m in wk_matches:
        home_raw = m.get("home_team") or m.get("homeTeam")
        away_raw = m.get("away_team") or m.get("awayTeam")
        home = normalize_team_name(home_raw) or home_raw
        away = normalize_team_name(away_raw) or away_raw
        date = (m.get("utc_date") or m.get("date") or "").split("T")[0]
        mid = m.get("id") or m.get("match_id")
        # IMPORTANT: The server cache key uses RAW team names. Use that for lookup
        # to avoid missing entries and recomputing with slightly different settings.
        key_pred_server = f"{mid}_{home_raw}_{away_raw}"
        key_pred_local = f"{mid}_{home}_{away}"
        pred: Optional[Dict[str, Any]] = None
        pred_from_server_cache = False
        if server_pred_cache:
            pred = server_pred_cache.get(key_pred_server)
            pred_from_server_cache = pred is not None
        if pred is None:
            pred = _prediction_cache.get(key_pred_local)
        if not pred:
            try:
                raw = advanced_ml_predictor.predict_match(home, away, league=code)
                if raw:
                    pred = _normalize_prediction(raw, home, away)
                    _prediction_cache[key_pred_local] = pred
            except Exception:
                pred = None
        # Actual scores/result
        hs = m.get("home_score") if "home_score" in m else m.get("homeScore")
        as_ = m.get("away_score") if "away_score" in m else m.get("awayScore")
        status_completed = m.get("status") in ["FINISHED", "COMPLETED"] or m.get(
            "is_completed"
        )
        actual_result = None
        if hs is not None and as_ is not None and status_completed:
            if hs > as_:
                actual_result = "H"
            elif as_ > hs:
                actual_result = "A"
            else:
                actual_result = "D"
        # Market consensus lookup
        market_rec = None
        if date:
            market_rec = historic_idx.get(f"{date}|{home.lower()}|{away.lower()}")
        if not market_rec:
            market_rec = historic_idx.get(f"{home}|{away}".lower())
        market_probs = None
        overround = None
        market_source = None
        preferred_decimals = None  # for EV when available from CSV historics
        ev_bookmaker = None
        if market_rec:
            # Prefer single-bookmaker (Bet365) from historic if present & configured, else consensus
            cons = (
                market_rec.get("preferred_implied")
                or market_rec.get("consensus_implied")
                or {}
            )
            h = cons.get("H")
            d = cons.get("D")
            a = cons.get("A")
            if all(isinstance(v, (int, float)) for v in (h, d, a)):
                total = h + d + a
                if total > 0:
                    market_probs = {"H": h / total, "D": d / total, "A": a / total}
                else:
                    market_probs = {"H": h, "D": d, "A": a}
            overround = (
                market_rec.get("preferred_overround")
                if market_rec.get("preferred_implied")
                else market_rec.get("consensus_overround")
            )
            market_source = (
                "historic_bet365"
                if market_rec.get("preferred_implied")
                else "historic_consensus"
            )
            # Capture preferred bookmaker decimals for EV when present from CSV historics
            try:
                pd_map = market_rec.get("preferred_decimals")
                if isinstance(pd_map, dict) and all(
                    k in pd_map for k in ("H", "D", "A")
                ):
                    preferred_decimals = {
                        k: float(pd_map.get(k))
                        if isinstance(pd_map.get(k), (int, float))
                        else None
                        for k in ("H", "D", "A")
                    }
                    ev_bookmaker = market_rec.get("preferred_bookmaker")
            except Exception:
                preferred_decimals = None
        # If still not found, try live real odds index from The Odds API
        if market_probs is None and live_idx:
            rec = None
            if date:
                rec = live_idx.get(f"{date}|{home.lower()}|{away.lower()}")
            if rec is None:
                rec = live_idx.get(f"{home}|{away}".lower())
            if rec:
                cons = rec.get("consensus_implied") or {}
                h = cons.get("H")
                d = cons.get("D")
                a = cons.get("A")
                if all(isinstance(v, (int, float)) for v in (h, d, a)):
                    tot = h + d + a
                    if tot and tot > 0:
                        market_probs = {"H": h / tot, "D": d / tot, "A": a / tot}
                    else:
                        market_probs = {"H": h, "D": d, "A": a}
                    overround = rec.get("consensus_overround")
                    market_source = rec.get("source") or "live_odds_api"
        model_probs = None
        model_pick = None
        if pred:
            model_probs = {
                "H": pred.get("home_win_prob"),
                "D": pred.get("draw_prob"),
                "A": pred.get("away_win_prob"),
            }
            # Only apply calibration if we generated the prediction on-the-fly.
            # Cached server predictions are already calibrated (when calibration is active).
            if not pred_from_server_cache:
                model_probs = calibration_service.apply_if_ready(model_probs)
            model_pick = max(
                model_probs.items(),
                key=lambda kv: (kv[1] if isinstance(kv[1], (int, float)) else -1),
            )[0]
        market_pick = None
        if market_probs:
            market_pick = max(market_probs.items(), key=lambda kv: kv[1])[0]
        edge_for_model_pick = None
        edge_recommendation = False
        if model_probs and market_probs and model_pick:
            edge_for_model_pick = round(
                model_probs[model_pick] - market_probs.get(model_pick, 0.0), 4
            )
            # pick-gating: require both edge and minimum model probability for the pick
            try:
                pick_prob_ok = float(model_probs.get(model_pick, 0.0)) >= float(
                    prob_threshold
                )
            except Exception:
                pick_prob_ok = False
            edge_recommendation = (
                edge_for_model_pick is not None
                and edge_for_model_pick >= edge_threshold
                and pick_prob_ok
            )
        # Scoring metrics if actual result & both prob sets known
        if actual_result and model_probs and market_probs:
            total_evaluable += 1
            # log loss
            model_p = max(min(model_probs[actual_result], 1 - eps), eps)
            market_p = max(min(market_probs[actual_result], 1 - eps), eps)
            import math

            model_log_losses.append(-math.log(model_p))
            market_log_losses.append(-math.log(market_p))
            # Brier (multi-class: sum (p_i - y_i)^2)
            for probs, bucket in (
                (model_probs, model_brier),
                (market_probs, market_brier),
            ):
                b = 0.0
                for k in ("H", "D", "A"):
                    y = 1.0 if k == actual_result else 0.0
                    b += (probs[k] - y) ** 2
                bucket.append(b)
            if model_pick == actual_result:
                model_correct += 1
            if market_pick == actual_result:
                market_correct += 1
        # EV calculations (requires model probabilities and decimals)
        ev_outcomes = None
        ev_for_model_pick = None
        best_ev_outcome = None
        best_ev = None
        if model_probs and isinstance(preferred_decimals, dict):
            try:
                ev_map = {}
                for o in ("H", "D", "A"):
                    p = model_probs.get(o)
                    dec = preferred_decimals.get(o) if preferred_decimals else None
                    if (
                        isinstance(p, (int, float))
                        and isinstance(dec, (int, float))
                        and dec > 1.0
                    ):
                        ev_map[o] = round(p * dec - 1.0, 4)
                if ev_map:
                    ev_outcomes = ev_map
                    # EV for model's pick when available
                    if model_pick and model_pick in ev_map:
                        ev_for_model_pick = ev_map[model_pick]
                    # Best EV outcome across available prices
                    best_ev_outcome, best_ev = max(ev_map.items(), key=lambda kv: kv[1])
            except Exception:
                ev_outcomes = None
                ev_for_model_pick = None
                best_ev_outcome = None
                best_ev = None

        rows.append(
            {
                "match_id": mid,
                # Return raw names to match frontend cards; include normalized for diagnostics
                "home_team": home_raw,
                "away_team": away_raw,
                "home_team_normalized": home,
                "away_team_normalized": away,
                "date": date or None,
                "home_score": hs,
                "away_score": as_,
                "actual_result": actual_result,
                "model_probs": {
                    k: (round(v, 4) if isinstance(v, (int, float)) else None)
                    for k, v in (model_probs or {}).items()
                }
                if model_probs
                else None,
                "market_probs": {
                    k: (round(v, 4) if isinstance(v, (int, float)) else None)
                    for k, v in (market_probs or {}).items()
                }
                if market_probs
                else None,
                "model_pick": model_pick,
                "market_pick": market_pick,
                "edge_for_model_pick": edge_for_model_pick,
                "edge_recommendation": edge_recommendation,
                "market_overround": overround,
                "market_source": market_source,
                # EV analytics (when CSV historics had bookmaker decimals)
                "ev_outcomes": ev_outcomes,
                "ev_for_model_pick": ev_for_model_pick,
                "best_ev_outcome": best_ev_outcome,
                "best_ev": best_ev,
                "ev_bookmaker": ev_bookmaker,
                "preferred_decimals": preferred_decimals,
            }
        )
    # Aggregate metrics
    # Extra summaries: edge count, pick alignment, market source counts
    recommended_edges = sum(1 for r in rows if r.get("edge_recommendation"))
    pick_alignments = sum(
        1
        for r in rows
        if r.get("model_pick")
        and r.get("market_pick")
        and r["model_pick"] == r["market_pick"]
    )
    source_counts: Dict[str, int] = {}
    for r in rows:
        src = r.get("market_source") or "unknown"
        source_counts[src] = source_counts.get(src, 0) + 1

    agg = {
        "week": week,
        "matches_considered": len(wk_matches),
        "rows_produced": len(rows),
        "evaluable_with_actual_and_probs": total_evaluable,
        "model_accuracy": (model_correct / total_evaluable)
        if total_evaluable
        else None,
        "market_accuracy": (market_correct / total_evaluable)
        if total_evaluable
        else None,
        "model_log_loss": (sum(model_log_losses) / len(model_log_losses))
        if model_log_losses
        else None,
        "market_log_loss": (sum(market_log_losses) / len(market_log_losses))
        if market_log_losses
        else None,
        "model_brier": (sum(model_brier) / len(model_brier)) if model_brier else None,
        "market_brier": (sum(market_brier) / len(market_brier))
        if market_brier
        else None,
        "edge_threshold": edge_threshold,
        "prob_threshold": prob_threshold,
        "recommended_edges": recommended_edges,
        "edge_rate": (recommended_edges / len(rows)) if rows else None,
        "pick_alignment_rate": (pick_alignments / len(rows)) if rows else None,
        "market_source_counts": source_counts,
        # Live odds diagnostics (helps detect missing API key/region issues in UI)
        "live_odds_events_seen": len(live_idx),
        "live_odds_error": live_error,
    }
    # Advantage indicators (lower log loss / brier is better)
    if agg["model_log_loss"] is not None and agg["market_log_loss"] is not None:
        agg["model_log_loss_delta"] = round(
            agg["market_log_loss"] - agg["model_log_loss"], 4
        )
    if agg["model_brier"] is not None and agg["market_brier"] is not None:
        agg["model_brier_delta"] = round(agg["market_brier"] - agg["model_brier"], 4)
    out = {"aggregate": agg, "matches": rows}
    try:
        out["league"] = code
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Week Totals (Over/Under 2.5) vs Model vs Market
# ---------------------------------------------------------------------------
def compare_week_totals(
    week: int,
    line: float = 2.5,
    edge_threshold: float = 0.05,
    league: Optional[str] = None,
    use_live: Optional[bool] = None,
) -> Dict[str, Any]:
    """Compare totals market (Over/Under at given line) vs model vs actual for a week.

    Requirements:
      - Historic odds ingestion must include totals_2_5 per bookmaker and consensus_totals_2_5 in records.
      - Model provides over_2_5_goals_probability via advanced_ml_predictor.
    Produces metrics: accuracy, log loss (binary), brier, edges for Over/Under.
    """
    if not _prediction_cache:
        rebuild_predictions()
    code = normalize_league_code(league)
    service = get_league_service(code)
    matches = (
        service.get_all_matches()
        if hasattr(service, "get_all_matches")
        else EnhancedEPLService().get_all_matches()
    )
    weeks = game_week_service.organize_matches_by_week(matches)
    wk_matches = weeks.get(week, [])
    historic_idx = _load_historic_odds_index()
    # Live gating
    allow_live: bool = False
    try:
        if use_live is not None:
            allow_live = bool(use_live)
        else:
            allow_live = str(os.getenv("ODDS_COMPARE_USE_LIVE", "0")).strip() in (
                "1",
                "true",
                "True",
            )
    except Exception:
        allow_live = False
    # Backup live Bovada totals lookup (only if live enabled)
    bovada_totals: Dict[str, Dict[str, Any]] = {}
    if allow_live:
        try:
            fetcher = _get_bovada_fetcher(code)
            if fetcher is not None:
                bova = fetcher() if fetcher is not None else {"events": []}
                for ev in bova.get("events") or []:
                    h = normalize_team_name(ev.get("home_team")) or ev.get("home_team")
                    a = normalize_team_name(ev.get("away_team")) or ev.get("away_team")
                    key = f"{h}|{a}".lower()
                    for t in ev.get("totals") or []:
                        try:
                            l = (
                                float(t.get("line"))
                                if t.get("line") is not None
                                else None
                            )
                        except Exception:
                            l = None
                        if l is None:
                            continue
                        bovada_totals.setdefault(key, {})[l] = {
                            "over_prob": t.get("over_prob"),
                            "under_prob": t.get("under_prob"),
                            "market_line": l,
                            "source": "live_bovada",
                        }
        except Exception:
            pass

    # Try real goals market store first (if populated), then historic consensus
    def get_totals_from_store(date: str, home: str, away: str, line: float):
        try:
            g = goals_market_store.lookup_total(date, home, away, line)
            if g:
                over_p, bookmaker, used_line = g
                return {
                    "line": used_line,
                    "over_implied": over_p,
                    "under_implied": (1.0 - over_p)
                    if isinstance(over_p, (int, float))
                    else None,
                    "overround": None,
                    "source": "goals_market_store",
                }
        except Exception:
            return None
        return None

    # helper to pull totals consensus from historic record if present
    def get_totals_from_historic(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not rec:
            return None
        tot = rec.get("consensus_totals_2_5")
        if isinstance(tot, dict) and float(tot.get("line", 2.5)) == float(line):
            return tot
        # attempt derive from bookmakers if consensus absent
        bms = rec.get("bookmakers") or {}
        over_vals: List[float] = []
        under_vals: List[float] = []
        overrounds: List[float] = []
        for bm_data in bms.values():
            t = bm_data.get("totals_2_5") if isinstance(bm_data, dict) else None
            if not isinstance(t, dict):
                continue
            if float(t.get("line", 2.5)) != float(line):
                continue
            oi = t.get("over_implied")
            ui = t.get("under_implied")
            oor = t.get("overround")
            if isinstance(oi, (int, float)):
                over_vals.append(float(oi))
            if isinstance(ui, (int, float)):
                under_vals.append(float(ui))
            if isinstance(oor, (int, float)):
                overrounds.append(float(oor))
        if over_vals and under_vals:
            try:
                import statistics as stats

                return {
                    "line": line,
                    "over_implied": float(stats.median(over_vals)),
                    "under_implied": float(stats.median(under_vals)),
                    "overround": float(stats.median(overrounds))
                    if overrounds
                    else None,
                }
            except Exception:
                return {
                    "line": line,
                    "over_implied": sum(over_vals) / len(over_vals),
                    "under_implied": sum(under_vals) / len(under_vals),
                    "overround": sum(overrounds) / len(overrounds)
                    if overrounds
                    else None,
                }
        return None

    rows: List[Dict[str, Any]] = []
    model_log_losses: List[float] = []
    market_log_losses: List[float] = []
    model_briers: List[float] = []
    market_briers: List[float] = []
    model_correct = 0
    market_correct = 0
    total_evaluable = 0
    eps = 1e-12

    for m in wk_matches:
        home_raw = m.get("home_team") or m.get("homeTeam")
        away_raw = m.get("away_team") or m.get("awayTeam")
        home = normalize_team_name(home_raw) or home_raw
        away = normalize_team_name(away_raw) or away_raw
        date = (m.get("utc_date") or m.get("date") or "").split("T")[0]
        mid = m.get("id") or m.get("match_id")
        key_pred = f"{mid}_{home}_{away}"
        pred = _prediction_cache.get(key_pred)
        if not pred:
            try:
                # Pass league to engage league-aware feature builder for non-PL
                raw = advanced_ml_predictor.predict_match(home, away, league=code)
                if raw:
                    pred = _normalize_prediction(raw, home, away)
                    _prediction_cache[key_pred] = pred
            except Exception:
                pred = None

        hs = m.get("home_score") if "home_score" in m else m.get("homeScore")
        as_ = m.get("away_score") if "away_score" in m else m.get("awayScore")
        status_completed = m.get("status") in ["FINISHED", "COMPLETED"] or m.get(
            "is_completed"
        )
        actual_total = None
        actual_over = None
        if hs is not None and as_ is not None and status_completed:
            actual_total = hs + as_
            actual_over = 1 if (actual_total > line) else 0

        # Market totals lookup: prefer real store, then historic, finally Bovada for EPL
        market_store = get_totals_from_store(date, home, away, line)
        market_totals = market_store
        market_source_str = (market_store or {}).get("source") if market_store else None
        if not market_totals:
            market_rec = None
            if date:
                market_rec = historic_idx.get(f"{date}|{home.lower()}|{away.lower()}")
            if not market_rec:
                market_rec = historic_idx.get(f"{home}|{away}".lower())
            market_totals = get_totals_from_historic(market_rec) if market_rec else None
            market_source_str = "historic_totals_consensus" if market_totals else None
        # If no historic, try Bovada totals at matching line (any league supported by fetcher)
        if allow_live and (not market_totals) and date:
            key = f"{home.lower()}|{away.lower()}"
            by_match = bovada_totals.get(key) or {}
            # Support nearest-line tolerance (±0.25) for Bovada totals
            try:
                import os

                tol = float(os.getenv("BOVADA_LINE_TOLERANCE", "0.25"))
            except Exception:
                tol = 0.25
            chosen_line = None
            if by_match:
                # Exact first
                if line in by_match:
                    chosen_line = line
                else:
                    try:
                        # pick closest line within tolerance
                        candidates = list(by_match.keys())
                        if candidates:
                            chosen_line = min(
                                candidates, key=lambda L: abs(float(L) - float(line))
                            )
                            if abs(float(chosen_line) - float(line)) > tol:
                                chosen_line = None
                    except Exception:
                        chosen_line = None
            if chosen_line is not None:
                bt = by_match.get(chosen_line) or {}
                op = bt.get("over_prob")
                up = bt.get("under_prob")
                if isinstance(op, (int, float)) or isinstance(up, (int, float)):
                    try:
                        s = (float(op) if isinstance(op, (int, float)) else 0.0) + (
                            float(up) if isinstance(up, (int, float)) else 0.0
                        )
                        over_p = (
                            (float(op) / s)
                            if s > 0
                            else float(op)
                            if isinstance(op, (int, float))
                            else None
                        )
                    except Exception:
                        over_p = None
                    market_totals = {
                        "line": float(chosen_line),
                        "over_implied": over_p,
                        "under_implied": (1.0 - over_p)
                        if isinstance(over_p, (int, float))
                        else None,
                        "overround": None,
                    }
                    market_source_str = "live_bovada"
        market_over_prob = None
        if market_totals and all(
            isinstance(market_totals.get(k), (int, float))
            for k in ("over_implied", "under_implied")
        ):
            # normalize to sum to 1 even if rounding noise exists
            ov = float(market_totals["over_implied"])
            un = float(market_totals["under_implied"])
            s = ov + un
            if s > 0:
                market_over_prob = ov / s
            else:
                market_over_prob = ov

        # Model totals probability
        model_over_prob = None
        if pred:
            # Try to use predictor-provided over_2_5_goals_probability if available
            p_over = pred.get("over_2_5_goals_probability")
            if isinstance(p_over, (int, float)):
                model_over_prob = float(p_over)
            else:
                # approximate from predicted total goals via the predictor helper
                try:
                    total_pred = float(pred.get("total_goals", 2.6))
                    # fallback: use predictor's internal method via a small shim
                    # Since _calculate_over_probability is private, approximate with a smooth tanh-based proxy
                    import math

                    z = total_pred - line
                    prob_under = 0.5 * (1.0 + math.tanh(-z))
                    model_over_prob = max(0.05, min(0.95, 1.0 - prob_under))
                except Exception:
                    model_over_prob = None
            # Apply temperature scaling? Not applicable directly to totals; skip calibration_service here

        model_pick = None
        market_pick = None
        edge_for_model_pick = None
        edge_recommendation = False
        if model_over_prob is not None:
            model_pick = "Over" if model_over_prob >= 0.5 else "Under"
        if market_over_prob is not None:
            market_pick = "Over" if market_over_prob >= 0.5 else "Under"
        if model_over_prob is not None and market_over_prob is not None and model_pick:
            if model_pick == "Over":
                edge_for_model_pick = round(model_over_prob - market_over_prob, 4)
            else:
                edge_for_model_pick = round(
                    (1.0 - model_over_prob) - (1.0 - market_over_prob), 4
                )
            edge_recommendation = edge_for_model_pick >= edge_threshold

        # Metrics (binary)
        if (
            actual_over is not None
            and model_over_prob is not None
            and market_over_prob is not None
        ):
            total_evaluable += 1
            import math

            mp = model_over_prob if actual_over == 1 else (1.0 - model_over_prob)
            kp = market_over_prob if actual_over == 1 else (1.0 - market_over_prob)
            mp = max(min(mp, 1 - eps), eps)
            kp = max(min(kp, 1 - eps), eps)
            model_log_losses.append(-math.log(mp))
            market_log_losses.append(-math.log(kp))
            model_briers.append((model_over_prob - actual_over) ** 2)
            market_briers.append((market_over_prob - actual_over) ** 2)
            if (model_pick == "Over" and actual_over == 1) or (
                model_pick == "Under" and actual_over == 0
            ):
                model_correct += 1
            if (market_pick == "Over" and actual_over == 1) or (
                market_pick == "Under" and actual_over == 0
            ):
                market_correct += 1

        rows.append(
            {
                "match_id": mid,
                "home_team": home_raw,
                "away_team": away_raw,
                "date": date or None,
                "line": line,
                "home_score": hs,
                "away_score": as_,
                "actual_total": actual_total,
                "actual_over": actual_over,
                "model_over_prob": round(model_over_prob, 4)
                if isinstance(model_over_prob, (int, float))
                else None,
                "market_over_prob": round(market_over_prob, 4)
                if isinstance(market_over_prob, (int, float))
                else None,
                "market_line": (
                    market_totals.get("line")
                    if isinstance(market_totals, dict)
                    else None
                ),
                # Flag if a nearest-line fallback was used (e.g., requested 2.5 -> used 2.75)
                "market_line_fallback": (
                    isinstance(market_totals, dict)
                    and market_totals.get("line") is not None
                    and isinstance(line, (int, float))
                    and float(market_totals.get("line")) != float(line)
                ),
                "market_line_delta": (
                    abs(float(market_totals.get("line")) - float(line))
                    if (
                        isinstance(market_totals, dict)
                        and market_totals.get("line") is not None
                        and isinstance(line, (int, float))
                    )
                    else None
                ),
                "model_pick": model_pick,
                "market_pick": market_pick,
                "edge_for_model_pick": edge_for_model_pick,
                "edge_recommendation": edge_recommendation,
                "market_source": market_source_str,
            }
        )

    agg = {
        "week": week,
        "line": line,
        "matches_considered": len(wk_matches),
        "rows_produced": len(rows),
        "evaluable_with_actual_and_probs": total_evaluable,
        "model_accuracy": (model_correct / total_evaluable)
        if total_evaluable
        else None,
        "market_accuracy": (market_correct / total_evaluable)
        if total_evaluable
        else None,
        "model_log_loss": (sum(model_log_losses) / len(model_log_losses))
        if model_log_losses
        else None,
        "market_log_loss": (sum(market_log_losses) / len(market_log_losses))
        if market_log_losses
        else None,
        "model_brier": (sum(model_briers) / len(model_briers))
        if model_briers
        else None,
        "market_brier": (sum(market_briers) / len(market_briers))
        if market_briers
        else None,
        "edge_threshold": edge_threshold,
        "recommended_edges": sum(1 for r in rows if r.get("edge_recommendation")),
    }
    if agg["model_log_loss"] is not None and agg["market_log_loss"] is not None:
        agg["model_log_loss_delta"] = round(
            agg["market_log_loss"] - agg["model_log_loss"], 4
        )
    if agg["model_brier"] is not None and agg["market_brier"] is not None:
        agg["model_brier_delta"] = round(agg["market_brier"] - agg["model_brier"], 4)
    out = {"aggregate": agg, "matches": rows}
    try:
        out["league"] = code
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# First Half Totals (Over/Under) vs Model (optional market)
# ---------------------------------------------------------------------------
def compare_week_first_half_totals(
    week: int,
    line: float = 1.0,
    edge_threshold: float = 0.05,
    league: Optional[str] = None,
    use_live: Optional[bool] = None,
) -> Dict[str, Any]:
    if not _prediction_cache:
        rebuild_predictions()
    code = normalize_league_code(league)
    service = get_league_service(code)
    matches = (
        service.get_all_matches()
        if hasattr(service, "get_all_matches")
        else EnhancedEPLService().get_all_matches()
    )
    weeks = game_week_service.organize_matches_by_week(matches)
    wk_matches = weeks.get(week, [])
    historic_idx = _load_historic_odds_index()

    # Live gating
    allow_live: bool = False
    try:
        if use_live is not None:
            allow_live = bool(use_live)
        else:
            allow_live = str(os.getenv("ODDS_COMPARE_USE_LIVE", "0")).strip() in (
                "1",
                "true",
                "True",
            )
    except Exception:
        allow_live = False

    # Backup live Bovada first-half totals lookup
    bovada_fh: Dict[str, Dict[float, Dict[str, Any]]] = {}
    if allow_live:
        try:
            fetcher = _get_bovada_fetcher(code)
            if fetcher is not None:
                bova = fetcher() if fetcher is not None else {"events": []}
                for ev in bova.get("events") or []:
                    h = normalize_team_name(ev.get("home_team")) or ev.get("home_team")
                    a = normalize_team_name(ev.get("away_team")) or ev.get("away_team")
                    key = f"{h}|{a}".lower()
                    for t in ev.get("first_half_totals") or []:
                        try:
                            l = (
                                float(t.get("line"))
                                if t.get("line") is not None
                                else None
                            )
                        except Exception:
                            l = None
                        if l is None:
                            continue
                        bovada_fh.setdefault(key, {})[l] = {
                            "over_prob": t.get("over_prob"),
                            "under_prob": t.get("under_prob"),
                            "market_line": l,
                            "source": "live_bovada",
                        }
        except Exception:
            pass

    rows: List[Dict[str, Any]] = []
    model_log_losses: List[float] = []
    model_briers: List[float] = []
    market_log_losses: List[float] = []
    market_briers: List[float] = []
    model_correct = 0
    market_correct = 0
    total_evaluable = 0
    eps = 1e-12

    for m in wk_matches:
        home_raw = m.get("home_team") or m.get("homeTeam")
        away_raw = m.get("away_team") or m.get("awayTeam")
        home = normalize_team_name(home_raw) or home_raw
        away = normalize_team_name(away_raw) or away_raw
        date = (m.get("utc_date") or m.get("date") or "").split("T")[0]
        mid = m.get("id") or m.get("match_id")
        key_pred = f"{mid}_{home}_{away}"
        pred = _prediction_cache.get(key_pred)
        if not pred:
            try:
                raw = advanced_ml_predictor.predict_match(home, away, league=code)
                if raw:
                    pred = _normalize_prediction(raw, home, away)
                    _prediction_cache[key_pred] = pred
            except Exception:
                pred = None

        hs = m.get("home_score") if "home_score" in m else m.get("homeScore")
        as_ = m.get("away_score") if "away_score" in m else m.get("awayScore")
        status_completed = m.get("status") in ["FINISHED", "COMPLETED"] or m.get(
            "is_completed"
        )
        actual_first_half = None
        if hs is not None and as_ is not None and status_completed:
            fh_field = m.get("first_half_goals")
            if isinstance(fh_field, (int, float)):
                actual_first_half = int(fh_field)
            else:
                total_goals = hs + as_
                actual_first_half = int(round(total_goals * 0.45))

        # Market FH totals not expected in historic data; placeholders
        market_over_prob = None
        market_source = None
        market_line = None
        if date:
            _ = historic_idx.get(f"{date}|{home.lower()}|{away.lower()}")
        else:
            _ = historic_idx.get(f"{home}|{away}".lower())
        # Bovada backup by exact line
        # Bovada backup by nearest line within tolerance
        try:
            key = f"{home.lower()}|{away.lower()}"
            by_match = bovada_fh.get(key) or {}
            try:
                import os

                tol = float(os.getenv("BOVADA_LINE_TOLERANCE", "0.25"))
            except Exception:
                tol = 0.25
            chosen_line = None
            if market_over_prob is None and by_match:
                if line in by_match:
                    chosen_line = line
                else:
                    try:
                        candidates = list(by_match.keys())
                        if candidates:
                            chosen_line = min(
                                candidates, key=lambda L: abs(float(L) - float(line))
                            )
                            if abs(float(chosen_line) - float(line)) > tol:
                                chosen_line = None
                    except Exception:
                        chosen_line = None
            if market_over_prob is None and (chosen_line is not None):
                bt = by_match.get(chosen_line) or {}
                op = bt.get("over_prob")
                up = bt.get("under_prob")
                if isinstance(op, (int, float)) or isinstance(up, (int, float)):
                    s = (float(op) if isinstance(op, (int, float)) else 0.0) + (
                        float(up) if isinstance(up, (int, float)) else 0.0
                    )
                    market_over_prob = (
                        (float(op) / s)
                        if s > 0
                        else (float(op) if isinstance(op, (int, float)) else None)
                    )
                    market_source = "live_bovada"
                    market_line = bt.get("market_line")
        except Exception:
            pass

        # Model FH probability from predicted mean
        model_over_prob = None
        if pred:
            try:
                mu = pred.get("first_half_goals")
                mu = float(mu) if mu is not None else None
                if mu is None:
                    tg = float(pred.get("total_goals", 2.6))
                    mu = tg * 0.45
                import math

                z = mu - line
                pu = 0.5 * (1.0 + math.tanh(-z))
                model_over_prob = max(0.05, min(0.95, 1.0 - pu))
            except Exception:
                model_over_prob = None

        model_pick = (
            "Over"
            if (model_over_prob is not None and model_over_prob >= 0.5)
            else ("Under" if model_over_prob is not None else None)
        )
        market_pick = (
            "Over"
            if (market_over_prob is not None and market_over_prob >= 0.5)
            else ("Under" if market_over_prob is not None else None)
        )

        # Compute edge for model pick when market available
        edge_for_model_pick = None
        edge_recommendation = False
        if (
            model_pick
            and isinstance(model_over_prob, (int, float))
            and isinstance(market_over_prob, (int, float))
        ):
            if model_pick == "Over":
                edge_for_model_pick = round(
                    float(model_over_prob) - float(market_over_prob), 4
                )
            else:  # Under
                edge_for_model_pick = round(
                    (1.0 - float(model_over_prob)) - (1.0 - float(market_over_prob)), 4
                )
            edge_recommendation = (edge_for_model_pick is not None) and (
                edge_for_model_pick >= edge_threshold
            )

        # Compute edge for model pick when market available
        edge_for_model_pick = None
        edge_recommendation = False
        if (
            model_pick
            and isinstance(model_over_prob, (int, float))
            and isinstance(market_over_prob, (int, float))
        ):
            if model_pick == "Over":
                edge_for_model_pick = round(
                    float(model_over_prob) - float(market_over_prob), 4
                )
            else:  # Under
                edge_for_model_pick = round(
                    (1.0 - float(model_over_prob)) - (1.0 - float(market_over_prob)), 4
                )
            edge_recommendation = (edge_for_model_pick is not None) and (
                edge_for_model_pick >= edge_threshold
            )

        if actual_first_half is not None and model_over_prob is not None:
            total_evaluable += 1
            actual_over = 1 if (actual_first_half > line) else 0
            import math

            mp = model_over_prob if actual_over == 1 else (1.0 - model_over_prob)
            mp = max(min(mp, 1 - eps), eps)
            model_log_losses.append(-math.log(mp))
            model_briers.append((model_over_prob - actual_over) ** 2)
            if (model_pick == "Over" and actual_over == 1) or (
                model_pick == "Under" and actual_over == 0
            ):
                model_correct += 1
            if market_over_prob is not None:
                kp = market_over_prob if actual_over == 1 else (1.0 - market_over_prob)
                kp = max(min(kp, 1 - eps), eps)
                market_log_losses.append(-math.log(kp))
                market_briers.append((market_over_prob - actual_over) ** 2)
                if (market_pick == "Over" and actual_over == 1) or (
                    market_pick == "Under" and actual_over == 0
                ):
                    market_correct += 1

        rows.append(
            {
                "match_id": mid,
                "home_team": home_raw,
                "away_team": away_raw,
                "date": date or None,
                "actual_first_half_goals": actual_first_half,
                "line": line,
                "market_line": market_line,
                "model_over_prob": round(model_over_prob, 4)
                if isinstance(model_over_prob, (int, float))
                else None,
                "market_over_prob": round(market_over_prob, 4)
                if isinstance(market_over_prob, (int, float))
                else None,
                "model_pick": model_pick,
                "market_pick": market_pick,
                "edge_for_model_pick": None
                if (
                    market_over_prob is None
                    or model_over_prob is None
                    or model_pick is None
                )
                else (
                    round(model_over_prob - market_over_prob, 4)
                    if model_pick == "Over"
                    else round((1.0 - model_over_prob) - (1.0 - market_over_prob), 4)
                ),
                "edge_recommendation": False
                if (market_over_prob is None)
                else (
                    (
                        round(model_over_prob - market_over_prob, 4)
                        if model_pick == "Over"
                        else round(
                            (1.0 - model_over_prob) - (1.0 - market_over_prob), 4
                        )
                    )
                    >= edge_threshold
                    if model_pick
                    else False
                ),
                "market_source": market_source,
                # Flag if a nearest-line fallback was used (requested line vs returned market_line)
                "market_line_fallback": (
                    isinstance(market_line, (int, float))
                    and isinstance(line, (int, float))
                    and float(market_line) != float(line)
                )
                or False,
                "market_line_delta": (
                    abs(float(market_line) - float(line))
                    if (
                        isinstance(market_line, (int, float))
                        and isinstance(line, (int, float))
                    )
                    else None
                ),
            }
        )

    agg = {
        "week": week,
        "line": line,
        "matches_considered": len(wk_matches),
        "rows_produced": len(rows),
        "evaluable_with_actual_and_probs": total_evaluable,
        "model_accuracy": (model_correct / total_evaluable)
        if total_evaluable
        else None,
        "market_accuracy": (market_correct / total_evaluable)
        if (total_evaluable and market_log_losses)
        else None,
        "model_log_loss": (sum(model_log_losses) / len(model_log_losses))
        if model_log_losses
        else None,
        "market_log_loss": (sum(market_log_losses) / len(market_log_losses))
        if market_log_losses
        else None,
        "model_brier": (sum(model_briers) / len(model_briers))
        if model_briers
        else None,
        "market_brier": (sum(market_briers) / len(market_briers))
        if market_briers
        else None,
        "edge_threshold": edge_threshold,
        "recommended_edges": sum(1 for r in rows if r.get("edge_recommendation")),
    }
    if agg["model_log_loss"] is not None and agg["market_log_loss"] is not None:
        agg["model_log_loss_delta"] = round(
            agg["market_log_loss"] - agg["model_log_loss"], 4
        )
    if agg["model_brier"] is not None and agg["market_brier"] is not None:
        agg["model_brier_delta"] = round(agg["market_brier"] - agg["model_brier"], 4)
    out = {"aggregate": agg, "matches": rows}
    try:
        out["league"] = code
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Second Half Totals (Over/Under) vs Model (optional market)
# ---------------------------------------------------------------------------
def compare_week_second_half_totals(
    week: int,
    line: float = 1.0,
    edge_threshold: float = 0.05,
    league: Optional[str] = None,
    use_live: Optional[bool] = None,
) -> Dict[str, Any]:
    if not _prediction_cache:
        rebuild_predictions()
    code = normalize_league_code(league)
    service = get_league_service(code)
    matches = (
        service.get_all_matches()
        if hasattr(service, "get_all_matches")
        else EnhancedEPLService().get_all_matches()
    )
    weeks = game_week_service.organize_matches_by_week(matches)
    wk_matches = weeks.get(week, [])
    historic_idx = _load_historic_odds_index()

    # Backup live Bovada second-half totals lookup (only if live enabled)
    allow_live: bool = False
    try:
        if use_live is not None:
            allow_live = bool(use_live)
        else:
            allow_live = str(os.getenv("ODDS_COMPARE_USE_LIVE", "0")).strip() in (
                "1",
                "true",
                "True",
            )
    except Exception:
        allow_live = False
    bovada_sh: Dict[str, Dict[float, Dict[str, Any]]] = {}
    if allow_live:
        try:
            fetcher = _get_bovada_fetcher(code)
            if fetcher is not None:
                bova = fetcher() if fetcher is not None else {"events": []}
                for ev in bova.get("events") or []:
                    h = normalize_team_name(ev.get("home_team")) or ev.get("home_team")
                    a = normalize_team_name(ev.get("away_team")) or ev.get("away_team")
                    key = f"{h}|{a}".lower()
                    for t in ev.get("second_half_totals") or []:
                        try:
                            l = (
                                float(t.get("line"))
                                if t.get("line") is not None
                                else None
                            )
                        except Exception:
                            l = None
                        if l is None:
                            continue
                        bovada_sh.setdefault(key, {})[l] = {
                            "over_prob": t.get("over_prob"),
                            "under_prob": t.get("under_prob"),
                            "market_line": l,
                            "source": "live_bovada",
                        }
        except Exception:
            pass

    rows: List[Dict[str, Any]] = []
    model_log_losses: List[float] = []
    model_briers: List[float] = []
    market_log_losses: List[float] = []
    market_briers: List[float] = []
    model_correct = 0
    market_correct = 0
    total_evaluable = 0
    eps = 1e-12

    for m in wk_matches:
        home_raw = m.get("home_team") or m.get("homeTeam")
        away_raw = m.get("away_team") or m.get("awayTeam")
        home = normalize_team_name(home_raw) or home_raw
        away = normalize_team_name(away_raw) or away_raw
        date = (m.get("utc_date") or m.get("date") or "").split("T")[0]
        mid = m.get("id") or m.get("match_id")
        key_pred = f"{mid}_{home}_{away}"
        pred = _prediction_cache.get(key_pred)
        if not pred:
            try:
                raw = advanced_ml_predictor.predict_match(home, away, league=code)
                if raw:
                    pred = _normalize_prediction(raw, home, away)
                    _prediction_cache[key_pred] = pred
            except Exception:
                pred = None

        hs = m.get("home_score") if "home_score" in m else m.get("homeScore")
        as_ = m.get("away_score") if "away_score" in m else m.get("awayScore")
        status_completed = m.get("status") in ["FINISHED", "COMPLETED"] or m.get(
            "is_completed"
        )
        actual_second_half = None
        if hs is not None and as_ is not None and status_completed:
            total_goals = hs + as_
            fh_field = m.get("first_half_goals")
            if isinstance(fh_field, (int, float)):
                actual_second_half = int(total_goals - int(fh_field))
            else:
                actual_second_half = int(round(total_goals * 0.55))

        market_over_prob = None
        market_source = None
        market_line = None
        if date:
            _ = historic_idx.get(f"{date}|{home.lower()}|{away.lower()}")
        else:
            _ = historic_idx.get(f"{home}|{away}".lower())
        # Bovada backup by nearest line within tolerance (only if live enabled)
        try:
            key = f"{home.lower()}|{away.lower()}"
            by_match = bovada_sh.get(key) or {}
            try:
                import os

                tol = float(os.getenv("BOVADA_LINE_TOLERANCE", "0.25"))
            except Exception:
                tol = 0.25
            chosen_line = None
            if allow_live and market_over_prob is None and by_match:
                if line in by_match:
                    chosen_line = line
                else:
                    try:
                        candidates = list(by_match.keys())
                        if candidates:
                            chosen_line = min(
                                candidates, key=lambda L: abs(float(L) - float(line))
                            )
                            if abs(float(chosen_line) - float(line)) > tol:
                                chosen_line = None
                    except Exception:
                        chosen_line = None
            if allow_live and market_over_prob is None and (chosen_line is not None):
                bt = by_match.get(chosen_line) or {}
                op = bt.get("over_prob")
                up = bt.get("under_prob")
                if isinstance(op, (int, float)) or isinstance(up, (int, float)):
                    s = (float(op) if isinstance(op, (int, float)) else 0.0) + (
                        float(up) if isinstance(up, (int, float)) else 0.0
                    )
                    market_over_prob = (
                        (float(op) / s)
                        if s > 0
                        else (float(op) if isinstance(op, (int, float)) else None)
                    )
                    market_source = "live_bovada"
                    market_line = bt.get("market_line")
        except Exception:
            pass

        model_over_prob = None
        if pred:
            try:
                tg = float(pred.get("total_goals", 2.6))
                fh = pred.get("first_half_goals")
                fh = float(fh) if fh is not None else tg * 0.45
                mu = max(0.0, tg - fh)
                import math

                z = mu - line
                pu = 0.5 * (1.0 + math.tanh(-z))
                model_over_prob = max(0.05, min(0.95, 1.0 - pu))
            except Exception:
                model_over_prob = None

        model_pick = (
            "Over"
            if (model_over_prob is not None and model_over_prob >= 0.5)
            else ("Under" if model_over_prob is not None else None)
        )
        market_pick = (
            "Over"
            if (market_over_prob is not None and market_over_prob >= 0.5)
            else ("Under" if market_over_prob is not None else None)
        )

        if actual_second_half is not None and model_over_prob is not None:
            total_evaluable += 1
            actual_over = 1 if (actual_second_half > line) else 0
            import math

            mp = model_over_prob if actual_over == 1 else (1.0 - model_over_prob)
            mp = max(min(mp, 1 - eps), eps)
            model_log_losses.append(-math.log(mp))
            model_briers.append((model_over_prob - actual_over) ** 2)
            if (model_pick == "Over" and actual_over == 1) or (
                model_pick == "Under" and actual_over == 0
            ):
                model_correct += 1
            if market_over_prob is not None:
                kp = market_over_prob if actual_over == 1 else (1.0 - market_over_prob)
                kp = max(min(kp, 1 - eps), eps)
                market_log_losses.append(-math.log(kp))
                market_briers.append((market_over_prob - actual_over) ** 2)
                if (market_pick == "Over" and actual_over == 1) or (
                    market_pick == "Under" and actual_over == 0
                ):
                    market_correct += 1

        rows.append(
            {
                "match_id": mid,
                "home_team": home_raw,
                "away_team": away_raw,
                "date": date or None,
                "actual_second_half_goals": actual_second_half,
                "line": line,
                "market_line": market_line,
                "model_over_prob": round(model_over_prob, 4)
                if isinstance(model_over_prob, (int, float))
                else None,
                "market_over_prob": round(market_over_prob, 4)
                if isinstance(market_over_prob, (int, float))
                else None,
                "model_pick": model_pick,
                "market_pick": market_pick,
                "edge_for_model_pick": None,
                "edge_recommendation": False,
                "market_source": market_source,
                # Flag if a nearest-line fallback was used (requested line vs returned market_line)
                "market_line_fallback": (
                    isinstance(market_line, (int, float))
                    and isinstance(line, (int, float))
                    and float(market_line) != float(line)
                )
                or False,
                "market_line_delta": (
                    abs(float(market_line) - float(line))
                    if (
                        isinstance(market_line, (int, float))
                        and isinstance(line, (int, float))
                    )
                    else None
                ),
            }
        )

    agg = {
        "week": week,
        "line": line,
        "matches_considered": len(wk_matches),
        "rows_produced": len(rows),
        "evaluable_with_actual_and_probs": total_evaluable,
        "model_accuracy": (model_correct / total_evaluable)
        if total_evaluable
        else None,
        "market_accuracy": (market_correct / total_evaluable)
        if (total_evaluable and market_log_losses)
        else None,
        "model_log_loss": (sum(model_log_losses) / len(model_log_losses))
        if model_log_losses
        else None,
        "market_log_loss": (sum(market_log_losses) / len(market_log_losses))
        if market_log_losses
        else None,
        "model_brier": (sum(model_briers) / len(model_briers))
        if model_briers
        else None,
        "market_brier": (sum(market_briers) / len(market_briers))
        if market_briers
        else None,
        "edge_threshold": edge_threshold,
        "recommended_edges": sum(1 for r in rows if r.get("edge_recommendation")),
    }
    if agg["model_log_loss"] is not None and agg["market_log_loss"] is not None:
        agg["model_log_loss_delta"] = round(
            agg["market_log_loss"] - agg["model_log_loss"], 4
        )
    if agg["model_brier"] is not None and agg["market_brier"] is not None:
        agg["model_brier_delta"] = round(agg["market_brier"] - agg["model_brier"], 4)
    out = {"aggregate": agg, "matches": rows}
    try:
        out["league"] = code
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Corners Totals (Over/Under) vs Model (optional market)
# ---------------------------------------------------------------------------
def compare_week_corners_totals(
    week: int,
    line: float = 9.5,
    edge_threshold: float = 0.05,
    league: Optional[str] = None,
) -> Dict[str, Any]:
    if not _prediction_cache:
        rebuild_predictions()
    code = normalize_league_code(league)
    service = get_league_service(code)
    matches = (
        service.get_all_matches()
        if hasattr(service, "get_all_matches")
        else EnhancedEPLService().get_all_matches()
    )
    weeks = game_week_service.organize_matches_by_week(matches)
    wk_matches = weeks.get(week, [])
    historic_idx = _load_historic_odds_index()

    rows: List[Dict[str, Any]] = []
    model_log_losses: List[float] = []
    model_briers: List[float] = []
    market_log_losses: List[float] = []
    market_briers: List[float] = []
    model_correct = 0
    market_correct = 0
    total_evaluable = 0
    eps = 1e-12

    for m in wk_matches:
        home_raw = m.get("home_team") or m.get("homeTeam")
        away_raw = m.get("away_team") or m.get("awayTeam")
        home = normalize_team_name(home_raw) or home_raw
        away = normalize_team_name(away_raw) or away_raw
        date = (m.get("utc_date") or m.get("date") or "").split("T")[0]
        mid = m.get("id") or m.get("match_id")
        key_pred = f"{mid}_{home}_{away}"
        pred = _prediction_cache.get(key_pred)
        if not pred:
            try:
                raw = advanced_ml_predictor.predict_match(home, away, league=code)
                if raw:
                    pred = _normalize_prediction(raw, home, away)
                    _prediction_cache[key_pred] = pred
            except Exception:
                pred = None

        status_completed = m.get("status") in ["FINISHED", "COMPLETED"] or m.get(
            "is_completed"
        )
        actual_corners = None
        if status_completed:
            if "total_corners" in m and isinstance(
                m.get("total_corners"), (int, float)
            ):
                actual_corners = int(m.get("total_corners"))
            else:
                hc = m.get("home_corners")
                ac = m.get("away_corners")
                if isinstance(hc, (int, float)) and isinstance(ac, (int, float)):
                    actual_corners = int(hc + ac)
                else:
                    # Try lookup from corners actuals store
                    c_actual = corners_actuals_store.lookup(date, home, away)
                    if c_actual and isinstance(
                        c_actual.get("total_corners"), (int, float)
                    ):
                        actual_corners = int(c_actual["total_corners"])

        market_over_prob = None
        market_source = None
        market_line = None
        try:
            mkt = corners_market_store.lookup_total(date, home, away, line)
            if mkt:
                market_over_prob, market_source, market_line = mkt
        except Exception:
            pass

        model_over_prob = None
        if pred:
            try:
                mu = pred.get("total_corners")
                mu = float(mu) if mu is not None else None
                if mu is None:
                    tg = float(pred.get("total_goals", 2.6))
                    mu = max(6.0, tg * 4.0)
                # Use calibrated mapping and optional market blending
                model_over_prob = corners_calibration_service.predict_over_prob(
                    mu, line, week=week, market_over_prob=market_over_prob
                )
                # keep within [0.05, 0.95] for stability in logs
                model_over_prob = max(0.05, min(0.95, float(model_over_prob)))
            except Exception:
                model_over_prob = None

        model_pick = (
            "Over"
            if (model_over_prob is not None and model_over_prob >= 0.5)
            else ("Under" if model_over_prob is not None else None)
        )
        market_pick = (
            "Over"
            if (market_over_prob is not None and market_over_prob >= 0.5)
            else ("Under" if market_over_prob is not None else None)
        )

        # Compute edge fields for output when market prob is present
        edge_for_model_pick = None
        edge_recommendation = False
        if (
            model_pick
            and isinstance(model_over_prob, (int, float))
            and isinstance(market_over_prob, (int, float))
        ):
            if model_pick == "Over":
                edge_for_model_pick = round(
                    float(model_over_prob) - float(market_over_prob), 4
                )
            else:
                edge_for_model_pick = round(
                    (1.0 - float(model_over_prob)) - (1.0 - float(market_over_prob)), 4
                )
            edge_recommendation = (edge_for_model_pick is not None) and (
                edge_for_model_pick >= edge_threshold
            )

        if actual_corners is not None and model_over_prob is not None:
            total_evaluable += 1
            actual_over = 1 if (actual_corners > line) else 0
            import math

            mp = model_over_prob if actual_over == 1 else (1.0 - model_over_prob)
            mp = max(min(mp, 1 - eps), eps)
            model_log_losses.append(-math.log(mp))
            model_briers.append((model_over_prob - actual_over) ** 2)
            if (model_pick == "Over" and actual_over == 1) or (
                model_pick == "Under" and actual_over == 0
            ):
                model_correct += 1
            if market_over_prob is not None:
                kp = market_over_prob if actual_over == 1 else (1.0 - market_over_prob)
                kp = max(min(kp, 1 - eps), eps)
                market_log_losses.append(-math.log(kp))
                market_briers.append((market_over_prob - actual_over) ** 2)
                if (market_pick == "Over" and actual_over == 1) or (
                    market_pick == "Under" and actual_over == 0
                ):
                    market_correct += 1

        rows.append(
            {
                "match_id": mid,
                "home_team": home_raw,
                "away_team": away_raw,
                "date": date or None,
                "actual_total_corners": actual_corners,
                "line": line,
                "market_line": market_line,
                "model_over_prob": round(model_over_prob, 4)
                if isinstance(model_over_prob, (int, float))
                else None,
                "market_over_prob": round(market_over_prob, 4)
                if isinstance(market_over_prob, (int, float))
                else None,
                "model_pick": model_pick,
                "market_pick": market_pick,
                "edge_for_model_pick": edge_for_model_pick,
                "edge_recommendation": edge_recommendation,
                "market_source": market_source,
                # Flag if a nearest-line fallback was used by the market provider
                "market_line_fallback": (
                    isinstance(market_line, (int, float))
                    and isinstance(line, (int, float))
                    and float(market_line) != float(line)
                )
                or False,
                "market_line_delta": (
                    abs(float(market_line) - float(line))
                    if (
                        isinstance(market_line, (int, float))
                        and isinstance(line, (int, float))
                    )
                    else None
                ),
                "calibration": corners_calibration_service.status()
                if rows == []
                else None,
            }
        )

    agg = {
        "week": week,
        "line": line,
        "matches_considered": len(wk_matches),
        "rows_produced": len(rows),
        "evaluable_with_actual_and_probs": total_evaluable,
        "model_accuracy": (model_correct / total_evaluable)
        if total_evaluable
        else None,
        "market_accuracy": (market_correct / total_evaluable)
        if (total_evaluable and market_log_losses)
        else None,
        "model_log_loss": (sum(model_log_losses) / len(model_log_losses))
        if model_log_losses
        else None,
        "market_log_loss": (sum(market_log_losses) / len(market_log_losses))
        if market_log_losses
        else None,
        "model_brier": (sum(model_briers) / len(model_briers))
        if model_briers
        else None,
        "market_brier": (sum(market_briers) / len(market_briers))
        if market_briers
        else None,
        "edge_threshold": edge_threshold,
        "recommended_edges": sum(1 for r in rows if r.get("edge_recommendation")),
    }
    if agg["model_log_loss"] is not None and agg["market_log_loss"] is not None:
        agg["model_log_loss_delta"] = round(
            agg["market_log_loss"] - agg["model_log_loss"], 4
        )
    if agg["model_brier"] is not None and agg["market_brier"] is not None:
        agg["model_brier_delta"] = round(agg["market_brier"] - agg["model_brier"], 4)
    out = {"aggregate": agg, "matches": rows}
    try:
        out["league"] = code
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Team Totals: Goals (Over/Under) vs Model (optional market)
# ---------------------------------------------------------------------------
def compare_week_team_goals_totals(
    week: int,
    side: str = "home",
    line: float = 1.5,
    edge_threshold: float = 0.05,
    league: Optional[str] = None,
) -> Dict[str, Any]:
    s = side.lower()
    assert s in ("home", "away"), "side must be 'home' or 'away'"
    if not _prediction_cache:
        rebuild_predictions()
    code = normalize_league_code(league)
    service = get_league_service(code)
    matches = (
        service.get_all_matches()
        if hasattr(service, "get_all_matches")
        else EnhancedEPLService().get_all_matches()
    )
    weeks = game_week_service.organize_matches_by_week(matches)
    wk_matches = weeks.get(week, [])
    historic_idx = _load_historic_odds_index()

    # Backup live Bovada team totals lookup
    bovada_tt: Dict[str, Dict[str, Dict[float, Dict[str, Any]]]] = {}
    try:
        fetcher = _get_bovada_fetcher(code)
        if fetcher is not None:
            bova = fetcher() if fetcher is not None else {"events": []}
            for ev in bova.get("events") or []:
                h = normalize_team_name(ev.get("home_team")) or ev.get("home_team")
                a = normalize_team_name(ev.get("away_team")) or ev.get("away_team")
                key = f"{h}|{a}".lower()
                for t in ev.get("team_totals") or []:
                    try:
                        l = float(t.get("line")) if t.get("line") is not None else None
                    except Exception:
                        l = None
                    if l is None:
                        continue
                    side_key = (t.get("side") or "").lower()
                    if side_key not in ("home", "away"):
                        continue
                    bovada_tt.setdefault(key, {}).setdefault(side_key, {})[l] = {
                        "over_prob": t.get("over_prob"),
                        "under_prob": t.get("under_prob"),
                        "market_line": l,
                        "source": "live_bovada",
                    }
    except Exception:
        pass

    rows: List[Dict[str, Any]] = []
    model_log_losses: List[float] = []
    model_briers: List[float] = []
    market_log_losses: List[float] = []
    market_briers: List[float] = []
    model_correct = 0
    market_correct = 0
    total_evaluable = 0
    eps = 1e-12

    for m in wk_matches:
        home_raw = m.get("home_team") or m.get("homeTeam")
        away_raw = m.get("away_team") or m.get("awayTeam")
        home = normalize_team_name(home_raw) or home_raw
        away = normalize_team_name(away_raw) or away_raw
        date = (m.get("utc_date") or m.get("date") or "").split("T")[0]
        mid = m.get("id") or m.get("match_id")
        key_pred = f"{mid}_{home}_{away}"
        pred = _prediction_cache.get(key_pred)
        if not pred:
            try:
                raw = advanced_ml_predictor.predict_match(home, away, league=code)
                if raw:
                    pred = _normalize_prediction(raw, home, away)
                    _prediction_cache[key_pred] = pred
            except Exception:
                pred = None

        hs = m.get("home_score") if "home_score" in m else m.get("homeScore")
        as_ = m.get("away_score") if "away_score" in m else m.get("awayScore")
        status_completed = m.get("status") in ["FINISHED", "COMPLETED"] or m.get(
            "is_completed"
        )
        actual_team_goals = None
        if hs is not None and as_ is not None and status_completed:
            actual_team_goals = hs if s == "home" else as_

        # Markets for team totals likely absent; placeholders
        market_over_prob = None
        market_source = None
        market_line = None
        if date:
            _ = historic_idx.get(f"{date}|{home.lower()}|{away.lower()}")
        else:
            _ = historic_idx.get(f"{home}|{away}".lower())
        # Bovada backup by exact line and side
        # Bovada backup by nearest line within tolerance and side
        try:
            key = f"{home.lower()}|{away.lower()}"
            by_match = bovada_tt.get(key) or {}
            side_dict = by_match.get(s) or {}
            try:
                import os

                tol = float(os.getenv("BOVADA_LINE_TOLERANCE", "0.25"))
            except Exception:
                tol = 0.25
            chosen_line = None
            if market_over_prob is None and side_dict:
                if line in side_dict:
                    chosen_line = line
                else:
                    try:
                        candidates = list(side_dict.keys())
                        if candidates:
                            chosen_line = min(
                                candidates, key=lambda L: abs(float(L) - float(line))
                            )
                            if abs(float(chosen_line) - float(line)) > tol:
                                chosen_line = None
                    except Exception:
                        chosen_line = None
            if market_over_prob is None and (chosen_line is not None):
                bt = side_dict.get(chosen_line) or {}
                op = bt.get("over_prob")
                up = bt.get("under_prob")
                if isinstance(op, (int, float)) or isinstance(up, (int, float)):
                    ssum = (float(op) if isinstance(op, (int, float)) else 0.0) + (
                        float(up) if isinstance(up, (int, float)) else 0.0
                    )
                    market_over_prob = (
                        (float(op) / ssum)
                        if ssum > 0
                        else (float(op) if isinstance(op, (int, float)) else None)
                    )
                    market_source = "live_bovada"
                    market_line = bt.get("market_line")
        except Exception:
            pass

        # Model probability from predicted mean goals for the chosen side
        model_over_prob = None
        if pred:
            try:
                mu = float(pred.get("home_goals" if s == "home" else "away_goals"))
                import math

                z = mu - line
                pu = 0.5 * (1.0 + math.tanh(-z))
                model_over_prob = max(0.05, min(0.95, 1.0 - pu))
            except Exception:
                model_over_prob = None

        model_pick = (
            "Over"
            if (model_over_prob is not None and model_over_prob >= 0.5)
            else ("Under" if model_over_prob is not None else None)
        )
        market_pick = (
            "Over"
            if (market_over_prob is not None and market_over_prob >= 0.5)
            else ("Under" if market_over_prob is not None else None)
        )

        if actual_team_goals is not None and model_over_prob is not None:
            total_evaluable += 1
            actual_over = 1 if (actual_team_goals > line) else 0
            import math

            mp = model_over_prob if actual_over == 1 else (1.0 - model_over_prob)
            mp = max(min(mp, 1 - eps), eps)
            model_log_losses.append(-math.log(mp))
            model_briers.append((model_over_prob - actual_over) ** 2)
            if (model_pick == "Over" and actual_over == 1) or (
                model_pick == "Under" and actual_over == 0
            ):
                model_correct += 1
            if market_over_prob is not None:
                kp = market_over_prob if actual_over == 1 else (1.0 - market_over_prob)
                kp = max(min(kp, 1 - eps), eps)
                market_log_losses.append(-math.log(kp))
                market_briers.append((market_over_prob - actual_over) ** 2)
                if (market_pick == "Over" and actual_over == 1) or (
                    market_pick == "Under" and actual_over == 0
                ):
                    market_correct += 1

        rows.append(
            {
                "match_id": mid,
                "home_team": home_raw,
                "away_team": away_raw,
                "date": date or None,
                "side": s,
                "actual_team_goals": actual_team_goals,
                # For UI reconciliation of team goals O/U
                "actual_over": (
                    1
                    if (
                        isinstance(actual_team_goals, (int, float))
                        and isinstance(line, (int, float))
                        and float(actual_team_goals) > float(line)
                    )
                    else (
                        0
                        if (
                            isinstance(actual_team_goals, (int, float))
                            and isinstance(line, (int, float))
                        )
                        else None
                    )
                ),
                "line": line,
                "market_line": market_line,
                "model_over_prob": round(model_over_prob, 4)
                if isinstance(model_over_prob, (int, float))
                else None,
                "market_over_prob": round(market_over_prob, 4)
                if isinstance(market_over_prob, (int, float))
                else None,
                "model_pick": model_pick,
                "market_pick": market_pick,
                "edge_for_model_pick": None,
                "edge_recommendation": False,
                "market_source": market_source,
                # Flag if a nearest-line fallback was used (requested line vs returned market_line)
                "market_line_fallback": (
                    isinstance(market_line, (int, float))
                    and isinstance(line, (int, float))
                    and float(market_line) != float(line)
                )
                or False,
                "market_line_delta": (
                    abs(float(market_line) - float(line))
                    if (
                        isinstance(market_line, (int, float))
                        and isinstance(line, (int, float))
                    )
                    else None
                ),
            }
        )

    agg = {
        "week": week,
        "side": s,
        "line": line,
        "matches_considered": len(wk_matches),
        "rows_produced": len(rows),
        "evaluable_with_actual_and_probs": total_evaluable,
        "model_accuracy": (model_correct / total_evaluable)
        if total_evaluable
        else None,
        "market_accuracy": (market_correct / total_evaluable)
        if (total_evaluable and market_log_losses)
        else None,
        "model_log_loss": (sum(model_log_losses) / len(model_log_losses))
        if model_log_losses
        else None,
        "market_log_loss": (sum(market_log_losses) / len(market_log_losses))
        if market_log_losses
        else None,
        "model_brier": (sum(model_briers) / len(model_briers))
        if model_briers
        else None,
        "market_brier": (sum(market_briers) / len(market_briers))
        if market_briers
        else None,
        "edge_threshold": edge_threshold,
        "recommended_edges": sum(1 for r in rows if r.get("edge_recommendation")),
    }
    if agg["model_log_loss"] is not None and agg["market_log_loss"] is not None:
        agg["model_log_loss_delta"] = round(
            agg["market_log_loss"] - agg["model_log_loss"], 4
        )
    if agg["model_brier"] is not None and agg["market_brier"] is not None:
        agg["model_brier_delta"] = round(agg["market_brier"] - agg["model_brier"], 4)
    out = {"aggregate": agg, "matches": rows}
    try:
        out["league"] = code
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Team Totals: Corners (Over/Under) vs Model (optional market)
# ---------------------------------------------------------------------------
def compare_week_team_corners_totals(
    week: int,
    side: str = "home",
    line: float = 4.5,
    edge_threshold: float = 0.05,
    league: Optional[str] = None,
    use_live: Optional[bool] = None,
) -> Dict[str, Any]:
    s = side.lower()
    assert s in ("home", "away"), "side must be 'home' or 'away'"
    if not _prediction_cache:
        rebuild_predictions()
    code = normalize_league_code(league)
    service = get_league_service(code)
    matches = (
        service.get_all_matches()
        if hasattr(service, "get_all_matches")
        else EnhancedEPLService().get_all_matches()
    )
    weeks = game_week_service.organize_matches_by_week(matches)
    wk_matches = weeks.get(week, [])
    historic_idx = _load_historic_odds_index()

    # Backup live Bovada team-corners lookup by side and line (with tolerance)
    bovada_tc: Dict[str, Dict[str, Dict[float, Dict[str, Any]]]] = {}
    try:
        fetcher = _get_bovada_fetcher(code)
        if fetcher is not None:
            bova = fetcher() if fetcher is not None else {"events": []}
            for ev in bova.get("events") or []:
                h = normalize_team_name(ev.get("home_team")) or ev.get("home_team")
                a = normalize_team_name(ev.get("away_team")) or ev.get("away_team")
                key = f"{h}|{a}".lower()
                for t in ev.get("team_corners") or []:
                    try:
                        l = float(t.get("line")) if t.get("line") is not None else None
                    except Exception:
                        l = None
                    if l is None:
                        continue
                    side_key = (t.get("side") or "").lower()
                    if side_key not in ("home", "away"):
                        continue
                    bovada_tc.setdefault(key, {}).setdefault(side_key, {})[l] = {
                        "over_prob": t.get("over_prob"),
                        "under_prob": t.get("under_prob"),
                        "market_line": l,
                        "source": "live_bovada",
                    }
    except Exception:
        pass

    rows: List[Dict[str, Any]] = []
    model_log_losses: List[float] = []
    model_briers: List[float] = []
    market_log_losses: List[float] = []
    market_briers: List[float] = []
    model_correct = 0
    market_correct = 0
    total_evaluable = 0
    eps = 1e-12

    for m in wk_matches:
        home_raw = m.get("home_team") or m.get("homeTeam")
        away_raw = m.get("away_team") or m.get("awayTeam")
        home = normalize_team_name(home_raw) or home_raw
        away = normalize_team_name(away_raw) or away_raw
        date = (m.get("utc_date") or m.get("date") or "").split("T")[0]
        mid = m.get("id") or m.get("match_id")
        key_pred = f"{mid}_{home}_{away}"
        pred = _prediction_cache.get(key_pred)
        if not pred:
            try:
                raw = advanced_ml_predictor.predict_match(home, away)
                if raw:
                    pred = _normalize_prediction(raw, home, away)
                    _prediction_cache[key_pred] = pred
            except Exception:
                pred = None

        status_completed = m.get("status") in ["FINISHED", "COMPLETED"] or m.get(
            "is_completed"
        )
        actual_team_corners = None
        if status_completed:
            kc = m.get("home_corners" if s == "home" else "away_corners")
            if isinstance(kc, (int, float)):
                actual_team_corners = int(kc)
            else:
                # Try lookup from corners actuals store
                c_actual = corners_actuals_store.lookup(date, home, away)
                if c_actual:
                    v = c_actual.get("home_corners" if s == "home" else "away_corners")
                    if isinstance(v, (int, float)):
                        actual_team_corners = int(v)

        market_over_prob = None
        market_source = None
        market_line = None
        try:
            mkt = corners_market_store.lookup_team(date, home, away, s, line)
            if mkt:
                market_over_prob, market_source, market_line = mkt
        except Exception:
            pass
        # Bovada backup by nearest line and side if no stored market
        if market_over_prob is None:
            try:
                key = f"{home.lower()}|{away.lower()}"
                by_match = bovada_tc.get(key) or {}
                side_dict = by_match.get(s) or {}
                try:
                    import os

                    tol = float(os.getenv("BOVADA_LINE_TOLERANCE", "0.25"))
                except Exception:
                    tol = 0.25
                chosen_line = None
                if side_dict:
                    if line in side_dict:
                        chosen_line = line
                    else:
                        try:
                            candidates = list(side_dict.keys())
                            if candidates:
                                chosen_line = min(
                                    candidates,
                                    key=lambda L: abs(float(L) - float(line)),
                                )
                                if abs(float(chosen_line) - float(line)) > tol:
                                    chosen_line = None
                        except Exception:
                            chosen_line = None
                if chosen_line is not None:
                    bt = side_dict.get(chosen_line) or {}
                    op = bt.get("over_prob")
                    up = bt.get("under_prob")
                    if isinstance(op, (int, float)) or isinstance(up, (int, float)):
                        ssum = (float(op) if isinstance(op, (int, float)) else 0.0) + (
                            float(up) if isinstance(up, (int, float)) else 0.0
                        )
                        market_over_prob = (
                            (float(op) / ssum)
                            if ssum > 0
                            else (float(op) if isinstance(op, (int, float)) else None)
                        )
                        market_source = "live_bovada"
                        market_line = bt.get("market_line")
            except Exception:
                pass

        model_over_prob = None
        if pred:
            try:
                mu = pred.get("home_corners" if s == "home" else "away_corners")
                mu = float(mu) if mu is not None else None
                if mu is None:
                    # heuristic from total corners
                    tc = float(pred.get("total_corners", 10.0))
                    mu = tc * (0.52 if s == "home" else 0.48)
                import math

                z = mu - line
                pu = 0.5 * (1.0 + math.tanh(-z))
                model_over_prob = max(0.05, min(0.95, 1.0 - pu))
            except Exception:
                model_over_prob = None

        model_pick = (
            "Over"
            if (model_over_prob is not None and model_over_prob >= 0.5)
            else ("Under" if model_over_prob is not None else None)
        )
        market_pick = (
            "Over"
            if (market_over_prob is not None and market_over_prob >= 0.5)
            else ("Under" if market_over_prob is not None else None)
        )

        # Compute edge fields for output when market prob is present
        edge_for_model_pick = None
        edge_recommendation = False
        if (
            model_pick
            and isinstance(model_over_prob, (int, float))
            and isinstance(market_over_prob, (int, float))
        ):
            if model_pick == "Over":
                edge_for_model_pick = round(
                    float(model_over_prob) - float(market_over_prob), 4
                )
            else:
                edge_for_model_pick = round(
                    (1.0 - float(model_over_prob)) - (1.0 - float(market_over_prob)), 4
                )
            edge_recommendation = (edge_for_model_pick is not None) and (
                edge_for_model_pick >= edge_threshold
            )

        if actual_team_corners is not None and model_over_prob is not None:
            total_evaluable += 1
            actual_over = 1 if (actual_team_corners > line) else 0
            import math

            mp = model_over_prob if actual_over == 1 else (1.0 - model_over_prob)
            mp = max(min(mp, 1 - eps), eps)
            model_log_losses.append(-math.log(mp))
            model_briers.append((model_over_prob - actual_over) ** 2)
            if (model_pick == "Over" and actual_over == 1) or (
                model_pick == "Under" and actual_over == 0
            ):
                model_correct += 1
            if market_over_prob is not None:
                kp = market_over_prob if actual_over == 1 else (1.0 - market_over_prob)
                kp = max(min(kp, 1 - eps), eps)
                market_log_losses.append(-math.log(kp))
                market_briers.append((market_over_prob - actual_over) ** 2)
                if (market_pick == "Over" and actual_over == 1) or (
                    market_pick == "Under" and actual_over == 0
                ):
                    market_correct += 1

        rows.append(
            {
                "match_id": mid,
                "home_team": home_raw,
                "away_team": away_raw,
                "date": date or None,
                "side": s,
                "actual_team_corners": actual_team_corners,
                # For UI reconciliation of team corners O/U
                "actual_over": (
                    1
                    if (
                        isinstance(actual_team_corners, (int, float))
                        and isinstance(line, (int, float))
                        and float(actual_team_corners) > float(line)
                    )
                    else (
                        0
                        if (
                            isinstance(actual_team_corners, (int, float))
                            and isinstance(line, (int, float))
                        )
                        else None
                    )
                ),
                "line": line,
                "market_line": market_line,
                "model_over_prob": round(model_over_prob, 4)
                if isinstance(model_over_prob, (int, float))
                else None,
                "market_over_prob": round(market_over_prob, 4)
                if isinstance(market_over_prob, (int, float))
                else None,
                "model_pick": model_pick,
                "market_pick": market_pick,
                "edge_for_model_pick": edge_for_model_pick,
                "edge_recommendation": edge_recommendation,
                "market_source": market_source,
            }
        )

    agg = {
        "week": week,
        "side": s,
        "line": line,
        "matches_considered": len(wk_matches),
        "rows_produced": len(rows),
        "evaluable_with_actual_and_probs": total_evaluable,
        "model_accuracy": (model_correct / total_evaluable)
        if total_evaluable
        else None,
        "market_accuracy": (market_correct / total_evaluable)
        if (total_evaluable and market_log_losses)
        else None,
        "model_log_loss": (sum(model_log_losses) / len(model_log_losses))
        if model_log_losses
        else None,
        "market_log_loss": (sum(market_log_losses) / len(market_log_losses))
        if market_log_losses
        else None,
        "model_brier": (sum(model_briers) / len(model_briers))
        if model_briers
        else None,
        "market_brier": (sum(market_briers) / len(market_briers))
        if market_briers
        else None,
        "edge_threshold": edge_threshold,
        "recommended_edges": sum(1 for r in rows if r.get("edge_recommendation")),
    }
    if agg["model_log_loss"] is not None and agg["market_log_loss"] is not None:
        agg["model_log_loss_delta"] = round(
            agg["market_log_loss"] - agg["model_log_loss"], 4
        )
    if agg["model_brier"] is not None and agg["market_brier"] is not None:
        agg["model_brier_delta"] = round(agg["market_brier"] - agg["model_brier"], 4)
    out = {"aggregate": agg, "matches": rows}
    try:
        out["league"] = code
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Week Double Chance (1X, 12, X2) vs Model vs Market
# ---------------------------------------------------------------------------
def compare_week_double_chance(
    week: int, market: str = "1X", edge_threshold: float = 0.05
) -> Dict[str, Any]:
    """Compare Double Chance market vs model vs actual for a week.
    market in {'1X','12','X2'}.
    Model and market DC probabilities are derived from 1X2 probabilities.
    """
    mkt = market.upper()
    assert mkt in ("1X", "12", "X2"), "market must be '1X', '12', or 'X2'"
    if not _prediction_cache:
        rebuild_predictions()
    service = EnhancedEPLService()
    matches = service.get_all_matches()
    weeks = game_week_service.organize_matches_by_week(matches)
    wk_matches = weeks.get(week, [])
    historic_idx = _load_historic_odds_index()
    rows: List[Dict[str, Any]] = []
    model_log_losses: List[float] = []
    market_log_losses: List[float] = []
    model_briers: List[float] = []
    market_briers: List[float] = []
    model_correct = 0
    market_correct = 0
    total_evaluable = 0
    eps = 1e-12

    def dc_prob_from_probs(probs: Dict[str, float]) -> float:
        if mkt == "1X":
            return probs.get("H", 0.0) + probs.get("D", 0.0)
        if mkt == "12":
            return probs.get("H", 0.0) + probs.get("A", 0.0)
        return probs.get("D", 0.0) + probs.get("A", 0.0)  # X2

    for m in wk_matches:
        home_raw = m.get("home_team") or m.get("homeTeam")
        away_raw = m.get("away_team") or m.get("awayTeam")
        home = normalize_team_name(home_raw) or home_raw
        away = normalize_team_name(away_raw) or away_raw
        date = (m.get("utc_date") or m.get("date") or "").split("T")[0]
        mid = m.get("id") or m.get("match_id")
        key_pred = f"{mid}_{home}_{away}"
        pred = _prediction_cache.get(key_pred)
        if not pred:
            try:
                raw = advanced_ml_predictor.predict_match(home, away)
                if raw:
                    pred = _normalize_prediction(raw, home, away)
                    _prediction_cache[key_pred] = pred
            except Exception:
                pred = None
        hs = m.get("home_score") if "home_score" in m else m.get("homeScore")
        as_ = m.get("away_score") if "away_score" in m else m.get("awayScore")
        status_completed = m.get("status") in ["FINISHED", "COMPLETED"] or m.get(
            "is_completed"
        )
        actual_result = None
        if hs is not None and as_ is not None and status_completed:
            if hs > as_:
                actual_result = "H"
            elif as_ > hs:
                actual_result = "A"
            else:
                actual_result = "D"
        # Market probs via historic
        market_rec = None
        if date:
            market_rec = historic_idx.get(f"{date}|{home.lower()}|{away.lower()}")
        if not market_rec:
            market_rec = historic_idx.get(f"{home}|{away}".lower())
        market_probs = None
        market_source = None
        if market_rec:
            cons = (
                market_rec.get("preferred_implied")
                or market_rec.get("consensus_implied")
                or {}
            )
            h = cons.get("H")
            d = cons.get("D")
            a = cons.get("A")
            if all(isinstance(v, (int, float)) for v in (h, d, a)):
                s = h + d + a
                market_probs = (
                    {"H": h / s, "D": d / s, "A": a / s}
                    if s > 0
                    else {"H": h, "D": d, "A": a}
                )
                market_source = (
                    "historic_bet365"
                    if market_rec.get("preferred_implied")
                    else "historic_consensus"
                )
        model_probs = None
        if pred:
            mp = {
                "H": pred["home_win_prob"],
                "D": pred["draw_prob"],
                "A": pred["away_win_prob"],
            }
            model_probs = calibration_service.apply_if_ready(mp)
        model_dc = dc_prob_from_probs(model_probs) if model_probs else None
        market_dc = dc_prob_from_probs(market_probs) if market_probs else None
        # Edge & metrics
        model_pick = (
            "Yes"
            if (model_dc is not None and model_dc >= 0.5)
            else "No"
            if model_dc is not None
            else None
        )
        market_pick = (
            "Yes"
            if (market_dc is not None and market_dc >= 0.5)
            else "No"
            if market_dc is not None
            else None
        )
        edge_for_model_pick = None
        edge_recommendation = False
        if model_dc is not None and market_dc is not None and model_pick:
            edge_for_model_pick = round(
                (model_dc - market_dc)
                if model_pick == "Yes"
                else ((1 - model_dc) - (1 - market_dc)),
                4,
            )
            edge_recommendation = edge_for_model_pick >= edge_threshold
        if actual_result and model_dc is not None and market_dc is not None:
            total_evaluable += 1
            actual_yes = (
                1
                if (
                    (mkt == "1X" and actual_result in ("H", "D"))
                    or (mkt == "12" and actual_result in ("H", "A"))
                    or (mkt == "X2" and actual_result in ("D", "A"))
                )
                else 0
            )
            import math

            mpb = model_dc if actual_yes == 1 else (1.0 - model_dc)
            kpb = market_dc if actual_yes == 1 else (1.0 - market_dc)
            mpb = max(min(mpb, 1 - eps), eps)
            kpb = max(min(kpb, 1 - eps), eps)
            model_log_losses.append(-math.log(mpb))
            market_log_losses.append(-math.log(kpb))
            model_briers.append((model_dc - actual_yes) ** 2)
            market_briers.append((market_dc - actual_yes) ** 2)
            if (model_pick == "Yes" and actual_yes == 1) or (
                model_pick == "No" and actual_yes == 0
            ):
                model_correct += 1
            if (market_pick == "Yes" and actual_yes == 1) or (
                market_pick == "No" and actual_yes == 0
            ):
                market_correct += 1
        rows.append(
            {
                "match_id": mid,
                "home_team": home_raw,
                "away_team": away_raw,
                "date": date or None,
                "actual_result": actual_result,
                "market": mkt,
                "model_dc_prob": round(model_dc, 4)
                if isinstance(model_dc, (int, float))
                else None,
                "market_dc_prob": round(market_dc, 4)
                if isinstance(market_dc, (int, float))
                else None,
                "model_pick": model_pick,
                "market_pick": market_pick,
                "edge_for_model_pick": edge_for_model_pick,
                "edge_recommendation": edge_recommendation,
                "market_source": market_source,
            }
        )
    agg = {
        "week": week,
        "market": mkt,
        "matches_considered": len(wk_matches),
        "rows_produced": len(rows),
        "evaluable_with_actual_and_probs": total_evaluable,
        "model_accuracy": (model_correct / total_evaluable)
        if total_evaluable
        else None,
        "market_accuracy": (market_correct / total_evaluable)
        if total_evaluable
        else None,
        "model_log_loss": (sum(model_log_losses) / len(model_log_losses))
        if model_log_losses
        else None,
        "market_log_loss": (sum(market_log_losses) / len(market_log_losses))
        if market_log_losses
        else None,
        "model_brier": (sum(model_briers) / len(model_briers))
        if model_briers
        else None,
        "market_brier": (sum(market_briers) / len(market_briers))
        if market_briers
        else None,
        "edge_threshold": edge_threshold,
        "recommended_edges": sum(1 for r in rows if r.get("edge_recommendation")),
    }
    if agg["model_log_loss"] is not None and agg["market_log_loss"] is not None:
        agg["model_log_loss_delta"] = round(
            agg["market_log_loss"] - agg["model_log_loss"], 4
        )
    if agg["model_brier"] is not None and agg["market_brier"] is not None:
        agg["model_brier_delta"] = round(agg["market_brier"] - agg["model_brier"], 4)
    return {"aggregate": agg, "matches": rows}


# ---------------------------------------------------------------------------
# Week Draw No Bet (home/away) vs Model vs Market
# ---------------------------------------------------------------------------
def compare_week_dnb(
    week: int, side: str = "home", edge_threshold: float = 0.05
) -> Dict[str, Any]:
    """Compare Draw No Bet for a week.
    side in {'home','away'}. Pushes (draws) are excluded from metrics.
    DNB probabilities are derived from 1X2 probabilities: pH/(1-pD) or pA/(1-pD).
    """
    s = side.lower()
    assert s in ("home", "away"), "side must be 'home' or 'away'"
    if not _prediction_cache:
        rebuild_predictions()
    service = EnhancedEPLService()
    matches = service.get_all_matches()
    weeks = game_week_service.organize_matches_by_week(matches)
    wk_matches = weeks.get(week, [])
    historic_idx = _load_historic_odds_index()
    rows: List[Dict[str, Any]] = []
    model_log_losses: List[float] = []
    market_log_losses: List[float] = []
    model_briers: List[float] = []
    market_briers: List[float] = []
    model_correct = 0
    market_correct = 0
    total_evaluable = 0
    eps = 1e-12

    def dnb_from_probs(probs: Dict[str, float]) -> Optional[float]:
        pH, pD, pA = probs.get("H", 0.0), probs.get("D", 0.0), probs.get("A", 0.0)
        denom = 1.0 - pD
        if denom <= 0:
            return None
        return (pH / denom) if s == "home" else (pA / denom)

    for m in wk_matches:
        home_raw = m.get("home_team") or m.get("homeTeam")
        away_raw = m.get("away_team") or m.get("awayTeam")
        home = normalize_team_name(home_raw) or home_raw
        away = normalize_team_name(away_raw) or away_raw
        date = (m.get("utc_date") or m.get("date") or "").split("T")[0]
        mid = m.get("id") or m.get("match_id")
        key_pred = f"{mid}_{home}_{away}"
        pred = _prediction_cache.get(key_pred)
        if not pred:
            try:
                raw = advanced_ml_predictor.predict_match(home, away)
                if raw:
                    pred = _normalize_prediction(raw, home, away)
                    _prediction_cache[key_pred] = pred
            except Exception:
                pred = None
        hs = m.get("home_score") if "home_score" in m else m.get("homeScore")
        as_ = m.get("away_score") if "away_score" in m else m.get("awayScore")
        status_completed = m.get("status") in ["FINISHED", "COMPLETED"] or m.get(
            "is_completed"
        )
        actual_result = None
        if hs is not None and as_ is not None and status_completed:
            if hs > as_:
                actual_result = "H"
            elif as_ > hs:
                actual_result = "A"
            else:
                actual_result = "D"
        market_rec = None
        if date:
            market_rec = historic_idx.get(f"{date}|{home.lower()}|{away.lower()}")
        if not market_rec:
            market_rec = historic_idx.get(f"{home}|{away}".lower())
        market_probs = None
        market_source = None
        if market_rec:
            cons = (
                market_rec.get("preferred_implied")
                or market_rec.get("consensus_implied")
                or {}
            )
            h = cons.get("H")
            d = cons.get("D")
            a = cons.get("A")
            if all(isinstance(v, (int, float)) for v in (h, d, a)):
                ssum = h + d + a
                market_probs = (
                    {"H": h / ssum, "D": d / ssum, "A": a / ssum}
                    if ssum > 0
                    else {"H": h, "D": d, "A": a}
                )
                market_source = (
                    "historic_bet365"
                    if market_rec.get("preferred_implied")
                    else "historic_consensus"
                )
        model_probs = None
        if pred:
            mp = {
                "H": pred["home_win_prob"],
                "D": pred["draw_prob"],
                "A": pred["away_win_prob"],
            }
            model_probs = calibration_service.apply_if_ready(mp)
        model_dnb = dnb_from_probs(model_probs) if model_probs else None
        market_dnb = dnb_from_probs(market_probs) if market_probs else None
        model_pick = None
        market_pick = None
        if model_dnb is not None:
            model_pick = (
                ("Home DNB" if s == "home" else "Away DNB")
                if model_dnb >= 0.5
                else ("Opp DNB")
            )
        if market_dnb is not None:
            market_pick = (
                ("Home DNB" if s == "home" else "Away DNB")
                if market_dnb >= 0.5
                else ("Opp DNB")
            )
        edge_for_model_pick = None
        edge_recommendation = False
        if model_dnb is not None and market_dnb is not None and model_pick:
            if (s == "home" and model_pick == "Home DNB") or (
                s == "away" and model_pick == "Away DNB"
            ):
                edge_for_model_pick = round(model_dnb - market_dnb, 4)
            else:
                edge_for_model_pick = round((1.0 - model_dnb) - (1.0 - market_dnb), 4)
            edge_recommendation = edge_for_model_pick >= edge_threshold
        # Metrics: exclude draws (push)
        if (
            actual_result in ("H", "A")
            and model_dnb is not None
            and market_dnb is not None
        ):
            total_evaluable += 1
            actual_yes = (
                1
                if (
                    (s == "home" and actual_result == "H")
                    or (s == "away" and actual_result == "A")
                )
                else 0
            )
            import math

            mpb = model_dnb if actual_yes == 1 else (1.0 - model_dnb)
            kpb = market_dnb if actual_yes == 1 else (1.0 - market_dnb)
            mpb = max(min(mpb, 1 - eps), eps)
            kpb = max(min(kpb, 1 - eps), eps)
            model_log_losses.append(-math.log(mpb))
            market_log_losses.append(-math.log(kpb))
            model_briers.append((model_dnb - actual_yes) ** 2)
            market_briers.append((market_dnb - actual_yes) ** 2)
            if (
                (s == "home" and model_pick == "Home DNB")
                or (s == "away" and model_pick == "Away DNB")
            ) == (actual_yes == 1):
                model_correct += 1
            if (
                (s == "home" and market_pick == "Home DNB")
                or (s == "away" and market_pick == "Away DNB")
            ) == (actual_yes == 1):
                market_correct += 1
        rows.append(
            {
                "match_id": mid,
                "home_team": home_raw,
                "away_team": away_raw,
                "date": date or None,
                "actual_result": actual_result,
                "side": s,
                "model_dnb_prob": round(model_dnb, 4)
                if isinstance(model_dnb, (int, float))
                else None,
                "market_dnb_prob": round(market_dnb, 4)
                if isinstance(market_dnb, (int, float))
                else None,
                "model_pick": model_pick,
                "market_pick": market_pick,
                "edge_for_model_pick": edge_for_model_pick,
                "edge_recommendation": edge_recommendation,
                "market_source": market_source,
            }
        )
    agg = {
        "week": week,
        "side": s,
        "matches_considered": len(wk_matches),
        "rows_produced": len(rows),
        "evaluable_with_actual_and_probs": total_evaluable,
        "model_accuracy": (model_correct / total_evaluable)
        if total_evaluable
        else None,
        "market_accuracy": (market_correct / total_evaluable)
        if total_evaluable
        else None,
        "model_log_loss": (sum(model_log_losses) / len(model_log_losses))
        if model_log_losses
        else None,
        "market_log_loss": (sum(market_log_losses) / len(market_log_losses))
        if market_log_losses
        else None,
        "model_brier": (sum(model_briers) / len(model_briers))
        if model_briers
        else None,
        "market_brier": (sum(market_briers) / len(market_briers))
        if market_briers
        else None,
        "edge_threshold": edge_threshold,
        "recommended_edges": sum(1 for r in rows if r.get("edge_recommendation")),
    }
    if agg["model_log_loss"] is not None and agg["market_log_loss"] is not None:
        agg["model_log_loss_delta"] = round(
            agg["market_log_loss"] - agg["model_log_loss"], 4
        )
    if agg["model_brier"] is not None and agg["market_brier"] is not None:
        agg["model_brier_delta"] = round(agg["market_brier"] - agg["model_brier"], 4)
    return {"aggregate": agg, "matches": rows}


# ---------------------------------------------------------------------------
# Odds API integration (Tasks A,B,D)
# ---------------------------------------------------------------------------
def _compute_consensus_and_edges(
    odds_payload: Dict[str, Any], edge_threshold: float = 0.05
) -> Dict[str, Any]:
    """Derive median (consensus) implied probabilities across bookmakers and
    compute model vs market edges.

    Returns structure with per-event consensus + edges.
    Edge = model_probability - consensus_probability (positive favors model bet).
    """
    import statistics as stats

    events = odds_payload.get("events") or []
    # If caller provided league in payload, normalize it for predictor blending
    from ..services.league_manager import normalize_league_code as _norm

    code = _norm(odds_payload.get("league"))
    consensus_list: List[Dict[str, Any]] = []
    edges_list: List[Dict[str, Any]] = []
    pref_env = os.getenv("PREFERRED_BOOKMAKERS", "bet365,draftkings,fanduel,bovada")
    preferred_keys: Set[str] = {
        k.strip().lower() for k in pref_env.split(",") if k.strip()
    }
    for ev in events:
        home = normalize_team_name(ev.get("home_team"))
        away = normalize_team_name(ev.get("away_team"))
        bms = ev.get("bookmakers") or []
        # Filter to preferred if available
        filtered_bms = []
        for bm in bms:
            key = (bm.get("key") or "").lower()
            title = (bm.get("title") or "").lower()
            if key in preferred_keys or title in preferred_keys:
                filtered_bms.append(bm)
        use_bms = filtered_bms if filtered_bms else bms
        probs_acc: Dict[str, List[float]] = {"H": [], "D": [], "A": []}
        overrounds: List[float] = []
        used_keys: List[str] = []
        for bm in use_bms:
            h2h = bm.get("h2h") or {}
            ip = h2h.get("implied_probabilities") or {}
            ov = h2h.get("overround")
            for k in ("H", "D", "A"):
                v = ip.get(k)
                if v is not None and v > 0:
                    probs_acc[k].append(float(v))
            if isinstance(ov, (int, float)):
                overrounds.append(float(ov))
            if bm.get("key"):
                used_keys.append(str(bm.get("key")).lower())
        if any(probs_acc[k] for k in probs_acc):
            consensus_probs = {
                k: (round(stats.median(vs), 4) if vs else None)
                for k, vs in probs_acc.items()
            }
            consensus_overround = (
                round(stats.median(overrounds), 4) if overrounds else None
            )
        else:
            consensus_probs = {k: None for k in ("H", "D", "A")}
            consensus_overround = None
        consensus_entry = {
            "home_team": home,
            "away_team": away,
            "consensus_implied": consensus_probs,
            "consensus_overround": consensus_overround,
            "bookmakers_considered": sum(
                1
                for bm in use_bms
                if (bm.get("h2h") or {}).get("implied_probabilities")
            ),
            "bookmakers_used": used_keys,
        }
        consensus_list.append(consensus_entry)
        # Model prediction (on-demand) for edge calc
        model_prediction = None
        try:
            raw = advanced_ml_predictor.predict_match(home, away, league=code)
            if raw:
                model_prediction = _normalize_prediction(raw, home, away)
        except Exception:
            model_prediction = None
        if model_prediction and all(consensus_probs.values()):  # ensure no None
            model_probs = {
                "H": model_prediction["home_win_prob"],
                "D": model_prediction["draw_prob"],
                "A": model_prediction["away_win_prob"],
            }
            edges = {
                k: round(model_probs[k] - consensus_probs[k], 4)
                for k in ("H", "D", "A")
                if consensus_probs[k] is not None
            }
            recommended = [k for k, v in edges.items() if v >= edge_threshold]
            edges_list.append(
                {
                    "home_team": home,
                    "away_team": away,
                    "model_probs": {k: round(v, 4) for k, v in model_probs.items()},
                    "consensus_probs": consensus_probs,
                    "edges": edges,
                    "edge_threshold": edge_threshold,
                    "recommended_outcomes": recommended,
                }
            )
        else:
            edges_list.append(
                {
                    "home_team": home,
                    "away_team": away,
                    "model_probs": None
                    if not model_prediction
                    else {
                        "H": round(model_prediction["home_win_prob"], 4),
                        "D": round(model_prediction["draw_prob"], 4),
                        "A": round(model_prediction["away_win_prob"], 4),
                    },
                    "consensus_probs": consensus_probs,
                    "edges": None,
                    "edge_threshold": edge_threshold,
                    "recommended_outcomes": [],
                }
            )
    return {"consensus": consensus_list, "edges": edges_list}


def _write_output(name: str, payload: Dict[str, Any]):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = OUTPUT_DIR / f"{ts}_{name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def _load_cached_runs(prefix_filters: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
    """Load cached JSON artifacts whose filenames contain any of the prefix filters."""
    results: List[Tuple[str, Dict[str, Any]]] = []
    for p in sorted(OUTPUT_DIR.glob("*.json")):
        fname = p.name
        if not any(f in fname for f in prefix_filters):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            results.append((fname, data))
        except Exception:
            continue
    return results


def consolidated_weeks_report(weeks: List[int]) -> Dict[str, Any]:
    """Build a consolidated report across markets for given weeks by scanning cached outputs.
    Expected file name fragments per market (already produced by compare commands):
      - week{W}_odds_compare, week{W}_totals_compare, week{W}_firsthalf_compare,
        week{W}_secondhalf_compare, week{W}_corners_compare,
        week{W}_dc_*, week{W}_dnb_*, week{W}_team_goals_*, week{W}_team_corners_*
    """
    week_fragments = []
    for w in weeks:
        week_fragments.extend(
            [
                f"week{w}_odds_compare",
                f"week{w}_totals_compare",
                f"week{w}_firsthalf_compare",
                f"week{w}_secondhalf_compare",
                f"week{w}_corners_compare",
                f"week{w}_dc_",
                f"week{w}_dnb_",
                f"week{w}_team_goals_",
                f"week{w}_team_corners_",
            ]
        )
    loaded = _load_cached_runs(week_fragments)

    # Organize by week and market type
    report: Dict[str, Any] = {str(w): {"files": [], "markets": {}} for w in weeks}

    def _market_key_from_name(name: str) -> str:
        # Derive a simple market key from filename
        # Examples: week1_odds_compare.json -> odds; week2_dc_1X_compare -> dc_1X
        frag = name
        if "week" in frag and "_compare" in frag:
            core = frag.split("week", 1)[1]  # e.g., '1_dc_1X_compare...'
            parts = core.split("_")
            # drop leading week number
            if parts and parts[0].isdigit():
                parts = parts[1:]
            if parts and parts[-1].startswith("compare"):
                parts = parts[:-1]
            key = "_".join(parts) if parts else "unknown"
            return key
        return "unknown"

    for fname, data in loaded:
        # Parse week from name
        wk = None
        if "week" in fname:
            try:
                wk = int(fname.split("week", 1)[1].split("_", 1)[0])
            except Exception:
                wk = None
        if wk not in weeks:
            continue
        wk_key = str(wk)
        report[wk_key]["files"].append(fname)
        mkey = _market_key_from_name(fname)
        # Capture high-level aggregate if present
        agg = data.get("aggregate") if isinstance(data, dict) else None
        if not isinstance(report[wk_key]["markets"].get(mkey), list):
            report[wk_key]["markets"][mkey] = []
        report[wk_key]["markets"][mkey].append(agg or data)

    # Build small summary per week
    for wk in weeks:
        wk_key = str(wk)
        markets = report[wk_key]["markets"]
        summary = {}
        for m, entries in markets.items():
            # Try to aggregate common fields if available
            acc = [
                e.get("model_accuracy")
                for e in entries
                if isinstance(e, dict) and e.get("model_accuracy") is not None
            ]
            rec = [
                e.get("recommended_edges")
                for e in entries
                if isinstance(e, dict) and e.get("recommended_edges") is not None
            ]
            evaluable = [
                e.get("evaluable_with_actual_and_probs")
                for e in entries
                if isinstance(e, dict)
                and e.get("evaluable_with_actual_and_probs") is not None
            ]
            summary[m] = {
                "count": len(entries),
                "avg_model_accuracy": round(sum(acc) / len(acc), 3) if acc else None,
                "total_recommended_edges": int(sum(rec)) if rec else 0,
                "total_evaluable": int(sum(evaluable)) if evaluable else None,
            }
        report[wk_key]["summary"] = summary
    return report


def main():
    parser = argparse.ArgumentParser(description="Offline ML ops tasks")
    sub = parser.add_subparsers(dest="cmd")
    p_retrain = sub.add_parser("retrain", help="Retrain ML models and bump version")
    p_retrain.add_argument(
        "--level", choices=["major", "minor", "patch"], default="patch"
    )
    sub.add_parser("rebuild", help="Rebuild all predictions (reset cache in-memory)")
    p_sum = sub.add_parser(
        "summary", help="Summarize prediction distribution for a week"
    )
    p_sum.add_argument("--week", type=int, default=1)
    sub.add_parser(
        "mark-trained", help="Force-set model metadata is_trained=True (repair)"
    )
    p_full = sub.add_parser(
        "full",
        help="Retrain, rebuild, then summarize week 1 (or provided) in one process",
    )
    p_full.add_argument("--level", choices=["major", "minor", "patch"], default="patch")
    p_full.add_argument("--week", type=int, default=1)
    p_recon = sub.add_parser(
        "reconcile",
        help="Reconcile completed matches for a week (optionally using manual results)",
    )
    p_recon.add_argument("--week", type=int, default=1)
    p_recon.add_argument(
        "--manual", type=str, default=None, help="Path to manual results JSON file"
    )
    p_recon.add_argument(
        "--force-repredict",
        action="store_true",
        help="Force regeneration of predictions even if cached",
    )
    p_recon.add_argument(
        "--apply-goal-adjust",
        action="store_true",
        help="Apply historical bias correction offsets to goals before reconciliation",
    )
    p_fetch = sub.add_parser(
        "fetch-scores",
        help="Fetch final scores from Football-Data.org API and write manual results file",
    )
    p_fetch.add_argument("--week", type=int, default=1)
    p_fetch.add_argument("--season", type=int, default=2025)
    p_fetch.add_argument("--competition", type=str, default="PL")
    p_fetch.add_argument(
        "--reconcile",
        action="store_true",
        help="Immediately run reconciliation after fetch",
    )
    p_fetch.add_argument(
        "--force-repredict",
        action="store_true",
        help="Force regeneration during auto-reconcile",
    )
    p_diag = sub.add_parser(
        "diagnostics", help="Extended diagnostics (bias, calibration, adjustments)"
    )
    p_diag.add_argument(
        "--week", type=int, default=None, help="Limit diagnostics to a specific week"
    )
    p_diag.add_argument(
        "--calibration",
        action="store_true",
        help="Include probability calibration bins",
    )
    p_diag.add_argument(
        "--goal-adjust",
        action="store_true",
        help="Include suggested goal adjustment offsets",
    )
    p_odds = sub.add_parser(
        "ingest-odds", help="Ingest historic odds CSV files (local)"
    )
    p_odds.add_argument(
        "--path",
        type=str,
        default="data/odds",
        help="Directory or single CSV file path containing historic odds",
    )
    p_download_hist = sub.add_parser(
        "download-historic-odds", help="Download EPL historic CSV odds then ingest"
    )
    p_download_hist.add_argument(
        "--start-year",
        type=int,
        default=2020,
        help="Starting season year (e.g. 2020 for 2020-2021)",
    )
    p_download_hist.add_argument(
        "--end-year", type=int, default=2024, help="Ending season year inclusive"
    )
    p_download_hist.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing CSVs"
    )
    p_download_hist.add_argument(
        "--out-dir",
        type=str,
        default="data/odds",
        help="Target directory for downloaded CSVs",
    )
    p_market_rows = sub.add_parser(
        "market-rows",
        help="Show count and sample of market-derived historic training rows",
    )
    p_market_rows.add_argument(
        "--limit", type=int, default=5, help="Sample size to display"
    )
    p_live_odds = sub.add_parser(
        "live-odds",
        help="Fetch (best-effort) odds from Football-Data.org for a matchday",
    )
    p_live_odds.add_argument("--competition", type=str, default="PL")
    p_live_odds.add_argument("--season", type=int, default=2025)
    p_live_odds.add_argument("--matchday", type=int, default=1)
    p_live_odds.add_argument(
        "--deep",
        action="store_true",
        help="Per-match follow-up requests for missing odds",
    )
    p_odds_api = sub.add_parser(
        "odds-api", help="Fetch odds via external Odds API (requires ODDS_API_KEY)"
    )
    p_odds_api.add_argument(
        "--sport", type=str, default="soccer_epl", help="Sport key (e.g., soccer_epl)"
    )
    p_odds_api.add_argument(
        "--regions", type=str, default="eu", help="Regions parameter (api docs)"
    )
    p_odds_api.add_argument(
        "--markets", type=str, default="h2h", help="Markets (default h2h)"
    )
    p_odds_api.add_argument(
        "--edge-threshold",
        type=float,
        default=0.05,
        help="Edge threshold for recommendations",
    )
    # Calibration
    p_cal = sub.add_parser(
        "calibrate",
        help="Fit temperature scaling up to a given week (uses completed matches only)",
    )
    p_cal.add_argument(
        "--up-to-week",
        type=int,
        default=1,
        help="Include matches up to and including this week",
    )
    p_cmp = sub.add_parser(
        "week-odds-compare",
        help="Compare model vs historic market consensus vs actual outcomes for a week",
    )
    p_cmp.add_argument("--week", type=int, default=1)
    p_cmp.add_argument("--edge-threshold", type=float, default=0.05)
    p_cmp_tot = sub.add_parser(
        "week-totals-compare",
        help="Compare model vs market totals (O/U 2.5) vs actual for a week",
    )
    p_cmp_tot.add_argument("--week", type=int, default=1)
    p_cmp_tot.add_argument("--line", type=float, default=2.5)
    p_cmp_tot.add_argument("--edge-threshold", type=float, default=0.05)
    # First/Second Half totals
    p_fh = sub.add_parser(
        "week-firsthalf-compare",
        help="First Half totals compare (O/U line) vs actual for a week",
    )
    p_fh.add_argument("--week", type=int, default=1)
    p_fh.add_argument("--line", type=float, default=1.0)
    p_fh.add_argument("--edge-threshold", type=float, default=0.05)
    p_sh = sub.add_parser(
        "week-secondhalf-compare",
        help="Second Half totals compare (O/U line) vs actual for a week",
    )
    p_sh.add_argument("--week", type=int, default=1)
    p_sh.add_argument("--line", type=float, default=1.0)
    p_sh.add_argument("--edge-threshold", type=float, default=0.05)
    # Corners totals
    p_cor = sub.add_parser(
        "week-corners-compare",
        help="Corners totals compare (O/U line) vs actual for a week",
    )
    p_cor.add_argument("--week", type=int, default=1)
    p_cor.add_argument("--line", type=float, default=9.5)
    p_cor.add_argument("--edge-threshold", type=float, default=0.05)
    p_dc = sub.add_parser(
        "week-dc-compare", help="Double Chance compare (1X, 12, X2) vs market"
    )
    p_dc.add_argument("--week", type=int, default=1)
    p_dc.add_argument("--market", type=str, choices=["1X", "12", "X2"], default="1X")
    p_dc.add_argument("--edge-threshold", type=float, default=0.05)
    p_dnb = sub.add_parser(
        "week-dnb-compare", help="Draw No Bet compare (home/away) vs market"
    )
    p_dnb.add_argument("--week", type=int, default=1)
    p_dnb.add_argument("--side", type=str, choices=["home", "away"], default="home")
    p_dnb.add_argument("--edge-threshold", type=float, default=0.05)
    # Team totals
    p_tt_g = sub.add_parser(
        "week-team-goals-compare",
        help="Team goals totals compare (home/away) O/U vs actual",
    )
    p_tt_g.add_argument("--week", type=int, default=1)
    p_tt_g.add_argument("--side", type=str, choices=["home", "away"], default="home")
    p_tt_g.add_argument("--line", type=float, default=1.5)
    p_tt_g.add_argument("--edge-threshold", type=float, default=0.05)
    p_tt_c = sub.add_parser(
        "week-team-corners-compare",
        help="Team corners totals compare (home/away) O/U vs actual",
    )
    p_tt_c.add_argument("--week", type=int, default=1)
    p_tt_c.add_argument("--side", type=str, choices=["home", "away"], default="home")
    p_tt_c.add_argument("--line", type=float, default=4.5)
    p_tt_c.add_argument("--edge-threshold", type=float, default=0.05)
    # Fetch corners markets via Odds API
    p_cm = sub.add_parser(
        "fetch-corners-markets",
        help="Fetch corners totals markets via Odds API (requires ODDS_API_KEY)",
    )
    p_cm.add_argument("--sport", default="soccer_epl")
    p_cm.add_argument("--regions", default="eu")
    p_cm.add_argument("--historical", action="store_true")
    p_cm.add_argument(
        "--from",
        dest="ts_from",
        default=None,
        help="ISO datetime lower bound for historical fetch",
    )
    p_cm.add_argument(
        "--to",
        dest="ts_to",
        default=None,
        help="ISO datetime upper bound for historical fetch",
    )
    p_cm.add_argument("--markets", default="alternate_totals_corners")
    p_cm.add_argument("--bookmakers", default=None)
    p_cm.add_argument("--out", default="data/corners_market_lines_manual.json")
    # Backfill corners via Bovada
    p_bcm = sub.add_parser(
        "bovada-corners-backfill",
        help="Backfill corners totals and team corners from Bovada into markets file",
    )
    p_bcm.add_argument("--out", default="data/corners_market_lines_manual.json")
    # Consolidated report
    p_cons = sub.add_parser(
        "consolidated-report", help="Consolidate cached week results into one artifact"
    )
    p_cons.add_argument(
        "--weeks",
        type=str,
        default="1,2,3",
        help="Comma-separated week numbers (e.g., 1,2,3)",
    )
    # Corners calibration
    p_cal_cor = sub.add_parser(
        "calibrate-corners",
        help="Fit corners calibration on historical weeks and persist",
    )
    p_cal_cor.add_argument(
        "--up-to-week",
        type=int,
        default=6,
        help="Use completed weeks up to and including this week for calibration",
    )
    p_cal_cor.add_argument("--line", type=float, default=9.5)
    p_cal_cor.add_argument(
        "--threshold-week",
        type=int,
        default=7,
        help="Blend market odds from this week onward",
    )
    p_cal_cor.add_argument(
        "--blend-weight",
        type=float,
        default=0.25,
        help="Weight for market over prob in blending (0..0.9)",
    )
    # Odds snapshot
    p_snap = sub.add_parser(
        "odds-snapshot",
        help="Snapshot all week compares to a JSON file under data/odds_snapshots",
    )
    p_snap.add_argument("--week", type=int, default=1)
    p_snap.add_argument("--edge-threshold", type=float, default=0.05)
    # Walk-forward training pipeline
    p_wf = sub.add_parser(
        "walkforward",
        help="Walk-forward: predict→reconcile→calibrate→retrain for a week range",
    )
    p_wf.add_argument("--start-week", type=int, default=1)
    p_wf.add_argument("--end-week", type=int, default=5)
    p_wf.add_argument(
        "--retrain-level", choices=["major", "minor", "patch"], default="patch"
    )
    p_wf.add_argument(
        "--no-fetch",
        action="store_true",
        help="Do not fetch scores; use existing manual files if present",
    )
    p_wf.add_argument(
        "--apply-goal-adjust",
        action="store_true",
        help="Apply historical goal bias adjustment during reconciliation",
    )
    # Daily update
    p_daily = sub.add_parser(
        "daily-update",
        help="Run daily update: refresh schedule, fetch scores, reconcile, update odds, retrain, rebuild, snapshot",
    )
    p_daily.add_argument(
        "--retrain-level", choices=["major", "minor", "patch"], default="patch"
    )
    p_daily.add_argument("--no-closing", action="store_true")
    p_daily.add_argument("--no-snapshot", action="store_true")
    p_daily.add_argument("--no-bovada", action="store_true")
    p_daily.add_argument("--no-oddsapi-corners", action="store_true")
    p_daily.add_argument("--oddsapi-regions", default="eu,uk,us")
    p_daily.add_argument(
        "--oddsapi-bookmakers", default="pinnacle,bet365,williamhill,unibet,betfair_ex"
    )
    args = parser.parse_args()

    if args.cmd == "retrain":
        out = retrain(args.level)
        path = _write_output("retrain", out)
        print(json.dumps({"saved": path, **out}, indent=2))
    elif args.cmd == "rebuild":
        stats = rebuild_predictions()
        # also dump sample of first 10 predictions for inspection
        sample_items = list(_prediction_cache.items())[:10]
        sample = {k: v for k, v in sample_items}
        out = {"stats": stats, "sample": sample}
        path = _write_output("rebuild", out)
        print(json.dumps({"saved": path, **out}, indent=2))
    elif args.cmd == "mark-trained":
        # Direct metadata surgery
        meta_path = Path("app/ml/saved_models/model_metadata.pkl")
        if not meta_path.exists():
            print("No model_metadata.pkl present.")
            return
        try:
            import joblib

            meta = joblib.load(meta_path)
            meta["is_trained"] = True
            meta["training_repair_timestamp"] = datetime.utcnow().isoformat()  # type: ignore
            joblib.dump(meta, meta_path)
            print("model_metadata.pkl updated: is_trained=True")
        except Exception as e:
            print(f"Failed to update metadata: {e}")
        return
    elif args.cmd == "summary":
        if not _prediction_cache:
            rebuild_predictions()
        out = summarize_week(args.week)
        path = _write_output(f"week{args.week}_summary", out)
        print(json.dumps({"saved": path, **out}, indent=2))
    elif args.cmd == "full":
        retrain_out = retrain(args.level)
        rebuild_stats = rebuild_predictions()
        summary_out = summarize_week(args.week)
        aggregate = {
            "retrain": retrain_out,
            "rebuild_stats": rebuild_stats,
            "summary": summary_out,
        }
        path = _write_output(f"full_w{args.week}", aggregate)
        print(json.dumps({"saved": path, **aggregate}, indent=2))
    elif args.cmd == "reconcile":
        result = reconcile_week(
            args.week,
            manual_path=args.manual,
            force_repredict=args.force_repredict,
            apply_goal_adjust=args.apply_goal_adjust,
        )
        path = _write_output(f"reconcile_w{args.week}", result)
        print(json.dumps({"saved": path, **result}, indent=2))
    elif args.cmd == "fetch-scores":
        out = fetch_scores(
            args.competition,
            args.season,
            args.week,
            auto_reconcile=args.reconcile,
            force_repredict=args.force_repredict,
        )
        path = _write_output(f"fetch_scores_w{args.week}", out)
        print(json.dumps({"saved": path, **out}, indent=2))
    elif args.cmd == "diagnostics":
        out = diagnostics(
            args.week,
            calibration=getattr(args, "calibration", False),
            goal_adjust=getattr(args, "goal_adjust", False),
        )
        suffix = f"_w{args.week}" if args.week else "_all"
        path = _write_output(f"diagnostics{suffix}", out)
        print(json.dumps({"saved": path, **out}, indent=2))
    elif args.cmd == "ingest-odds":
        out = ingest_odds_path(args.path)
        path = _write_output("ingest_odds", out)
        print(json.dumps({"saved": path, **out}, indent=2))
    elif args.cmd == "download-historic-odds":
        if download_epl_history is None:
            out = {"error": "historic_odds_downloader not available"}
        else:
            dl = download_epl_history(
                args.start_year,
                args.end_year,
                out_dir=args.out_dir,
                overwrite=args.overwrite,
            )
            if "error" in dl:
                out = dl
            else:
                ingest = ingest_odds_path(args.out_dir)
                out = {"download": dl, "ingest": ingest}
        path = _write_output("download_historic_odds", out)
        print(json.dumps({"saved": path, **out}, indent=2))
    elif args.cmd == "market-rows":
        try:
            from ..services.historic_odds_service import market_training_rows

            rows = market_training_rows()
            count = len(rows)
            sample = rows[: args.limit]
            feature_keys = (
                sorted({k for r in sample for k in r.keys()}) if sample else []
            )
            out = {
                "count": count,
                "sample_size": len(sample),
                "sample": sample,
                "feature_keys": feature_keys,
            }
        except Exception as e:
            out = {"error": str(e)}
        path = _write_output("market_rows", out)
        print(json.dumps({"saved": path, **out}, indent=2))
    elif args.cmd == "live-odds":
        out = fetch_week_odds(
            args.competition, args.season, args.matchday, deep_lookup=args.deep
        )
        path = _write_output(f"live_odds_w{args.matchday}", out)
        print(json.dumps({"saved": path, **out}, indent=2))
    elif args.cmd == "odds-api":
        if fetch_h2h_odds is None:
            out = {"error": "odds_api_service not available"}
        else:
            odds_raw = fetch_h2h_odds(
                sport_key=args.sport, regions=args.regions, markets=args.markets
            )
            if "error" in odds_raw:
                out = odds_raw
            else:
                consensus_edges = _compute_consensus_and_edges(
                    odds_raw, edge_threshold=args.edge_threshold
                )
                out = {**odds_raw, **consensus_edges}
        path = _write_output(f"odds_api_{args.sport}", out)
        print(json.dumps({"saved": path, **out}, indent=2))
    elif args.cmd == "calibrate":
        out = calibration_service.calibrate_up_to_week(args.up_to_week)
        # include status snapshot
        status = calibration_service.status()
        payload = {"result": out, "status": status}
        path = _write_output(f"calibrate_up_to_week{args.up_to_week}", payload)
        print(json.dumps({"saved": path, **payload}, indent=2))
    elif args.cmd == "week-odds-compare":
        out = compare_week_odds(args.week, edge_threshold=args.edge_threshold)
        path = _write_output(f"week{args.week}_odds_compare", out)
        print(json.dumps({"saved": path, **out}, indent=2))
    elif args.cmd == "odds-snapshot":
        week = int(getattr(args, "week", 1))
        edge = float(getattr(args, "edge_threshold", 0.05))
        payload = {
            "week": week,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "edge_threshold": edge,
            "markets": {},
        }
        payload["markets"]["match_odds"] = compare_week_odds(week, edge_threshold=edge)
        payload["markets"]["totals_full"] = compare_week_totals(
            week, line=2.5, edge_threshold=edge
        )
        payload["markets"]["totals_first_half"] = compare_week_first_half_totals(
            week, line=1.0, edge_threshold=edge
        )
        payload["markets"]["totals_second_half"] = compare_week_second_half_totals(
            week, line=1.0, edge_threshold=edge
        )
        payload["markets"]["corners_totals"] = compare_week_corners_totals(
            week, line=9.5, edge_threshold=edge
        )
        payload["markets"]["team_goals_home"] = compare_week_team_goals_totals(
            week, side="home", line=1.5, edge_threshold=edge
        )
        payload["markets"]["team_goals_away"] = compare_week_team_goals_totals(
            week, side="away", line=1.5, edge_threshold=edge
        )
        payload["markets"]["team_corners_home"] = compare_week_team_corners_totals(
            week, side="home", line=4.5, edge_threshold=edge
        )
        payload["markets"]["team_corners_away"] = compare_week_team_corners_totals(
            week, side="away", line=4.5, edge_threshold=edge
        )
        out_dir = Path("data/odds_snapshots")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"week_{week}_odds_snapshot_{ts}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {"saved": str(out_path), "markets": list(payload["markets"].keys())},
                indent=2,
            )
        )
    elif args.cmd == "walkforward":
        out = walkforward_train(
            start_week=getattr(args, "start_week", 1),
            end_week=getattr(args, "end_week", 5),
            fetch_scores_first=(not getattr(args, "no_fetch", False)),
            retrain_level=getattr(args, "retrain_level", "patch"),
            apply_goal_adjust=getattr(args, "apply_goal_adjust", False),
        )
        print(json.dumps(out, indent=2))
    elif args.cmd == "daily-update":
        out = daily_update(
            retrain_level=getattr(args, "retrain_level", "patch"),
            capture_closing=(not getattr(args, "no_closing", False)),
            generate_snapshot=(not getattr(args, "no_snapshot", False)),
            include_bovada_corners=(not getattr(args, "no_bovada", False)),
            include_odds_api_corners=(not getattr(args, "no_oddsapi_corners", False)),
            odds_api_regions=getattr(args, "oddsapi_regions", "eu,uk,us"),
            odds_api_bookmakers=getattr(
                args,
                "oddsapi_bookmakers",
                "pinnacle,bet365,williamhill,unibet,betfair_ex",
            ),
        )
        path = _write_output("daily_update", out)
        print(json.dumps({"saved": path, **out}, indent=2))
    elif args.cmd == "week-totals-compare":
        out = compare_week_totals(
            args.week,
            line=getattr(args, "line", 2.5),
            edge_threshold=getattr(args, "edge_threshold", 0.05),
        )
        path = _write_output(f"week{args.week}_totals_compare", out)
        print(json.dumps({"saved": path, **out}, indent=2))
    elif args.cmd == "week-firsthalf-compare":
        out = compare_week_first_half_totals(
            args.week,
            line=getattr(args, "line", 1.0),
            edge_threshold=getattr(args, "edge_threshold", 0.05),
        )
        path = _write_output(f"week{args.week}_firsthalf_compare", out)
        print(json.dumps({"saved": path, **out}, indent=2))
    elif args.cmd == "week-secondhalf-compare":
        out = compare_week_second_half_totals(
            args.week,
            line=getattr(args, "line", 1.0),
            edge_threshold=getattr(args, "edge_threshold", 0.05),
        )
        path = _write_output(f"week{args.week}_secondhalf_compare", out)
        print(json.dumps({"saved": path, **out}, indent=2))
    elif args.cmd == "week-corners-compare":
        out = compare_week_corners_totals(
            args.week,
            line=getattr(args, "line", 9.5),
            edge_threshold=getattr(args, "edge_threshold", 0.05),
        )
        path = _write_output(f"week{args.week}_corners_compare", out)
        print(json.dumps({"saved": path, **out}, indent=2))
    elif args.cmd == "week-dc-compare":
        out = compare_week_double_chance(
            args.week,
            market=getattr(args, "market", "1X"),
            edge_threshold=getattr(args, "edge_threshold", 0.05),
        )
        path = _write_output(f"week{args.week}_dc_{args.market}_compare", out)
        print(json.dumps({"saved": path, **out}, indent=2))
    elif args.cmd == "week-dnb-compare":
        out = compare_week_dnb(
            args.week,
            side=getattr(args, "side", "home"),
            edge_threshold=getattr(args, "edge_threshold", 0.05),
        )
        path = _write_output(f"week{args.week}_dnb_{args.side}_compare", out)
        print(json.dumps({"saved": path, **out}, indent=2))
    elif args.cmd == "week-team-goals-compare":
        out = compare_week_team_goals_totals(
            args.week,
            side=getattr(args, "side", "home"),
            line=getattr(args, "line", 1.5),
            edge_threshold=getattr(args, "edge_threshold", 0.05),
        )
        path = _write_output(f"week{args.week}_team_goals_{args.side}_compare", out)
        print(json.dumps({"saved": path, **out}, indent=2))
    elif args.cmd == "week-team-corners-compare":
        out = compare_week_team_corners_totals(
            args.week,
            side=getattr(args, "side", "home"),
            line=getattr(args, "line", 4.5),
            edge_threshold=getattr(args, "edge_threshold", 0.05),
        )
        path = _write_output(f"week{args.week}_team_corners_{args.side}_compare", out)
        print(json.dumps({"saved": path, **out}, indent=2))
    elif args.cmd == "fetch-corners-markets":
        res = fetch_corners_totals_from_odds_api(
            sport_key=getattr(args, "sport", "soccer_epl"),
            regions=getattr(args, "regions", "eu"),
            historical=getattr(args, "historical", False),
            ts_from=getattr(args, "ts_from", None),
            ts_to=getattr(args, "ts_to", None),
            markets=getattr(args, "markets", "alternate_totals_corners"),
            bookmakers=getattr(args, "bookmakers", None),
        )
        if "error" in res:
            print(json.dumps(res, indent=2))
        else:
            import os, json as _json

            os.makedirs("data", exist_ok=True)
            out_path = getattr(args, "out", "data/corners_market_lines_manual.json")
            # Merge with existing file to accumulate records across runs
            existing: List[Dict[str, Any]] = []
            try:
                if os.path.exists(out_path):
                    with open(out_path, "r", encoding="utf-8") as f:
                        prev = _json.load(f)
                        if isinstance(prev, dict) and isinstance(
                            prev.get("records"), list
                        ):
                            existing = prev["records"]
            except Exception:
                existing = []
            new_recs = res.get("records") or []

            # Deduplicate by (date, home, away, line, bookmaker)
            def _k(r):
                return (
                    str(r.get("date"))[:19],
                    (r.get("home_team") or r.get("home") or "").lower(),
                    (r.get("away_team") or r.get("away") or "").lower(),
                    float(r.get("line")) if r.get("line") is not None else None,
                    (r.get("bookmaker") or "").lower(),
                )

            seen = set(_k(r) for r in existing if isinstance(r, dict))
            merged = list(existing)
            for r in new_recs:
                try:
                    key = _k(r)
                    if key not in seen:
                        merged.append(r)
                        seen.add(key)
                except Exception:
                    merged.append(r)
            with open(out_path, "w", encoding="utf-8") as f:
                _json.dump(
                    {
                        "records": merged,
                        "source": "oddsapi",
                        "historical": res.get("historical"),
                    },
                    f,
                    indent=2,
                )
            totals, team = reload_market_store()
            print(
                _json.dumps(
                    {
                        "saved": out_path,
                        "added": len(new_recs),
                        "total_records": len(merged),
                        "reloaded": {"totals": totals, "team": team},
                    },
                    indent=2,
                )
            )
    elif args.cmd == "bovada-corners-backfill":
        # Use Bovada odds service to extract corners totals and team corners, then append to markets JSON
        out_path = getattr(args, "out", "data/corners_market_lines_manual.json")
        from ..services.bovada_odds_service import (
            fetch_pl_odds as _fetch_bov,
            _american_to_decimal as _ml_to_dec,
        )
        import os, json as _json

        os.makedirs("data", exist_ok=True)
        payload = _fetch_bov() if _fetch_bov is not None else {"events": []}
        events = payload.get("events") or []
        rows: List[Dict[str, Any]] = []
        for ev in events:
            date = ev.get("commence_time")
            home = normalize_team_name(ev.get("home_team")) or ev.get("home_team")
            away = normalize_team_name(ev.get("away_team")) or ev.get("away_team")
            # Corners totals
            for t in ev.get("corners_totals") or []:
                try:
                    line = float(t.get("line")) if t.get("line") is not None else None
                except Exception:
                    line = None
                if not home or not away or not date or line is None:
                    continue
                op = t.get("over_prob")
                up = t.get("under_prob")
                oml = t.get("over_ml")
                uml = t.get("under_ml")
                over_odds = _ml_to_dec(float(oml)) if oml is not None else None
                under_odds = _ml_to_dec(float(uml)) if uml is not None else None
                # Fallback only if ML missing: approximate from probs
                if not over_odds and isinstance(op, (int, float)) and float(op) > 0:
                    over_odds = 1.0 / float(op)
                if not under_odds and isinstance(up, (int, float)) and float(up) > 0:
                    under_odds = 1.0 / float(up)
                if over_odds and under_odds:
                    rows.append(
                        {
                            "date": date,
                            "home_team": home,
                            "away_team": away,
                            "line": float(line),
                            "over_odds": round(float(over_odds), 4),
                            "under_odds": round(float(under_odds), 4),
                            "bookmaker": "Bovada",
                            "market_type": "total",
                        }
                    )
            # Team corners
            for t in ev.get("team_corners") or []:
                try:
                    line = float(t.get("line")) if t.get("line") is not None else None
                except Exception:
                    line = None
                side = (t.get("side") or "").lower()
                if side not in ("home", "away"):
                    continue
                if not home or not away or not date or line is None:
                    continue
                op = t.get("over_prob")
                up = t.get("under_prob")
                oml = t.get("over_ml")
                uml = t.get("under_ml")
                over_odds = _ml_to_dec(float(oml)) if oml is not None else None
                under_odds = _ml_to_dec(float(uml)) if uml is not None else None
                if not over_odds and isinstance(op, (int, float)) and float(op) > 0:
                    over_odds = 1.0 / float(op)
                if not under_odds and isinstance(up, (int, float)) and float(up) > 0:
                    under_odds = 1.0 / float(up)
                if over_odds and under_odds:
                    rows.append(
                        {
                            "date": date,
                            "home_team": home,
                            "away_team": away,
                            "side": side,
                            "line": float(line),
                            "over_odds": round(float(over_odds), 4),
                            "under_odds": round(float(under_odds), 4),
                            "bookmaker": "Bovada",
                            "market_type": "team",
                        }
                    )
        # Merge into target JSON, dedupe
        existing: List[Dict[str, Any]] = []
        try:
            if os.path.exists(out_path):
                with open(out_path, "r", encoding="utf-8") as f:
                    prev = _json.load(f)
                    if isinstance(prev, dict) and isinstance(prev.get("records"), list):
                        existing = prev["records"]
        except Exception:
            existing = []

        def _k(r):
            return (
                str(r.get("date"))[:19],
                (r.get("home_team") or r.get("home") or "").lower(),
                (r.get("away_team") or r.get("away") or "").lower(),
                float(r.get("line")) if r.get("line") is not None else None,
                (r.get("bookmaker") or "").lower(),
                (r.get("market_type") or "total").lower(),
                (r.get("side") or "").lower(),
            )

        seen = set(_k(r) for r in existing if isinstance(r, dict))
        merged = list(existing)
        added = 0
        for r in rows:
            try:
                key = _k(r)
                if key not in seen:
                    merged.append(r)
                    seen.add(key)
                    added += 1
            except Exception:
                merged.append(r)
                added += 1
        with open(out_path, "w", encoding="utf-8") as f:
            _json.dump(
                {"records": merged, "source": "bovada", "added_now": added}, f, indent=2
            )
        totals, team = reload_market_store()
        print(
            _json.dumps(
                {
                    "saved": out_path,
                    "added_now": added,
                    "total_records": len(merged),
                    "reloaded": {"totals": totals, "team": team},
                },
                indent=2,
            )
        )
    elif args.cmd == "consolidated-report":
        weeks = [
            int(w.strip())
            for w in str(getattr(args, "weeks", "1,2,3")).split(",")
            if w.strip().isdigit()
        ]
        out = consolidated_weeks_report(weeks)
        path = _write_output(f"consolidated_weeks_{'-'.join(map(str,weeks))}", out)
        print(json.dumps({"saved": path, **out}, indent=2))
    elif args.cmd == "calibrate-corners":
        # Build dataset from weeks 1..up_to_week
        up_to = int(getattr(args, "up_to_week", 6))
        line = float(getattr(args, "line", 9.5))
        service = EnhancedEPLService()
        matches = service.get_all_matches()
        weeks = game_week_service.organize_matches_by_week(matches)
        # default to EPL for this calibration tool
        from ..services.league_manager import normalize_league_code as _norm

        code = _norm("PL")
        zs: List[float] = []
        ys: List[int] = []
        used = 0
        for w in range(1, up_to + 1):
            for m in weeks.get(w, []):
                home_raw = m.get("home_team") or m.get("homeTeam")
                away_raw = m.get("away_team") or m.get("awayTeam")
                home = normalize_team_name(home_raw) or home_raw
                away = normalize_team_name(away_raw) or away_raw
                date = (m.get("utc_date") or m.get("date") or "").split("T")[0]
                # Get actual total corners
                actual_corners = None
                if "total_corners" in m and isinstance(
                    m.get("total_corners"), (int, float)
                ):
                    actual_corners = int(m.get("total_corners"))
                else:
                    hc = m.get("home_corners")
                    ac = m.get("away_corners")
                    if isinstance(hc, (int, float)) and isinstance(ac, (int, float)):
                        actual_corners = int(hc + ac)
                    else:
                        c_actual = corners_actuals_store.lookup(date, home, away)
                        if c_actual and isinstance(
                            c_actual.get("total_corners"), (int, float)
                        ):
                            actual_corners = int(c_actual["total_corners"])
                if actual_corners is None:
                    continue
                # Get model's mean corners (or proxy)
                if not _prediction_cache:
                    rebuild_predictions()
                mid = m.get("id") or m.get("match_id")
                key_pred = f"{mid}_{home}_{away}"
                pred = _prediction_cache.get(key_pred)
                if not pred:
                    try:
                        raw = advanced_ml_predictor.predict_match(
                            home, away, league=code
                        )
                        if raw:
                            pred = _normalize_prediction(raw, home, away)
                            _prediction_cache[key_pred] = pred
                    except Exception:
                        pred = None
                if not pred:
                    continue
                mu = pred.get("total_corners")
                mu = float(mu) if mu is not None else None
                if mu is None:
                    tg = float(pred.get("total_goals", 2.6))
                    mu = max(6.0, tg * 4.0)
                z = float(mu) - line
                y = 1 if (actual_corners > line) else 0
                zs.append(z)
                ys.append(y)
                used += 1
        from ..services.corners_calibration_service import corners_calibration_service

        params = corners_calibration_service.fit(zs, ys, up_to_week=up_to)
        corners_calibration_service.set_market_blend(
            getattr(args, "threshold_week", 7), getattr(args, "blend_weight", 0.25)
        )
        path = corners_calibration_service.save()
        out = {"fitted": params.__dict__, "samples": used, "saved": path}
        save_path = _write_output(f"calibrate_corners_up_to_w{up_to}", out)
        print(json.dumps({"saved": save_path, **out}, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
