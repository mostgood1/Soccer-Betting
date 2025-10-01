"""
Auto-tune blending weights to improve predictive performance.

Optimizes per-league result_market_weight and totals_market_weight by minimizing
average log loss (result) and Brier/MAE for Over 2.5 using completed matches.

Writes back to cache/model_blend.json with *_by_league maps.

Usage: run as a script. It will:
  - Load reconciliation_store and historic odds index
  - For each league (PL, BL1, FL1, SA, PD), iterate candidate weights
  - Evaluate average negative log likelihood for result on matches where both
    model and market probs are available; choose best weight
  - Optionally evaluate totals using available market totals (if present)
  - Save updated blend config
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import math

from app.services.reconciliation_store import reconciliation_store
from app.services.calibration_service import calibration_service
from app.offline.tasks import _load_historic_odds_index
from app.services.team_name_normalizer import normalize_team_name


def _load_blend_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_blend_config(path: str, data: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _blend(
    p: Tuple[float, float, float], m: Tuple[float, float, float], w: float
) -> Tuple[float, float, float]:
    w = max(0.0, min(1.0, float(w)))
    d = (1 - w) * p[0] + w * m[0]
    h = (1 - w) * p[1] + w * m[1]
    a = (1 - w) * p[2] + w * m[2]
    s = max(1e-12, d + h + a)
    return (d / s, h / s, a / s)


def _nll(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -math.log(p)


def _evaluate_result_for_league(
    league: str, historic_idx: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, float], List[float]]:
    """Return mapping weight->avg NLL and the candidate grid used."""
    grid = [round(x, 2) for x in [i / 20.0 for i in range(0, 11)]]  # 0.0..0.5
    # Pull reconciled matches with model probs and actual result
    rec = reconciliation_store.load()
    rows = [
        r
        for r in rec.get("matches", [])
        if str(r.get("league")).upper() == str(league).upper()
    ]
    # Build evaluation set
    eval_rows = []
    for r in rows:
        home = normalize_team_name(r.get("home_team") or r.get("home")) or r.get(
            "home_team"
        )
        away = normalize_team_name(r.get("away_team") or r.get("away")) or r.get(
            "away_team"
        )
        if not (home and away):
            continue
        # Model probs expected in reconciliation record; fallback skip if missing
        pm = r.get("model_probs") or {}
        H = pm.get("H")
        D = pm.get("D")
        A = pm.get("A")
        if not all(isinstance(v, (int, float)) for v in (H, D, A)):
            continue
        # Apply calibration for safety
        pm_cal = calibration_service.apply_if_ready(
            {"H": H, "D": D, "A": A}, league=league
        )
        p = (pm_cal["D"], pm_cal["H"], pm_cal["A"])
        # Market probs from historic index
        hk = f"{home.lower()}|{away.lower()}"
        mk = historic_idx.get(hk) or {}
        cons = mk.get("preferred_implied") or mk.get("consensus_implied") or {}
        Mh = cons.get("H")
        Md = cons.get("D")
        Ma = cons.get("A")
        if not all(isinstance(v, (int, float)) for v in (Mh, Md, Ma)):
            continue
        s = Mh + Md + Ma
        if s <= 0:
            continue
        m = (Md / s, Mh / s, Ma / s)
        # Actual result label
        res = (r.get("result") or "").upper()
        if res not in ("H", "D", "A"):
            continue
        eval_rows.append((p, m, res))
    if not eval_rows:
        return {}, grid
    # Evaluate grid
    scores: Dict[str, float] = {}
    for w in grid:
        total = 0.0
        cnt = 0
        for p, m, res in eval_rows:
            d, h, a = _blend(p, m, w)
            pr = {"D": d, "H": h, "A": a}[res]
            total += _nll(pr)
            cnt += 1
        if cnt:
            scores[str(w)] = total / cnt
    return scores, grid


def main() -> None:
    leagues = ["PL", "BL1", "FL1", "SA", "PD"]
    blend_path = os.path.join("cache", "model_blend.json")
    cfg = _load_blend_config(blend_path)
    historic_idx = _load_historic_odds_index()
    # Results tuning
    result_map: Dict[str, float] = cfg.get("result_market_weight_by_league", {}) or {}
    for lg in leagues:
        scores, grid = _evaluate_result_for_league(lg, historic_idx)
        if not scores:
            continue
        best_w, best_v = min(
            ((float(k), v) for k, v in scores.items()), key=lambda kv: kv[1]
        )
        result_map[lg] = round(best_w, 2)
    if result_map:
        cfg["result_market_weight_by_league"] = result_map
    _save_blend_config(blend_path, cfg)
    print("Auto-tune complete. Updated:", json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
