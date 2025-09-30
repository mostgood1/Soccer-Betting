from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
from statistics import mean
import math

from .reconciliation_store import reconciliation_store
from ..offline.tasks import _load_historic_odds_index
from .team_name_normalizer import normalize_team_name
from .calibration_service import calibration_service


def _nll(p: float) -> float:
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    return -math.log(p)


def compute_model_vs_market_summary(
    leagues: Optional[List[str]] = None,
    weeks_back: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute aggregate metrics comparing model vs market implied probabilities.

    Returns dict keyed by league with fields:
      - count
      - model: {avg_nll, avg_brier, accuracy}
      - market: {avg_nll, avg_brier, accuracy}
    """
    lg_set = {s.upper() for s in (leagues or ["PL", "BL1", "FL1", "SA", "PD"])}
    hist_idx = _load_historic_odds_index()
    data = reconciliation_store.load()
    rows = data.get("matches", [])

    out: Dict[str, Any] = {}
    for lg in lg_set:
        nll_m: List[float] = []
        nll_k: List[float] = []
        br_m: List[float] = []
        br_k: List[float] = []
        acc_m = 0
        acc_k = 0
        cnt = 0
        for r in rows:
            if str(r.get("league")).upper() != lg:
                continue
            # optional weeks_back filter if week present
            if weeks_back and isinstance(r.get("game_week"), int):
                gw = int(r.get("game_week"))
                # assume current max week as max seen in data
                # filter last N weeks relative to max week in dataset per league
                # first pass collect max; we can approximate by simple threshold later
                pass
            pm = r.get("model_probs") or {}
            if not all(isinstance(pm.get(k), (int, float)) for k in ("H", "D", "A")):
                continue
            # Use calibrated probabilities for fairness
            pmc = calibration_service.apply_if_ready({"H": pm.get("H"), "D": pm.get("D"), "A": pm.get("A")}, league=lg)
            home = normalize_team_name(r.get("home_team") or r.get("home")) or r.get("home_team")
            away = normalize_team_name(r.get("away_team") or r.get("away")) or r.get("away_team")
            if not (home and away):
                continue
            mk_rec = hist_idx.get(f"{home.lower()}|{away.lower()}") or {}
            cons = mk_rec.get("preferred_implied") or mk_rec.get("consensus_implied") or {}
            if not all(isinstance(cons.get(k), (int, float)) for k in ("H", "D", "A")):
                continue
            s = cons["H"] + cons["D"] + cons["A"]
            if not s or s <= 0:
                continue
            mk = {k: float(cons[k]) / s for k in ("H", "D", "A")}
            res = (r.get("result") or "").upper()
            if res not in ("H", "D", "A"):
                continue
            # log loss
            nll_m.append(_nll(pmc[res]))
            nll_k.append(_nll(mk[res]))
            # brier
            br_m.append(sum((pmc[k] - (1.0 if k == res else 0.0)) ** 2 for k in ("H", "D", "A")))
            br_k.append(sum((mk[k] - (1.0 if k == res else 0.0)) ** 2 for k in ("H", "D", "A")))
            # accuracy
            pick_m = max((("H", pmc["H"]), ("D", pmc["D"]), ("A", pmc["A"])) , key=lambda kv: kv[1])[0]
            pick_k = max((("H", mk["H"]), ("D", mk["D"]), ("A", mk["A"])) , key=lambda kv: kv[1])[0]
            if pick_m == res:
                acc_m += 1
            if pick_k == res:
                acc_k += 1
            cnt += 1
        out[lg] = {
            "count": cnt,
            "model": {
                "avg_nll": (mean(nll_m) if nll_m else None),
                "avg_brier": (mean(br_m) if br_m else None),
                "accuracy": (acc_m / cnt if cnt else None),
            },
            "market": {
                "avg_nll": (mean(nll_k) if nll_k else None),
                "avg_brier": (mean(br_k) if br_k else None),
                "accuracy": (acc_k / cnt if cnt else None),
            },
        }
    return out
