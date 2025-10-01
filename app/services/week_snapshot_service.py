"""Week snapshot & reconciliation service.

Responsibilities:
 - Capture a "closing" snapshot of model + market state for all matches in a week.
 - Finalize a week: attach actual scores/results, compute metrics, and optionally
   prepare incremental training rows.
 - Provide aggregated report combining baseline, closing, and final metrics.

Snapshots are stored under data/week_snapshots/ as JSON for auditability.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import math
import random

from .betting_odds_service import BettingOddsService
from .game_week_service import (
    game_week_service,
    reconciliation_service as model_reconciliation_service,
)
from .enhanced_epl_service_v2 import EnhancedEPLService
from .reconciliation_store import reconciliation_store
from .baseline_prediction_store import baseline_prediction_store
from ..ml.advanced_predictor import advanced_ml_predictor
from .league_manager import (
    list_supported as list_supported_leagues,
    get_service as get_league_service,
)
from .team_name_normalizer import normalize_team_name

_SNAPSHOT_DIR = Path("data/week_snapshots")
_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


class WeekSnapshotService:
    def __init__(self):
        self.betting_service = BettingOddsService()
        self.epl_service = EnhancedEPLService()

    # --------------- File helpers ---------------
    def _closing_path(self, week: int) -> Path:
        return _SNAPSHOT_DIR / f"week_{week}_closing.json"

    def _final_path(self, week: int) -> Path:
        return _SNAPSHOT_DIR / f"week_{week}_final.json"

    def load_closing(self, week: int) -> Optional[Dict[str, Any]]:
        p = self._closing_path(week)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    def load_final(self, week: int) -> Optional[Dict[str, Any]]:
        p = self._final_path(week)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    def is_week_locked(self, week: int) -> bool:
        """A week is considered locked if a final snapshot exists.
        Locked weeks must not have predictions recomputed or mutated by API calls.
        """
        try:
            return self._final_path(week).exists()
        except Exception:
            return False

    # --------------- Capture Closing ---------------
    def capture_closing(self, week: int, force: bool = False) -> Dict[str, Any]:
        existing = self.load_closing(week)
        if existing and not force:
            return {"status": "exists", "week": week, "snapshot": existing}

        matches = self._week_matches(week)
        rows: List[Dict[str, Any]] = []
        ts = datetime.utcnow().isoformat()
        for m in matches:
            home = m.get("home_team") or m.get("homeTeam")
            away = m.get("away_team") or m.get("awayTeam")
            if not home or not away:
                continue
            # Market odds (consensus simulated)
            odds = self.betting_service.get_match_odds(home, away)
            # Extract flat odds from mock structure when present
            home_odds = odds.get("home_odds")
            draw_odds = odds.get("draw_odds")
            away_odds = odds.get("away_odds")
            if home_odds is None and isinstance(odds.get("market_odds"), dict):
                mw = odds["market_odds"].get("match_winner", {})
                home_odds = (mw.get("home") or {}).get("odds")
                draw_odds = (mw.get("draw") or {}).get("odds")
                away_odds = (mw.get("away") or {}).get("odds")
            normalized_input = {
                "home_odds": home_odds,
                "draw_odds": draw_odds,
                "away_odds": away_odds,
            }
            market_probs = self._normalize_market_odds(normalized_input)
            # Model prediction (fresh, not cache-bound)
            model_pred = advanced_ml_predictor.predict_match(home, away)
            model_probs = {
                "H": float(model_pred.get("home_win_probability", 0.33)),
                "D": float(model_pred.get("draw_probability", 0.34)),
                "A": float(model_pred.get("away_win_probability", 0.33)),
            }
            # Feature row (for incremental training later)
            feature_row = advanced_ml_predictor.get_feature_row(home, away)
            rows.append(
                {
                    "match_id": m.get("id") or m.get("match_id"),
                    "home_team": home,
                    "away_team": away,
                    "date": m.get("date"),
                    "closing_timestamp": ts,
                    "market_raw": {
                        "home_odds": home_odds,
                        "draw_odds": draw_odds,
                        "away_odds": away_odds,
                        "overround": odds.get("overround"),
                    },
                    "market_probs": market_probs,
                    "model_probs": model_probs,
                    "model_total_goals": model_pred.get("total_goals"),
                    "model_result_pick": model_pred.get("match_result"),
                    "model_result_confidence": model_pred.get(
                        "match_result_confidence"
                    ),
                    "feature_row": feature_row,
                }
            )
        snapshot = {
            "week": week,
            "captured_at": ts,
            "match_count": len(rows),
            "rows": rows,
            "model_version": advanced_ml_predictor.model_version,
        }
        self._closing_path(week).write_text(
            json.dumps(snapshot, indent=2), encoding="utf-8"
        )
        return {"status": "created", "week": week, "snapshot": snapshot}

    # --------------- Finalize Week ---------------
    def finalize_week(
        self,
        week: int,
        retrain: bool = False,
        version_bump: str = "minor",
        edge_threshold: float = 0.05,
        prob_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        closing = self.load_closing(week)
        if not closing:
            raise ValueError("No closing snapshot present. Capture closing first.")
        final_existing = self.load_final(week)
        if final_existing:
            return {"status": "exists", "week": week, "final": final_existing}
        matches = self._week_matches(week)
        # Overlay manual results written by offline.fetch_scores so we can finalize
        # even if the upstream Football-Data cache hasn't refreshed yet.
        try:
            from ..services.team_name_normalizer import normalize_team_name as _norm
            from pathlib import Path as _P
            import json as _json

            p = _P("data") / f"manual_results_week{week}.json"
            if p.exists():
                manual_rows = []
                try:
                    manual_rows = _json.loads(p.read_text(encoding="utf-8")) or []
                except Exception:
                    manual_rows = []
                if manual_rows:
                    # Build indices into the local matches list
                    idx_by_id = {}
                    idx_by_pair = {}
                    for m in matches:
                        mid = m.get("id") or m.get("match_id")
                        ht = _norm(
                            m.get("home_team")
                            or m.get("homeTeam")
                            or (m.get("home") or {}).get("name")
                        )
                        at = _norm(
                            m.get("away_team")
                            or m.get("awayTeam")
                            or (m.get("away") or {}).get("name")
                        )
                        if mid is not None:
                            idx_by_id[mid] = m
                        if ht and at:
                            idx_by_pair[(ht, at)] = m
                    for r in manual_rows:
                        mid = r.get("match_id")
                        ht = _norm(r.get("home_team") or r.get("home"))
                        at = _norm(r.get("away_team") or r.get("away"))
                        target = idx_by_id.get(mid) if mid is not None else None
                        if not target and ht and at:
                            target = idx_by_pair.get((ht, at))
                        if not target:
                            continue
                        hs = r.get("home_score")
                        as_ = r.get("away_score")
                        if hs is None or as_ is None:
                            continue
                        target["home_score"] = hs
                        target["away_score"] = as_
                        target["status"] = "COMPLETED"
                        target["is_completed"] = True
        except Exception:
            # Best-effort overlay only
            pass
        index = {(m.get("id") or m.get("match_id")): m for m in matches}
        enriched_rows = []
        completed = 0
        for row in closing["rows"]:
            mid = row.get("match_id")
            m = index.get(mid)
            if not m:
                continue
            home_score = m.get("home_score") or m.get("homeScore")
            away_score = m.get("away_score") or m.get("awayScore")
            is_completed = (
                m.get("status") in ["FINISHED", "COMPLETED"]
                or m.get("is_completed")
                or (home_score is not None and away_score is not None)
            )
            actual_result = None
            if is_completed and home_score is not None and away_score is not None:
                completed += 1
                if home_score > away_score:
                    actual_result = "H"
                elif away_score > home_score:
                    actual_result = "A"
                else:
                    actual_result = "D"
            r = dict(row)
            r.update(
                {
                    "home_score": home_score,
                    "away_score": away_score,
                    "is_completed": is_completed,
                    "actual_result": actual_result,
                }
            )
            enriched_rows.append(r)
        # Compute edge analytics before metrics (adds edge fields)
        self._compute_edge_analytics(
            enriched_rows,
            edge_threshold=edge_threshold,
            prob_threshold=prob_threshold,
        )
        metrics = self._compute_metrics(enriched_rows)

        final_snapshot = {
            "week": week,
            "finalized_at": datetime.utcnow().isoformat(),
            "rows": enriched_rows,
            "completed_matches": completed,
            "metrics": metrics,
            "model_version_pre_retrain": advanced_ml_predictor.model_version,
        }

        # Optional retraining using incremental data
        retrain_info = None
        if retrain and completed > 0:
            incremental_df = self._build_incremental_training(enriched_rows)
            # Combine with synthetic base for stability
            base_df = advanced_ml_predictor.generate_training_data(1200)
            import pandas as pd

            combined = pd.concat([base_df, incremental_df], ignore_index=True)
            new_version = advanced_ml_predictor.retrain_and_bump(
                training_data=combined, level=version_bump
            )
            retrain_info = {
                "new_model_version": new_version,
                "incremental_rows": len(incremental_df),
                "combined_size": len(combined),
            }
            final_snapshot["model_version_post_retrain"] = new_version
        self._final_path(week).write_text(
            json.dumps(final_snapshot, indent=2), encoding="utf-8"
        )
        return {
            "status": "finalized",
            "week": week,
            "final": final_snapshot,
            "retrain": retrain_info,
        }

    def simulate_finalize_week(
        self,
        week: int,
        persist: bool = False,
        edge_threshold: float = 0.05,
        prob_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Produce a simulated finalization for a week when real scores are not yet available.
        Does NOT overwrite an existing real final snapshot unless persist=True and no real final exists.
        Scores/results are sampled from model probabilities; useful for dry-runs of the pipeline.
        """
        closing = self.load_closing(week)
        if not closing:
            raise ValueError("No closing snapshot present. Capture closing first.")
        real_final = self.load_final(week)
        if real_final:
            return {
                "status": "already-finalized-real",
                "week": week,
                "final": real_final,
            }
        simulated_rows = []
        for row in closing["rows"]:
            r = dict(row)
            model_probs = r.get("model_probs") or {}
            outcomes = ["H", "D", "A"]
            weights = [model_probs.get(o, 1 / 3) for o in outcomes]
            s = sum(weights) or 1.0
            weights = [w / s for w in weights]
            # Sample result
            pick = random.choices(outcomes, weights=weights, k=1)[0]
            r["actual_result"] = pick
            # Light score fabrication (not used in core metrics except presence)
            base_total = r.get("model_total_goals") or 2.6
            jitter = random.uniform(-0.6, 0.6)
            est_total = max(0, base_total + jitter)
            if pick == "H":
                home_score = max(
                    1, int(round(est_total * 0.6 + random.uniform(-0.3, 0.3)))
                )
                away_score = max(
                    0, int(round(est_total - home_score + random.uniform(-0.3, 0.3)))
                )
            elif pick == "A":
                away_score = max(
                    1, int(round(est_total * 0.55 + random.uniform(-0.3, 0.3)))
                )
                home_score = max(
                    0, int(round(est_total - away_score + random.uniform(-0.3, 0.3)))
                )
            else:  # Draw
                home_score = max(
                    0, int(round(est_total / 2 + random.uniform(-0.3, 0.3)))
                )
                away_score = home_score
            r["home_score"] = home_score
            r["away_score"] = away_score
            r["is_completed"] = True
            simulated_rows.append(r)
        # Edge analytics & metrics
        self._compute_edge_analytics(
            simulated_rows,
            edge_threshold=edge_threshold,
            prob_threshold=prob_threshold,
        )
        metrics = self._compute_metrics(simulated_rows)
        snapshot = {
            "week": week,
            "finalized_at": datetime.utcnow().isoformat(),
            "rows": simulated_rows,
            "completed_matches": len(simulated_rows),
            "metrics": metrics,
            "model_version_pre_retrain": advanced_ml_predictor.model_version,
            "simulated": True,
        }
        if persist:
            # Only persist if no real final exists still
            if not self.load_final(week):
                self._final_path(week).write_text(
                    json.dumps(snapshot, indent=2), encoding="utf-8"
                )
        return {
            "status": "simulated",
            "week": week,
            "final": snapshot,
            "persisted": persist,
        }

    # --------------- Multi-league utilities ---------------
    def reconcile_historic_weeks_all_leagues(self, up_to_week: int) -> Dict[str, Any]:
        """Walk all supported leagues and reconcile completed matches up to a week.
        Uses current model predictions; writes into reconciliation_store.
        """
        leagues = (
            list_supported_leagues()
        )  # returns [{"code": "PL", "name": "..."}, ...]
        totals = {"matches": 0, "reconciled": 0, "by_league": {}}
        # Ensure models are loaded
        try:
            if not advanced_ml_predictor.is_trained:
                advanced_ml_predictor.load_models()
        except Exception:
            pass
        for lg in leagues:
            # Normalize to a league code string
            code = (lg.get("code") if isinstance(lg, dict) else str(lg)).upper()
            try:
                svc = get_league_service(code)
            except Exception:
                continue
            matches = (
                svc.get_all_matches()
                if hasattr(svc, "get_all_matches")
                else self.epl_service.get_all_matches()
            )
            weeks = game_week_service.organize_matches_by_week(matches)
            rec_count = 0
            considered = 0
            for w in range(1, int(up_to_week) + 1):
                for m in weeks.get(w, []):
                    hs = (
                        m.get("home_score") if "home_score" in m else m.get("homeScore")
                    )
                    as_ = (
                        m.get("away_score") if "away_score" in m else m.get("awayScore")
                    )
                    status_completed = m.get("status") in [
                        "FINISHED",
                        "COMPLETED",
                    ] or m.get("is_completed")
                    if hs is None or as_ is None or not status_completed:
                        continue
                    home = normalize_team_name(m.get("home_team") or m.get("homeTeam"))
                    away = normalize_team_name(m.get("away_team") or m.get("awayTeam"))
                    if not (home and away):
                        continue
                    try:
                        raw = advanced_ml_predictor.predict_match(
                            home, away, league=code
                        )
                        if not raw:
                            continue
                        pred = {
                            "home_goals": float(raw.get("home_goals", 1.2)),
                            "away_goals": float(raw.get("away_goals", 1.1)),
                            "total_goals": float(raw.get("total_goals", 2.5)),
                            "home_win_prob": float(
                                raw.get("home_win_probability", 0.34)
                            ),
                            "draw_prob": float(raw.get("draw_probability", 0.32)),
                            "away_win_prob": float(
                                raw.get("away_win_probability", 0.34)
                            ),
                        }
                        rec_match = dict(m)
                        rec_match["home_score"] = hs
                        rec_match["away_score"] = as_
                        rec_match["game_week"] = w
                        rec_match["league"] = getattr(svc, "code", code)
                        rec = model_reconciliation_service.reconcile_match_predictions(
                            rec_match, pred
                        )
                        reconciliation_store.upsert(rec_match, rec)
                        rec_count += 1
                        considered += 1
                    except Exception:
                        continue
            totals["by_league"][code] = {
                "reconciled": rec_count,
                "considered": considered,
            }
            totals["reconciled"] += rec_count
            totals["matches"] += considered
        reconciliation_store.compute_aggregates()
        return totals

    def calibrate_all_leagues_up_to_week(self, up_to_week: int) -> Dict[str, Any]:
        """Fit per-league temperature calibration using completed matches up to week."""
        from .calibration_service import calibration_service

        out = {"by_league": {}, "up_to_week": int(up_to_week)}
        for lg in list_supported_leagues():
            code = (lg.get("code") if isinstance(lg, dict) else str(lg)).upper()
            try:
                res = calibration_service.calibrate_up_to_week(
                    int(up_to_week), league=code
                )
                out["by_league"][code] = res
            except Exception as e:
                out["by_league"][code] = {"trained": False, "error": str(e)}
        out["status"] = calibration_service.status()
        return out

    # --------------- Report ---------------
    def build_report(self, week: int) -> Dict[str, Any]:
        closing = self.load_closing(week)
        final = self.load_final(week)
        baseline = baseline_prediction_store.get_week(week)
        baseline_deltas = None
        if baseline and closing:
            # Build index for closing rows
            close_index = {}
            for r in closing.get("rows", []):
                mk = f"{r.get('match_id')}_{r.get('home_team')}_{r.get('away_team')}"
                close_index[mk] = r
            deltas = []
            agg_shift = {"H": 0.0, "D": 0.0, "A": 0.0}
            counted = 0
            for p in baseline.get("predictions", []):
                mk = p.get("match_key")
                base_pred = p.get("prediction") or {}
                closing_row = close_index.get(mk)
                if not closing_row:
                    continue
                model_probs = closing_row.get("model_probs") or {}
                delta_obj = {
                    "match_key": mk,
                    "baseline": {
                        "H": base_pred.get("home_win_prob"),
                        "D": base_pred.get("draw_prob"),
                        "A": base_pred.get("away_win_prob"),
                    },
                    "closing_model": model_probs,
                    "delta": {},
                }
                valid = True
                for k_map in [
                    ("H", "home_win_prob"),
                    ("D", "draw_prob"),
                    ("A", "away_win_prob"),
                ]:
                    k, bp = k_map
                    bval = base_pred.get(bp)
                    mval = model_probs.get(k)
                    if bval is None or mval is None:
                        valid = False
                        continue
                    delta_obj["delta"][k] = mval - bval
                if delta_obj["delta"]:
                    for k, v in delta_obj["delta"].items():
                        agg_shift[k] += v
                    counted += 1
                deltas.append(delta_obj)
            if counted:
                avg_shift = {k: agg_shift[k] / counted for k in agg_shift}
            else:
                avg_shift = None
            baseline_deltas = {
                "matches_compared": counted,
                "average_shift": avg_shift,
                "details": deltas,
            }
        return {
            "week": week,
            "baseline_snapshot": baseline,
            "closing_snapshot": closing,
            "final_snapshot": final,
            "metrics": final.get("metrics") if final else None,
            "baseline_closing_deltas": baseline_deltas,
        }

    # --------------- Helpers ---------------
    def _week_matches(self, week: int) -> List[Dict[str, Any]]:
        matches = self.epl_service.get_all_matches()
        weeks = game_week_service.organize_matches_by_week(matches)
        return weeks.get(week, [])

    def _normalize_market_odds(
        self, odds: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        """Convert raw decimal odds to implied probabilities (simple inverse and normalized).
        Returns dict with keys H,D,A (no-vig normalized across the three).
        """
        try:
            h = odds.get("home_odds")
            d = odds.get("draw_odds")
            a = odds.get("away_odds")
            if not all([h, d, a]):
                return None
            if any(o in (None, 0) for o in [h, d, a]):
                return None
            inv = [1 / float(h), 1 / float(d), 1 / float(a)]
            s = sum(inv)
            if not (s and math.isfinite(s)):
                return None
            return {"H": inv[0] / s, "D": inv[1] / s, "A": inv[2] / s}
        except Exception:
            return None

    def _compute_metrics(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        eval_rows = [r for r in rows if r.get("actual_result") and r.get("model_probs")]
        if not eval_rows:
            return {"matches_evaluated": 0}
        correct = 0
        log_losses = []
        brier_components = []  # accumulate per outcome vector
        realized_roi_sum = 0.0
        realized_roi_count = 0
        for r in eval_rows:
            model_probs = r["model_probs"]
            actual = r["actual_result"]
            pick = max(model_probs.items(), key=lambda kv: kv[1])[0]
            if pick == actual:
                correct += 1
            # probability actual
            p_actual = model_probs.get(actual, 1e-9)
            log_losses.append(-math.log(max(p_actual, 1e-9)))
            # Brier
            for outcome in ["H", "D", "A"]:
                y = 1.0 if outcome == actual else 0.0
                p = model_probs.get(outcome, 0.0)
                brier_components.append((p - y) ** 2)
            # Realized ROI if we selected a betting edge (choose top edge if flagged)
            edge_sel = r.get("edge_selection")
            if (
                edge_sel
                and edge_sel.get("selected_outcome")
                and edge_sel.get("decimal_odds")
            ):
                realized_roi_count += 1
                if actual == edge_sel["selected_outcome"]:
                    realized_roi_sum += edge_sel["decimal_odds"] - 1.0
                else:
                    realized_roi_sum -= 1.0
        n = len(eval_rows)
        return {
            "matches_evaluated": n,
            "accuracy": correct / n if n else None,
            "log_loss": sum(log_losses) / n if n else None,
            "brier_score": sum(brier_components) / (n * 3) if n else None,
            "realized_roi_sum": realized_roi_sum if realized_roi_count else None,
            "realized_roi_avg": (realized_roi_sum / realized_roi_count)
            if realized_roi_count
            else None,
            "realized_roi_bets": realized_roi_count,
        }

    def _build_incremental_training(self, rows: List[Dict[str, Any]]):
        import pandas as pd

        data = []
        for r in rows:
            if r.get("actual_result") is None:
                continue
            feat = r.get("feature_row") or {}
            # Ensure all model feature columns present
            row = {c: feat.get(c, 0.0) for c in advanced_ml_predictor.feature_columns}
            # Targets
            hs = r.get("home_score")
            as_ = r.get("away_score")
            if hs is None or as_ is None:
                continue
            row["home_goals"] = hs
            row["away_goals"] = as_
            row["total_goals"] = hs + as_
            # match_result numeric: Draw=0 Home=1 Away=2 per existing training
            if hs > as_:
                row["match_result"] = 1
            elif as_ > hs:
                row["match_result"] = 2
            else:
                row["match_result"] = 0
            data.append(row)
        if not data:
            return pd.DataFrame(columns=advanced_ml_predictor.feature_columns)
        return pd.DataFrame(data)

    # --------------- Edge Analytics ---------------
    def _compute_edge_analytics(
        self,
        rows: List[Dict[str, Any]],
        edge_threshold: float = 0.05,
        prob_threshold: float = 0.5,
    ):
        """Annotate rows with edge analytics and aggregate summary fields.
        Adds per row:
          edge_details: list of per-outcome dicts
          edge_selection: chosen outcome (highest positive edge >= threshold)
        """
        total_selected = 0
        expected_roi_sum = 0.0
        total_edge_sum = 0.0
        selection_count = 0
        for r in rows:
            model_probs = r.get("model_probs") or {}
            market_probs = r.get("market_probs") or {}
            raw = r.get("market_raw") or {}
            details = []
            for outcome, odd_key in [
                ("H", "home_odds"),
                ("D", "draw_odds"),
                ("A", "away_odds"),
            ]:
                model_p = model_probs.get(outcome)
                market_p = market_probs.get(outcome)
                decimal_odds = raw.get(odd_key)
                edge = None
                expected_roi = None
                if model_p is not None and market_p is not None:
                    edge = model_p - market_p
                if model_p is not None and decimal_odds:
                    # Expected ROI for 1-unit stake: (p*odds) - 1
                    expected_roi = model_p * decimal_odds - 1.0
                details.append(
                    {
                        "outcome": outcome,
                        "model_prob": model_p,
                        "market_prob": market_p,
                        "edge": edge,
                        "decimal_odds": decimal_odds,
                        "expected_roi": expected_roi,
                    }
                )
            # Choose selection
            candidates = []
            for d in details:
                try:
                    if (
                        d["edge"] is not None
                        and d["edge"] >= edge_threshold
                        and d["decimal_odds"]
                        and d.get("model_prob") is not None
                        and float(d.get("model_prob")) >= float(prob_threshold)
                    ):
                        candidates.append(d)
                except Exception:
                    # Be conservative: skip invalid rows rather than failing the whole week
                    continue
            selection = None
            if candidates:
                selection = max(candidates, key=lambda x: x["edge"])
                total_selected += 1
                total_edge_sum += selection["edge"] or 0.0
                if selection.get("expected_roi") is not None:
                    expected_roi_sum += selection["expected_roi"]
                    selection_count += 1
            r["edge_details"] = details
            if selection:
                r["edge_selection"] = {
                    "selected_outcome": selection["outcome"],
                    "edge": selection["edge"],
                    "decimal_odds": selection["decimal_odds"],
                    "expected_roi": selection["expected_roi"],
                }
            else:
                r["edge_selection"] = None
        agg = {
            "edge_threshold": edge_threshold,
            "prob_threshold": prob_threshold,
            "bets_placed": total_selected,
            "avg_edge": (total_edge_sum / total_selected) if total_selected else None,
            "expected_roi_sum": expected_roi_sum if selection_count else None,
            "expected_roi_avg": (expected_roi_sum / selection_count)
            if selection_count
            else None,
        }
        # Attach aggregate to first row for easy retrieval (non-invasive) & return separately if needed
        if rows:
            rows[0]["_edge_aggregate"] = agg
        return agg


week_snapshot_service = WeekSnapshotService()
