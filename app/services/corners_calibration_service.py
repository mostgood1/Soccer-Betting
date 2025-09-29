"""Corners calibration and market blending service.

Calibrates a logistic mapping from z = (predicted_mean_corners - line) to
Over probability using historical weeks' actuals. Starting from a configured
threshold week, optionally blend calibrated probability with market Over
probability when available.

Persistence: saves/loads small JSON under data/corners_calibration.json.
"""
from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _default_path() -> str:
    return os.path.join(_repo_root(), "data", "corners_calibration.json")


@dataclass
class CornersCalibrationParams:
    intercept: float = 0.0
    slope: float = 1.0
    z_scale: float = 1.0
    l2_lambda: float = 0.1
    up_to_week: int = 0
    n_samples: int = 0
    threshold_week: int = 7
    blend_weight: float = 0.25
    last_fit_utc: Optional[str] = None


class CornersCalibrationService:
    def __init__(self) -> None:
        self.params = CornersCalibrationParams()
        # Try load defaults
        try:
            self.load()
        except Exception:
            pass

    # --- Core math helpers ---
    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _predict_raw(self, z: float) -> float:
        zt = z / (
            self.params.z_scale
            if self.params.z_scale and self.params.z_scale > 0
            else 1.0
        )
        return self._sigmoid(self.params.intercept + self.params.slope * zt)

    # --- API ---
    def fit(
        self, zs: List[float], ys: List[int], up_to_week: int
    ) -> CornersCalibrationParams:
        """Fit logistic regression with single feature z using Newton-Raphson.
        ys must be binary (0/1). Returns fitted params.
        """
        if not zs or not ys or len(zs) != len(ys):
            # keep defaults
            self.params.up_to_week = up_to_week
            self.params.n_samples = 0
            self.params.last_fit_utc = datetime.utcnow().isoformat()
            return self.params
        # Feature scaling for stability
        mean_z = sum(zs) / len(zs)
        var_z = sum((z - mean_z) ** 2 for z in zs) / len(zs)
        std_z = max(var_z**0.5, 1e-6)
        z_scaled = [(z - mean_z) / std_z for z in zs]
        # Start weights small
        w0 = 0.0
        w1 = 0.5
        lam = self.params.l2_lambda if self.params.l2_lambda is not None else 0.1
        for _ in range(25):
            g0 = 0.0
            g1 = 0.0
            h00 = 0.0
            h01 = 0.0
            h11 = 0.0
            for z, y in zip(z_scaled, ys):
                p = 1.0 / (1.0 + math.exp(-(w0 + w1 * z)))
                r = p * (1 - p)
                g0 += p - y
                g1 += (p - y) * z
                h00 += r
                h01 += r * z
                h11 += r * z * z
            # L2 regularization
            g0 += lam * w0
            g1 += lam * w1
            h00 += lam
            h11 += lam
            # Solve 2x2: H * delta = g
            det = h00 * h11 - h01 * h01
            if abs(det) < 1e-9:
                break
            inv_h00 = h11 / det
            inv_h01 = -h01 / det
            inv_h11 = h00 / det
            d0 = inv_h00 * g0 + inv_h01 * g1
            d1 = inv_h01 * g0 + inv_h11 * g1
            w0 -= d0
            w1 -= d1
            # Optional damping for stability
            if abs(d0) < 1e-6 and abs(d1) < 1e-6:
                break
        # Save params
        self.params.intercept = float(w0)
        self.params.slope = float(w1)
        self.params.z_scale = float(std_z)
        self.params.up_to_week = int(up_to_week)
        self.params.n_samples = int(len(ys))
        self.params.last_fit_utc = datetime.utcnow().isoformat()
        return self.params

    def set_market_blend(self, threshold_week: int, blend_weight: float) -> None:
        self.params.threshold_week = int(threshold_week)
        # Clamp weight to [0, 0.9]
        self.params.blend_weight = max(0.0, min(float(blend_weight), 0.9))

    def predict_over_prob(
        self,
        mu: float,
        line: float,
        week: Optional[int] = None,
        market_over_prob: Optional[float] = None,
    ) -> float:
        z = float(mu) - float(line)
        base = self._predict_raw(z)
        # If week >= threshold and we have market, blend
        if (
            week is not None
            and week >= self.params.threshold_week
            and market_over_prob is not None
        ):
            w = self.params.blend_weight
            return (1.0 - w) * base + w * float(market_over_prob)
        return base

    def save(self, path: Optional[str] = None) -> str:
        path = path or _default_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.params), f, indent=2)
        return path

    def load(self, path: Optional[str] = None) -> CornersCalibrationParams:
        path = path or _default_path()
        if not os.path.exists(path):
            return self.params
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in (data or {}).items():
            if hasattr(self.params, k):
                setattr(self.params, k, v)
        return self.params

    def status(self) -> Dict[str, Any]:
        return asdict(self.params)


# Module singleton
corners_calibration_service = CornersCalibrationService()
