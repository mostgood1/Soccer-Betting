import json
import os
import sys
import time
from typing import Any, Dict, List

import requests

BASE_URL = os.getenv("SMOKE_BASE_URL", "http://127.0.0.1:8040").rstrip("/")
LEAGUES = os.getenv("SMOKE_LEAGUES", "PL,BL1,FL1,SA,PD").split(",")
TIMEOUT = float(os.getenv("SMOKE_TIMEOUT", "25"))


def get_json(
    path: str, *, params: Dict[str, Any] | None = None, timeout: float | None = None
):
    url = f"{BASE_URL}{path}"
    resp = requests.get(url, params=params, timeout=timeout or TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    results: List[Dict[str, Any]] = []

    # quick health
    try:
        h = get_json("/health", timeout=5)
        status = h.get("status") or h.get("overall_status") or h
        print(f"HEALTH: {status}", file=sys.stderr)
    except Exception as e:
        print(f"HEALTH_ERROR: {e}", file=sys.stderr)

    for lg in [l.strip() for l in LEAGUES if l.strip()]:
        row: Dict[str, Any] = {"league": lg}
        try:
            gw = get_json("/api/game-weeks", params={"league": lg}, timeout=20)
            cw = gw.get("current_week") or gw.get("currentWeek")
            row["week"] = cw
            if not cw:
                row["error"] = "no current_week"
                results.append(row)
                continue

            wk = get_json(f"/api/game-weeks/{cw}", params={"league": lg}, timeout=30)
            matches = wk.get("matches", [])
            row["matches"] = len(matches)
            row["withPredictions"] = sum(1 for m in matches if m.get("predictions"))

            wod = get_json(
                f"/api/betting/odds/week/{cw}", params={"league": lg}, timeout=20
            )
            if isinstance(wod, dict) and "count" in wod:
                row["weekOdds"] = wod["count"]
            elif isinstance(wod, list):
                row["weekOdds"] = len(wod)
            else:
                row["weekOdds"] = (
                    len(wod.get("matches", [])) if isinstance(wod, dict) else 0
                )

            oc = get_json(
                f"/api/game-weeks/{cw}/odds-compare", params={"league": lg}, timeout=25
            )
            row["oddsCompare"] = (
                len(oc.get("matches", [])) if isinstance(oc, dict) else 0
            )

            tc = get_json(
                f"/api/game-weeks/{cw}/totals-compare",
                params={"league": lg, "line": 2.5},
                timeout=25,
            )
            row["totalsCompare"] = (
                len(tc.get("matches", [])) if isinstance(tc, dict) else 0
            )

            cc = get_json(
                f"/api/game-weeks/{cw}/corners-compare",
                params={"league": lg, "line": 9.5},
                timeout=25,
            )
            row["cornersCompare"] = (
                len(cc.get("matches", [])) if isinstance(cc, dict) else 0
            )

            brand = get_json("/api/branding/teams", params={"league": lg}, timeout=10)
            row["brandingCount"] = (
                len(brand) if isinstance(brand, list) else brand.get("count", 0)
            )

        except Exception as e:
            row["error"] = str(e)
        results.append(row)

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
