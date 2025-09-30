import os
import sys
from pathlib import Path
import json
from typing import Any, Dict, List

# Ensure fast startup and consistent policy for this probe
os.environ.setdefault("ML_SKIP_STARTUP_TRAIN", "1")
os.environ.setdefault("ALLOW_ON_DEMAND_PREDICTIONS", "0")

# Ensure project root is on sys.path so `import app` works regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)

LEAGUES = ["PL", "BL1", "FL1", "SA", "PD"]
SMOKE_FULL = os.getenv("SMOKE_FULL", "0") == "1"


def get(path: str, params: Dict[str, Any] | None = None):
    r = client.get(path, params=params)
    r.raise_for_status()
    return r.json()


def main():
    out: List[Dict[str, Any]] = []
    # Health
    h = get("/health")
    print(f"HEALTH: {h.get('status')}")

    for lg in LEAGUES:
        row: Dict[str, Any] = {"league": lg}
        try:
            gw = get("/api/game-weeks", params={"league": lg})
            cw = gw.get("current_week") or gw.get("currentWeek")
            row["week"] = cw
            if not cw:
                row["error"] = "no current_week"
                out.append(row)
                continue

            wk = get(f"/api/game-weeks/{cw}", params={"league": lg})
            matches = wk.get("matches", [])
            row["matches"] = len(matches)
            row["withPredictions"] = sum(1 for m in matches if m.get("predictions"))

            if SMOKE_FULL:
                wod = get(f"/api/betting/odds/week/{cw}", params={"league": lg})
                if isinstance(wod, dict) and "count" in wod:
                    row["weekOdds"] = wod["count"]
                elif isinstance(wod, list):
                    row["weekOdds"] = len(wod)
                else:
                    row["weekOdds"] = len(wod.get("matches", [])) if isinstance(wod, dict) else 0

                oc = get(f"/api/game-weeks/{cw}/odds-compare", params={"league": lg})
                row["oddsCompare"] = len(oc.get("matches", [])) if isinstance(oc, dict) else 0

                tc = get(
                    f"/api/game-weeks/{cw}/totals-compare",
                    params={"league": lg, "line": 2.5},
                )
                row["totalsCompare"] = len(tc.get("matches", [])) if isinstance(tc, dict) else 0

                cc = get(
                    f"/api/game-weeks/{cw}/corners-compare",
                    params={"league": lg, "line": 9.5},
                )
                row["cornersCompare"] = len(cc.get("matches", [])) if isinstance(cc, dict) else 0

            brand = get("/api/branding/teams", params={"league": lg})
            row["brandingCount"] = (len(brand) if isinstance(brand, list) else brand.get("count", 0))
        except Exception as e:
            row["error"] = str(e)
        out.append(row)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
