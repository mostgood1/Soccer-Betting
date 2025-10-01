import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Faster startup in scripts
os.environ.setdefault("ML_SKIP_STARTUP_TRAIN", "1")
os.environ.setdefault("ALLOW_ON_DEMAND_PREDICTIONS", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.week_snapshot_service import week_snapshot_service
from app.services.reconciliation_store import reconciliation_store
from app.services.calibration_service import calibration_service


def main():
    # Heuristic: use the maximum current week across leagues by peeking at reconciled data or default 8
    up_to_week = int(os.getenv("HIST_UP_TO_WEEK", "8"))
    print(f"[hist] Reconciling all leagues up to week {up_to_week} ...")
    recon = week_snapshot_service.reconcile_historic_weeks_all_leagues(up_to_week)
    print(json.dumps({"reconcile": recon}, indent=2))

    print(f"[hist] Calibrating all leagues up to week {up_to_week} ...")
    cal = week_snapshot_service.calibrate_all_leagues_up_to_week(up_to_week)
    print(json.dumps({"calibration": cal}, indent=2))

    # Snapshot reconciliation store
    dump = reconciliation_store.dump()
    print(
        json.dumps(
            {
                "season_stats": dump.get("season_stats"),
                "weekly_stats_keys": list((dump.get("weekly_stats") or {}).keys())[:10],
                "last_updated": dump.get("last_updated"),
                "calibration_status": calibration_service.status(),
                "completed_at": datetime.utcnow().isoformat() + "Z",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
