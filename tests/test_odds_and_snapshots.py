import os
from pathlib import Path
from fastapi.testclient import TestClient

# Make app load fast in tests
os.environ.setdefault("ML_SKIP_STARTUP_TRAIN", "1")
os.environ.setdefault("ALLOW_ON_DEMAND_PREDICTIONS", "0")

from app.main import app  # noqa: E402
from app.services.betting_odds_service import BettingOddsService  # noqa: E402


def test_decimal_to_american():
    svc = BettingOddsService()
    # Favorite example: 1.67 decimal ~ -149 American
    assert abs(svc._decimal_to_american(1.67) + 149) <= 2
    # Underdog example: 2.4 decimal ~ +140 American
    assert abs(svc._decimal_to_american(2.4) - 140) <= 2
    # Edge cases
    assert svc._decimal_to_american(1.0) == 0
    assert svc._decimal_to_american(0.0) == 0


def test_odds_snapshot_listing(tmp_path: Path, monkeypatch):
    client = TestClient(app)
    # Create a temp odds_snapshots dir with a couple files
    base = tmp_path / "odds_snapshots"
    base.mkdir(parents=True, exist_ok=True)
    (base / "week_5_odds_snapshot_20250926T000000Z.json").write_text("{}", encoding="utf-8")
    (base / "week_6_odds_snapshot_20250926T010000Z.json").write_text("{}", encoding="utf-8")

    # Point app to temp dir by monkeypatching Path in this module's scope
    # Simple approach: change CWD for this test so relative Path("data/odds_snapshots") resolves under tmp
    data_dir = tmp_path
    # create 'data' folder and move the snapshots inside proper structure
    real_data_dir = data_dir / "data"
    real_data_dir.mkdir(exist_ok=True)
    for f in base.iterdir():
        # Move files under data/odds_snapshots
        dest_dir = real_data_dir / "odds_snapshots"
        dest_dir.mkdir(parents=True, exist_ok=True)
        f.rename(dest_dir / f.name)

    # Run listing with temporary CWD
    old_cwd = os.getcwd()
    os.chdir(str(data_dir))
    try:
        r = client.get("/api/admin/odds-snapshots")
        assert r.status_code == 200
        files = r.json().get("files", [])
        assert len(files) == 2

        r5 = client.get("/api/admin/odds-snapshots?week=5")
        assert r5.status_code == 200
        only5 = r5.json().get("files", [])
        assert len(only5) == 1
        assert only5[0]["week"] == 5

        # Fetch one file's contents
        fname = only5[0]["name"]
        rf = client.get(f"/api/admin/odds-snapshots/{fname}")
        assert rf.status_code == 200
        assert rf.json().get("file") == fname
    finally:
        os.chdir(old_cwd)
