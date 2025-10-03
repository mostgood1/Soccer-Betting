import os
import json
from pathlib import Path
from fastapi.testclient import TestClient

# Speed up and make deterministic
os.environ.setdefault("ML_SKIP_STARTUP_TRAIN", "1")
os.environ.setdefault("ALLOW_ON_DEMAND_PREDICTIONS", "0")
os.environ.setdefault("DISABLE_PROVIDER_CALLS", "1")

from app.main import app  # noqa: E402
from app.services.odds_csv_store import append_h2h_from_bovada  # noqa: E402


class DummyLeagueService:
    code = "PL"

    def __init__(self, matches):
        self._matches = matches

    def get_all_matches(self):
        return self._matches


def test_week_odds_uses_csv_fallback(monkeypatch, tmp_path: Path):
    client = TestClient(app)

    # Isolate all file I/O to temp dir so CSVs are written under tmp/data/odds_history
    old_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        # Create a single Week 7 fixture that our CSV row will match
        dt = "2099-10-04T14:00:00Z"  # far future to bypass future_only filtering
        matches = [
            {
                "home_team": "Test FC",
                "away_team": "Rail Club",
                "utc_date": dt,
                "matchday": 7,
            }
        ]

        # Patch get_league_service so the API uses our dummy service
        import app.main as main_mod

        monkeypatch.setattr(main_mod, "get_league_service", lambda league=None: DummyLeagueService(matches))

        # Write a matching H2H CSV row via helper using a synthetic Bovada-like event
        ev = {
            "home_team": "Test FC",
            "away_team": "Rail Club",
            "commence_time": dt,
            "h2h": {"H": 0.55, "D": 0.25, "A": 0.20},
            # Optional decimals (will be preferred if present)
            "h2h_decimal": {"H": 1.82, "D": 4.0, "A": 5.0},
        }
        rows = append_h2h_from_bovada("PL", [ev], week=7)
        assert rows >= 3  # three outcomes

        # Call week odds endpoint in cache-only mode; using future_only=false to avoid time filters
        r = client.get("/api/betting/odds/week/7", params={"league": "PL", "limit": 1, "cache_only": True, "future_only": False})
        assert r.status_code == 200
        payload = r.json()
        assert payload.get("week") == 7
        assert payload.get("league") == "PL"
        assert payload.get("count", 0) >= 1
        m = payload["matches"][0]
        assert m["home_team"] == "Test FC"
        assert m["away_team"] == "Rail Club"
        odds = m.get("odds") or {}
        # Provider should be csv-historic (fallback path)
        assert odds.get("provider") in {"csv-historic", "closing-snapshot"}
        mw = ((odds.get("market_odds") or {}).get("match_winner") or {})
        assert isinstance(mw, dict) and any(mw.values()), "match_winner odds missing"
        # Check at least one decimal odds exists (from our CSV)
        assert (mw.get("home") or {}).get("odds")
    finally:
        os.chdir(old_cwd)
