import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Deterministic, fast
os.environ.setdefault("ML_SKIP_STARTUP_TRAIN", "1")
os.environ.setdefault("ALLOW_ON_DEMAND_PREDICTIONS", "0")
os.environ.setdefault("DISABLE_PROVIDER_CALLS", "1")

from app.main import app  # noqa: E402
from app.services.odds_csv_store import append_h2h_from_bovada  # noqa: E402

SUPPORTED = ["PL", "BL1", "FL1", "SA", "PD"]


class DummyLeagueService:
    def __init__(self, code: str, matches):
        self.code = code
        self._matches = matches

    def get_all_matches(self):
        return self._matches


@pytest.mark.parametrize("league", SUPPORTED)
def test_all_leagues_week_odds_csv_fallback(league: str, monkeypatch, tmp_path: Path):
    client = TestClient(app)
    # Isolate I/O under tmp
    old = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        dt = "2099-10-05T15:30:00Z"
        home = f"Home {league}"
        away = f"Away {league}"
        matches = [
            {"home_team": home, "away_team": away, "utc_date": dt, "matchday": 7}
        ]

        # Always return dummy service regardless of requested league
        import app.main as main_mod

        monkeypatch.setattr(
            main_mod,
            "get_league_service",
            lambda l=None: DummyLeagueService(league, matches),
        )

        # Write a CSV H2H row for this league
        ev = {
            "home_team": home,
            "away_team": away,
            "commence_time": dt,
            "h2h": {"H": 0.5, "D": 0.28, "A": 0.22},
            "h2h_decimal": {"H": 2.0, "D": 3.6, "A": 4.55},
        }
        wrote = append_h2h_from_bovada(league, [ev], week=7)
        assert wrote >= 3

        # Ensure no TTL collisions from prior iterations
        client.post("/api/admin/week-odds-cache/clear")

        # Call API (cache-only) and verify odds present
        r = client.get(
            f"/api/betting/odds/week/7",
            params={
                "league": league,
                "limit": 1,
                "cache_only": True,
                "future_only": False,
            },
        )
        assert r.status_code == 200
        payload = r.json()
        assert payload.get("league") == league
        assert payload.get("count", 0) >= 1
        m = payload["matches"][0]
        assert m["home_team"] == home and m["away_team"] == away
        odds = m.get("odds") or {}
        # Accept csv-historic (preferred) or closing-snapshot in fallback edges
        assert (odds.get("provider") in {"csv-historic", "closing-snapshot"}) or (
            odds.get("market_odds") is not None
        )
        mw = (odds.get("market_odds") or {}).get("match_winner") or {}
        assert isinstance(mw, dict) and any(
            mw.values()
        ), f"match_winner missing for {league}"
    finally:
        os.chdir(old)
