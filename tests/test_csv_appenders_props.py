import os
import csv
from pathlib import Path
from fastapi.testclient import TestClient

# Fast, deterministic env
os.environ.setdefault("ML_SKIP_STARTUP_TRAIN", "1")
os.environ.setdefault("ALLOW_ON_DEMAND_PREDICTIONS", "0")
os.environ.setdefault("DISABLE_PROVIDER_CALLS", "1")

from app.main import app  # noqa: E402
from app.services import odds_csv_store as store  # noqa: E402


def _count_rows(p: Path) -> int:
    if not p.exists():
        return 0
    with p.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        rows = list(r)
    return max(0, len(rows) - 1)  # minus header


def test_extended_market_appenders_write_csvs(monkeypatch, tmp_path: Path):
    # Redirect CSV base dir to tmp
    base = tmp_path / "data" / "odds_history"
    base.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(store, "BASE_DIR", base, raising=True)

    league = "PL"
    dt = "2099-10-10T12:00:00Z"
    ev = {
        "home_team": "Testers",
        "away_team": "Rivals",
        "commence_time": dt,
        "h2h": {"H": 0.4, "D": 0.3, "A": 0.3},
        "h2h_decimal": {"H": 2.5, "D": 3.3, "A": 3.3},
        "btts": {"yes_prob": 0.5, "no_prob": 0.5, "yes_ml": 100, "no_ml": 100},
        "totals": [{"line": 2.5, "over_prob": 0.51, "under_prob": 0.49}],
        "first_half_totals": [{"line": 1.0, "over_prob": 0.55, "under_prob": 0.45}],
        "second_half_totals": [{"line": 1.5, "over_prob": 0.5, "under_prob": 0.5}],
        "team_totals": [{"side": "home", "line": 1.5, "over_prob": 0.52, "under_prob": 0.48}],
        "corners_totals": [{"line": 9.5, "over_prob": 0.52, "under_prob": 0.48}],
        "team_corners": [{"side": "away", "line": 4.5, "over_prob": 0.5, "under_prob": 0.5}],
        "double_chance": {"1X": {"prob": 0.7}, "X2": {"prob": 0.65}, "12": {"prob": 0.8}},
        "draw_no_bet": {"home_prob": 0.6, "away_prob": 0.4},
        "asian_handicap": [{"line": -0.5, "home_prob": 0.55, "away_prob": 0.45}],
        "cards_totals": [{"line": 3.5, "over_prob": 0.5, "under_prob": 0.5}],
        "corners_handicap": [{"line": -1.5, "home_prob": 0.52, "away_prob": 0.48}],
    }

    # Write rows via all appenders
    total = 0
    total += store.append_h2h_from_bovada(league, [ev], week=7)
    total += store.append_btts_from_bovada(league, [ev], week=7)
    total += store.append_totals_from_bovada(league, [ev], week=7)
    total += store.append_first_half_totals_from_bovada(league, [ev], week=7)
    total += store.append_second_half_totals_from_bovada(league, [ev], week=7)
    total += store.append_team_goals_totals_from_bovada(league, [ev], week=7)
    total += store.append_corners_totals_from_bovada(league, [ev], week=7)
    total += store.append_team_corners_from_bovada(league, [ev], week=7)
    total += store.append_double_chance_from_bovada(league, [ev], week=7)
    total += store.append_dnb_from_bovada(league, [ev], week=7)
    total += store.append_asian_handicap_from_bovada(league, [ev], week=7)
    total += store.append_cards_totals_from_bovada(league, [ev], week=7)
    total += store.append_corners_handicap_from_bovada(league, [ev], week=7)

    assert total >= 3 + 2 + 2 + 2 + 2 + 2 + 2 + 3 + 2 + 2 + 2 + 2  # h2h + btts + many OU/dc/dnb/ah/cards/corners

    # Verify files exist and have rows
    def p(market: str) -> Path:
        return base / f"{market}_{league}.csv"

    expected = {
        "h2h": 3,
        "btts": 2,
        "totals": 2,
        "first_half_totals": 2,
        "second_half_totals": 2,
        "team_goals_totals": 2,
        "corners_totals": 2,
        "team_corners_totals": 2,
        "double_chance": 3,
        "dnb": 2,
        "asian_handicap": 2,
        "cards_totals": 2,
        "corners_handicap": 2,
    }

    for market, min_rows in expected.items():
        path = p(market)
        assert path.exists(), f"CSV not created: {path}"
        rows = _count_rows(path)
        assert rows >= min_rows, f"Expected at least {min_rows} rows in {path}, got {rows}"
