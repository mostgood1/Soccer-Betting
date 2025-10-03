"""
Generate manual goals totals market lines for a given week and leagues (default PL, PD)
from local Football-Data fixtures. This is a fallback to enable expanded markets
when providers do not return totals. It appends to data/goals_market_lines_manual.json
and avoids duplicating existing records.

Schema per record:
  {
    "date": ISO datetime (from fixture utcDate),
    "home_team": str,
    "away_team": str,
    "line": float,         # goals line
    "over_odds": float,    # decimal odds
    "under_odds": float,   # decimal odds
    "bookmaker": "manual"
  }

Usage (example):
  python -m scripts.generate_manual_goals_lines --week 7 --leagues PL,PD --line 2.5 --over 1.91 --under 1.91
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


FILE_MAP = {
    "PL": "football_data_PL_2025_2026.json",
    "BL1": "football_data_BL1_2025_2026.json",
    "FL1": "football_data_FL1_2025_2026.json",
    "SA": "football_data_SA_2025_2026.json",
    "PD": "football_data_PD_2025_2026.json",
}


def load_fixtures(league: str) -> List[Dict[str, Any]]:
    path = Path("data") / FILE_MAP[league]
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    # Football-Data style payload usually has matches
    if isinstance(data, dict) and isinstance(data.get("matches"), list):
        return data["matches"]
    # Or it could be a flat list of matches
    if isinstance(data, list):
        return data
    return []


def extract_week_matches(
    fixtures: List[Dict[str, Any]], week: int
) -> List[Tuple[str, str, str]]:
    """Return list of tuples (utc_date, home_name, away_name) for given matchday/week."""
    out: List[Tuple[str, str, str]] = []
    for m in fixtures:
        try:
            md = m.get("matchday") or m.get("game_week") or m.get("round")
            # Some payloads use strings like "REGULAR_SEASON - 7"
            if isinstance(md, str):
                # Try to extract the last integer in the string
                import re

                nums = re.findall(r"\d+", md)
                md_val = int(nums[-1]) if nums else None
            else:
                md_val = int(md) if md is not None else None
            if md_val != int(week):
                continue
            utc_date = m.get("utcDate") or m.get("utc_date") or m.get("date")
            home_name = (
                (m.get("homeTeam") or {}).get("name")
                if isinstance(m.get("homeTeam"), dict)
                else m.get("home_team") or m.get("home")
            )
            away_name = (
                (m.get("awayTeam") or {}).get("name")
                if isinstance(m.get("awayTeam"), dict)
                else m.get("away_team") or m.get("away")
            )
            if utc_date and home_name and away_name:
                out.append((str(utc_date), str(home_name), str(away_name)))
        except Exception:
            continue
    return out


def load_existing(out_path: Path) -> List[Dict[str, Any]]:
    if not out_path.exists():
        return []
    try:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            return [r for r in payload["records"] if isinstance(r, dict)]
    except Exception:
        return []
    return []


def record_key(r: Dict[str, Any]) -> Tuple[str, str, str, float]:
    date = str(r.get("date"))[:19]
    home = str(r.get("home_team") or r.get("home") or "").lower()
    away = str(r.get("away_team") or r.get("away") or "").lower()
    try:
        line = float(r.get("line")) if r.get("line") is not None else None
    except Exception:
        line = None
    return date, home, away, float(line) if isinstance(line, (int, float)) else 0.0


def upsert_records(
    out_path: Path, new_records: List[Dict[str, Any]]
) -> Tuple[int, int]:
    existing = load_existing(out_path)
    existing_keys = {record_key(r) for r in existing}
    merged = list(existing)
    added = 0
    for r in new_records:
        k = record_key(r)
        if k in existing_keys:
            continue
        merged.append(r)
        existing_keys.add(k)
        added += 1
    out_payload = {"records": merged, "source": "manual-script"}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    return added, len(merged)


def main():
    parser = argparse.ArgumentParser(
        description="Generate manual goals totals lines from fixtures"
    )
    parser.add_argument("--week", type=int, default=7, help="Matchday/week number")
    parser.add_argument(
        "--leagues",
        type=str,
        default="PL,PD",
        help="Comma-separated league codes (PL,BL1,FL1,SA,PD)",
    )
    parser.add_argument("--line", type=float, default=2.5, help="Goals line to apply")
    parser.add_argument("--over", type=float, default=1.91, help="Over decimal odds")
    parser.add_argument("--under", type=float, default=1.91, help="Under decimal odds")
    args = parser.parse_args()

    leagues = [c.strip().upper() for c in args.leagues.split(",") if c.strip()]
    out_path = Path("data/goals_market_lines_manual.json")
    to_add: List[Dict[str, Any]] = []
    for lg in leagues:
        if lg not in FILE_MAP:
            continue
        fixtures = load_fixtures(lg)
        week_matches = extract_week_matches(fixtures, args.week)
        for date, home, away in week_matches:
            try:
                to_add.append(
                    {
                        "date": date,
                        "home_team": home,
                        "away_team": away,
                        "line": float(args.line),
                        "over_odds": float(args.over),
                        "under_odds": float(args.under),
                        "bookmaker": "manual",
                    }
                )
            except Exception:
                continue
    added, total = upsert_records(out_path, to_add)
    print(
        json.dumps(
            {
                "success": True,
                "added": added,
                "total_records": total,
                "output": str(out_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
