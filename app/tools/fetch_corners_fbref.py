"""
FBref Corners Importer
----------------------
Attempts to fetch real corners actuals from FBref via the soccerdata library
for specified league game weeks (PL, BL1, FL1, SA, PD), keyed by date/home/away,
and writes them to data/corners_actuals_2025_26.csv for consumption by
corners_actuals_store.

Usage:
        python -m app.tools.fetch_corners_fbref --league BL1 --weeks 1,2,3 \
            [--out data/corners_actuals_2025_26.csv] [--use-fbref]

Notes:
 - Primary path (offline-friendly): reads local Football-Data.co.uk CSVs that
     include HC (home corners) and AC (away corners) columns if present under
     data/ or data/odds.
 - Optional path (--use-fbref): FBref table shapes can vary. We search across
     several team match stat tables for columns containing 'corner'. Requires
     network access and may be slow/blocked in some environments.
 - We map matches using EnhancedEPLService fixtures (date/home/away) to team
   match rows (team vs opponent at date). Team names are normalized.
 - Existing CSV entries are preserved and updated when overlapping keys.
"""
from __future__ import annotations
import argparse
import csv
import os
from typing import Dict, Any, Iterable, Optional, Tuple, List

import pandas as pd

try:
    import soccerdata as sd
except Exception as e:  # pragma: no cover
    sd = None  # type: ignore

from app.services.league_manager import (
    get_service as get_league_service,
    normalize_league_code,
)
from app.services.game_week_service import game_week_service
from app.services.team_name_normalizer import normalize_team_name


def _read_existing_csv(path: str) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return idx
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = _key(row.get("date"), row.get("home_team"), row.get("away_team"))
                if key:
                    idx[key] = row
    except Exception:
        pass
    return idx


def _key(
    date: Optional[str], home: Optional[str], away: Optional[str]
) -> Optional[str]:
    if not (date and home and away):
        return None
    d = (date.split("T", 1)[0]).strip()
    h = (normalize_team_name(home) or str(home)).lower()
    a = (normalize_team_name(away) or str(away)).lower()
    return f"{d}|{h}|{a}"


def _coerce_int(v: Any) -> Optional[int]:
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return None


def _find_corner_column(df: pd.DataFrame) -> Optional[str]:
    # Look for a column containing 'corner' substring
    for c in df.columns:
        cl = str(c).lower()
        if "corner" in cl:
            return c
    return None


def _gather_fbref_corners() -> Optional[pd.DataFrame]:
    if sd is None:
        return None
    fb = sd.FBref()
    candidates: List[pd.DataFrame] = []
    stat_types = ["summary", "standard", "misc"]
    for st in stat_types:
        try:
            df = fb.read_team_match_stats(stat_type=st)
            if isinstance(df, pd.DataFrame) and not df.empty:
                corner_col = _find_corner_column(df)
                # Require the essentials we need to match rows
                has_keys = all(k in df.columns for k in ["date", "team", "opponent"])
                if corner_col and has_keys:
                    candidates.append(df)
        except Exception:
            continue
    if not candidates:
        return None
    # Prefer the first usable candidate
    return candidates[0]


def _filter_by_competition(df: pd.DataFrame, league_code: str) -> pd.DataFrame:
    """Best-effort filter of FBref team-match stats for a specific league.

    We search for a column that identifies competition/league/country and filter by
    known names per league code. If no such column is found, return df unchanged.
    """
    try:
        code = normalize_league_code(league_code)
    except Exception:
        code = league_code
    comp_map = {
        "PL": ["Premier League", "England"],
        "BL1": ["Bundesliga", "Germany"],
        "FL1": ["Ligue 1", "France"],
        "SA": ["Serie A", "Italy"],
        "PD": ["La Liga", "Spain", "Primera Division"],
    }
    synonyms = comp_map.get(code, [])
    if not isinstance(df, pd.DataFrame) or df.empty or not synonyms:
        return df
    # Find a likely competition column
    cols = list(df.columns)
    cand_cols = [
        c
        for c in cols
        if any(
            k in str(c).lower() for k in ["comp", "league", "competition", "country"]
        )
    ]
    if not cand_cols:
        return df
    # OR-filter rows containing any synonym in any candidate column
    mask = pd.Series([False] * len(df))
    for c in cand_cols:
        try:
            ser = df[c].astype(str).str.lower()
            for s in synonyms:
                mask = mask | ser.str.contains(str(s).lower(), na=False)
        except Exception:
            continue
    try:
        out = df.loc[mask]
        return out if not out.empty else df
    except Exception:
        return df


def _scan_local_football_data_csvs() -> List[str]:
    roots = [os.path.join("data"), os.path.join("data", "odds")]
    found: List[str] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if not name.lower().endswith(".csv"):
                continue
            path = os.path.join(root, name)
            try:
                # Quick header sniff without loading full file
                head = pd.read_csv(path, nrows=1)
                cols = {c.lower() for c in head.columns}
                if {"date", "hometeam", "awayteam", "hc", "ac"}.issubset(cols):
                    found.append(path)
            except Exception:
                continue
    return found


def _lookup_from_football_data(
    df: pd.DataFrame, date: str, home: str, away: str
) -> Optional[Tuple[int, int, int]]:
    # Normalize schema and attempt to match
    try:
        # Football-Data CSVs typically use day-first format (e.g., 15/08/2025)
        # Make this explicit to avoid pandas warnings and accidental mis-parsing
        d = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True).dt.strftime(
            "%Y-%m-%d"
        )
    except Exception:
        d = df["Date"].astype(str).str.slice(0, 10)
    ht = df["HomeTeam"].astype(str)
    at = df["AwayTeam"].astype(str)
    mask = (
        (d == date.split("T", 1)[0])
        & (ht.str.lower() == (normalize_team_name(home) or home).lower())
        & (at.str.lower() == (normalize_team_name(away) or away).lower())
    )
    rows = df.loc[mask]
    if rows.empty:
        return None
    hc = _coerce_int(rows.iloc[0].get("HC"))
    ac = _coerce_int(rows.iloc[0].get("AC"))
    if hc is None or ac is None:
        return None
    return hc, ac, hc + ac


def _lookup_match_corners(
    df: pd.DataFrame, date: str, home: str, away: str
) -> Optional[Tuple[int, int, int]]:
    corner_col = _find_corner_column(df)
    if not corner_col:
        return None
    # Normalize values for matching
    date_only = date.split("T", 1)[0]
    h = normalize_team_name(home) or home
    a = normalize_team_name(away) or away
    # FBref sometimes stores dates as datetime; coerce to string for compare
    df_local = df.copy()
    try:
        # Normalize to YYYY-MM-DD; local sources may also be day-first
        df_local["__date__"] = pd.to_datetime(
            df_local["date"], dayfirst=True, errors="coerce"
        ).dt.strftime("%Y-%m-%d")
    except Exception:
        df_local["__date__"] = df_local["date"].astype(str).str.slice(0, 10)
    df_local["__team__"] = df_local["team"].astype(str)
    df_local["__opp__"] = df_local["opponent"].astype(str)

    # Rows where team=home and opponent=away at date
    mask_home = (
        (df_local["__date__"] == date_only)
        & (df_local["__team__"].str.lower() == h.lower())
        & (df_local["__opp__"].str.lower() == a.lower())
    )
    # Rows where team=away and opponent=home at date
    mask_away = (
        (df_local["__date__"] == date_only)
        & (df_local["__team__"].str.lower() == a.lower())
        & (df_local["__opp__"].str.lower() == h.lower())
    )
    row_h = df_local.loc[mask_home]
    row_a = df_local.loc[mask_away]
    if row_h.empty or row_a.empty:
        return None
    hc = _coerce_int(row_h.iloc[0][corner_col])
    ac = _coerce_int(row_a.iloc[0][corner_col])
    if hc is None or ac is None:
        return None
    return hc, ac, hc + ac


def import_weeks(
    weeks: Iterable[int], out_path: str, use_fbref: bool = False, league: str = "PL"
) -> Dict[str, Any]:
    code = normalize_league_code(league)
    svc = get_league_service(code)
    # Prefer get_all_matches if available; else fallback to EPL service signature
    matches = (
        svc.get_all_matches() if hasattr(svc, "get_all_matches") else svc.get_matches()
    )
    weeks_map = game_week_service.organize_matches_by_week(matches)
    # Try local Football-Data CSVs first (offline-friendly)
    local_csvs = _scan_local_football_data_csvs()
    fbref_df = None
    if use_fbref:
        fbref_df = _gather_fbref_corners()
        if fbref_df is not None:
            fbref_df = _filter_by_competition(fbref_df, code)

    existing = _read_existing_csv(out_path)
    written = 0
    considered = 0
    for w in weeks:
        wk = weeks_map.get(int(w), [])
        for m in wk:
            considered += 1
            date = m.get("utc_date") or m.get("date")
            home = m.get("home_team") or m.get("homeTeam")
            away = m.get("away_team") or m.get("awayTeam")
            key = _key(date, home, away)
            if not key:
                continue
            trip = None
            # Attempt local CSVs
            for csv_path in local_csvs:
                try:
                    df = pd.read_csv(csv_path)
                    # Normalize column names to expected case
                    df = df.rename(columns={c: c.strip() for c in df.columns})
                    if all(
                        col in df.columns
                        for col in ["Date", "HomeTeam", "AwayTeam", "HC", "AC"]
                    ):
                        trip = _lookup_from_football_data(
                            df, str(date), str(home), str(away)
                        )
                        if trip is not None:
                            break
                except Exception:
                    continue
            # Fallback to FBref if requested
            if trip is None and fbref_df is not None:
                trip = _lookup_match_corners(fbref_df, str(date), str(home), str(away))
            if trip is None:
                continue
            hc, ac, tc = trip
            existing[key] = {
                "date": str(date).split("T", 1)[0],
                "home_team": home,
                "away_team": away,
                "home_corners": str(hc),
                "away_corners": str(ac),
                "total_corners": str(tc),
            }
            written += 1

    # Write back out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "date",
            "home_team",
            "away_team",
            "home_corners",
            "away_corners",
            "total_corners",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in existing.values():
            writer.writerow(rec)

    return {
        "success": True,
        "weeks": list(weeks),
        "considered": considered,
        "wrote": written,
        "league": code,
        "out": out_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Import corners actuals from FBref via soccerdata"
    )
    parser.add_argument(
        "--league",
        type=str,
        default="PL",
        help="League code: PL, BL1, FL1, SA, PD (default: PL)",
    )
    parser.add_argument(
        "--weeks",
        type=str,
        default="1,2,3",
        help="Comma-separated game weeks (e.g., 1,2,3)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join("data", "corners_actuals_2025_26.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--use-fbref",
        action="store_true",
        help="Allow FBref network fetch if local CSVs are unavailable",
    )
    args = parser.parse_args()
    # Support ranges like 1-5,7,9-10
    raw = [s.strip() for s in str(args.weeks).split(",") if s.strip()]
    weeks_list: List[int] = []
    for token in raw:
        if "-" in token:
            try:
                a, b = token.split("-", 1)
                a_i = int(a)
                b_i = int(b)
                if a_i <= b_i:
                    weeks_list.extend(list(range(a_i, b_i + 1)))
                else:
                    weeks_list.extend(list(range(b_i, a_i + 1)))
            except Exception:
                continue
        else:
            try:
                weeks_list.append(int(token))
            except Exception:
                continue
    weeks = sorted(set(weeks_list))
    result = import_weeks(
        weeks, args.out, use_fbref=bool(args.use_fbref), league=str(args.league)
    )
    print(result)


if __name__ == "__main__":
    main()
