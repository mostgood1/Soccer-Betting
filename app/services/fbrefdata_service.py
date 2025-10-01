"""
FBrefData integration service.

Provides helper functions to backfill matches (schedule/results) and player stats
from the fbrefdata package as our primary historical source when available.

All network calls are optional and gated. In tests or environments without the
package installed, the service remains inactive (is_available=False).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


def _season_string_for_today(today: Optional[datetime] = None) -> str:
    d = today or datetime.utcnow()
    year = d.year
    # Season starts in Aug; if month < Aug, season started previous year
    start_year = year if d.month >= 8 else year - 1
    return f"{start_year}-{start_year+1}"


@dataclass
class BackfillResult:
    league: str
    season: str
    schedule_rows: int = 0
    player_rows: int = 0
    artifacts: List[str] = None
    notes: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "league": self.league,
            "season": self.season,
            "schedule_rows": self.schedule_rows,
            "player_rows": self.player_rows,
            "artifacts": self.artifacts or [],
            "notes": self.notes or [],
        }


class FbrefDataService:
    def __init__(self) -> None:
        self._fd = None
        self.is_available = False
        try:
            # Import lazily to keep test startup fast
            import fbrefdata as fd  # type: ignore

            self._fd = fd
            self.is_available = True
        except Exception:
            self._fd = None
            self.is_available = False

        # Base folder for persisted artifacts
        self.base_dir = Path(os.getenv("FBREFDATA_BASE_DIR", "data/fbref"))
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def map_league_code(league: str) -> Tuple[str, List[str]]:
        code = (league or "PL").upper()
        # Default competition name and tokens for filtering when using combined dataset
        mapping: Dict[str, Tuple[str, List[str]]] = {
            "PL": ("ENG-Premier League", ["Premier League"]),
            "BL1": ("GER-Bundesliga", ["Bundesliga"]),
            "FL1": ("FRA-Ligue 1", ["Ligue 1"]),
            "SA": ("ITA-Serie A", ["Serie A"]),
            "PD": ("ESP-La Liga", ["La Liga", "Primera División"]),
        }
        return mapping.get(code, ("ENG-Premier League", ["Premier League"]))

    @staticmethod
    def _filter_competition(df, tokens: List[str]):
        try:
            import pandas as pd  # type: ignore

            if df is None or df.empty:
                return df
            cols = [
                c
                for c in ["Comp", "Competition", "League", "comp", "competition"]
                if c in df.columns
            ]
            if not cols:
                return df
            mask = None
            for c in cols:
                series = df[c].astype(str).str.lower()
                submask = None
                for tok in tokens:
                    t = (tok or "").lower()
                    if not t:
                        continue
                    m = series.str.contains(t, na=False)
                    submask = m if submask is None else (submask | m)
                mask = submask if mask is None else (mask | submask)
            if mask is None:
                return df
            filtered = df[mask]
            # Fall back to original if we filtered out everything unexpectedly
            return filtered if not filtered.empty else df
        except Exception:
            return df

    def _ensure_dir(self, league: str, season: str) -> Path:
        comp = league.upper()
        out = self.base_dir / comp / season
        out.mkdir(parents=True, exist_ok=True)
        return out

    def backfill(
        self,
        league: str,
        season: Optional[str] = None,
        include_schedule: bool = True,
        include_players: bool = True,
    ) -> BackfillResult:
        """Fetch and persist schedule and player stats for a league/season.

        - Writes CSVs under data/fbref/{LEAGUE}/{SEASON}/
        - Returns counts and artifact paths
        - Safe no-op when fbrefdata is not available
        """
        season_str = season or _season_string_for_today()
        res = BackfillResult(league=league.upper(), season=season_str, artifacts=[], notes=[])
        # Lazy import if not available (handles runtime installs without restart)
        if not self.is_available or self._fd is None:
            try:
                import fbrefdata as fd  # type: ignore

                self._fd = fd
                self.is_available = True
            except Exception as ie:
                self._fd = None
                self.is_available = False
                # record why import failed
                if hasattr(ie, "__class__"):
                    res.notes.append(f"fbrefdata import error: {ie.__class__.__name__}: {ie}")
        if not self.is_available:
            res.notes.append("fbrefdata not installed or unavailable; skipped")
            return res

        comp, tokens = self.map_league_code(league)
        out_dir = self._ensure_dir(league, season_str)

        # Inject browser-like headers to reduce likelihood of 403 blocks (best-effort)
        _requests_patched = False
        try:
            import requests  # type: ignore
            orig_request = requests.sessions.Session.request

            def _patched_request(self_sess, method, url, **kwargs):  # type: ignore
                hdrs = kwargs.get("headers") or {}
                # Only set if not explicitly provided
                hdrs.setdefault(
                    "User-Agent",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
                )
                hdrs.setdefault("Accept-Language", "en-US,en;q=0.9")
                hdrs.setdefault(
                    "Accept",
                    "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                )
                # fbref often requires a referer from its own domain
                if isinstance(url, str) and url.startswith("https://fbref.com/"):
                    hdrs.setdefault("Referer", "https://fbref.com/en/")
                kwargs["headers"] = hdrs
                return orig_request(self_sess, method, url, **kwargs)

            requests.sessions.Session.request = _patched_request  # type: ignore
            _requests_patched = True
        except Exception:
            requests = None  # type: ignore
            orig_request = None  # type: ignore

        try:
            # Disable provider calls in tests when requested
            if os.getenv("DISABLE_PROVIDER_CALLS", "0") == "1":
                res.notes.append("DISABLE_PROVIDER_CALLS=1; skipped network fetch")
                return res

            try:
                fb = self._fd.FBref(comp, season_str)  # type: ignore[attr-defined]
                comp_used = comp
            except Exception as e:
                # Fallback to combined dataset and filter later
                try:
                    fb = self._fd.FBref("Big 5 European Leagues Combined", season_str)  # type: ignore[attr-defined]
                    comp_used = "Big 5 European Leagues Combined"
                    res.notes.append(
                        f"Init fallback to combined dataset due to: {e}"
                    )
                except Exception as ee:
                    res.notes.append(f"Failed to init FBrefData: {e}\n{ee}")
                    return res
            # Schedule / results
            if include_schedule:
                try:
                    sch = fb.read_schedule()  # pandas DataFrame
                    # If using combined dataset or as safety, filter by competition tokens
                    sch = self._filter_competition(sch, tokens)
                    if sch is not None:
                        path = out_dir / "schedule.csv"
                        sch.to_csv(path, index=False)
                        res.schedule_rows = int(getattr(sch, "shape", [0, 0])[0])
                        res.artifacts.append(str(path))
                except Exception as e:
                    res.notes.append(f"schedule failed: {e}")
                    try:
                        # Fallback: build schedule from baked Football-Data fixtures
                        built = self._fallback_schedule_from_baked(league, season_str, out_dir)
                        if built > 0:
                            res.schedule_rows = built
                            res.notes.append("schedule fallback: baked football-data")
                    except Exception as ee:
                        res.notes.append(f"schedule fallback failed: {ee}")

            # Player stats (standard + shooting as a minimum)
            if include_players:
                total_rows = 0
                try:
                    try:
                        std = fb.read_player_standard_stats()
                    except Exception:
                        std = None
                    if std is not None:
                        std = self._filter_competition(std, tokens)
                        p = out_dir / "players_standard.csv"
                        std.to_csv(p, index=False)
                        total_rows += int(getattr(std, "shape", [0, 0])[0])
                        res.artifacts.append(str(p))
                except Exception as e:
                    res.notes.append(f"player_standard failed: {e}")

                try:
                    try:
                        shooting = fb.read_player_shooting()
                    except Exception:
                        shooting = None
                    if shooting is not None:
                        shooting = self._filter_competition(shooting, tokens)
                        p = out_dir / "players_shooting.csv"
                        shooting.to_csv(p, index=False)
                        total_rows += int(getattr(shooting, "shape", [0, 0])[0])
                        res.artifacts.append(str(p))
                except Exception as e:
                    res.notes.append(f"player_shooting failed: {e}")

                # Optional blocks that may not be supported for all comps
                for method, fname in [
                    ("read_player_passing", "players_passing.csv"),
                    ("read_player_misc", "players_misc.csv"),
                    ("read_player_defense", "players_defense.csv"),
                    ("read_goalkeeping", "goalkeeping.csv"),
                ]:
                    try:
                        if hasattr(fb, method):
                            df = getattr(fb, method)()
                            if df is not None:
                                df = self._filter_competition(df, tokens)
                                p = out_dir / fname
                                df.to_csv(p, index=False)
                                total_rows += int(getattr(df, "shape", [0, 0])[0])
                                res.artifacts.append(str(p))
                    except Exception:
                        # best-effort; ignore
                        pass

                res.player_rows = total_rows

        finally:
            # Restore original request method
            if _requests_patched:
                try:
                    requests.sessions.Session.request = orig_request  # type: ignore
                except Exception:
                    pass
        return res

    def _fallback_schedule_from_baked(self, league: str, season: str, out_dir: Path) -> int:
        """Write schedule.csv using baked Football-Data JSON embedded in the image.

        This ensures artifacts exist even if fbref scraping is blocked. The columns
        are a best-effort schedule representation.
        """
        import json
        import pandas as pd  # type: ignore

        code = (league or "PL").upper()
        # Map season like 2025-2026 to baked file suffix 2025_2026
        suffix = season.replace("-", "_")
        name_map = {
            "PL": f"/app/baked/football_data_PL_{suffix}.json",
            "BL1": f"/app/baked/football_data_BL1_{suffix}.json",
            "FL1": f"/app/baked/football_data_FL1_{suffix}.json",
            "SA": f"/app/baked/football_data_SA_{suffix}.json",
            "PD": f"/app/baked/football_data_PD_{suffix}.json",
        }
        path = name_map.get(code)
        if not path or not Path(path).exists():
            # Also try non-code EPL filenames for backward compat
            alt = f"/app/baked/football_data_epl_{suffix}.json"
            if code == "PL" and Path(alt).exists():
                path = alt
            else:
                raise FileNotFoundError(f"no baked file for {league} {season}")
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        # Expect either a list of fixtures or an object with key 'fixtures'/'matches'
        fixtures = []
        if isinstance(raw, list):
            fixtures = raw
        elif isinstance(raw, dict):
            fixtures = (
                raw.get("fixtures")
                or raw.get("matches")
                or raw.get("records")
                or []
            )
        rows = []
        for m in fixtures:
            try:
                date = m.get("utcDate") or m.get("date") or m.get("commence_time")
                if not date and m.get("start_time_ms"):
                    try:
                        ts = int(m.get("start_time_ms")) / 1000.0
                        date = datetime.utcfromtimestamp(ts).isoformat() + "Z"
                    except Exception:
                        date = None
                home = (
                    m.get("homeTeam") or m.get("home_team") or (m.get("home") or {}).get("name")
                )
                away = (
                    m.get("awayTeam") or m.get("away_team") or (m.get("away") or {}).get("name")
                )
                if not (home and away and date):
                    continue
                rows.append(
                    {
                        "date": date,
                        "home_team": home,
                        "away_team": away,
                        "league": code,
                        "season": season,
                    }
                )
            except Exception:
                continue
        if not rows:
            return 0
        df = pd.DataFrame(rows)
        out = out_dir / "schedule.csv"
        df.to_csv(out, index=False)
        return int(df.shape[0])

    def ingest_local(
        self,
        league: str,
        season: Optional[str] = None,
    ) -> BackfillResult:
        """Copy locally provided CSVs into fbref artifacts folder.

        Expects files in data/manual_fbref_import/{LEAGUE}/{SEASON}/ with names:
          - schedule.csv
          - players_standard.csv
          - players_shooting.csv
          - players_passing.csv (optional)
          - players_misc.csv (optional)
          - players_defense.csv (optional)
          - goalkeeping.csv (optional)
        """
        season_str = season or _season_string_for_today()
        res = BackfillResult(league=league.upper(), season=season_str, artifacts=[], notes=[])
        src_dir = Path("data/manual_fbref_import") / league.upper() / season_str
        if not src_dir.exists():
            res.notes.append(f"source directory not found: {src_dir}")
            return res
        out_dir = self._ensure_dir(league, season_str)
        files = [
            "schedule.csv",
            "players_standard.csv",
            "players_shooting.csv",
            "players_passing.csv",
            "players_misc.csv",
            "players_defense.csv",
            "goalkeeping.csv",
        ]
        total_players = 0
        try:
            import pandas as pd  # type: ignore
        except Exception:
            pd = None  # type: ignore
        for fname in files:
            src = src_dir / fname
            if not src.exists():
                continue
            dst = out_dir / fname
            try:
                # Use pandas to validate + count rows when available
                nrows = 0
                if pd is not None:
                    try:
                        df = pd.read_csv(src)
                        nrows = int(getattr(df, "shape", [0, 0])[0])
                        df.to_csv(dst, index=False)
                    except Exception:
                        # Fallback to raw copy if parsing fails
                        dst.write_bytes(src.read_bytes())
                else:
                    dst.write_bytes(src.read_bytes())
                if fname == "schedule.csv":
                    res.schedule_rows = nrows if nrows else res.schedule_rows
                else:
                    total_players += nrows
                res.artifacts.append(str(dst))
            except Exception as e:
                res.notes.append(f"failed to ingest {fname}: {e}")
        if total_players:
            res.player_rows = total_players
        return res


# Singleton-ish accessor
fbrefdata_service = FbrefDataService()
