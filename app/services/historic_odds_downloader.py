"""Historic EPL Odds Downloader.

Downloads Premier League historical odds/result CSVs from football-data.co.uk.
Source patterns (publicly available):
  https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv
Where season_code is e.g. 2425 for 2024-2025 (concat two-digit start & end years).

We only automate downloading; licensing/usage terms remain with the data provider.

Functions:
  season_code(year) -> str  (year is starting season year, e.g. 2024 -> '2425')
  download_epl_history(start_year, end_year, out_dir='data/odds', overwrite=False)

Returns structured summary including downloaded files, skipped, and errors.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import os

try:  # requests may already be in requirements
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

BASE_URL = "https://www.football-data.co.uk/mmz4281"


def season_code(year: int) -> str:
    y1 = str(year)[2:]
    y2 = str(year + 1)[2:]
    return f"{y1}{y2}"


def download_epl_history(start_year: int, end_year: int, out_dir: str = "data/odds", overwrite: bool = False, timeout: int = 15) -> Dict[str, Any]:
    if requests is None:
        return {"error": "requests not available"}
    if end_year < start_year:
        return {"error": "end_year must be >= start_year"}
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    downloaded: List[str] = []
    skipped: List[str] = []
    errors: List[Dict[str, str]] = []
    for year in range(start_year, end_year + 1):
        code = season_code(year)
        url = f"{BASE_URL}/{code}/E0.csv"
        dest = out_path / f"EPL_{year}_{year+1}.csv"
        if dest.exists() and not overwrite:
            skipped.append(dest.name)
            continue
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code != 200:
                errors.append({"season": f"{year}-{year+1}", "status": str(resp.status_code), "url": url})
                continue
            text = resp.text
            # Basic sanity: Must have header containing HomeTeam,AwayTeam
            if 'HomeTeam' not in text or 'AwayTeam' not in text:
                errors.append({"season": f"{year}-{year+1}", "error": "missing expected columns", "url": url})
                continue
            dest.write_text(text, encoding='utf-8')
            downloaded.append(dest.name)
        except Exception as e:  # pragma: no cover
            errors.append({"season": f"{year}-{year+1}", "error": str(e), "url": url})
    return {
        "start_year": start_year,
        "end_year": end_year,
        "downloaded": downloaded,
        "skipped": skipped,
        "errors": errors,
        "output_dir": str(out_path)
    }
