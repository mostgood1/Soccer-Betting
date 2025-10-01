"""
Render production smoke test for the Soccer Betting API.

Usage:
  python -m scripts.prod_probe --base https://<your-app>.onrender.com --token <REFRESH_CRON_TOKEN>

Environment variables:
  RENDER_BASE_URL  - default base URL if --base not passed
  REFRESH_CRON_TOKEN - default bearer token if --token not passed

This script:
  - Pings health and config endpoints
  - Optionally rehydrates baked data (safe idempotent) if token supplied
  - Triggers hourly cron equivalents (refresh-bovada, snapshot-csv)
  - Checks cron-summary for timestamps
  - Fetches example odds routes (week and match)

Exit codes:
  0 success, non-zero on failure.
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from typing import Any, Dict, Optional

import requests


def _hdrs(token: Optional[str]) -> Dict[str, str]:
    h = {"Accept": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _get(base: str, path: str, token: Optional[str] = None, **kwargs) -> requests.Response:
    url = base.rstrip("/") + path
    timeout = kwargs.pop("timeout", 30)
    return requests.get(url, headers=_hdrs(token), timeout=timeout, **kwargs)


def _post(base: str, path: str, token: Optional[str] = None, **kwargs) -> requests.Response:
    url = base.rstrip("/") + path
    timeout = kwargs.pop("timeout", 60)
    return requests.post(url, headers=_hdrs(token), timeout=timeout, **kwargs)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=os.getenv("RENDER_BASE_URL", ""), help="Render base URL, e.g. https://soccer-betting.onrender.com")
    p.add_argument("--token", default=os.getenv("REFRESH_CRON_TOKEN", ""), help="Bearer token for protected cron/admin endpoints")
    p.add_argument("--league", default="PL", help="League to spot-check for odds week route")
    p.add_argument("--week", type=int, default=6, help="Week number to test, defaults to 6")
    p.add_argument("--home", default="Arsenal", help="Home team for match odds probe")
    p.add_argument("--away", default="Chelsea", help="Away team for match odds probe")
    p.add_argument("--skip-cron", action="store_true", help="Skip triggering cron endpoints")
    args = p.parse_args()

    base = args.base.strip()
    token = args.token.strip() or None
    if not base:
        print("ERROR: --base or RENDER_BASE_URL is required", file=sys.stderr)
        return 2

    failures = []

    def step(name: str, fn):
        print(f"\n==> {name}")
        try:
            return fn()
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failures.append(name)
            return None

    # Health
    def health():
        r = _get(base, "/health")
        print(r.status_code, r.text[:300])
        r.raise_for_status()
        j = r.json()
        assert j.get("status") == "healthy"
        return j

    step("health", health)

    # Config status
    def config():
        r = _get(base, "/api/admin/status/config")
        print(r.status_code, r.text[:300])
        r.raise_for_status()
        return r.json()

    cfg = step("config", config) or {}

    # Hydration status
    def hydration():
        r = _get(base, "/api/admin/status/hydration")
        print(r.status_code, r.text[:300])
        r.raise_for_status()
        return r.json()

    step("hydration", hydration)

    # Optional: rehydrate baked league files (safe/idempotent) if token available
    if token:
        def rehydrate():
            r = _post(base, "/api/admin/data/rehydrate", token=token)
            print(r.status_code, r.text[:300])
            r.raise_for_status()
            return r.json()

        step("rehydrate", rehydrate)

    # Cron: refresh odds + snapshot CSV
    if token and not args.skip_cron:
        def refresh_bovada():
            r = _post(base, "/api/cron/refresh-bovada", token=token)
            print(r.status_code, r.text[:300])
            r.raise_for_status()
            return r.json()

        def snapshot_csv():
            r = _post(base, "/api/cron/snapshot-csv?league=ALL&include_odds_api=true", token=token)
            print(r.status_code, r.text[:300])
            r.raise_for_status()
            return r.json()

        step("refresh-bovada", refresh_bovada)
        time.sleep(1)
        step("snapshot-csv", snapshot_csv)

    # Cron summary
    def cron_summary():
        r = _get(base, "/api/admin/status/cron-summary")
        print(r.status_code, r.text[:300])
        r.raise_for_status()
        return r.json()

    step("cron-summary", cron_summary)

    # Odds, week
    def odds_week():
        r = _get(base, f"/api/betting/odds/week/{args.week}?league={args.league}&limit=3")
        print(r.status_code, r.text[:300])
        r.raise_for_status()
        j = r.json()
        assert isinstance(j, dict)
        # odds may be empty depending on providers; don't fail hard
        return j

    step("odds-week", odds_week)

    # Odds, match
    def odds_match():
        r = _get(base, f"/api/betting/odds/{args.home}/{args.away}")
        print(r.status_code, r.text[:300])
        r.raise_for_status()
        return r.json()

    step("odds-match", odds_match)

    # Optional: upcoming (if deployed route exists)
    def odds_upcoming():
        # Use cache-only and include odds to validate EU aggregator coverage
        r = _get(base, "/api/betting/odds/upcoming?prefetch=false&limit=5&include_odds=true&cache_only=true", timeout=60)
        print(r.status_code, r.text[:300])
        if r.status_code == 404:
            print("[warn] upcoming route not present on this deployment")
            return {"skipped": True}
        r.raise_for_status()
        return r.json()

    step("odds-upcoming", odds_upcoming)

    print("\nSummary:")
    if failures:
        print(f"FAILED steps: {failures}")
        return 1
    print("All checks completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
