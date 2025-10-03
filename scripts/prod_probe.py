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


def _get(
    base: str, path: str, token: Optional[str] = None, **kwargs
) -> requests.Response:
    url = base.rstrip("/") + path
    timeout = kwargs.pop("timeout", 30)
    return requests.get(url, headers=_hdrs(token), timeout=timeout, **kwargs)


def _post(
    base: str, path: str, token: Optional[str] = None, **kwargs
) -> requests.Response:
    url = base.rstrip("/") + path
    timeout = kwargs.pop("timeout", 60)
    return requests.post(url, headers=_hdrs(token), timeout=timeout, **kwargs)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base",
        default=os.getenv("RENDER_BASE_URL", ""),
        help="Render base URL, e.g. https://soccer-betting.onrender.com",
    )
    p.add_argument(
        "--token",
        default=os.getenv("REFRESH_CRON_TOKEN", ""),
        help="Bearer token for protected cron/admin endpoints",
    )
    p.add_argument(
        "--league",
        default="PL",
        help="Primary league to print details for (the script will still validate ALL leagues)",
    )
    p.add_argument(
        "--week", type=int, default=6, help="Week number to test, defaults to 6"
    )
    p.add_argument("--home", default="Arsenal", help="Home team for match odds probe")
    p.add_argument("--away", default="Chelsea", help="Away team for match odds probe")
    p.add_argument(
        "--skip-cron", action="store_true", help="Skip triggering cron endpoints"
    )
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
            r = _post(
                base,
                "/api/cron/snapshot-csv?league=ALL&include_odds_api=true",
                token=token,
            )
            print(r.status_code, r.text[:300])
            r.raise_for_status()
            return r.json()

        step("refresh-bovada", refresh_bovada)
        time.sleep(1)
        step("snapshot-csv", snapshot_csv)

    # Kick precompute-recommendations (requires token). Do per-league calls to avoid a single long request.
    if token and not args.skip_cron:

        def precompute_one(lg: str):
            r = _post(
                base,
                "/api/cron/precompute-recommendations",
                token=token,
                params={
                    "league": lg,
                    "edge_threshold": "0.03",
                    "prob_threshold": "0.5",
                },
                timeout=120,
            )
            print("precompute", lg, r.status_code, r.text[:200])
            r.raise_for_status()
            return r.json()

        for lg in ["PL", "BL1", "FL1", "SA", "PD"]:
            try:
                step(f"precompute-{lg}", lambda lg=lg: precompute_one(lg))
                time.sleep(0.5)
            except Exception:
                failures.append(f"precompute-{lg}")

    # Cron summary
    def cron_summary():
        r = _get(base, "/api/admin/status/cron-summary", timeout=60)
        print(r.status_code, r.text[:300])
        r.raise_for_status()
        return r.json()

    # Allow a brief moment for cron-status files to be written
    time.sleep(0.5)
    step("cron-summary", cron_summary)

    # Odds, match (single sample)
    def odds_match():
        r = _get(base, f"/api/betting/odds/{args.home}/{args.away}")
        print(r.status_code, r.text[:300])
        r.raise_for_status()
        return r.json()

    step("odds-match", odds_match)

    # Validate ALL leagues for week odds, edges, and recommendations
    ALL_LEAGUES = ["PL", "BL1", "FL1", "SA", "PD"]

    def check_league(lg: str):
        out: Dict[str, Any] = {"league": lg}
        # Week odds (limit 3)
        rw = _get(base, f"/api/betting/odds/week/{args.week}?league={lg}&limit=3")
        print(lg, "odds-week:", rw.status_code, rw.text[:160])
        try:
            rw.raise_for_status()
            out["odds_week_count"] = len((rw.json() or {}).get("matches", []))
        except Exception as e:
            out["odds_week_error"] = str(e)
        # Week edges (fast Bovada-only path)
        re = _get(
            base,
            f"/api/betting/edges/week/{args.week}?league={lg}&limit=3&threshold=0.01&allow_on_demand=false&fast=true",
        )
        print(lg, "edges-week:", re.status_code, re.text[:160])
        try:
            re.raise_for_status()
            out["edges_week_count"] = len((re.json() or {}).get("matches", []))
        except Exception as e:
            out["edges_week_error"] = str(e)
        # Recommendations latest
        rr = _get(base, "/api/recommendations/latest", params={"league": lg})
        print(lg, "recs-latest:", rr.status_code, rr.text[:160])
        if rr.status_code == 404:
            out["recs_latest"] = "missing"
        else:
            try:
                rr.raise_for_status()
                out["recs_latest_count"] = len((rr.json() or {}).get("matches", []))
            except Exception as e:
                out["recs_latest_error"] = str(e)
        return out

    per_league = []
    print("\n==> Validating ALL leagues")
    for lg in ALL_LEAGUES:
        try:
            per_league.append(check_league(lg))
            time.sleep(0.3)
        except Exception as e:
            per_league.append({"league": lg, "error": str(e)})

    # Upcoming multi-league with odds (cache-only to avoid live calls)
    def odds_upcoming_all():
        r = _get(
            base,
            "/api/betting/odds/upcoming?leagues=ALL&prefetch=false&limit=5&include_odds=true&cache_only=true",
            timeout=90,
        )
        print("upcoming-all:", r.status_code, r.text[:300])
        if r.status_code == 404:
            return {"skipped": True}
        r.raise_for_status()
        j = r.json()
        leagues = j.get("leagues", {}) if isinstance(j, dict) else {}
        return {"leagues": {k: v.get("count", 0) for k, v in leagues.items()}}

    up = step("odds-upcoming-all", odds_upcoming_all)

    # Verify latest recommendations exist for primary league (detail print)
    def recommendations_latest():
        r = _get(base, "/api/recommendations/latest", params={"league": args.league})
        print(r.status_code, r.text[:300])
        if r.status_code == 404:
            print("[warn] recommendations not yet available for this league")
            return {"skipped": True}
        r.raise_for_status()
        j = r.json()
        # Expect a dict with "matches" list
        assert isinstance(j, dict)
        _ = j.get("matches", [])
        return {"count": len(_)}

    step("recommendations-latest", recommendations_latest)

    # Print compact per-league summary at the end
    print("\nPer-league summary:")
    try:
        for row in per_league:
            print(json.dumps(row, indent=2))
    except Exception:
        pass

    print("\nSummary:")
    if failures:
        print(f"FAILED steps: {failures}")
        return 1
    print("All checks completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
