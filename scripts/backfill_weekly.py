"""One-shot backfill utility to generate weekly artifacts for completed weeks
across all leagues. Uses normalization-aware odds loader to maximize matches.

Usage (from repo root):
  python -m scripts.backfill_weekly [--league PL] [--up-to 38]
"""
from __future__ import annotations

import argparse
from typing import List, Optional

from app.services.weekly_files_service import (
    backfill_all_leagues,
    backfill_completed_weeks,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill weekly artifacts")
    parser.add_argument(
        "--league",
        "-l",
        default=None,
        help="League code (PL, BL1, FL1, SA, PD) or omit for all",
    )
    parser.add_argument(
        "--up-to",
        "-u",
        type=int,
        default=None,
        help="Only process weeks <= this number",
    )
    args = parser.parse_args()

    include = ["odds", "predictions", "results"]
    if args.league:
        res = backfill_completed_weeks(
            args.league, up_to_week=args.up_to, include=include
        )
        print(res)
    else:
        res = backfill_all_leagues(up_to_week=args.up_to, include=include)
        print(res)


if __name__ == "__main__":
    main()
