import argparse
import json
import sys
import urllib.request
import urllib.error


def fetch_json(url: str, timeout: int = 15):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = r.read().decode("utf-8", errors="ignore")
            return json.loads(data)
    except Exception as e:
        return {"error": str(e), "url": url}


def summarize_week(base: str, league: str, week: int):
    d = fetch_json(f"{base}/api/game-weeks/{week}?league={league}")
    o = fetch_json(
        f"{base}/api/betting/odds/week/{week}?league={league}&future_only=false&cache_only=true"
    )
    r = fetch_json(f"{base}/api/game-weeks/{week}/reconciliation?league={league}")
    matches = (d or {}).get("matches") or []
    m_count = len(matches)
    completed_with_scores = 0
    for m in matches:
        is_completed = m.get("is_completed") or (m.get("status") or "").upper() in (
            "FINISHED",
            "COMPLETED",
        )
        if is_completed and (
            m.get("home_score") is not None and m.get("away_score") is not None
        ):
            completed_with_scores += 1
    odds_rows = 0
    odds_with_totals = 0
    odds_missing_any_or_totals = 0
    sample_missing = []
    om = (o or {}).get("matches") or []
    odds_rows = len(om)
    for row in om:
        od = (row or {}).get("odds") or {}
        mo = od.get("market_odds") or {}
        has_any = bool(mo)
        has_totals = bool(mo.get("totals"))
        if has_totals:
            odds_with_totals += 1
        if not has_any or not has_totals:
            odds_missing_any_or_totals += 1
            if len(sample_missing) < 3:
                sample_missing.append(
                    {
                        "home": row.get("home_team"),
                        "away": row.get("away_team"),
                        "has_any": has_any,
                        "has_totals": has_totals,
                    }
                )
    recs = (r or {}).get("reconciliations") or []
    rec_count = len(recs)
    return {
        "week": week,
        "detail_error": (d or {}).get("detail") if isinstance(d, dict) else None,
        "odds_error": (o or {}).get("detail") if isinstance(o, dict) else None,
        "recon_error": (r or {}).get("detail") if isinstance(r, dict) else None,
        "matches": m_count,
        "completed_with_scores": completed_with_scores,
        "odds_rows": odds_rows,
        "odds_with_totals": odds_with_totals,
        "odds_missing_any_or_totals": odds_missing_any_or_totals,
        "reconciliations": rec_count,
        "samples_missing": sample_missing,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8073")
    p.add_argument("--league", default="PL")
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=7)
    args = p.parse_args()
    out = []
    for wk in range(args.start, args.end + 1):
        out.append(summarize_week(args.base, args.league, wk))
    print(
        json.dumps(
            {"league": args.league, "base": args.base, "coverage": out}, indent=2
        )
    )


if __name__ == "__main__":
    sys.exit(main())
