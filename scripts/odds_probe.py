import os
from app.services.betting_odds_service import betting_odds_service as S
from app.services.league_manager import get_service
from app.services.team_name_normalizer import normalize_team_name as norm

# Ensure provider calls are allowed
os.environ.pop("DISABLE_PROVIDER_CALLS", None)

print("Prefetching Bovada snapshots...")
res = S.prefetch_bovada()
print("prefetch_bovada:", res)

leagues = ["PL", "BL1", "FL1", "SA", "PD"]
for lg in leagues:
    svc = get_service(lg)
    matches = (
        svc.get_all_matches() if hasattr(svc, "get_all_matches") else svc.get_matches()
    )
    upcoming = [
        m
        for m in matches
        if (m.get("status") or "").lower() in ("scheduled", "timed", "postponed", "")
    ]
    ev_cnt = len((S._bovada_cache.get(lg) or {}).get("events") or [])
    print(f"\nLEAGUE {lg}: bovada_events={ev_cnt} upcoming={len(upcoming)}")
    for m in upcoming[:4]:
        home = (
            m.get("home_team") or m.get("homeTeam") or (m.get("home") or {}).get("name")
        )
        away = (
            m.get("away_team") or m.get("awayTeam") or (m.get("away") or {}).get("name")
        )
        dt = m.get("date") or m.get("utc_date")
        odds = S.get_match_odds(home, away, dt, prefer_bovada_only=True)
        mw = (odds or {}).get("market_odds", {}).get("match_winner")
        provider = (odds or {}).get("provider")
        print(
            "  ",
            (norm(home) or home),
            "vs",
            (norm(away) or away),
            "-> provider:",
            provider,
            "H2H:",
            bool(mw),
        )
