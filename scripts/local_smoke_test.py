import os
import sys
from pathlib import Path
from pprint import pprint

# Ensure repository root is on sys.path so `import app` resolves
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from app.main import app

# Optionally set a cron token so admin endpoints can be exercised
os.environ.setdefault("REFRESH_CRON_TOKEN", "dev-token")

client = TestClient(app)

print("== ROOT ==")
r = client.get("/")
print(r.status_code)
print(r.headers.get("content-type"))
print(r.text[:120])

print("\n== HEALTH ==")
print(client.get("/health").json())

print("\n== LEAGUES ==")
print(client.get("/api/leagues").json())

print("\n== BY-DATE (1 day ahead, include completed) ==")
resp = client.get("/api/games/by-date?days_ahead=1&days_back=0&include_completed=true")
print(resp.status_code)
data = resp.json()
print(
    {
        k: len(data.get(k, []))
        if isinstance(data.get(k), list)
        else type(data.get(k)).__name__
        for k in data.keys()
    }
)

print("\n== WEEK ODDS (PL, week=6) ==")
print(client.get("/api/betting/odds/week/6?league=PL&limit=3").json().get("count"))

print("\n== PREDICTIONS (PL, week=6) ==")
print(client.get("/api/predictions/week/6?league=PL").json().get("count"))

print("\n== COVERAGE STATUS ==")
print(client.get("/api/admin/status/coverage").status_code)

print("\n== ENSURE FULL COVERAGE (composite) ==")
headers = {"Authorization": "Bearer dev-token"}
resp = client.post(
    "/api/admin/ensure-full-coverage?include_odds_api=true", headers=headers
)
print(resp.status_code)
try:
    js = resp.json()
    print(
        {
            "steps": [s.get("name") for s in js.get("steps", [])],
            "coverage_keys": list(js.get("coverage", {}).keys())[:5],
        }
    )
except Exception as e:
    print("ensure-full-coverage response parse error:", e)

print("\n== MULTI-LEAGUE CHECKS ==")
for lg in ["BL1", "FL1", "SA", "PD"]:
    try:
        r1 = client.get(
            f"/api/games/by-date?leagues={lg}&days_ahead=30&days_back=0&include_completed=false"
        )
        r2 = client.get(f"/api/game-weeks?league={lg}")
        print(
            lg,
            "by-date status:",
            r1.status_code,
            "groups:",
            len((r1.json() or {}).get("groups") or []),
            "game-weeks status:",
            r2.status_code,
        )
    except Exception as e:
        print(lg, "error:", e)
