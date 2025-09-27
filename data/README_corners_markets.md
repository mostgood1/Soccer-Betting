Corners Markets Data (Totals and Team Totals)

Provide real market lines and odds for corners so the comparisons can include market metrics and edges. Drop one or more of these files:

- data/corners_market_lines_2025_26.csv
- data/corners_market_lines_2025_26.json
- data/corners_market_lines_manual.csv
- data/corners_market_lines_manual.json

CSV format (you can mix total and team rows):

# Total corners O/U
date,home_team,away_team,line,over_odds,under_odds,bookmaker,market_type
2025-08-16,Tottenham,Brentford,9.5,1.90,1.90,Bet365,total

# Team corners O/U
date,home_team,away_team,side,line,over_odds,under_odds,bookmaker,market_type
2025-08-16,Tottenham,Brentford,home,4.5,1.83,1.95,Bet365,team

Notes
- Dates must be YYYY-MM-DD. Team names should match our normalizer (e.g., "Man City" or "Manchester City" both OK).
- Odds are decimal. We'll compute implied probabilities and normalize across Over/Under.
- If multiple bookmakers/lines exist for the same match, last one wins for that exact line; we also fallback to the closest line when comparing.
- Lines starting with # are allowed as comments and are ignored.

JSON format
Either an array of rows with the same fields or an object with key "records" (preferred) or "markets":
[
  { "date":"2025-08-16", "home_team":"Tottenham", "away_team":"Brentford", "line":9.5, "over_odds":1.9, "under_odds":1.9, "bookmaker":"Bet365", "market_type":"total" },
  { "date":"2025-08-16", "home_team":"Tottenham", "away_team":"Brentford", "side":"home", "line":4.5, "over_odds":1.83, "under_odds":1.95, "bookmaker":"Bet365", "market_type":"team" }
]

Usage
- After adding/refreshing files, re-run the offline comparisons for weeks 1–3:
  - python -m app.offline.tasks week-corners-compare --week 1
  - python -m app.offline.tasks week-team-corners-compare --week 1 --side home
  - python -m app.offline.tasks week-team-corners-compare --week 1 --side away
  - Repeat for weeks 2 and 3, then rebuild the consolidated report:
  - python -m app.offline.tasks consolidated-report --weeks 1,2,3

API/UI
- The consolidated report now includes market summaries for corners where provided.