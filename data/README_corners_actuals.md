Corners Actuals Data (EPL 2025-26)

Provide real match corners actuals to enable evaluation of corners markets.

Accepted files (any that exist will be loaded at startup):
- data/corners_actuals_2025_26.json
- data/corners_actuals_2025_26.csv
- data/corners_actuals_manual.json
- data/corners_actuals_manual.csv

Schema
- date: YYYY-MM-DD (or full ISO string; we'll extract the date part)
- home_team: string (display name; we'll normalize internally)
- away_team: string
- home_corners: integer
- away_corners: integer
- total_corners: integer (optional; computed as home+away if missing)

Notes
- Team names are normalized via app/services/team_name_normalizer.py, so common variations are handled.
- If both home_corners and away_corners are present, total_corners will be computed when missing.
- This data is used by:
  - app/services/enhanced_epl_service_v2.py to attach per-match corners fields
  - app/offline/tasks.py compare_week_corners_totals and compare_week_team_corners_totals for evaluation

Workflow
1) Paste real corners data into one of the accepted files (CSV is easiest; see sample rows in corners_actuals_2025_26.csv)
2) Restart the API or re-run the offline comparisons.
3) The consolidated report will now show evaluable counts > 0 for corners markets where data is present.
