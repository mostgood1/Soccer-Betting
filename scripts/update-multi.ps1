# Refresh all leagues and optionally snapshot a week; run from repo root
param(
  [int]$Week = 6,
  [switch]$Snapshot,
  [switch]$RebuildPL
)

$BaseUrl = "http://127.0.0.1:8000"

# Ensure server is up; if not, try to start it in a new window
try {
  $ping = Invoke-RestMethod -Uri "$BaseUrl/api/debug/ping" -TimeoutSec 3
} catch {
  Write-Host "API not responding on 8000; attempting to start server..." -ForegroundColor Yellow
  Start-Process powershell -ArgumentList "-NoProfile","-Command",".\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000" -WindowStyle Hidden
  Start-Sleep -Seconds 3
}

# Bulk refresh all leagues real-data only
$refresh = Invoke-RestMethod -Uri "$BaseUrl/api/admin/data/leagues/refresh-all" -Method Post
$refresh | ConvertTo-Json -Depth 6

if ($Snapshot) {
  $codes = "PL,BL1,FL1,SA,PD"
  $snap = Invoke-RestMethod -Uri "$BaseUrl/api/admin/data/leagues/refresh-all?leagues=$codes&snapshot_week=$Week" -Method Get
  $snap | ConvertTo-Json -Depth 6
}

if ($RebuildPL) {
  $re = Invoke-RestMethod -Uri "$BaseUrl/api/admin/data/leagues/refresh?league=PL&rebuild_predictions=true" -Method Get
  $re | ConvertTo-Json -Depth 6
}

Write-Host "Done." -ForegroundColor Green
