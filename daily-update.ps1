param(
  [ValidateSet('major','minor','patch')][string]$RetrainLevel = 'patch',
  [switch]$NoClosing,
  [switch]$NoSnapshot,
  [switch]$NoBovada,
  [switch]$NoOddsApiCorners,
  [string]$OddsApiRegions = 'eu,uk,us',
  [string]$OddsApiBookmakers = 'pinnacle,bet365,williamhill,unibet,betfair_ex',
  [switch]$VerboseLog
)

$ErrorActionPreference = 'Stop'
# Ensure consistent stdout/stderr encoding
$env:PYTHONIOENCODING = 'utf-8'

# Move to repo root (this script's directory)
Set-Location -Path $PSScriptRoot

# Resolve Python interpreter (prefer venv)
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$pythonExe = if (Test-Path $venvPython) { $venvPython } else { 'python' }

# Build arguments for daily update
$cmdArgs = @(
  '-m','app.offline.tasks','daily-update',
  '--retrain-level', $RetrainLevel,
  '--oddsapi-regions', $OddsApiRegions,
  '--oddsapi-bookmakers', $OddsApiBookmakers
)
if ($NoClosing) { $cmdArgs += '--no-closing' }
if ($NoSnapshot) { $cmdArgs += '--no-snapshot' }
if ($NoBovada) { $cmdArgs += '--no-bovada' }
if ($NoOddsApiCorners) { $cmdArgs += '--no-oddsapi-corners' }

# Prepare logging
$logDir = Join-Path $PSScriptRoot 'cache\logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$ts = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$logPath = Join-Path $logDir "daily-update-$ts.log"

Write-Host "Running daily update..." -ForegroundColor Cyan
Write-Host "Python: $pythonExe" -ForegroundColor DarkGray
Write-Host ("Args: " + ($cmdArgs -join ' ')) -ForegroundColor DarkGray
Write-Host "Logging to: $logPath" -ForegroundColor DarkGray

# Execute and tee output to log
try {
  & $pythonExe @cmdArgs 2>&1 | Tee-Object -FilePath $logPath | ForEach-Object {
    if ($VerboseLog) { $_ } else { $_ }
  }
  $exit = $LASTEXITCODE
} catch {
  Write-Error $_
  $exit = 1
}

if ($exit -ne 0) {
  Write-Host "Daily update FAILED. Exit code: $exit" -ForegroundColor Red
  Write-Host "See log: $logPath" -ForegroundColor Red
  exit $exit
}

Write-Host "Daily update completed successfully." -ForegroundColor Green
Write-Host "Log file: $logPath" -ForegroundColor Green
