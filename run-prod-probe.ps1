param(
  [string]$BaseUrl,
  [string]$Token,
  [string]$League,
  [int]$Week,
  [string]$HomeTeam,
  [string]$AwayTeam,
  [switch]$SkipCron
)

$ErrorActionPreference = 'Stop'

if (-not $BaseUrl) { $BaseUrl = $env:RENDER_BASE_URL }
if (-not $Token) { $Token = $env:REFRESH_CRON_TOKEN }
if (-not $League) { $League = 'PL' }
if (-not $Week) { $Week = 6 }
if (-not $HomeTeam) { $HomeTeam = 'Arsenal' }
if (-not $AwayTeam) { $AwayTeam = 'Chelsea' }

if (-not $BaseUrl) { Write-Error "Provide --BaseUrl or set RENDER_BASE_URL" }

$argsList = @('--base', $BaseUrl, '--league', $League, '--week', $Week, '--home', $HomeTeam, '--away', $AwayTeam)
if ($Token) { $argsList += @('--token', $Token) }
if ($SkipCron) { $argsList += '--skip-cron' }

& "$PSScriptRoot/.venv/Scripts/python.exe" -m scripts.prod_probe @argsList
exit $LASTEXITCODE
