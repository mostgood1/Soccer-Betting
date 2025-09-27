# Launch Uvicorn in a detached PowerShell process so that interactive commands in the main terminal
# don't terminate the web server. This avoids the issue where sending another command into the same
# terminal session stops the running server (triggering FastAPI shutdown events).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\run-detached.ps1 -Port 8070 -Synthetic 1
#
# Parameters:
#   -Port <int>              : Port to bind (default 8070)
#   -Synthetic <0|1>         : If 1 sets ML_FORCE_SYNTHETIC=1 (force fully synthetic training data)
#   -SkipStartupTrain <0|1>  : If 1 sets ML_SKIP_STARTUP_TRAIN=1 (lazy load models only)
#   -KeepAlive <0|1>         : If 1 sets APP_DEBUG_KEEPALIVE=1 (internal async no-op loop)
#   -LogLevel <level>        : Uvicorn log level (info|warning|debug) default info
#
# The process is started with Start-Process so it remains active after this script returns.
# A lightweight transcript file is written to .\uvicorn-detached.log (overwritten each launch).

param(
    [int]$Port = 8070,
    [ValidateSet('0','1')][string]$Synthetic = '1',
    [ValidateSet('0','1')][string]$SkipStartupTrain = '0',
    [ValidateSet('0','1')][string]$KeepAlive = '1',
    [ValidateSet('critical','error','warning','info','debug','trace')][string]$LogLevel = 'info'
)

$ErrorActionPreference = 'Stop'

Write-Host "[DETACHED] Launching Uvicorn on 127.0.0.1:$Port (synthetic=$Synthetic skipTrain=$SkipStartupTrain keepAlive=$KeepAlive log=$LogLevel)" -ForegroundColor Cyan

# Build environment block
$envBlock = @{
    'ML_FORCE_SYNTHETIC'   = $Synthetic
    'ML_SKIP_STARTUP_TRAIN'= $SkipStartupTrain
    'APP_DEBUG_KEEPALIVE'  = $KeepAlive
}

# Resolve python path
$python = Join-Path -Path (Resolve-Path .\.venv\Scripts).Path -ChildPath 'python.exe'
if (-not (Test-Path $python)) {
    Write-Host "[DETACHED] python.exe not found in .venv. Please create/activate venv first." -ForegroundColor Red
    exit 1
}

$arguments = @('-m','uvicorn','app.main:app','--host','127.0.0.1','--port',"$Port",'--log-level',"$LogLevel")

# Optional: disable reload to reduce spurious restarts
if (-not $arguments.Contains('--reload')) { $arguments += '--no-use-colors' }

# Prepare log file
$logFile = Join-Path (Get-Location) 'uvicorn-detached.log'
if (Test-Path $logFile) { Remove-Item $logFile -Force }

# Construct start info
$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = $python
$startInfo.Arguments = ($arguments -join ' ')
$startInfo.WorkingDirectory = (Get-Location).Path
$startInfo.UseShellExecute = $false
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true

foreach ($k in $envBlock.Keys) {
    $startInfo.EnvironmentVariables[$k] = $envBlock[$k]
}

$process = New-Object System.Diagnostics.Process
$process.StartInfo = $startInfo

# Event handlers to tee output to log file
$stdOutHandler = [System.Diagnostics.DataReceivedEventHandler]{ param($sender,$e) if ($e.Data) { Add-Content -Path $logFile -Value $e.Data } }
$stdErrHandler = [System.Diagnostics.DataReceivedEventHandler]{ param($sender,$e) if ($e.Data) { Add-Content -Path $logFile -Value $e.Data } }

$null = $process.Start()
$process.BeginOutputReadLine()
$process.BeginErrorReadLine()
$process.add_OutputDataReceived($stdOutHandler)
$process.add_ErrorDataReceived($stdErrHandler)

Write-Host "[DETACHED] Started PID $($process.Id). Logs: $logFile" -ForegroundColor Green
Write-Host "[DETACHED] To stop: Stop-Process -Id $($process.Id)" -ForegroundColor Yellow
