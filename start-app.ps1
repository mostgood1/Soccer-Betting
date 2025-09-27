# EPL Betting Platform Launcher - Single PowerShell File
# Simplified version with basic ASCII characters

param(
    [switch]$dev,
    [switch]$quiet,
    [switch]$help,
    [switch]$noReload,
    [int]$port = 8000
)

if ($help) {
    Write-Host ""
    Write-Host "EPL BETTING PLATFORM LAUNCHER" -ForegroundColor Cyan
    Write-Host "=============================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\start-app.ps1 [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -dev     Development mode with detailed logging" -ForegroundColor Gray
    Write-Host "  -quiet   Quiet mode with minimal output" -ForegroundColor Gray
    Write-Host "  -noReload Disable Uvicorn file reload (stability)" -ForegroundColor Gray
    Write-Host "  -port    Override default port (default 8000)" -ForegroundColor Gray
    Write-Host "  -help    Show this help message" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\start-app.ps1           # Normal startup" -ForegroundColor Gray
    Write-Host "  .\start-app.ps1 -dev      # Development mode" -ForegroundColor Gray
    Write-Host "  .\start-app.ps1 -quiet    # Quiet startup" -ForegroundColor Gray
    Write-Host ""
    exit 0
}

function Write-Success { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Error { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }
function Write-Warning { param($msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Info { param($msg) Write-Host "[INFO] $msg" -ForegroundColor Cyan }

# Check if running from correct directory
if (!(Test-Path "app\main.py")) {
    Write-Error "Please run this script from the Soccer-Betting project root directory"
    exit 1
}

# Banner
if (!$quiet) {
    Clear-Host
    Write-Host ""
    Write-Host "EPL BETTING PLATFORM - PREDICTABLE ENGINE v3.0" -ForegroundColor Cyan
    Write-Host "===============================================" -ForegroundColor Gray
    Write-Host "Real Data | ML Predictions | Game Week Cards" -ForegroundColor White
    Write-Host "===============================================" -ForegroundColor Gray
    Write-Host ""
}

# Check Virtual Environment
Write-Info "Checking virtual environment..."

if (!(Test-Path ".venv\Scripts\python.exe")) {
    Write-Error "Virtual environment not found. Please run: python -m venv .venv"
    exit 1
}

if (!$quiet) { Write-Success "Virtual environment found" }

# Activate virtual environment
try {
    & ".\.venv\Scripts\Activate.ps1"
    if (!$quiet) { Write-Success "Virtual environment activated" }
} catch {
    Write-Error "Failed to activate virtual environment: $($_.Exception.Message)"
    exit 1
}

# Check Dependencies
if (!$quiet) { Write-Info "Checking dependencies..." }

$requiredPackages = @("fastapi", "uvicorn", "pandas")
$missingPackages = @()

foreach ($package in $requiredPackages) {
    try {
        python -c "import $package" 2>$null
        if ($LASTEXITCODE -ne 0) { $missingPackages += $package }
    } catch {
        $missingPackages += $package
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Warning "Installing missing packages..."
    pip install -r requirements.txt --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install dependencies"
        exit 1
    }
}

if (!$quiet) { Write-Success "Dependencies OK" }

# Health Checks
if (!$quiet) { Write-Info "Running health checks..." }

# Create directories if needed
if (!(Test-Path "data")) { New-Item -ItemType Directory -Path "data" -Force | Out-Null }
if (!(Test-Path "cache")) { New-Item -ItemType Directory -Path "cache" -Force | Out-Null }

# Check frontend files
$frontendFiles = @("frontend/index.html", "frontend/styles.css", "frontend/app.js")
foreach ($file in $frontendFiles) {
    if (!(Test-Path $file)) {
        Write-Error "Missing frontend file: $file"
        exit 1
    }
}

if (!$quiet) { Write-Success "Frontend files verified" }

# Start Server
$serverHost = "127.0.0.1"

# Kill existing processes on port
try {
    $existingProcess = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($existingProcess) {
        Write-Warning "Freeing port $port..."
        $processId = $existingProcess.OwningProcess
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
} catch {
    # Ignore - port might not be in use
}

if (!$quiet) {
    Write-Host ""
    Write-Host "STARTING EPL BETTING PLATFORM" -ForegroundColor Green -BackgroundColor Black
    Write-Host "=============================" -ForegroundColor Green
    Write-Host "Server URL: http://$serverHost`:$port" -ForegroundColor White
    Write-Host "API Docs:   http://$serverHost`:$port/docs" -ForegroundColor White
    Write-Host "Game Weeks: http://$serverHost`:$port/#game-weeks" -ForegroundColor White
    Write-Host ""
    Write-Host "Features:" -ForegroundColor Yellow
    Write-Host "- Game Week Cards with Model Reconciliation" -ForegroundColor Gray
    Write-Host "- 380+ EPL Fixtures with Real Data" -ForegroundColor Gray
    Write-Host "- ML Predictions and Confidence Scores" -ForegroundColor Gray
    Write-Host "- Player Statistics and Performance" -ForegroundColor Gray
    Write-Host ""
    Write-Warning "Press Ctrl+C to stop the server"
    Write-Host ""
}

# Prepare and run startup command (stable uvicorn invocation)
$logLevel = if ($dev) { "debug" } else { "warning" }
$pythonExe = ".\.venv\Scripts\python.exe"
$args = @('-m','uvicorn','app.main:app','--host', $serverHost, '--port', $port, '--log-level', $logLevel)
if ($dev -and -not $noReload) { $args += '--reload' }
# Ensure app stays alive by default unless explicitly disabled
if (-not $env:APP_DEBUG_KEEPALIVE) { $env:APP_DEBUG_KEEPALIVE = '1' }
if ($env:ML_SKIP_STARTUP_TRAIN) { Write-Host "[INFO] ML_SKIP_STARTUP_TRAIN=$($env:ML_SKIP_STARTUP_TRAIN)" -ForegroundColor Cyan }

try {
    if ($quiet) {
        Write-Host "Server starting at http://$serverHost`:$port" -ForegroundColor Green
    }
    
    & $pythonExe @args
    
} catch {
    Write-Error "Failed to start server: $($_.Exception.Message)"
    Write-Host ""
    Write-Warning "Troubleshooting:"
    Write-Host "1. Ensure port $port is available" -ForegroundColor Gray
    Write-Host "2. Check virtual environment activation" -ForegroundColor Gray
    Write-Host "3. Verify all dependencies are installed" -ForegroundColor Gray
    Write-Host ""
    exit 1
} finally {
    if (!$quiet) { Write-Host "" }
}