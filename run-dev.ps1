# Simple dev runner to avoid quoting issues inside inline PowerShell -Command
$env:ML_SKIP_STARTUP_TRAIN = '1'
$port = 8051
Write-Host "Starting MINIMAL test server on port $port" -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -m uvicorn minimal_app:app --host 127.0.0.1 --port $port --log-level debug
