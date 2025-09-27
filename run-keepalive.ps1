# Dedicated script to launch with keepalive env vars (workaround for inline -Command parsing)
$env:ML_SKIP_STARTUP_TRAIN = '1'
$env:APP_DEBUG_KEEPALIVE = '1'
$port = 8060
Write-Host "Launching keepalive server on port $port" -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port $port --log-level debug --no-use-colors
