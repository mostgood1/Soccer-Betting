param(
    [int[]]$Ports = @(8000,8070,8071,8072)
)

foreach ($p in $Ports) {
  Write-Host "Stopping listeners on port $p if any..."
  $lines = netstat -ano | Select-String ":$p\s" | ForEach-Object { $_.ToString() }
  foreach ($line in $lines) {
    $parts = $line -split "\s+"
    if ($parts.Length -ge 5) {
      $procId = $parts[-1]
      try {
        if ($procId -match '^[0-9]+$') {
          Write-Host "Killing PID $procId for port $p" -ForegroundColor Yellow
          Stop-Process -Id [int]$procId -Force -ErrorAction SilentlyContinue
        }
      } catch {}
    }
  }
}
Write-Host "Done."
