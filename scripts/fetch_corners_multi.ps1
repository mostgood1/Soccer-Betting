param(
  [string]$League = "PL",
  [string]$Weeks = "1-38",
  [string]$Out = "data/corners_actuals_2025_26.csv",
  [switch]$UseFBref
)

# Expand ranges like 1-5,7,9-10 to comma list
function Expand-Weeks($spec) {
  $parts = $spec -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ }
  $nums = @()
  foreach ($p in $parts) {
    if ($p -match '^(\d+)-(\d+)$') {
      $start = [int]$Matches[1]; $end = [int]$Matches[2]
      for ($i=$start; $i -le $end; $i++) { $nums += $i }
    } else {
      $nums += [int]$p
    }
  }
  return ($nums | Sort-Object -Unique) -join ','
}

$weeksExpanded = Expand-Weeks $Weeks
$use = if ($UseFBref) { "--use-fbref" } else { "" }

Write-Host "Fetching corners for $League weeks $weeksExpanded -> $Out"
& "C:/Users/mostg/OneDrive/Coding/Soccer-Betting/.venv/Scripts/python.exe" `
  -m app.tools.fetch_corners_fbref `
  --league $League `
  --weeks $weeksExpanded `
  --out $Out `
  $use
