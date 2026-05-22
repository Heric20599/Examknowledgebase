# Invoke GET /internal/cron/ping every 15 minutes via Windows Task Scheduler.
# Requires: API running (uvicorn), CRON_SECRET in .env and in the task environment.
param(
    [string]$ApiBaseUrl = $env:API_BASE_URL,
    [string]$CronSecret = $env:CRON_SECRET
)

if (-not $ApiBaseUrl) { $ApiBaseUrl = "http://127.0.0.1:8000" }
$ApiBaseUrl = $ApiBaseUrl.TrimEnd("/")

if (-not $CronSecret) {
    Write-Error "Set CRON_SECRET (same value as in the API .env)."
    exit 1
}

$uri = "$ApiBaseUrl/internal/cron/ping"
$headers = @{ "X-Cron-Secret" = $CronSecret }

try {
    $response = Invoke-RestMethod -Uri $uri -Method Get -Headers $headers -TimeoutSec 60
    Write-Output ($response | ConvertTo-Json -Compress)
    exit 0
} catch {
    Write-Error $_
    exit 1
}
