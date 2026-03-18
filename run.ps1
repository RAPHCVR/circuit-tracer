$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$appPath = Join-Path $projectRoot "app.py"
$venvStreamlit = Join-Path $projectRoot ".venv\Scripts\streamlit.exe"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

Set-Location $projectRoot

Write-Host "Starting Streamlit Neural Dashboard..." -ForegroundColor Cyan

if (Test-Path $venvStreamlit) {
  & $venvStreamlit run $appPath
  exit $LASTEXITCODE
}

if (Test-Path $venvPython) {
  & $venvPython -m streamlit run $appPath
  exit $LASTEXITCODE
}

if (Get-Command uv -ErrorAction SilentlyContinue) {
  $env:UV_CACHE_DIR = Join-Path $projectRoot ".uv-cache"
  uv run streamlit run $appPath
  exit $LASTEXITCODE
}

Write-Error "Could not find uv or a local .venv. Create a venv and install deps, or install uv."
