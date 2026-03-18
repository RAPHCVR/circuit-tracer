#!/usr/bin/env bash
set -euo pipefail

# Prefer the project venv when present; fall back to uv otherwise.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
app_path="${project_root}/app.py"

cd "${project_root}"

if [[ -x "${project_root}/.venv/bin/streamlit" ]]; then
  "${project_root}/.venv/bin/streamlit" run "${app_path}"
elif [[ -x "${project_root}/.venv/bin/python" ]]; then
  "${project_root}/.venv/bin/python" -m streamlit run "${app_path}"
elif command -v uv >/dev/null 2>&1; then
  export UV_CACHE_DIR="${project_root}/.uv-cache"
  uv run streamlit run "${app_path}"
else
  streamlit run "${app_path}"
fi
