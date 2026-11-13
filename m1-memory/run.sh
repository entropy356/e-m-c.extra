#!/usr/bin/env bash
# 启动 M1 记忆前端。凭据从导出了 AGENT_* / TAVILY_API_KEY 的 shell 片段读取。
#   ENV_FILE=/path/to/agent-env.sh PORT=3000 ./run.sh
set -euo pipefail
BASE="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE"

for candidate in "${ENV_FILE:-}" "$BASE/.env"; do
  if [ -n "$candidate" ] && [ -f "$candidate" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$candidate"
    set +a
  fi
done

RUNTIME="${EMC_RUNTIME_PATH:-$BASE/../../e-m-c.runtime/src}"
[ -d "$BASE/vendor" ] && RUNTIME="$BASE/vendor"
export PYTHONPATH="$BASE:$RUNTIME${PYTHONPATH:+:$PYTHONPATH}"

exec python3 "$BASE/server.py"
