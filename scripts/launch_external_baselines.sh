#!/usr/bin/env bash
# 一つの親プロセスが全比較を逐次実行する。SSH切断後も継続する。
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
EXTERNAL_ROOT="${1:-$(dirname "$REPO_ROOT")/UMAPing_external_baselines}"
EXTERNAL_ROOT="$(cd "$EXTERNAL_ROOT" && pwd -P)"
CONTROL_PYTHON="$(cat "$EXTERNAL_ROOT/control-python.txt")"
unset VIRTUAL_ENV CONDA_PREFIX PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg KERAS_BACKEND=tensorflow
export TF_FORCE_GPU_ALLOW_GROWTH=true
LOG_ROOT="$REPO_ROOT/runs_external_baselines/logs"
"$CONTROL_PYTHON" - "$REPO_ROOT" "$LOG_ROOT" <<'PY'
from pathlib import Path
import sys
repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / 'src'))
from umaping.experiments.external import validate_output
validate_output(repo, repo / 'runs', Path(sys.argv[2]))
PY
mkdir -p "$LOG_ROOT"
STAMP="$(date -u '+%Y%m%dT%H%M%SZ')_$$"
OUTPUT_ROOT="$REPO_ROOT/runs_external_baselines/$STAMP"
LOGFILE="$LOG_ROOT/$STAMP.log"
nohup "$CONTROL_PYTHON" "$REPO_ROOT/scripts/run_external_baselines.py" suite \
  --repo-root "$REPO_ROOT" --external-root "$EXTERNAL_ROOT" \
  --output-dir "$OUTPUT_ROOT" --device auto > "$LOGFILE" 2>&1 < /dev/null &
SUITE_PID=$!
printf '%s\n' "$SUITE_PID" > "$LOG_ROOT/$STAMP.pid"
echo "PID: $SUITE_PID"
echo "結果: $OUTPUT_ROOT/summary.csv"
echo "ログ: $LOGFILE"
printf '監視: tail -f %q\n' "$LOGFILE"
