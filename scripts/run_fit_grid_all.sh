#!/usr/bin/env bash
# 既存成果物は上書きせず、FitGrid移植→最終比較→raw監査→temporalを順次実行。
# ログ例: nohup bash scripts/run_fit_grid_all.sh > "$LOG_FILE" 2>&1 &
set -Eeuo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
RAW_FILE="${RAW_FILE:-}"
GROUP_COLUMN="${GROUP_COLUMN:-}"
CELL_TYPE_COLUMN="${CELL_TYPE_COLUMN:-}"
SOURCE_RUN="${SOURCE_RUN:-runs/embryoid_body/main}"
BENCHMARK_RUN="${BENCHMARK_RUN:-results/embryo_repulsion_estimators/recovery_20260915T104741Z_406100}"
RUN_ROOT="${RUN_ROOT:-runs/embryoid_body}"
START_STAGE=1
GROUP_ORDER=()

usage() {
  cat <<'HELP'
使い方: bash scripts/run_fit_grid_all.sh [オプション]
  --raw-file PATH       注釈付きrawファイル。省略時はdata/raw/embryoid_body内の
                        h5ad/loomが1個の場合だけ選択（downloadなし）。
  --group-column NAME   時間列。省略時はPython CLIが一意な候補だけ自動検出。
  --group-order LABEL   年代順に1ラベルずつ繰り返して指定。
  --cell-type-column NAME  時間と別のcell-type annotation。
  --device cuda|cpu    既定cuda。
  --start-stage 1|2|3|4  既定1。完了済み段階を明示的に飛ばす場合のみ指定。
  --help

環境変数: PYTHON_BIN（既定.venv/bin/python）、RAW_FILE、GROUP_COLUMN、
CELL_TYPE_COLUMN、DEVICE、SOURCE_RUN、BENCHMARK_RUN、RUN_ROOT。
seedは0,1,2、全queryを使用。既存出力があれば停止し、削除や再利用はしない。
--start-stageは途中checkpoint再開ではない。失敗段階の出力が残っている場合は、
その出力を保護したまま別のRUN_ROOTを指定する必要がある。
HELP
}
need_value() {
  if (( $# < 2 )) || [[ -z "$2" ]]; then
    printf '値が必要です: %s\n' "$1" >&2
    exit 2
  fi
}
while (( $# )); do
  case "$1" in
    --raw-file) need_value "$@"; RAW_FILE="$2"; shift 2 ;;
    --group-column) need_value "$@"; GROUP_COLUMN="$2"; shift 2 ;;
    --group-order) need_value "$@"; GROUP_ORDER+=("$2"); shift 2 ;;
    --cell-type-column) need_value "$@"; CELL_TYPE_COLUMN="$2"; shift 2 ;;
    --device) need_value "$@"; DEVICE="$2"; shift 2 ;;
    --start-stage) need_value "$@"; START_STAGE="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) printf '不明なオプション: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$DEVICE" == cuda || "$DEVICE" == cpu ]] || { echo 'deviceはcudaまたはcpuです' >&2; exit 2; }
[[ "$START_STAGE" =~ ^[1-4]$ ]] || { echo 'start-stageは1〜4です' >&2; exit 2; }
command -v "$PYTHON_BIN" >/dev/null || { echo "Pythonがありません: $PYTHON_BIN" >&2; exit 2; }

STAGE='preflight'
trap 'code=$?; printf "[%s] FAILED stage=%s exit=%s line=%s\n" "$(date -u +%FT%TZ)" "$STAGE" "$code" "$LINENO" >&2; exit "$code"' ERR
trap 'printf "[%s] INTERRUPTED stage=%s\n" "$(date -u +%FT%TZ)" "$STAGE" >&2; exit 130' INT TERM
run() {
  printf '[%s] COMMAND:' "$(date -u +%FT%TZ)"
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

printf '[%s] START repo=%s pid=%s start_stage=%s\n' "$(date -u +%FT%TZ)" "$REPO_DIR" "$$" "$START_STAGE"
printf 'commit=%s\n' "$(git rev-parse HEAD)"
printf 'Python=%s device=%s\n' "$PYTHON_BIN" "$DEVICE"
export PYTHONUNBUFFERED=1
# 長時間の段階1/2に入る前に、rawパス・device・出力の衝突だけ検査する。
# 実時間点の監査と学習は必ず段階3/4で行う。
if [[ -z "$RAW_FILE" ]]; then
  RAW_FILE="$("$PYTHON_BIN" - <<'PY'
from pathlib import Path
import sys
root = Path('data/raw/embryoid_body')
files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ('.h5ad', '.loom'))
if len(files) != 1:
    print(f'rawの自動選択不可: {root} 内のh5ad/loom候補数={len(files)}。--raw-fileで対象を指定してください。', file=sys.stderr)
    for path in files:
        print(path, file=sys.stderr)
    sys.exit(2)
print(files[0].resolve())
PY
)"
fi
[[ -f "$RAW_FILE" ]] || { echo "rawがありません: $RAW_FILE" >&2; exit 2; }
printf 'raw=%s\n' "$RAW_FILE"
run "$PYTHON_BIN" - "$DEVICE" "$RUN_ROOT" "$START_STAGE" <<'PY'
from pathlib import Path
import sys
import torch
import umaping.fit_grid_experiment
if sys.argv[1] == 'cuda' and not torch.cuda.is_available():
    raise SystemExit('CUDAが使用できません。GPU環境を確認するか --device cpu を指定してください。')
root, start = Path(sys.argv[2]), int(sys.argv[3])
outputs = [root/'temporal_holdout_uniform', root/'temporal_holdout_fit_grid', root/'comparisons/temporal_holdout']
for seed in (1, 2):
    outputs += [root/f'temporal_seed_{seed}_uniform', root/f'temporal_seed_{seed}_fit_grid', root/'comparisons'/f'temporal_seed_{seed}']
if start <= 1:
    outputs.append(root/'fit_grid')
if start <= 2:
    outputs.append(root/'comparisons/fit_grid_vs_main')
conflicts = [str(p) for p in outputs if p.exists() or p.is_symlink()]
if conflicts:
    raise SystemExit('既存出力は上書きしません:\n'+'\n'.join(conflicts))
PY

TEMPORAL_ARGS=(--data "$RAW_FILE")
if [[ -n "$GROUP_COLUMN" ]]; then TEMPORAL_ARGS+=(--group-column "$GROUP_COLUMN"); fi
if (( ${#GROUP_ORDER[@]} )); then TEMPORAL_ARGS+=(--group-order "${GROUP_ORDER[@]}"); fi

if (( START_STAGE <= 1 )); then
  STAGE='1/4 promote'
  printf '[%s] START %s\n' "$(date -u +%FT%TZ)" "$STAGE"
  run "$PYTHON_BIN" -u -m umaping.fit_grid_experiment promote \
    --source "$SOURCE_RUN" --benchmark "$BENCHMARK_RUN" --output "$RUN_ROOT/fit_grid"
  printf '[%s] DONE %s\n' "$(date -u +%FT%TZ)" "$STAGE"
fi
if (( START_STAGE <= 2 )); then
  STAGE='2/4 compare'
  printf '[%s] START %s\n' "$(date -u +%FT%TZ)" "$STAGE"
  run "$PYTHON_BIN" -u -m umaping.fit_grid_experiment compare \
    --uniform "$SOURCE_RUN" --fit-grid "$RUN_ROOT/fit_grid" \
    --output "$RUN_ROOT/comparisons/fit_grid_vs_main" --standard-suite --device "$DEVICE"
  printf '[%s] DONE %s\n' "$(date -u +%FT%TZ)" "$STAGE"
fi
# --start-stage 4でも監査は省略しない。
STAGE='3/4 audit'
printf '[%s] START %s\n' "$(date -u +%FT%TZ)" "$STAGE"
run "$PYTHON_BIN" -u -m umaping.fit_grid_experiment audit "${TEMPORAL_ARGS[@]}"
printf '[%s] DONE %s\n' "$(date -u +%FT%TZ)" "$STAGE"

STAGE='4/4 temporal'
printf '[%s] START %s\n' "$(date -u +%FT%TZ)" "$STAGE"
if [[ -n "$CELL_TYPE_COLUMN" ]]; then TEMPORAL_ARGS+=(--cell-type-column "$CELL_TYPE_COLUMN"); fi
run "$PYTHON_BIN" -u -m umaping.fit_grid_experiment temporal "${TEMPORAL_ARGS[@]}" \
  --config experiments/configs/embryoid_body.yaml --run-root "$RUN_ROOT" --seeds 0 1 2 --device "$DEVICE"
printf '[%s] DONE %s\n' "$(date -u +%FT%TZ)" "$STAGE"
printf '[%s] ALL COMPLETE: %s/comparisons/fit_grid_vs_main ; %s/comparisons/temporal_holdout\n' \
  "$(date -u +%FT%TZ)" "$RUN_ROOT" "$RUN_ROOT"
