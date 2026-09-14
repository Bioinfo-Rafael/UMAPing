#!/usr/bin/env bash
# 現在のPython環境に触れず、リポジトリ外に比較用環境を作る。
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
EXTERNAL_ROOT="${1:-$(dirname "$REPO_ROOT")/UMAPing_external_baselines}"
mkdir -p "$EXTERNAL_ROOT"
EXTERNAL_ROOT="$(cd "$EXTERNAL_ROOT" && pwd -P)"
case "$EXTERNAL_ROOT/" in
  "$REPO_ROOT/"*) echo '外部環境の保存先をリポジトリ外にしてください。' >&2; exit 1 ;;
esac
# 環境変数の変更はこの子シェルだけ。activate/pip/uv syncは使わない。
unset VIRTUAL_ENV CONDA_PREFIX PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export UV_PYTHON_INSTALL_DIR="$EXTERNAL_ROOT/python"
export UV_PYTHON_BIN_DIR="$EXTERNAL_ROOT/bin"
export UV_CACHE_DIR="$EXTERNAL_ROOT/uv-cache"
mkdir -p "$EXTERNAL_ROOT/bin" "$EXTERNAL_ROOT/setup_logs" "$EXTERNAL_ROOT/venvs"
if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
elif [ -x "$EXTERNAL_ROOT/bin/uv" ]; then
  UV_BIN="$EXTERNAL_ROOT/bin/uv"
else
  curl -LsSf https://astral.sh/uv/0.8.22/install.sh -o "$EXTERNAL_ROOT/uv-install.sh"
  UV_UNMANAGED_INSTALL="$EXTERNAL_ROOT/bin" sh "$EXTERNAL_ROOT/uv-install.sh"
  UV_BIN="$EXTERNAL_ROOT/bin/uv"
fi
"$UV_BIN" python install 3.11.11
CONTROL_PYTHON="$("$UV_BIN" python find --managed-python 3.11.11)"
printf '%s\n' "$CONTROL_PYTHON" > "$EXTERNAL_ROOT/control-python.txt"

# 既存の無関係なvenvやsymlinkへインストールしない。
"$CONTROL_PYTHON" - "$REPO_ROOT" "$EXTERNAL_ROOT" <<'PY'
from pathlib import Path
import sys
repo, root = map(lambda p: Path(p).resolve(), sys.argv[1:])
for name in ('parametric_umap', 'numap', 'paramrepulsor'):
    target = root / 'venvs' / name
    if target.is_symlink() or (target.exists() and not (target / '.umaping-external-owned').is_file()):
        raise SystemExit(f'専用環境と確認できない既存パスです: {target}')
    if not target.resolve().is_relative_to(root) or target.resolve().is_relative_to(repo):
        raise SystemExit(f'環境の実パスが不正です: {target}')
PY

setup_one() (
  set -euo pipefail
  baseline="$1"
  env_dir="$EXTERNAL_ROOT/venvs/$baseline"
  if [ ! -d "$env_dir" ]; then
    "$UV_BIN" venv --python "$CONTROL_PYTHON" "$env_dir"
    touch "$env_dir/.umaping-external-owned"
  fi
  rm -f "$env_dir/.ready"
  env_python="$env_dir/bin/python"
  "$env_python" -c 'import sys; assert sys.version_info[:2] == (3, 11)'
  if [ "$baseline" = parametric_umap ]; then
    torch_index=https://download.pytorch.org/whl/cpu
    torch_version='2.6.0+cpu'
  else
    torch_index=https://download.pytorch.org/whl/cu124
    torch_version='2.6.0+cu124'
  fi
  "$UV_BIN" pip install --python "$env_python" --index-url "$torch_index" "torch==$torch_version"
  printf 'torch==%s\n' "$torch_version" > "$env_dir/torch-constraint.txt"
  "$UV_BIN" pip install --python "$env_python" -c "$env_dir/torch-constraint.txt" \
    -r "$REPO_ROOT/scripts/external_requirements/$baseline.txt" "$REPO_ROOT"
  "$UV_BIN" pip check --python "$env_python"
  # インポートのみ。データ取得・fit/transformは環境構築中には呼ばない。
  "$env_python" - "$baseline" <<'PY'
import sys
from umaping.data.preprocessing import load_prepared_dataset
from umaping.evaluation.embedding import evaluate_embedding
from umaping.evaluation.advanced import multi_k_recall_and_ndcg
if sys.argv[1] == 'parametric_umap':
    from umap.parametric_umap import ParametricUMAP
elif sys.argv[1] == 'numap':
    from numap import NUMAP
else:
    from parampacmap import ParamPaCMAP
print('インポート確認成功:', sys.argv[1])
PY
  "$UV_BIN" pip freeze --python "$env_python" > "$env_dir/requirements.freeze.txt"
  git -C "$REPO_ROOT" rev-parse HEAD > "$env_dir/.ready"
)

for baseline in parametric_umap numap paramrepulsor; do
  echo "環境構築: ${baseline}（$EXTERNAL_ROOT/setup_logs/$baseline.log）"
  # if関数呼出しだとbashのerrexitが無効になるため、独立したsubshellの終了値を取得。
  set +e
  setup_one "$baseline" > "$EXTERNAL_ROOT/setup_logs/$baseline.log" 2>&1
  setup_status=$?
  set -e
  if [ "$setup_status" -eq 0 ]; then
    echo "OK: $baseline"
  else
    echo "FAILED: ${baseline}（残りの環境構築は継続します）"
    tail -n 15 "$EXTERNAL_ROOT/setup_logs/$baseline.log"
  fi
done
echo 'OOS-UMAPは公式の学習・OOS実行コード不足のため利用不可として記録します。'
echo "準備終了: $EXTERNAL_ROOT"
