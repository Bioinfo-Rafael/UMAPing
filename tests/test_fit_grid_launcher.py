"""launcherの順序・引数保持・失敗時停止。重い実験は実行しない。"""
import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_fit_grid_all.sh'


def invoke(tmp_path, fail='', start='1'):
    raw = tmp_path / 'raw with spaces.h5ad'
    raw.touch()
    capture = tmp_path / 'calls.jsonl'
    stub = tmp_path / 'python-stub'
    stub.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
with open(os.environ['CAPTURE'], 'a') as f:
    f.write(json.dumps(sys.argv[1:])+'\\n')
if '-m' in sys.argv:
    command = sys.argv[sys.argv.index('-m')+2]
    if command == os.environ.get('FAIL'):
        sys.exit(7)
''')
    stub.chmod(0o755)
    env = dict(os.environ, PYTHON_BIN=str(stub), CAPTURE=str(capture), FAIL=fail,
               RUN_ROOT=str(tmp_path / 'new runs'))
    result = subprocess.run(['bash',str(SCRIPT),'--raw-file',str(raw),'--device','cpu',
                             '--group-column','sample labels',
                             '--group-order','day 0','--group-order','day 3',
                             '--cell-type-column','cell type','--start-stage',start],
                            env=env,text=True,capture_output=True)
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    commands = [args for args in calls if '-m' in args]
    return result, commands, raw


def test_launcher_order_and_quoted_arguments(tmp_path):
    result,calls,raw = invoke(tmp_path)
    assert result.returncode == 0, result.stderr
    assert [args[3] for args in calls] == ['promote','compare','audit','temporal']
    for args in calls[2:]:
        assert args[args.index('--data')+1] == str(raw)
        assert args[args.index('--group-column')+1] == 'sample labels'
        start = args.index('--group-order')
        assert args[start+1:start+3] == ['day 0','day 3']
    assert calls[-1][calls[-1].index('--cell-type-column')+1] == 'cell type'
    assert 'ALL COMPLETE' in result.stdout


def test_launcher_stops_at_failed_comparison(tmp_path):
    result,calls,_ = invoke(tmp_path,fail='compare')
    assert result.returncode == 7
    assert [args[3] for args in calls] == ['promote','compare']
    assert 'FAILED stage=2/4 compare exit=7' in result.stderr
    assert 'ALL COMPLETE' not in result.stdout


def test_launcher_temporal_start_still_audits(tmp_path):
    result,calls,_ = invoke(tmp_path,start='4')
    assert result.returncode == 0
    assert [args[3] for args in calls] == ['audit','temporal']
