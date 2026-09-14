#!/usr/bin/env python3
"""選択された隔離環境のPythonで外部比較を実行する。"""
from pathlib import Path
import sys

# 親プロセスではUMAPingの依存packageすら不要。実験workerには専用環境を使う。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from umaping.experiments.external import main

if __name__ == "__main__":
    raise SystemExit(main())
