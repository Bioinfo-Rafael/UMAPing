#!/usr/bin/env python3
"""既存の通常環境から、インストールし直さず比較CLIを呼ぶ。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from umaping.comparison.__main__ import main

if __name__ == "__main__":
    main()
