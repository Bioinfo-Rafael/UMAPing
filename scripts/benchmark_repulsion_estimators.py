#!/usr/bin/env python3
"""既存環境で実行する凍結Embryo repulsion teacher benchmark。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from umaping.repulsion_estimators.runner import main

if __name__ == '__main__':
    main()
