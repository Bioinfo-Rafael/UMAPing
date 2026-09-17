"""python -m umaping.fit_grid_experiment --help"""
import argparse
from pathlib import Path
import json
import torch


def main(argv=None):
    parser = argparse.ArgumentParser(description='凍結FitGridの移植・最終埋め込み比較・時間holdout（自動downloadなし）')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('promote')
    p.add_argument('--source', type=Path, default=Path('runs/embryoid_body/main'))
    p.add_argument('--benchmark', type=Path, default=Path('results/embryo_repulsion_estimators/recovery_20260915T104741Z_406100'))
    p.add_argument('--output', type=Path, default=Path('runs/embryoid_body/fit_grid'))
    p = sub.add_parser('compare')
    p.add_argument('--uniform', type=Path, default=Path('runs/embryoid_body/main'))
    p.add_argument('--fit-grid', type=Path, default=Path('runs/embryoid_body/fit_grid'))
    p.add_argument('--output', type=Path, default=Path('runs/embryoid_body/comparisons/fit_grid_vs_main'))
    p.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    p.add_argument('--standard-suite', action='store_true', help='新規評価用コピー上で通常評価・分析・advancedも実行')
    for command in ('audit', 'temporal'):
        p = sub.add_parser(command)
        p.add_argument('--data', type=Path, required=True, help='実在するraw h5ad/loom等。auditのみPreparedDataset npzも可能')
        p.add_argument('--group-column', help='省略時は時間列候補が唯一の場合だけ自動検出')
        p.add_argument('--group-order', nargs='+', help='実ラベルの明示的な年代順。数値・day_・E・数値窓・ISO日時は自動整列可')
        if command == 'temporal':
            p.add_argument('--config', type=Path, default=Path('experiments/configs/embryoid_body.yaml'))
            p.add_argument('--run-root', type=Path, default=Path('runs/embryoid_body'))
            p.add_argument('--device', default='cpu', choices=['cpu','cuda'])
            p.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
            p.add_argument('--cell-type-column')
    p = sub.add_parser('summarize', help='完了済みseedだけを集計。学習・推論なし')
    p.add_argument('--run-root', type=Path, default=Path('runs/embryoid_body'))
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'promote':
        from .materialize import materialize
        print(materialize(args.source, args.benchmark, args.output))
    elif args.command == 'compare':
        from .comparison import compare
        compare(args.uniform, args.fit_grid, args.output, torch.device(args.device), args.standard_suite)
    elif args.command == 'audit':
        from .temporal import audit_file
        report = audit_file(args.data, args.group_column, args.group_order)
        if not report['valid_temporal_split']:
            parser.exit(2, '5点以上の年代順を検証できません。学習は実行しません。\n')
    elif args.command == 'summarize':
        from .summary import summarize
        print(summarize(args.run_root, args.seeds, args.output))
    else:
        from .temporal import run_temporal
        run_temporal(args)


if __name__ == '__main__':
    main()
