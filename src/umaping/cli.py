"""The `umaping` command-line interface.

    umaping download  --dataset {coil20,coil100,pancreas}
    umaping train     --config configs/X.yaml --run-dir runs/X/main
    umaping evaluate  --run-dir runs/X/main
    umaping analyze   --run-dir runs/X/main
    umaping pipeline  --config configs/X.yaml --run-dir runs/X/main

`train` runs preprocessing -> graph -> retriever -> spectral -> reference
flow -> repulsion field (checkpointed, resumable). `pipeline` additionally
runs `evaluate` and `analyze` afterwards. See README.md for full usage.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from umaping.config import Config
from umaping.utils.device import resolve_device
from umaping.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _guard_run_dir(run_dir: Path, resume: bool) -> None:
    """Refuse to silently reuse a run directory that already has completed
    stages unless --resume was passed explicitly."""
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists() and not resume:
        raise SystemExit(
            f"Run directory '{run_dir}' already contains a previous run (metadata.json exists). "
            "Pass --resume to continue it, or point --run-dir at an empty directory to start fresh."
        )


def cmd_download(args: argparse.Namespace) -> None:
    from umaping.data.coil import download_coil
    from umaping.data.pancreas import download_pancreas

    raw_dir = Path(args.data_root) / args.dataset
    if args.dataset in ("coil20", "coil100"):
        path = download_coil(args.dataset, raw_dir)
        logger.info("%s ready at %s", args.dataset, path)
    elif args.dataset == "pancreas":
        path = download_pancreas(raw_dir)
        logger.info("pancreas.h5ad ready at %s", path)
    else:
        raise SystemExit(f"Unknown dataset '{args.dataset}'")


def cmd_train(args: argparse.Namespace) -> None:
    from umaping.pipeline import run_training

    cfg = Config.load(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    device = resolve_device(args.device)
    run_dir = Path(args.run_dir)
    _guard_run_dir(run_dir, args.resume)
    set_seed(cfg.seed)

    logger.info("Training on device=%s, dataset=%s, run_dir=%s", device, cfg.dataset.name, run_dir)
    run_training(cfg, run_dir, device)
    logger.info("Training complete. Checkpoints + memory under %s", run_dir)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from umaping.pipeline import run_evaluation

    device = resolve_device(args.device)
    logger.info("Evaluating run at %s on device=%s", args.run_dir, device)
    run_evaluation(args.run_dir, device)
    logger.info("Evaluation complete. Metrics under %s/metrics", args.run_dir)


def cmd_analyze(args: argparse.Namespace) -> None:
    from umaping.pipeline import run_analysis

    device = resolve_device(args.device)
    logger.info("Generating figures for run at %s on device=%s", args.run_dir, device)
    run_analysis(args.run_dir, device)
    logger.info("Analysis complete. Figures under %s/figures", args.run_dir)


def cmd_pipeline(args: argparse.Namespace) -> None:
    from umaping.pipeline import run_analysis, run_evaluation, run_training

    cfg = Config.load(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    device = resolve_device(args.device)
    run_dir = Path(args.run_dir)
    _guard_run_dir(run_dir, args.resume)
    set_seed(cfg.seed)

    logger.info("Running full pipeline on device=%s, dataset=%s, run_dir=%s", device, cfg.dataset.name, run_dir)
    run_training(cfg, run_dir, device)
    run_evaluation(run_dir, device)
    run_analysis(run_dir, device)
    logger.info("Pipeline complete. See %s/metrics and %s/figures", run_dir, run_dir)


def _add_device_seed_resume(p: argparse.ArgumentParser, with_config: bool) -> None:
    if with_config:
        p.add_argument("--config", required=True, help="Path to a YAML config (see configs/).")
        p.add_argument("--seed", type=int, default=None, help="Overrides config.seed if given.")
        p.add_argument("--resume", action="store_true", help="Continue a run-dir that already has completed stages.")
    p.add_argument("--run-dir", required=True, help="Directory for checkpoints/memory/metrics/figures.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="umaping", description="Inductive, single-query approximation of UMAP.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_download = subparsers.add_parser("download", help="Download a raw dataset (COIL-20/100 or pancreas).")
    p_download.add_argument("--dataset", required=True, choices=["coil20", "coil100", "pancreas"])
    p_download.add_argument("--data-root", default="data/raw", help="Root directory to download datasets under.")
    p_download.set_defaults(func=cmd_download)

    p_train = subparsers.add_parser("train", help="Preprocessing -> graph -> retriever -> spectral -> flow -> repulsion.")
    _add_device_seed_resume(p_train, with_config=True)
    p_train.set_defaults(func=cmd_train)

    p_eval = subparsers.add_parser("evaluate", help="Compute all metrics for a trained run.")
    _add_device_seed_resume(p_eval, with_config=False)
    p_eval.set_defaults(func=cmd_evaluate)

    p_analyze = subparsers.add_parser("analyze", help="Generate all figures for a trained/evaluated run.")
    _add_device_seed_resume(p_analyze, with_config=False)
    p_analyze.set_defaults(func=cmd_analyze)

    p_pipeline = subparsers.add_parser("pipeline", help="train -> evaluate -> analyze, chained.")
    _add_device_seed_resume(p_pipeline, with_config=True)
    p_pipeline.set_defaults(func=cmd_pipeline)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
