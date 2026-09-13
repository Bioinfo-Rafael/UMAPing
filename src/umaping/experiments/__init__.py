"""Reusable experiment infrastructure for Experiments B/C/D (see
experiments/README.md): a common baseline interface (`experiments.baselines`),
an experiment registry + runner (`experiments.registry`, `experiments.runner`),
and the reference-set-size scaling benchmark (`experiments.scale_benchmark`).

This package deliberately does not reimplement anything `pipeline.py`
already does: `runner.run_experiment` calls the existing `run_training` /
`run_evaluation` / `run_analysis` / `run_advanced_analysis` functions
directly, so every new dataset gets the full existing metric/figure suite
for free, on top of the additional baselines/outputs this package adds.
"""

from __future__ import annotations
