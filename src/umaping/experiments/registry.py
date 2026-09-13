"""Experiment registry: maps a short experiment name (as passed to
``umaping experiment --experiment <name>``) to its config file, dataset
name, download function, and citation/source metadata -- all pulled
directly from the corresponding `data/*.py` module (single source of
truth), never duplicated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    dataset_name: str
    config_path: str
    download_fn: Callable[[str | Path], Path]
    citation: str
    source_url: str


_REGISTRY: dict[str, ExperimentSpec] = {}


def _register(spec: ExperimentSpec) -> None:
    _REGISTRY[spec.name] = spec


def get_experiment(name: str) -> ExperimentSpec:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown experiment '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_experiments() -> list[str]:
    return sorted(_REGISTRY)


def _build_registry() -> None:
    from umaping.data import embryoid_body, fashion_mnist, hong_ed, mnist_oos, organoid, twenty_newsgroups

    _register(
        ExperimentSpec(
            name="fashion_mnist",
            dataset_name="fashion_mnist",
            config_path="experiments/configs/fashion_mnist.yaml",
            download_fn=fashion_mnist.download_fashion_mnist,
            citation=fashion_mnist.CITATION,
            source_url=fashion_mnist.SOURCE_URL,
        )
    )
    _register(
        ExperimentSpec(
            name="twenty_newsgroups",
            dataset_name="twenty_newsgroups",
            config_path="experiments/configs/twenty_newsgroups.yaml",
            download_fn=twenty_newsgroups.download_twenty_newsgroups,
            citation=twenty_newsgroups.CITATION,
            source_url=twenty_newsgroups.SOURCE_URL,
        )
    )
    _register(
        ExperimentSpec(
            name="mnist_oos",
            dataset_name="mnist_oos",
            config_path="experiments/configs/mnist_oos.yaml",
            download_fn=mnist_oos.download_mnist,
            citation=mnist_oos.CITATION,
            source_url=mnist_oos.SOURCE_URL,
        )
    )
    _register(
        ExperimentSpec(
            name="hong_ed",
            dataset_name="hong_ed",
            config_path="experiments/configs/hong_ed.yaml",
            download_fn=hong_ed.download_hong_ed,
            citation=hong_ed.CITATION,
            source_url=hong_ed.SOURCE_URL,
        )
    )
    _register(
        ExperimentSpec(
            name="organoid",
            dataset_name="organoid",
            config_path="experiments/configs/organoid.yaml",
            download_fn=organoid.download_organoid,
            citation=organoid.CITATION,
            source_url=organoid.SOURCE_URL,
        )
    )
    _register(
        ExperimentSpec(
            name="embryoid_body",
            dataset_name="embryoid_body",
            config_path="experiments/configs/embryoid_body.yaml",
            download_fn=embryoid_body.download_embryoid_body,
            citation=embryoid_body.CITATION,
            source_url=embryoid_body.SOURCE_URL,
        )
    )


_build_registry()
