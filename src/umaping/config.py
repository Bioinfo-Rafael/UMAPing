"""Typed configuration schema, loaded from the YAML files under ``configs/``.

Every experiment is fully described by one YAML file plus CLI overrides for
``--device``, ``--seed`` and ``--resume``. The loaded :class:`Config` is
re-serialized verbatim into ``<run-dir>/config.yaml`` so a run directory is
self-describing (see ``utils/io.py`` for the atomic writer).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

from umaping.utils.io import load_yaml, save_yaml

_ACTIVATIONS = {"gelu", "relu", "silu", "tanh", "leaky_relu"}


@dataclass
class UMAPConfig:
    """Fuzzy-simplicial-set + low-dimensional-kernel hyperparameters, matching
    umap-learn's own ``n_neighbors``/``min_dist``/``spread``/``negative_sample_rate``
    conventions (see docs/method.md and RUNTIME_CHECKS.md for the verified
    correspondence to the current umap-learn implementation)."""

    n_neighbors: int = 15
    embedding_dim: int = 2
    min_dist: float = 0.1
    spread: float = 1.0
    negative_sample_rate: float = 5.0
    local_connectivity: float = 1.0
    set_op_mix_ratio: float = 1.0
    smooth_knn_n_iter: int = 64
    smooth_knn_bandwidth: float = 1.0
    smooth_knn_min_k_dist_scale: float = 1e-3
    # Numerical floor for sigma / denominators throughout graph + force code.
    eps: float = 1e-8


@dataclass
class RetrieverConfig:
    """DPR-style dual encoder trained against the *directed* fuzzy membership
    matrix Mu (never the symmetrized graph W)."""

    hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    retrieval_dim: int = 128
    activation: str = "gelu"
    temperature: float = 0.1
    candidate_multiplier: int = 4
    min_candidates: int = 64
    ann_backend: str = "exact"  # "exact" (default, deterministic) or "pynndescent"
    batch_size: int = 256
    n_random_negatives: int = 64
    exclude_known_neighbors_from_negatives: bool = True
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    steps: int = 2000
    log_every: int = 50


@dataclass
class SpectralConfig:
    """SpectralNet-style pointwise encoder trained on the *symmetric* graph W."""

    hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    raw_output_dim: int = 3  # r = embedding_dim + 1
    activation: str = "gelu"
    orthogonality_weight: float = 1.0
    edge_batch_size: int = 4096
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    steps: int = 2000
    calibration_target_scale: float = 10.0
    log_every: int = 50


@dataclass
class FlowConfig:
    """Reference mean UMAP dynamics (Y_ref(0) -> trajectory checkpoints)."""

    n_steps: int = 200
    initial_alpha: float = 1.0
    dynamics_negative_samples: int = 64  # M_neg for B_i(t) during trajectory build
    grad_clip: float | None = 4.0
    checkpoint_stride: int = 1  # 1 = store every step


@dataclass
class RepulsionConfig:
    """B_phi(y, t): residual MLP distilling the mean negative field B_X(y, t)."""

    hidden_dim: int = 256
    n_residual_blocks: int = 4
    time_embed_dim: int = 32
    activation: str = "gelu"
    jitter_sigma: float = 0.1
    teacher_negative_samples: int = 64
    weight_by_row_mass: bool = False
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    steps: int = 4000
    batch_size: int = 512
    log_every: int = 100


@dataclass
class EvalConfig:
    k: int = 15
    oracle_repulsion_samples: int = 2048
    repulsion_oracle_query_subset: int = 50
    trustworthiness_n_neighbors: int = 15


@dataclass
class DatasetConfig:
    name: str
    raw_dir: str = "data/raw"
    input_dim: int = 256
    # Dataset-specific knobs (pca_dim/use_pca/image_size for COIL,
    # n_hvg/n_pcs/reference_techs/query_techs for pancreas). Read by the
    # corresponding module in umaping.data.
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    dataset: DatasetConfig
    umap: UMAPConfig = field(default_factory=UMAPConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    repulsion: RepulsionConfig = field(default_factory=RepulsionConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 0
    float_dtype: str = "float32"

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = load_yaml(path)
        cfg = _from_dict(cls, raw)
        cfg.validate()
        return cfg

    def save(self, path: str | Path) -> None:
        save_yaml(path, dataclasses.asdict(self))

    def validate(self) -> None:
        errors: list[str] = []
        if self.spectral.raw_output_dim != self.umap.embedding_dim + 1:
            errors.append(
                "spectral.raw_output_dim must equal umap.embedding_dim + 1 "
                f"(got raw_output_dim={self.spectral.raw_output_dim}, "
                f"embedding_dim={self.umap.embedding_dim})"
            )
        if self.umap.n_neighbors < 1:
            errors.append("umap.n_neighbors must be >= 1")
        if self.retriever.min_candidates < self.eval.k:
            errors.append("retriever.min_candidates must be >= eval.k")
        if self.retriever.activation not in _ACTIVATIONS:
            errors.append(f"retriever.activation must be one of {_ACTIVATIONS}")
        if self.spectral.activation not in _ACTIVATIONS:
            errors.append(f"spectral.activation must be one of {_ACTIVATIONS}")
        if self.repulsion.activation not in _ACTIVATIONS:
            errors.append(f"repulsion.activation must be one of {_ACTIVATIONS}")
        if self.retriever.ann_backend not in {"exact", "pynndescent"}:
            errors.append("retriever.ann_backend must be 'exact' or 'pynndescent'")
        if self.dataset.input_dim <= 0:
            errors.append("dataset.input_dim must be positive")
        if errors:
            raise ValueError("Invalid config:\n  - " + "\n  - ".join(errors))

    def candidate_pool_size(self) -> int:
        """M = max(4 * k, 64), the default candidate-recall size before rerank."""
        return max(self.retriever.candidate_multiplier * self.umap.n_neighbors, self.retriever.min_candidates)


def _from_dict(cls, data: dict[str, Any] | None):
    data = dict(data or {})
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        field_type = hints[f.name]
        if dataclasses.is_dataclass(field_type) and isinstance(value, dict):
            value = _from_dict(field_type, value)
        kwargs[f.name] = value
    return cls(**kwargs)
