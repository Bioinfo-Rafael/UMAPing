"""All figure generation (docs/method.md; see OUTPUTS in README). Uses the
non-interactive Agg backend throughout since runs happen on headless remote
machines."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from umaping.dynamics import ReferenceTrajectory, mean_negative_field  # noqa: E402
from umaping.models.repulsion import RepulsionField  # noqa: E402


def plot_embedding(
    reference_embedding: np.ndarray,
    reference_label_sets: dict[str, np.ndarray],
    path: str | Path,
    title: str,
    query_embedding: np.ndarray | None = None,
    query_label_sets: dict[str, np.ndarray] | None = None,
) -> None:
    """One panel per label set (e.g. just "object_id" for COIL; both
    "celltype" and "tech" for pancreas)."""
    label_names = list(reference_label_sets.keys()) or ["_none"]
    n_panels = len(label_names)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.2), squeeze=False)
    axes = axes[0]

    for ax, label_name in zip(axes, label_names):
        if label_name == "_none":
            ax.scatter(reference_embedding[:, 0], reference_embedding[:, 1], s=6, alpha=0.6, c="tab:blue", label="reference")
        else:
            ref_labels = reference_label_sets[label_name]
            categories = pd.Categorical(ref_labels)
            ax.scatter(
                reference_embedding[:, 0],
                reference_embedding[:, 1],
                c=categories.codes,
                cmap="tab20",
                s=6,
                alpha=0.6,
                label="reference",
            )
            if query_embedding is not None and query_label_sets is not None and label_name in query_label_sets:
                q_codes = pd.Categorical(query_label_sets[label_name], categories=categories.categories).codes
                ax.scatter(
                    query_embedding[:, 0],
                    query_embedding[:, 1],
                    c=q_codes,
                    cmap="tab20",
                    s=18,
                    marker="x",
                    alpha=0.9,
                    label="query",
                )
        if query_embedding is not None and (query_label_sets is None or label_name not in query_label_sets):
            ax.scatter(query_embedding[:, 0], query_embedding[:, 1], c="black", s=10, marker="x", alpha=0.6, label="query")

        ax.set_title(f"{title}" + (f" -- {label_name}" if label_name != "_none" else ""))
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        ax.legend(loc="best", fontsize=8, markerscale=2)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss_curve(curves: dict[str, list[float]], path: str | Path, title: str, ylabel: str = "loss") -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, values in curves.items():
        if len(values) == 0:
            continue
        ax.plot(values, linewidth=1, label=name)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_retriever_training(losses: list[float], accuracies: list[float], path: str | Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(losses, color="tab:blue", linewidth=1, label="InfoNCE loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(accuracies, color="tab:orange", linewidth=1, alpha=0.7, label="batch top-1 accuracy")
    ax2.set_ylabel("batch top-1 accuracy", color="tab:orange")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.suptitle("Retriever training")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_retriever_recall(metrics: dict[str, float], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    names = list(metrics.keys())
    values = [metrics[n] for n in names]
    ax.bar(names, values, color="tab:blue")
    ax.set_ylim(0, max(1.05, max(values) * 1.1 if values else 1.0))
    ax.set_ylabel("score")
    ax.set_title("Retriever recall metrics")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_query_trajectories(
    reference_embedding: np.ndarray,
    trajectories: list[np.ndarray],
    path: str | Path,
    max_trajectories: int = 25,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(reference_embedding[:, 0], reference_embedding[:, 1], c="lightgray", s=4, alpha=0.5, label="reference")
    for traj in trajectories[:max_trajectories]:
        ax.plot(traj[:, 0], traj[:, 1], linewidth=0.8, alpha=0.7, color="tab:blue")
        ax.scatter([traj[0, 0]], [traj[0, 1]], c="tab:green", s=18, marker="o", zorder=3)
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], c="tab:red", s=24, marker="*", zorder=3)
    ax.set_title("Query trajectories (green = y*(0), red = final y*)")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_repulsion_field(
    model: RepulsionField,
    trajectory: ReferenceTrajectory,
    a: float,
    b: float,
    t: float,
    embedding_range: tuple[tuple[float, float], tuple[float, float]],
    device: torch.device,
    path: str | Path,
    resolution: int = 18,
    mc_samples: int = 256,
    seed: int = 0,
) -> None:
    (xmin, xmax), (ymin, ymax) = embedding_range
    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)

    rng = np.random.default_rng(seed)
    n = trajectory.positions.shape[1]
    idx = rng.integers(0, n, size=min(mc_samples, n))
    y_neg = torch.as_tensor(trajectory.positions_at(t, indices=idx), dtype=torch.float32, device=device)
    y_grid_t = torch.as_tensor(grid_points, dtype=torch.float32, device=device)

    model = model.to(device).eval()
    with torch.no_grad():
        teacher = mean_negative_field(y_grid_t, y_neg, a, b).cpu().numpy()
        t_t = torch.full((grid_points.shape[0],), t, dtype=torch.float32, device=device)
        pred = model(y_grid_t, t_t).cpu().numpy()

    error = np.linalg.norm(pred - teacher, axis=1).reshape(resolution, resolution)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    for ax, field, name in zip(axes[:2], [pred, teacher], ["Predicted B_phi", "Monte-Carlo teacher"]):
        u = field[:, 0].reshape(resolution, resolution)
        v = field[:, 1].reshape(resolution, resolution)
        ax.quiver(grid_x, grid_y, u, v, angles="xy", scale_units="xy", pivot="mid")
        ax.set_title(f"{name} (t={t:.2f})")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    im = axes[2].imshow(error, origin="lower", extent=[xmin, xmax, ymin, ymax], cmap="magma", aspect="auto")
    axes[2].set_title("||predicted - teacher||")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
