"""チェックポイントごとの四分木。連続時刻ではfieldを補間する。"""
from dataclasses import dataclass
import numpy as np
import torch
from .base import Teacher
from umaping.umap_forces import g_minus


@dataclass
class Cell:
    mass: int
    centroid: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    children: list
    points: np.ndarray | None = None


def build_tree(points, leaf_size=16, depth=0):
    lower, upper = points.min(0), points.max(0)
    center = (lower + upper) / 2
    cell = Cell(len(points), points.mean(0), lower, upper, [])
    if len(points) <= leaf_size or depth >= 40 or np.max(upper-lower) <= 1e-10:
        cell.points = points
        return cell
    quadrant = (points[:, 0] >= center[0]).astype(int) + 2 * (points[:, 1] >= center[1]).astype(int)
    cell.children = [build_tree(points[quadrant == j], leaf_size, depth+1) for j in range(4) if np.any(quadrant == j)]
    return cell


def sources(tree, query, theta):
    centers, masses = [], []
    stack = [tree]
    while stack:
        cell = stack.pop()
        distance = np.linalg.norm(query-cell.centroid)
        inside = np.all(query >= cell.lower) and np.all(query <= cell.upper)
        if cell.points is not None:
            centers.extend(cell.points)
            masses.extend([1] * cell.mass)
        elif not inside and distance > 1e-12 and np.max(cell.upper-cell.lower) / distance < theta:
            centers.append(cell.centroid)
            masses.append(cell.mass)
        else:
            stack.extend(cell.children)
    return np.asarray(centers, dtype=np.float32), np.asarray(masses, dtype=np.float32)


class BarnesHut(Teacher):
    def __init__(self, *args, theta=.5, **kwargs):
        super().__init__(*args, **kwargs)
        if theta < 0:
            raise ValueError('theta must be nonnegative')
        self.theta = theta
        self.trees = [build_tree(p) for p in self.trajectory.positions]

    @torch.no_grad()
    def estimate(self, y, t, anchor):
        queries = y.detach().cpu().numpy()
        lo, hi, frac = self.trajectory._bracket(t.detach().cpu().numpy())
        outputs, interactions = [], 0
        # CPU traversal + device kernel. Host/device transfers are included in timings.
        for i, query in enumerate(queries):
            total = torch.zeros(2, device=self.device)
            for checkpoint, weight in ((lo[i], 1-frac[i]), (hi[i], frac[i])):
                if weight == 0:
                    continue
                centers, masses = sources(self.trees[checkpoint], query, self.theta)
                z = torch.as_tensor(centers, device=self.device)
                mass = torch.as_tensor(masses, device=self.device)
                total += weight * (g_minus(y[i], z, self.a, self.b, clip=self.clip) * mass[:, None]).sum(0) / self.n
                interactions += len(masses)
            outputs.append(total)
        self.diagnostics = dict(force_interactions=interactions / len(y), score_seconds=0., theta=self.theta)
        return torch.stack(outputs)
