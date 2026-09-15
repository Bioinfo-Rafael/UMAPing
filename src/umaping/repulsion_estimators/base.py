"""凍結trajectory上のteacherと評価専用exact oracle。"""
from __future__ import annotations
import numpy as np
import torch
from umaping.dynamics import TorchTrajectoryView, exact_all_reference_mean_field
from umaping.umap_forces import g_minus


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


class Teacher:
    stochastic = False

    def __init__(self, trajectory, a, b, device='cpu', clip=4.0, seed=0):
        self.trajectory = trajectory
        self.device = torch.device(device)
        self.view = TorchTrajectoryView(trajectory, self.device)
        self.n = trajectory.positions.shape[1]
        self.a, self.b, self.clip = a, b, clip
        self.rng = np.random.default_rng(seed)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.diagnostics = {}

    def forces(self, y, t, indices):
        positions = self.view.positions_at_many(t.repeat_interleave(indices.shape[1]), indices.reshape(-1))
        return g_minus(y[:, None], positions.reshape(len(y), indices.shape[1], 2),
                       self.a, self.b, clip=self.clip)

    def estimate(self, y, t, anchor):
        raise NotImplementedError


@torch.no_grad()
def exact_field(trajectory, y, t, a, b, clip=4.0, query_chunk=32, reference_chunk=2048):
    """既存の全点和実装をquery軸でも分割。学習teacherには使用しない。"""
    result = []
    for start in range(0, len(y), query_chunk):
        stop = start + query_chunk
        result.append(exact_all_reference_mean_field(
            y[start:stop], trajectory, t[start:stop].cpu().numpy(), a, b, y.device,
            clip=clip, chunk_size=reference_chunk))
    return torch.cat(result)


class TrainingQueries:
    """stepだけで再現可能。teacher乱数やモデル乱数から独立。"""
    def __init__(self, trajectory, device, batch_size, jitter_sigma, seed):
        self.view = TorchTrajectoryView(trajectory, torch.device(device))
        self.device = torch.device(device)
        self.batch_size, self.jitter_sigma, self.seed = batch_size, jitter_sigma, seed

    def __call__(self, step):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
        anchor = torch.as_tensor(rng.integers(self.view.positions.shape[1], size=self.batch_size), device=self.device)
        t = torch.as_tensor(rng.uniform(float(self.view.times[0]), float(self.view.times[-1]), self.batch_size),
                            dtype=torch.float32, device=self.device)
        jitter = torch.as_tensor(rng.standard_normal((self.batch_size, 2)), dtype=torch.float32, device=self.device)
        return anchor, t, self.view.positions_at_many(t, anchor) + self.jitter_sigma * jitter
