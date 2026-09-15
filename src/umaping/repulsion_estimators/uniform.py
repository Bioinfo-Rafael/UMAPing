"""既存と同じ復元抽出・単純平均。selfも候補に含む。"""
import torch
from .base import Teacher


class UniformMC(Teacher):
    stochastic = True

    def __init__(self, *args, samples=64, **kwargs):
        super().__init__(*args, **kwargs)
        if samples < 1:
            raise ValueError('samples must be positive')
        self.samples = samples

    @torch.no_grad()
    def estimate(self, y, t, anchor):
        indices = torch.as_tensor(self.rng.integers(self.n, size=(len(y), self.samples)), device=self.device)
        self.diagnostics = {'force_interactions': self.samples, 'score_seconds': 0.0}
        return self.forces(y, t, indices).mean(1)
