"""Dual Top-Lを全評価し、補集合は順位indexの写像で一様復元抽出。"""
import time
import torch
from .dual_importance import DualProposal
from .base import synchronize


class DualTopL(DualProposal):
    stochastic = True

    def __init__(self, *args, top_l=32, tail_samples=32, **kwargs):
        super().__init__(*args, **kwargs)
        if top_l < 0 or tail_samples < 1:
            raise ValueError('Invalid Top-L budget')
        self.top_l, self.tail_samples = min(top_l, self.n), tail_samples

    @torch.no_grad()
    def estimate(self, y, t, anchor):
        outputs, score_time = [], 0.
        for start in range(0, len(y), self.score_batch):
            stop = start + self.score_batch
            synchronize(self.device)
            begin = time.perf_counter()
            top = self.scores(anchor[start:stop]).topk(self.top_l, dim=1).indices
            synchronize(self.device)
            score_time += time.perf_counter() - begin
            head = self.forces(y[start:stop], t[start:stop], top).sum(1) if self.top_l else torch.zeros_like(y[start:stop])
            rest = self.n - self.top_l
            if rest:
                rank = torch.as_tensor(self.rng.integers(rest, size=(len(top), self.tail_samples)), device=self.device)
                # 排除indexを昇順に飛び越す。棄却抽出の無限loopを避ける。
                for forbidden in top.sort(1).values.unbind(1):
                    rank = rank + (rank >= forbidden[:, None]).long()
                head = head + rest * self.forces(y[start:stop], t[start:stop], rank).mean(1)
            outputs.append(head / self.n)
        self.diagnostics = dict(force_interactions=self.top_l + (self.tail_samples if self.n > self.top_l else 0),
                                score_seconds=score_time)
        return torch.cat(outputs)
