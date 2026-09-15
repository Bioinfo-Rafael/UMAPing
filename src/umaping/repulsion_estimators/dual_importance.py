"""Dual表現のみのproposal。確率行列は小さいanchorバッチごとに破棄。"""
import time
import numpy as np
import torch
from .base import Teacher, synchronize


@torch.no_grad()
def hubness_vector(q, k, temperature, topk=32, chunk=256):
    if temperature <= 0 or topk < 1 or chunk < 1:
        raise ValueError('Invalid hubness parameters')
    h = torch.empty(len(k), dtype=torch.float64, device=k.device)
    count = min(topk, len(q))
    for j in range(0, len(k), chunk):
        keys = k[j:j+chunk].double()
        best = torch.empty((0, len(keys)), dtype=torch.float64, device=k.device)
        for i in range(0, len(q), chunk):
            scores = q[i:i+chunk].double() @ keys.T / temperature
            merged = torch.cat([best, scores])
            best = merged.topk(min(count, len(merged)), dim=0).values
        h[j:j+chunk] = best.mean(0)
    return h


class DualProposal(Teacher):
    def __init__(self, *args, queries, keys, temperature=.1, proposal_temperature=1.,
                 mixture=1., hubness=None, beta=1., score_batch=16, **kwargs):
        super().__init__(*args, **kwargs)
        if temperature <= 0 or proposal_temperature <= 0 or not 0 <= mixture <= 1 or score_batch < 1:
            raise ValueError('Invalid proposal parameters')
        self.q = torch.as_tensor(queries, dtype=torch.float64, device=self.device).detach()
        self.k = torch.as_tensor(keys, dtype=torch.float64, device=self.device).detach()
        if self.q.shape != self.k.shape or len(self.k) != self.n:
            raise ValueError('Q/K/trajectory shapes disagree')
        if not torch.isfinite(self.q).all() or not torch.isfinite(self.k).all():
            raise ValueError('Non-finite Dual embeddings')
        self.temperature, self.proposal_temperature = temperature, proposal_temperature
        self.mixture, self.beta, self.score_batch = mixture, beta, score_batch
        self.h = None if hubness is None else torch.as_tensor(hubness, device=self.device, dtype=torch.float64)
        if self.h is not None and (self.h.shape != (self.n,) or not torch.isfinite(self.h).all()):
            raise ValueError('Invalid hubness vector')

    def scores(self, anchors):
        return self.q[anchors] @ self.k.T / self.temperature

    def probabilities(self, anchors):
        scores = self.scores(anchors)
        if self.h is not None:
            scores = scores - self.beta * self.h
        logp = torch.log_softmax(scores / self.proposal_temperature, dim=1)
        p = logp.exp()
        p = (1 - self.mixture) / self.n + self.mixture * p
        p = p / p.sum(1, keepdim=True)
        # 純ISのsupportをunderflowで失ったら黙って確率をfloorしない。
        if not torch.isfinite(p).all() or (p <= 0).any():
            raise FloatingPointError('Proposal lost full support; try the explicitly labelled lambda=0.9 ablation')
        return p


class DualImportance(DualProposal):
    stochastic = True

    def __init__(self, *args, samples=64, **kwargs):
        super().__init__(*args, **kwargs)
        if samples < 1:
            raise ValueError('samples must be positive')
        self.samples = samples

    @torch.no_grad()
    def estimate(self, y, t, anchor):
        outputs, weights, ess, entropies, maxima, minima = [], [], [], [], [], []
        score_time = 0.
        for start in range(0, len(y), self.score_batch):
            stop = start + self.score_batch
            synchronize(self.device)
            begin = time.perf_counter()
            p = self.probabilities(anchor[start:stop])
            synchronize(self.device)
            score_time += time.perf_counter() - begin
            idx = torch.multinomial(p, self.samples, replacement=True, generator=self.generator)
            sampled = p.gather(1, idx)
            w = 1 / (self.n * sampled)
            outputs.append((self.forces(y[start:stop], t[start:stop], idx).double() * w[..., None]).mean(1).float())
            weights.append(w.flatten())
            ess.append(w.sum(1).square() / w.square().sum(1))
            entropies.append(-(p * p.log()).sum(1))
            maxima.append(p.max(1).values)
            minima.append(sampled.min())
        w = torch.cat(weights)
        self.diagnostics = dict(force_interactions=self.samples, score_seconds=score_time,
                                entropy=float(torch.cat(entropies).mean()), max_probability=float(torch.cat(maxima).max()),
                                min_sampled_probability=float(torch.stack(minima).min()), mean_weight=float(w.mean()),
                                max_weight=float(w.max()), ess=float(torch.cat(ess).mean()),
                                weights=w.cpu().numpy())
        return torch.cat(outputs)
