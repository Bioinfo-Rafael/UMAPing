"""CIC deposit + zero-padded linear FFT convolution + bilinear lookup。"""
import numpy as np
import torch
from scipy.signal import fftconvolve
from umaping.umap_forces import g_minus
from .base import Teacher


class GridField(Teacher):
    def __init__(self, *args, grid_size=128, jitter_sigma=.1, **kwargs):
        super().__init__(*args, **kwargs)
        if grid_size < 4:
            raise ValueError('grid_size must be >=4')
        self.grid_size = grid_size
        points = self.trajectory.positions
        lower, upper = points.min(axis=(0, 1)), points.max(axis=(0, 1))
        margin = np.maximum(6 * jitter_sigma, np.maximum(.05 * (upper-lower), 1e-3))
        self.lower, self.upper = lower-margin, upper+margin
        self.spacing = (self.upper-self.lower) / (grid_size-1)
        offsets = np.arange(-grid_size+1, grid_size)
        dx, dy = np.meshgrid(offsets*self.spacing[0], offsets*self.spacing[1], indexing='ij')
        delta = torch.as_tensor(np.stack([dx, dy], -1), dtype=torch.float32)
        # 元kernelを成分clip込みでそのまま適用。FFTの循環畳み込みは禁止。
        kernel = g_minus(delta, torch.zeros_like(delta), self.a, self.b, clip=self.clip).numpy()
        fields = []
        for checkpoint in points:
            scaled = (checkpoint-self.lower) / self.spacing
            index = np.floor(scaled).astype(int)
            fraction = scaled-index
            mass = np.zeros((grid_size, grid_size), dtype=np.float64)
            for ox, oy in ((0,0), (0,1), (1,0), (1,1)):
                weight = (fraction[:,0] if ox else 1-fraction[:,0]) * (fraction[:,1] if oy else 1-fraction[:,1])
                np.add.at(mass, (index[:,0]+ox, index[:,1]+oy), weight)
            if not np.isclose(mass.sum(), self.n):
                raise ValueError('CIC mass conservation failed')
            components = []
            for axis in range(2):
                full = fftconvolve(mass, kernel[...,axis], mode='full')
                components.append(full[grid_size-1:2*grid_size-1, grid_size-1:2*grid_size-1] / self.n)
            fields.append(np.stack(components, -1).astype(np.float32))
        self.fields = torch.as_tensor(np.stack(fields), device=self.device)
        self.lower_t = torch.as_tensor(self.lower, device=self.device)
        self.spacing_t = torch.as_tensor(self.spacing, device=self.device)
        self.outside_total = 0

    def spatial(self, y, checkpoint):
        scaled = (y-self.lower_t) / self.spacing_t
        base = scaled.floor().long().clamp(0, self.grid_size-2)
        fraction = scaled-base
        result = torch.zeros_like(y)
        for ox, oy in ((0,0), (0,1), (1,0), (1,1)):
            weight = (fraction[:,0] if ox else 1-fraction[:,0]) * (fraction[:,1] if oy else 1-fraction[:,1])
            result += self.fields[checkpoint, base[:,0]+ox, base[:,1]+oy] * weight[:,None]
        return result

    @torch.no_grad()
    def estimate(self, y, t, anchor):
        outside = ((y < self.lower_t) | (y > self.lower_t + self.spacing_t*(self.grid_size-1))).any(1)
        self.outside_total += int(outside.sum())
        self.diagnostics = dict(force_interactions=8, grid_cells=self.grid_size**2,
                                grid_bytes=self.fields.numel()*self.fields.element_size(),
                                outside_count=self.outside_total, score_seconds=0.)
        if outside.any():
            raise ValueError(f'Grid query outside bounds: {int(outside.sum())}; no wrapping/clamping/exact teacher fallback')
        time = t.double().clamp(self.view.times[0], self.view.times[-1])
        hi = torch.searchsorted(self.view.times, time, right=True).clamp(1, len(self.view.times)-1)
        lo = hi-1
        frac = ((time-self.view.times[lo]) / (self.view.times[hi]-self.view.times[lo])).float()
        return (1-frac[:,None])*self.spatial(y, lo) + frac[:,None]*self.spatial(y, hi)
