"""SpectralNet-style pointwise Spectral Encoder.

The trained network alone only produces an ``r = embedding_dim + 1``
dimensional raw output. Turning that into a calibrated 2D initialization
``y*_0`` for a single unseen point requires three *frozen, reference-only*
linear transforms, bundled here as :class:`SpectralCalibration`:

1. a whitening matrix so ``Z^T Z / N ~= I`` exactly (not just approximately,
   as the soft training penalty only encourages);
2. a projection onto the top non-trivial eigenvectors of the small projected
   operator ``H = Z^T L_sym Z``;
3. a center + scale calibration so coordinates sit at a scale suitable for
   the UMAP force equations.

No reference graph computation is needed at query time: applying a
:class:`SpectralCalibration` is just two small matrix multiplies plus an
affine rescale.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from umaping.models.mlp import MLP


class SpectralEncoderNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], raw_output_dim: int, activation: str = "gelu"):
        super().__init__()
        self.net = MLP(input_dim, hidden_dims, raw_output_dim, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class SpectralCalibration:
    whitening: np.ndarray  # (r, r): Z = raw @ whitening satisfies Z^T Z / N ~= I
    projection: np.ndarray  # (r, embedding_dim): non-trivial eigenvectors of H = Z^T L_sym Z
    mean: np.ndarray  # (embedding_dim,) reference-only center, subtracted after projection
    scale: float  # reference-only scale, applied after centering
    retained_eigenvalues: np.ndarray  # (embedding_dim,) diagnostic
    discarded_eigenvalue: float  # trivial mode's eigenvalue, diagnostic

    def apply(self, raw: np.ndarray) -> np.ndarray:
        raw = np.atleast_2d(raw)
        z = raw @ self.whitening
        y0 = z @ self.projection
        y0 = (y0 - self.mean) * self.scale
        return y0

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            whitening=self.whitening,
            projection=self.projection,
            mean=self.mean,
            scale=np.asarray(self.scale, dtype=np.float64),
            retained_eigenvalues=self.retained_eigenvalues,
            discarded_eigenvalue=np.asarray(self.discarded_eigenvalue, dtype=np.float64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SpectralCalibration":
        data = np.load(path)
        return cls(
            whitening=data["whitening"],
            projection=data["projection"],
            mean=data["mean"],
            scale=float(data["scale"]),
            retained_eigenvalues=data["retained_eigenvalues"],
            discarded_eigenvalue=float(data["discarded_eigenvalue"]),
        )


class SpectralEmbedder:
    """``x -> neural network -> stored linear transforms -> y0``, with no
    dependency on any other query point and no reference-graph access."""

    def __init__(self, encoder: SpectralEncoderNet, calibration: SpectralCalibration, device: str = "cpu"):
        self.encoder = encoder.to(device)
        self.encoder.eval()
        self.calibration = calibration
        self.device = torch.device(device)

    @torch.no_grad()
    def raw_forward(self, x: np.ndarray) -> np.ndarray:
        x_t = torch.as_tensor(np.atleast_2d(x), dtype=torch.float32, device=self.device)
        return self.encoder(x_t).cpu().numpy()

    def embed(self, x: np.ndarray) -> np.ndarray:
        raw = self.raw_forward(x)
        return self.calibration.apply(raw)

    def embed_one(self, x: np.ndarray) -> np.ndarray:
        return self.embed(x)[0]
