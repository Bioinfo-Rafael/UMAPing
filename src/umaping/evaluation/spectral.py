"""Spectral encoder evaluation (docs/method.md, Evaluation section): graph
Rayleigh energy, orthogonality error, eigenvalues of the projected operator
H, and subspace/principal-angle distance against an exact scipy `eigsh`
solution. The exact solver is evaluation-only -- never a training label."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import scipy.sparse as sp
from scipy.linalg import subspace_angles
from scipy.sparse.linalg import eigsh

from umaping.graph import normalized_laplacian


@dataclass
class SpectralMetrics:
    rayleigh_energy: float
    orthogonality_error: float
    projected_eigenvalues: list[float]
    exact_eigenvalues: list[float]
    principal_angles_degrees: list[float]
    max_principal_angle_degrees: float

    def to_dict(self) -> dict:
        return asdict(self)


def rayleigh_energy(z: np.ndarray, w: sp.csr_matrix) -> float:
    """trace(Z^T L_sym Z) == (1/2) sum_ij W_ij || z_i/sqrt(D_ii) - z_j/sqrt(D_jj) ||^2."""
    l_sym = normalized_laplacian(w)
    return float(np.trace(z.T @ (l_sym @ z)))


def orthogonality_error(z: np.ndarray) -> float:
    n, r = z.shape
    gram = (z.T @ z) / n
    return float(np.linalg.norm(gram - np.eye(r), ord="fro"))


def exact_laplacian_eigenvectors(w: sp.csr_matrix, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Smallest `n_components` eigenpairs of L_sym via sparse eigsh (shift-invert
    around 0, the standard trick for robustly finding the smallest eigenvalues)."""
    l_sym = normalized_laplacian(w)
    n = l_sym.shape[0]
    k = min(n_components, n - 1)
    try:
        eigval, eigvec = eigsh(l_sym, k=k, sigma=0.0, which="LM")
    except Exception:
        eigval, eigvec = eigsh(l_sym, k=k, which="SM")
    order = np.argsort(eigval)
    return eigval[order], eigvec[:, order]


def evaluate_spectral(
    z_whitened: np.ndarray,
    w: sp.csr_matrix,
    projection: np.ndarray,
    embedding_dim: int,
) -> SpectralMetrics:
    """`z_whitened`: (N, r) whitened raw reference outputs (Z^TZ/N ~= I).
    `projection`: (r, embedding_dim) retained (non-trivial) eigenvectors of
    H = Z^T L_sym Z, as stored in `SpectralCalibration`."""
    l_sym = normalized_laplacian(w)
    h = z_whitened.T @ (l_sym @ z_whitened)
    h = (h + h.T) / 2.0
    h_eigval = np.linalg.eigh(h)[0]

    energy = rayleigh_energy(z_whitened, w)
    ortho_err = orthogonality_error(z_whitened)

    exact_eigval, exact_eigvec = exact_laplacian_eigenvectors(w, n_components=embedding_dim + 1)
    our_subspace = z_whitened @ projection
    our_basis, _ = np.linalg.qr(our_subspace)
    exact_basis = exact_eigvec[:, 1:]  # discard the trivial (lowest) exact mode too

    angles_deg = np.degrees(subspace_angles(our_basis, exact_basis))

    return SpectralMetrics(
        rayleigh_energy=energy,
        orthogonality_error=ortho_err,
        projected_eigenvalues=[float(v) for v in h_eigval],
        exact_eigenvalues=[float(v) for v in exact_eigval],
        principal_angles_degrees=[float(v) for v in angles_deg],
        max_principal_angle_degrees=float(np.max(angles_deg)) if len(angles_deg) else 0.0,
    )
