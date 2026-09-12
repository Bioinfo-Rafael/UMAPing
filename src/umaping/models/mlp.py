"""Generic configurable MLP shared by the retriever and spectral encoder."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


def get_activation(name: str) -> nn.Module:
    try:
        return _ACTIVATIONS[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown activation '{name}'. Valid: {sorted(_ACTIVATIONS)}") from exc


class MLP(nn.Module):
    """Plain feed-forward network: Linear -> activation, repeated over
    ``hidden_dims``, followed by an unactivated output projection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(get_activation(activation))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected last dim {self.input_dim}, got {x.shape[-1]}")
        return self.net(x)


@torch.no_grad()
def batched_forward(model: nn.Module, x: np.ndarray, device, batch_size: int = 4096) -> np.ndarray:
    """Run `model` over all of `x` in chunks, e.g. for evaluating an encoder
    on a full N~16k-point reference set without holding one huge activation
    tensor. Always runs in eval mode."""
    was_training = model.training
    model.eval()
    model.to(device)
    x_t = torch.as_tensor(np.asarray(x, dtype=np.float32))
    outputs = []
    for start in range(0, x_t.shape[0], batch_size):
        chunk = x_t[start : start + batch_size].to(device)
        outputs.append(model(chunk).cpu().numpy())
    model.train(was_training)
    return np.concatenate(outputs, axis=0)
