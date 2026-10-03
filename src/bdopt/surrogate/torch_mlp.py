"""PyTorch surrogate (replaces the Keras/TensorFlow model of the original notebooks)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .base import Surrogate

_ACT = {"relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU, "softplus": nn.Softplus}


def _make_net(n_in: int, n_out: int, hidden: tuple[int, ...], activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), _ACT[activation]()]
        prev = h
    layers.append(nn.Linear(prev, n_out))
    return nn.Sequential(*layers)


class TorchMLP(Surrogate):
    """Multi-output MLP trained with Adam (full batch), validation-based early stopping and an optional
    ensemble (``n_ensemble > 1`` gives an uncertainty estimate from the members' spread).

    The default architecture ``(32, 64, 32)`` / ReLU / Adam(1e-3) is the one used in the platform's
    Keras notebooks.
    """

    def __init__(
        self,
        hidden: tuple[int, ...] = (32, 64, 32),
        activation: str = "relu",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 3000,
        patience: int = 200,
        val_fraction: float = 0.2,
        n_ensemble: int = 1,
        seed: int = 0,
    ):
        if activation not in _ACT:
            raise ValueError(f"activation must be one of {sorted(_ACT)}")
        self.hidden, self.activation, self.lr = tuple(hidden), activation, lr
        self.weight_decay, self.epochs, self.patience = weight_decay, epochs, patience
        self.val_fraction, self.n_ensemble, self.seed = val_fraction, n_ensemble, seed
        self.has_uncertainty = n_ensemble > 1

    def _train_one(self, U: torch.Tensor, Z: torch.Tensor, seed: int) -> nn.Module:
        g = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        n = len(U)
        n_val = int(round(self.val_fraction * n)) if n >= 10 and self.val_fraction > 0 else 0
        perm = torch.randperm(n, generator=g)
        va, tr = perm[:n_val], perm[n_val:]
        net = _make_net(U.shape[1], Z.shape[1], self.hidden, self.activation)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        best, best_state, since = float("inf"), None, 0
        for _ in range(self.epochs):
            net.train()
            opt.zero_grad()
            loss = loss_fn(net(U[tr]), Z[tr])
            loss.backward()
            opt.step()
            monitor = loss_fn(net(U[va]), Z[va]).item() if n_val else loss.item()
            if monitor < best - 1e-9:
                best, since = monitor, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                since += 1
                if since >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        return net.eval()

    def _fit(self, U, Z):
        Ut = torch.as_tensor(U, dtype=torch.float32)
        Zt = torch.as_tensor(Z, dtype=torch.float32)
        # tiny networks: multi-threading only adds overhead (~10x slower measured on 4 cores)
        prev = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            self.nets_ = [self._train_one(Ut, Zt, self.seed + k) for k in range(self.n_ensemble)]
        finally:
            torch.set_num_threads(prev)

    def _members(self, U) -> np.ndarray:
        Ut = torch.as_tensor(U, dtype=torch.float32)
        with torch.no_grad():
            return np.stack([net(Ut).numpy().astype(float) for net in self.nets_])

    def _predict(self, U):
        return self._members(U).mean(axis=0)

    def _predict_std(self, U):
        if self.n_ensemble < 2:
            return None
        return self._members(U).std(axis=0)
