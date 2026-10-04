"""More PyTorch surrogates: Monte-Carlo-dropout network, multi-task and deep-kernel Gaussian processes.

The Gaussian processes are implemented directly in PyTorch (dense Cholesky, marginal likelihood
maximised with L-BFGS / Adam): with the data sizes of a building study (tens to a few hundred
simulations) nothing more is needed.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .base import Surrogate
from .torch_mlp import TorchMLP, _make_net

_DT = torch.float64
_MIN_LS = 0.05  # same lower bound on length scales as GPSurrogate (degenerate "white noise" optimum)


def _matern52(x1: torch.Tensor, x2: torch.Tensor, ls: torch.Tensor) -> torch.Tensor:
    a, b = x1 / ls, x2 / ls
    d2 = (a**2).sum(-1)[:, None] + (b**2).sum(-1)[None, :] - 2 * a @ b.T
    r = torch.sqrt(torch.clamp(d2, min=1e-12))
    s5 = np.sqrt(5.0) * r
    return (1 + s5 + 5.0 / 3.0 * r**2) * torch.exp(-s5)


def _rbf(x1: torch.Tensor, x2: torch.Tensor, ls: torch.Tensor) -> torch.Tensor:
    a, b = x1 / ls, x2 / ls
    d2 = (a**2).sum(-1)[:, None] + (b**2).sum(-1)[None, :] - 2 * a @ b.T
    return torch.exp(-0.5 * torch.clamp(d2, min=0.0))


_KERN = {"matern52": _matern52, "rbf": _rbf}


class MCDropoutMLP(TorchMLP):
    """MLP with dropout kept active at prediction time (Gal & Ghahramani, 2016): the spread of ``n_mc``
    stochastic forward passes is an approximate Bayesian predictive uncertainty. A light-weight stand-in
    for a Bayesian neural network."""

    has_uncertainty = True

    def __init__(self, hidden: tuple[int, ...] = (64, 64), dropout: float = 0.1, n_mc: int = 50,
                 activation: str = "silu", lr: float = 3e-3, weight_decay: float = 1e-4, epochs: int = 3000,
                 patience: int = 300, val_fraction: float = 0.0, seed: int = 0):
        super().__init__(hidden=hidden, activation=activation, lr=lr, weight_decay=weight_decay,
                         epochs=epochs, patience=patience, val_fraction=val_fraction, n_ensemble=1, seed=seed)
        self.dropout, self.n_mc = dropout, n_mc
        self.has_uncertainty = True

    def _members(self, U) -> np.ndarray:
        Ut = torch.as_tensor(U, dtype=torch.float32)
        net = self.nets_[0]
        net.train()  # dropout on
        torch.manual_seed(self.seed + 12345)
        with torch.no_grad():
            out = np.stack([net(Ut).numpy().astype(float) for _ in range(self.n_mc)])
        net.eval()
        return out

    def _predict_std(self, U):
        return self._members(U).std(axis=0)


class MultiTaskGPSurrogate(Surrogate):
    """Multi-task Gaussian process (intrinsic coregionalisation model).

    All objectives share one ARD kernel over the parameters and are coupled by a learned task covariance
    ``B = L L^T + diag(kappa)``, so an objective that is cheap to learn (e.g. electricity) helps the ones
    that are not. Provides a predictive standard deviation per objective.
    """

    has_uncertainty = True

    def __init__(self, kernel: str = "matern52", n_restarts: int = 3, max_iter: int = 80, seed: int = 0):
        if kernel not in _KERN:
            raise ValueError(f"kernel must be one of {sorted(_KERN)}")
        self.kernel, self.n_restarts, self.max_iter, self.seed = kernel, n_restarts, max_iter, seed

    def _build(self, p: dict, X: torch.Tensor, Xs: torch.Tensor | None = None):
        ls = _MIN_LS + F.softplus(p["ls"])
        k = _KERN[self.kernel]
        m = p["L"].shape[0]
        Lt = torch.tril(p["L"])
        B = Lt @ Lt.T + torch.diag(F.softplus(p["kappa"]) + 1e-6)
        noise = F.softplus(p["noise"]) + 1e-7
        return ls, k, B, noise, m

    def _nll(self, p, X, y):
        ls, k, B, noise, m = self._build(p, X)
        n = len(X)
        K = torch.kron(B, k(X, X, ls)) + torch.diag(noise.repeat_interleave(n)) + 1e-8 * torch.eye(m * n, dtype=_DT)
        Lc = torch.linalg.cholesky(K)
        a = torch.cholesky_solve(y[:, None], Lc)
        return 0.5 * (y[:, None] * a).sum() + torch.log(torch.diagonal(Lc)).sum()

    def _closure(self, opt, p, X, y):
        def closure():
            opt.zero_grad()
            loss = self._nll(p, X, y)
            loss.backward()
            return loss

        return closure

    def _fit(self, U, Z):
        torch.manual_seed(self.seed)
        X = torch.as_tensor(U, dtype=_DT)
        n, d = U.shape
        m = Z.shape[1]
        y = torch.as_tensor(Z.T.reshape(-1), dtype=_DT)  # task-major
        best, best_nll = None, float("inf")
        for r in range(self.n_restarts):
            ls0 = [0.3, 0.8, 2.0, 0.5][r % 4]
            p = {
                "ls": torch.full((d,), float(np.log(np.expm1(max(ls0 - _MIN_LS, 1e-3)))), dtype=_DT),
                "L": (0.5 * torch.eye(m, dtype=_DT) + 0.05 * torch.randn(m, m, dtype=_DT)),
                "kappa": torch.full((m,), -2.0, dtype=_DT),
                "noise": torch.full((m,), -5.0, dtype=_DT),
            }
            for v in p.values():
                v.requires_grad_(True)
            opt = torch.optim.LBFGS(list(p.values()), lr=0.5, max_iter=self.max_iter, line_search_fn="strong_wolfe")

            try:
                opt.step(self._closure(opt, p, X, y))
                val = float(self._nll(p, X, y).detach())
            except (RuntimeError, torch.linalg.LinAlgError):
                continue
            if np.isfinite(val) and val < best_nll:
                best, best_nll = {k: v.detach().clone() for k, v in p.items()}, val
        if best is None:
            raise RuntimeError("multi-task GP: marginal likelihood optimisation failed")
        ls, k, B, noise, _ = self._build(best, X)
        K = torch.kron(B, k(X, X, ls)) + torch.diag(noise.repeat_interleave(n)) + 1e-8 * torch.eye(m * n, dtype=_DT)
        self._Lc = torch.linalg.cholesky(K)
        self._alpha = torch.cholesky_solve(y[:, None], self._Lc)
        self.params_, self._X, self._B, self._ls, self._m, self._n = best, X, B, ls, m, n
        self.task_covariance_ = B.numpy()

    def _cross(self, U):
        Xs = torch.as_tensor(U, dtype=_DT)
        return torch.kron(self._B, _KERN[self.kernel](Xs, self._X, self._ls)), Xs

    def _predict(self, U):
        Ks, _ = self._cross(U)
        mean = (Ks @ self._alpha).reshape(self._m, len(U)).T
        return mean.numpy()

    def _predict_std(self, U):
        Ks, Xs = self._cross(U)
        V = torch.linalg.solve_triangular(self._Lc, Ks.T, upper=False)
        prior = torch.diag(self._B).repeat_interleave(len(Xs))  # unit signal variance per point
        var = torch.clamp(prior - (V**2).sum(0), min=1e-12).reshape(self._m, len(U)).T
        return torch.sqrt(var).numpy()


class DeepKernelGPSurrogate(Surrogate):
    """Deep-kernel Gaussian process (Wilson et al., 2016): a small neural network maps the parameters to a
    feature space where a Matern/RBF kernel acts; network, kernel and noise are trained together by
    maximising the marginal likelihood (one output scale and noise per objective). Useful when the response
    is smooth only after a non-linear change of coordinates."""

    has_uncertainty = True

    def __init__(self, hidden: tuple[int, ...] = (32, 16), n_features: int = 4, kernel: str = "matern52",
                 epochs: int = 300, lr: float = 0.01, weight_decay: float = 1e-4, seed: int = 0):
        if kernel not in _KERN:
            raise ValueError(f"kernel must be one of {sorted(_KERN)}")
        self.hidden, self.n_features, self.kernel = tuple(hidden), n_features, kernel
        self.epochs, self.lr, self.weight_decay, self.seed = epochs, lr, weight_decay, seed

    def _kmat(self, net, F1, F2, ls):
        return _KERN[self.kernel](net(F1), net(F2), ls)

    def _fit(self, U, Z):
        torch.manual_seed(self.seed)
        X = torch.as_tensor(U, dtype=_DT)
        Y = torch.as_tensor(Z, dtype=_DT)
        n, d = U.shape
        m = Y.shape[1]
        net = _make_net(d, self.n_features, self.hidden, "tanh").to(_DT)
        raw_ls = torch.zeros(self.n_features, dtype=_DT, requires_grad=True)
        raw_amp = torch.zeros(m, dtype=_DT, requires_grad=True)
        raw_noise = torch.full((m,), -4.0, dtype=_DT, requires_grad=True)
        opt = torch.optim.Adam(
            [{"params": net.parameters(), "weight_decay": self.weight_decay}, {"params": [raw_ls, raw_amp, raw_noise]}],
            lr=self.lr,
        )

        def nll():
            ls = _MIN_LS + F.softplus(raw_ls)
            Kb = self._kmat(net, X, X, ls)
            total = 0.0
            for j in range(m):
                K = F.softplus(raw_amp[j]) * Kb + (F.softplus(raw_noise[j]) + 1e-6) * torch.eye(n, dtype=_DT)
                Lc = torch.linalg.cholesky(K)
                a = torch.cholesky_solve(Y[:, j : j + 1], Lc)
                total = total + 0.5 * (Y[:, j : j + 1] * a).sum() + torch.log(torch.diagonal(Lc)).sum()
            return total

        best, best_state = float("inf"), None
        for _ in range(self.epochs):
            opt.zero_grad()
            try:
                loss = nll()
            except (RuntimeError, torch.linalg.LinAlgError):
                break
            loss.backward()
            opt.step()
            if loss.item() < best:
                best = loss.item()
                best_state = (
                    {k: v.clone() for k, v in net.state_dict().items()},
                    raw_ls.detach().clone(), raw_amp.detach().clone(), raw_noise.detach().clone(),
                )
        if best_state is None:
            raise RuntimeError("deep-kernel GP: training failed")
        net.load_state_dict(best_state[0])
        net.eval()
        ls = _MIN_LS + F.softplus(best_state[1])
        amp, noise = F.softplus(best_state[2]), F.softplus(best_state[3]) + 1e-6
        with torch.no_grad():
            Kb = self._kmat(net, X, X, ls)
            self._chol, self._alpha = [], []
            for j in range(m):
                K = amp[j] * Kb + noise[j] * torch.eye(n, dtype=_DT)
                Lc = torch.linalg.cholesky(K)
                self._chol.append(Lc)
                self._alpha.append(torch.cholesky_solve(Y[:, j : j + 1], Lc))
        self._net, self._X, self._ls, self._amp, self._m = net, X, ls, amp, m

    def _ks(self, U):
        Xs = torch.as_tensor(U, dtype=_DT)
        with torch.no_grad():
            return self._kmat(self._net, Xs, self._X, self._ls)

    def _predict(self, U):
        Ks = self._ks(U)
        return torch.cat([self._amp[j] * Ks @ self._alpha[j] for j in range(self._m)], dim=1).numpy()

    def _predict_std(self, U):
        Ks = self._ks(U)
        out = []
        for j in range(self._m):
            V = torch.linalg.solve_triangular(self._chol[j], (self._amp[j] * Ks).T, upper=False)
            out.append(torch.sqrt(torch.clamp(self._amp[j] - (V**2).sum(0), min=1e-12)))
        return torch.stack(out, dim=1).numpy()
