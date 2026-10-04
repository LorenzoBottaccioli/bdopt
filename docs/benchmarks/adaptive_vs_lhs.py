"""Adaptive sampling against a plain Latin hypercube on three analytic responses.

Reproduces the table in docs/guide/adaptive-sampling.md.

    python docs/benchmarks/adaptive_vs_lhs.py

For each response and method: 3 seeds, 50 training points (14 initial + 36 adaptive) and 10 held-out points
used only to measure convergence; the error is the RMSE of a Gaussian-process surrogate on 3000 independent
points, divided by the standard deviation of the response.
"""

import warnings

import numpy as np

from bdopt import FunctionEvaluator, NullSelector, Parameter, Problem, RangeParameter, sampling
from bdopt.adaptive import AdaptiveSampler
from bdopt.surrogate import GPSurrogate

warnings.filterwarnings("ignore")
METHODS = ["flola_voronoi", "lola_voronoi", "cv_voronoi", "nipv", "probabilistic", "gp_variance",
           "latin_hypercube", "random", "lhs"]
SEEDS = 3


def gp():
    return GPSurrogate(n_restarts=3, max_extra_restarts=8)


def bump(r):    # narrow local feature on a smooth background
    return [0.3 * r["p0"] + np.exp(-(((r["p0"] - 0.7) ** 2 + (r["p1"] - 0.3) ** 2) / 0.012))]


def step(r):    # sharp transition across a hyperplane-like surface
    return [np.tanh(25 * (r["p0"] + 0.5 * r["p1"] - 0.9)) + 0.2 * r["p2"]]


def smooth(r):  # smooth baseline
    return [np.sin(3 * r["p0"]) + r["p1"] ** 2 + 0.3 * r["p2"]]


RESPONSES = {"smooth": smooth, "bump": bump, "step": step}


def error(model, problem, big, truth):
    p = model.predict(big).to_numpy()
    return float(np.sqrt(((p - truth) ** 2).mean() / truth.var()))


for kind, function in RESPONSES.items():
    names = [f"p{k}" for k in range(3)]
    problem = Problem([Parameter(NullSelector(n), RangeParameter(0, 1), name=n) for n in names], ["f"])
    evaluator = FunctionEvaluator(problem, function)
    big = sampling.lhs(problem, 3000, seed=99)
    truth = evaluator.evaluate(big).to_numpy()
    res = {}
    for method in METHODS:
        errors = []
        for seed in range(SEEDS):
            if method == "lhs":
                train = evaluator.evaluate(sampling.lhs(problem, 50, seed=seed), keep_input=True)
                errors.append(error(gp().fit(train, problem), problem, big, truth))
            else:
                result = AdaptiveSampler(problem, evaluator, method, surrogate=gp(), seed=seed).run(
                    n_initial=14, n_test=10, max_iter=36, tol=0, verbose=False
                )
                errors.append(error(result.surrogate, problem, big, truth))
        res[method] = (np.mean(errors), np.std(errors))
    print(kind, {k: f"{m:.3f}±{s:.3f}" for k, (m, s) in res.items()}, flush=True)
