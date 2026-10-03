"""Adaptive sampling against a plain Latin hypercube on three analytic responses.

Reproduces the table in docs/guide/adaptive-sampling.md.

    python docs/benchmarks/adaptive_vs_lhs.py

For each response and method: 4 seeds, 50 training points (14 initial + 36 adaptive) and 10 held-out points
used only to measure convergence; the error is the RMSE of a Gaussian-process surrogate on 3000 independent
points, divided by the standard deviation of the response.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np

from bdopt import *
from bdopt.adaptive import AdaptiveSampler
from bdopt.surrogate import GPSurrogate

METHODS=["flola_voronoi","lola_voronoi","cv_voronoi","nipv","probabilistic","gp_variance","latin_hypercube","random","lhs"]
def gp(): return GPSurrogate(n_restarts=3, max_extra_restarts=8)
def bump(r):    # narrow local feature on a smooth background
    return [0.3*r["p0"]+np.exp(-(((r["p0"]-0.7)**2+(r["p1"]-0.3)**2)/0.012))]
def step(r):    # sharp transition across a hyperplane-like surface
    return [np.tanh(25*(r["p0"]+0.5*r["p1"]-0.9))+0.2*r["p2"]]
def smooth(r):  # smooth baseline
    return [np.sin(3*r["p0"])+r["p1"]**2+0.3*r["p2"]]
def make(kind,d):
    P=[Parameter(NullSelector(f"p{k}"),RangeParameter(0,1),name=f"p{k}") for k in range(d)]
    pr=Problem(P,["f"])
    if kind=="bump":     # narrow local feature on a smooth background
        f=bump
    elif kind=="step":   # sharp transition across a hyperplane-like surface
        f=step
    else:                # smooth baseline
        f=smooth
    return pr, FunctionEvaluator(pr,f)
for kind in ["smooth","bump","step"]:
    pr,ev=make(kind,3)
    big=sampling.lhs(pr,3000,seed=99); truth=ev.evaluate(big).to_numpy()
    def err(train):
        p=gp().fit(train,pr).predict(big).to_numpy(); return float(np.sqrt(((p-truth)**2).mean()/truth.var()))
    res={}
    for method in METHODS:
        es=[]
        for seed in range(4):
            if method=="lhs":
                es.append(err(ev.evaluate(sampling.lhs(pr,50,seed=seed),keep_input=True)))
            else:
                r=AdaptiveSampler(pr,ev,method,surrogate=gp(),seed=seed).run(n_initial=14,n_test=10,max_iter=36,tol=0,verbose=False)
                es.append(err(r.data))
        res[method]=(np.mean(es),np.std(es))
    print(kind, {k:f"{m:.3f}±{s:.3f}" for k,(m,s) in res.items()}, flush=True)
