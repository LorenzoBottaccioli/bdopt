from __future__ import annotations

import pandas as pd

from ...optimize import nsga2


def NSGAII(evaluator, evaluations: int, population_size: int = 20, **kwargs) -> pd.DataFrame:  # noqa: N802
    """BESOS-style NSGA-II: runs ``evaluations // population_size`` generations and returns the
    non-dominated designs (parameters + objectives, plus a ``pareto-optimal`` column)."""
    n_gen = max(1, evaluations // population_size)
    front = nsga2(evaluator.problem, evaluator, pop_size=population_size, n_gen=n_gen)
    df = front.to_df()
    df["pareto-optimal"] = True
    return df
