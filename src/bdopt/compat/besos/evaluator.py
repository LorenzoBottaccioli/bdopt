from __future__ import annotations

from ...evaluator import EnergyPlusEvaluator, FunctionEvaluator


class EvaluatorEP(EnergyPlusEvaluator):
    """``EvaluatorEP(problem, building, out_dir=, err_dir=, epw_file=, progress_bar=, error_mode=)``."""

    def __init__(
        self,
        problem,
        building,
        out_dir="outputdir",
        err_dir=None,
        epw_file=None,
        error_mode="Failfast",
        progress_bar=False,
        ep_path=None,
        **kwargs,
    ):
        super().__init__(
            problem,
            building,
            epw=epw_file,
            out_dir=out_dir,
            err_dir=err_dir,
            progress=bool(progress_bar),
            on_error="raise" if error_mode == "Failfast" else "nan",
            eplus=ep_path,
            **kwargs,
        )


class EvaluatorGeneric(FunctionEvaluator):
    """``EvaluatorGeneric(func, problem)``: ``func`` receives the parameter values as a list."""

    def __init__(self, evaluation_func, problem, *a, **k):
        names = problem.parameter_names

        def func(row):
            return evaluation_func([row[n] for n in names])

        super().__init__(problem, func)
