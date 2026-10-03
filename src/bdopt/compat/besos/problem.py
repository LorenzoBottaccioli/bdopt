from __future__ import annotations

from ...problem import Problem


class EPProblem(Problem):
    """``EPProblem(inputs, outputs)``: BESOS name for :class:`bdopt.Problem`."""

    def __init__(self, inputs=(), outputs=(), minimize_outputs=None, constraints=(), *a, **k):
        super().__init__(list(inputs), list(outputs), list(constraints))
        if minimize_outputs is not None:
            for o, m in zip(self.objectives, minimize_outputs, strict=False):
                o.direction = "min" if m else "max"

    # BESOS attribute names
    @property
    def inputs(self):
        return self.parameters

    @property
    def outputs(self):
        return self.objectives
