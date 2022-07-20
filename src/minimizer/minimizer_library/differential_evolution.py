from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution

from src.minimizer.interfaces.minimizer import Minimizer


class DifferentialEvolution(Minimizer):
    """Differential evolution algorithm for minimizing functions implemented within the Minimizer interface

    See
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
    for more details of the underlying algorithm.
    """

    def __init__(self, display: bool = False, maxiter: int = 1000):
        """
        Parameters
        ----------
        display : bool
            display the details of the algorithm
        maxiter : int
            maximal iterations of the algorithm
        """
        self.display = display
        self._number_evaluations_last_call = None
        self._maxiter = maxiter

    def __call__(
            self, function: Callable, upper_bounds: np.ndarray, lower_bounds: np.ndarray,
    ) -> np.ndarray:
        t_initial = (upper_bounds + lower_bounds) / 2

        res = differential_evolution(
            func=function,
            x0=t_initial,
            disp=self.display,
            tol=1e-5,
            bounds=[
                (lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))
            ],
            maxiter=self._maxiter,
        )

        self.result = res
        self._number_evaluations_last_call = res.nfev
        return res.x
