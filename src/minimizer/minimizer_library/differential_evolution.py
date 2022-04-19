from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution

from src.minimizer.interfaces.minimizer import Minimizer


class DifferentialEvolution(Minimizer):
    """
    ...for harder to minimize functions. Needs more maximum_likelihood_estimations though.
    """

    def __init__(self,
                 display=False):
        self.display = display
        self._number_evaluations_last_call = None

    def __call__(self, function: Callable,
                 upper_bounds: np.ndarray,
                 lower_bounds: np.ndarray) -> np.ndarray:
        t_initial = (upper_bounds + lower_bounds) / 2

        res = differential_evolution(func=function,
                                     x0=t_initial,
                                     disp=self.display,
                                     tol=1e-5,
                                     bounds=[(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))],
                                     maxiter=10,
                                     )

        self.result = res
        self._number_evaluations_last_call = res.nfev
        return res.x

    @property
    def number_evaluations_last_call(self):
        return self._number_evaluations_last_call
