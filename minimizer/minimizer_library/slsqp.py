from typing import Callable

import numpy as np
from scipy.optimize import minimize

from minimizer.interfaces.minimizer import Minimizer


class SLSQP(Minimizer):
    def __init__(self,
                 display=False):
        self.display = display
        self._number_evaluations_last_call = None

    def __call__(self, function: Callable,
                 upper_bounds: np.ndarray,
                 lower_bounds: np.ndarray) -> np.ndarray:
        t_initial = (upper_bounds + lower_bounds) / 2
        res = minimize(function,
                       t_initial,
                       method='SLSQP',
                       bounds=[(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))],
                       tol=1e-7,
                       options={
                           'disp': self.display,
                       })

        self.result = res
        self._number_evaluations_last_call = res.nfev
        return res.x

    @property
    def number_evaluations_last_call(self):
        return self._number_evaluations_last_call
