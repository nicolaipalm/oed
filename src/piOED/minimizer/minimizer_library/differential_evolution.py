from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution

from piOED.minimizer.interfaces.minimizer import Minimizer
from scipy.optimize import LinearConstraint, NonlinearConstraint


class DifferentialEvolution(Minimizer):
    """Differential evolution algorithm for minimizing functions implemented within the Minimizer interface

    See
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
    for more details of the underlying algorithm.
    """

    def __init__(self, display: bool = False, maxiter: int = 1000, workers=1, init_sampling: str = 'latinhypercube'):
        """
        Parameters
        ----------
        display : bool
            display the details of the algorithm
        maxiter : int
            maximal iterations of the algorithm
        workers : int
            Number of parallel processes for optimization tasks. -1 uses all available CPUs
        init_sampling : str
            Sampling strategy of optimizer initialization.
        """
        self.display = display
        self._number_evaluations_last_call = None
        self._maxiter = maxiter
        self.init_sampling = init_sampling
        self.workers = workers

    def __call__(
            self,
            function: Callable,
            fcn_args: tuple,
            upper_bounds: np.ndarray,
            lower_bounds: np.ndarray,
            constraints: {LinearConstraint, NonlinearConstraint}=(),
    ) -> np.ndarray:
        t_initial = (upper_bounds + lower_bounds) / 2

        res = differential_evolution(
            func=function,
            args=fcn_args,
            x0=t_initial,
            disp=self.display,
            tol=1e-5,
            bounds=[
                (lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))
            ],
            maxiter=self._maxiter,
            updating='deferred',
            init=self.init_sampling,
            workers=self.workers,
            constraints=constraints,
        )

        self.result = res
        self._number_evaluations_last_call = res.nfev
        return res.x
