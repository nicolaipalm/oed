import numpy as np
from scipy.stats import qmc

from src.experiments.interfaces.design_of_experiment import (
    Experiment,
)


class LatinHypercube(Experiment):
    """Latin Hypercube design implemented within the experiment interface

    See https://en.wikipedia.org/wiki/Latin_hypercube_sampling.
    """
    def __init__(
        self,
        number_designs: int,
        lower_bounds_design: np.ndarray,
        upper_bounds_design: np.ndarray,
    ):
        """

        Parameters
        ----------
        number_designs : int
            The number of experimental experiments over which the maximization is taken

        lower_bounds_design : np.ndarray
            Lower bounds for an experimental experiments x
            with each entry representing the lower bound of the respective entry of x

        upper_bounds_design :  np.ndarray
            Lower bounds for an experimental experiments x
            with each entry representing the lower bound of the respective entry of x
        """
        self._design = qmc.scale(
            qmc.LatinHypercube(d=len(lower_bounds_design)).random(n=number_designs),
            lower_bounds_design,
            upper_bounds_design,
        )

    @property
    def name(self) -> str:
        return "LH"

    @property
    def designs(self) -> np.ndarray:
        return self._design
