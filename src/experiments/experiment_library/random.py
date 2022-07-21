from random import uniform

import numpy as np

from src.experiments.interfaces.design_of_experiment import (
    Experiment,
)


class Random(Experiment):
    """Random, i.e. uniformly distributed, experiment implemented within the experiment interface

    Each experimental design is (independently) drawn from a uniform distribution.
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
            The number of experimental experiment over which the maximization is taken

        lower_bounds_design : np.ndarray
            Lower bounds for an experimental experiment x
            with each entry representing the lower bound of the respective entry of x

        upper_bounds_design :  np.ndarray
            Lower bounds for an experimental experiment x
            with each entry representing the lower bound of the respective entry of x
        """
        self._design = np.array(
            [
                [
                    uniform(lower_bounds_design[index], upper_bounds_design[index])
                    for index in range(len(lower_bounds_design))
                ]
                for _ in range(number_designs)
            ]
        )

    @property
    def name(self) -> str:
        return "Random"

    @property
    def experiment(self) -> np.ndarray:
        return self._design
