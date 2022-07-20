from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class Minimizer(ABC):
    """Interface for minimizer of functions

    """
    @abstractmethod
    def __call__(
        self, function: Callable, upper_bounds: np.ndarray, lower_bounds: np.ndarray
    ) -> np.ndarray:
        # TODO: write interface for function: input ndarray and output float
        """Calculate the minimium of function within the boundaries provided


        Parameters
        ----------
        function : Callable
            Must be applied to a np.ndarray and return a float
        upper_bounds : np.ndarray
            Lower bounds for input with each entry representing
            the lower bound for the respective entry of the input of the function
        lower_bounds : np.ndarray
            Lower bounds for input with each entry representing
            the lower bound for the respective entry of the input of the function

        Returns
        -------
        np.ndarray
            minimium of function within the boundaries provided
        """
        pass
