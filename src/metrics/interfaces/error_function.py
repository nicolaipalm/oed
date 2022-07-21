from abc import ABC, abstractmethod

import numpy as np


class ErrorFunction(ABC):
    """Interface for error functions

    An error function is applied to two numpy arrays and return a float which is interpreted as the error corresponding
    to the arrays.
    """
    @abstractmethod
    def __call__(self, y1: np.ndarray, y2: np.ndarray) -> float:
        """

        Parameters
        ----------
        y1 : np.ndarray
            First array on which the error function is evaluated
        y2 : np.ndarray
            Second array on which the error function is evaluated

        Returns
        -------
        float
            Error function evaluated at y1 and y2.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the error function
        Returns
        -------
        str
            Name of the error function
        """
        pass
