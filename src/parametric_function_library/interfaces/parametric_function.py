from abc import ABC, abstractmethod

import numpy as np


class ParametricFunction(ABC):
    """Interface used to represent a parametric function f_{theta}(x) indexed by theta and evaluated at x

    Methods
    -------
    __call__

    """

    @abstractmethod
    def __call__(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Evaluate the function f_{-} at theta and x

        Parameters
        ----------
        theta : np.ndarray
            Parameter of the parametric model
        x : np.ndarray
            Point on which function should be evaluated

        Returns
        -------
        np.ndarray
            Parametric function f_theta(x) evaluated at theta and x
        """
        pass

    @abstractmethod
    def partial_derivative(
            self, theta: np.ndarray, x: np.ndarray, parameter_index: int
    ) -> np.ndarray:
        """ Partial derivative at theta_{parameter_index} of the function f_{-}(x)
        Parameters
        ----------
        theta : np.ndarray
            Parameter of the parametric model
        x : np.ndarray
            Point on which function should be evaluated
        parameter_index : ind
            index (of theta) on which the partial derivative is taken

        Returns
        -------
        np.ndarray
            Partial derivative at theta_{parameter_index} evaluated at theta and x
        """
        pass

    @abstractmethod
    def second_partial_derivative(
            self,
            theta: np.ndarray,
            x: np.ndarray,
            parameter1_index: int,
            parameter2_index: int,
    ) -> np.ndarray:
        """ Second Partial derivative at theta_{parameter1_index} and theta_{parameter2_index} of the function f_{-}(x)
        Parameters
        ----------
        theta : np.ndarray
            Parameter of the parametric model
        x : np.ndarray
            Point on which function should be evaluated
        parameter1_index : ind
            index (of theta) on which the first partial derivative is taken

        parameter2_index : ind
            index (of theta) on which the second partial derivative is taken

        Returns
        -------
        np.ndarray
            Partial derivative at theta_{parameter1_index} and theta_{parameter2_index} evaluated at theta and x
        """
        pass
