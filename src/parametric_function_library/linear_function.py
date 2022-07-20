import numpy as np

from src.parametric_function_library.interfaces.parametric_function import (
    ParametricFunction,
)


class LinearFunction(ParametricFunction):
    """Implementation of a linear function with the parametric function interface

    Theta represents the underlying matrix, i.e. f_theta(x) = theta[1:]*x+ theta[0]
    This linear function is restricted to a one dimensional output, i.e. the underlying matrix theta is of dimension 1xn
    """
    def __call__(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.array([np.sum(theta[1:] * x) + theta[0]])

    def partial_derivative(
        self, theta: np.ndarray, x: np.ndarray, parameter_index: int
    ) -> np.ndarray:
        if parameter_index == 0:
            return np.ones(1)
        else:
            return x[parameter_index - 1]

    def second_partial_derivative(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        parameter1_index: int,
        parameter2_index: int,
    ) -> np.ndarray:
        return np.zeros(1)
