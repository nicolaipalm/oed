import numpy as np

from src.parametric_function_library.interfaces.parametric_function import ParametricFunction


class LinearFunction(ParametricFunction):
    def __call__(self, theta: np.ndarray, x: np.ndarray) -> float:
        return np.sum(theta[1:] * x) + theta[0]

    def partial_derivative(self, theta: np.ndarray, x: np.ndarray, parameter_index: int) -> float:
        if parameter_index == 0:
            return 1
        else:
            return x[parameter_index - 1]

    def second_partial_derivative(self, theta: np.ndarray, x: np.ndarray, parameter1_index: int,
                                  parameter2_index: int) -> float:
        return 0
