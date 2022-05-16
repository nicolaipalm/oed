import numpy as np

from src.minimizer.interfaces.minimizer import Minimizer
from src.parametric_function_library.interfaces.parametric_function import (
    ParametricFunction,
)
from src.statistical_models.interfaces.statistical_model import StatisticalModel


class GaussianNoiseModel(StatisticalModel):
    def __init__(
            self,
            function: ParametricFunction,
            lower_bounds_theta: np.ndarray,
            upper_bounds_theta: np.ndarray,
            lower_bounds_x: np.ndarray,
            upper_bounds_x: np.ndarray,
            sigma: float = 1,
    ) -> None:
        self._function = function
        self._var = sigma ** 2  # variance
        self._lower_bounds_theta = lower_bounds_theta
        self._upper_bounds_theta = upper_bounds_theta
        self._lower_bounds_x = lower_bounds_x
        self._upper_bounds_x = upper_bounds_x

    def __call__(self, x: np.ndarray, theta: np.ndarray) -> float:
        return self._function(theta=theta, x=x)

    def random(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return np.random.normal(
            loc=self._function(theta=theta, x=x), scale=np.sqrt(self._var)
        )

    def calculate_fisher_information(
            self, theta: np.ndarray, i: int, j: int, x0: np.ndarray
    ):
        return (1 / self._var * np.dot(np.array([self._function.partial_derivative(theta=theta, x=x_k, parameter_index=i)for x_k in x0]).T,
                                       np.array([self._function.partial_derivative(theta=theta, x=x_k, parameter_index=j) for x_k in x0])))

    def calculate_fisher_information_matrix(
            self, x0: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        k = len(theta)
        return np.array(
            [
                [
                    self.calculate_fisher_information(theta=theta, x0=x0, i=i, j=j)
                    for i in range(k)
                ]
                for j in range(k)
            ]
        )

    def calculate_likelihood(
            self, x0: np.ndarray, y: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        # TODO: implement correctly plus use derivative known in minimizer
        pass

    def calculate_log_likelihood(
            self, x0: np.ndarray, y: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        return -np.sum(
            (y - np.array([self._function(theta=theta, x=x) for x in x0])) ** 2
        )

    def calculate_partial_derivative_log_likelihood(
            self, x0: np.ndarray, y: np.ndarray, theta: np.ndarray, parameter_index: int
    ) -> np.ndarray:
        return -np.sum(
            (y - np.array([self._function(theta=theta, x=x) for x in x0]))
            * np.array(
                [
                    self._function.partial_derivative(
                        theta=theta, x=x, parameter_index=parameter_index
                    )
                    for x in x0
                ]
            )
        )

    def calculate_maximum_likelihood_estimation(
            self, x0: np.ndarray, y: np.ndarray, minimizer: Minimizer
    ) -> np.ndarray:
        return minimizer(
            function=lambda theta: -self.calculate_log_likelihood(
                theta=theta, y=y, x0=x0
            ),
            lower_bounds=self.lower_bounds_theta,
            upper_bounds=self.upper_bounds_theta,
        )

    @property
    def lower_bounds_theta(self) -> np.ndarray:
        return self._lower_bounds_theta

    @property
    def upper_bounds_theta(self) -> np.ndarray:
        return self._upper_bounds_theta

    @property
    def lower_bounds_x(self) -> np.ndarray:
        return self._lower_bounds_x

    @property
    def upper_bounds_x(self) -> np.ndarray:
        return self._upper_bounds_x

    @property
    def name(self) -> str:
        return "Gaussian white noise model"
