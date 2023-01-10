import numpy as np

from piOED.minimizer.interfaces.minimizer import Minimizer
from piOED.parametric_function_library.interfaces.parametric_function import (
    ParametricFunction,
)
from piOED.statistical_models.interfaces.statistical_model import StatisticalModel


class GaussianNoiseModel(StatisticalModel):
    """Implementation of the statistical model induced by a function with white Gaussian noise
    ...within the StatisticalModel interface

    We specify a function f and a variance standard deviation sigma. The statistical model at some experimental experiment x
    is then given by the normal distribution N(f(x),sigma^2).
    Accordingly, given an experiment x0 consisting of experimental experiment x_1,...,x_n, the corresponding
    statistical model is then given by the multivariate normal distribution with mean vector (f(x))_{x \in x0}
    and covariance matrix diagonal matrix with all diagonal entries equal to sigma**2.
    """

    def __init__(
            self,
            function: ParametricFunction,
            lower_bounds_theta: np.ndarray,
            upper_bounds_theta: np.ndarray,
            lower_bounds_x: np.ndarray,
            upper_bounds_x: np.ndarray,
            sigma: float = 1,
    ) -> None:
        """
        Parameters
        ----------
        function : ParametricFunction
            Parametric function parametrized by theta.
        sigma : float
            Standard deviation of the underlying white noise in each component (default is 1)
        """
        self._function = function
        self._var = sigma ** 2
        self._lower_bounds_theta = lower_bounds_theta
        self._upper_bounds_theta = upper_bounds_theta
        self._lower_bounds_x = lower_bounds_x
        self._upper_bounds_x = upper_bounds_x

    def __call__(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return self._function(theta=theta, x=x)

    def random(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return np.random.normal(
            loc=self._function(theta=theta, x=x), scale=np.sqrt(self._var)
        )

    def calculate_fisher_information(
            self, theta: np.ndarray, i: int, j: int, x0: np.ndarray
    ):
        return (1 / self._var * np.dot(np.array(
            [self._function.partial_derivative(theta=theta, x=x_k, parameter_index=i) for x_k in x0]).flatten().T,
                                       np.array(
                                           [self._function.partial_derivative(theta=theta, x=x_k, parameter_index=j) for
                                            x_k in x0]).flatten()))

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

    def calculate_log_likelihood(
            self, theta: np.ndarray, *args  # x0: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        return np.sum(
            (args[1] - np.array([self._function(theta=theta, x=x) for x in args[0]])) ** 2
        )

    def calculate_likelihood(
            self, x0: np.ndarray, y: np.ndarray, theta: np.ndarray
    ) -> float:
        return np.exp(self.calculate_log_likelihood(theta, x0, y))

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
            function=self.calculate_log_likelihood,
            fcn_args=(x0, y),
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
