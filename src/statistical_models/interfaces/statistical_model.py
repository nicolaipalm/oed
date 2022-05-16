from abc import ABC, abstractmethod

import numpy as np

from src.minimizer.interfaces.minimizer import Minimizer


class StatisticalModel(ABC):
    def __call__(self, x: np.ndarray, theta: np.ndarray):
        """
        If the statistical model hase an underlying function, this can be called here.
        :param x:
        :type x:
        :param theta:
        :type theta:
        :return:
        :rtype:
        """
        raise NotImplementedError

    @abstractmethod
    def random(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def calculate_fisher_information_matrix(
            self, x0: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        pass

    def calculate_cramer_rao_lower_bound(
            self, x0: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        return np.linalg.pinv(
            self.calculate_fisher_information_matrix(x0=x0, theta=theta),
            #+ 2e-7 * np.identity(len(theta)),
            hermitian=True
        )

    def calculate_determinant_fisher_information_matrix(
            self, x0: np.ndarray, theta: np.ndarray
    ) -> float:
        return np.linalg.det(self.calculate_fisher_information_matrix(x0, theta))

    @abstractmethod
    def calculate_likelihood(
            self, x0: np.ndarray, y: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        pass

    def calculate_maximum_likelihood_estimation(
            self,
            x0: np.ndarray,
            y: np.ndarray,
            minimizer: Minimizer,
    ) -> np.ndarray:
        return minimizer(
            function=lambda theta: -self.calculate_likelihood(theta=theta, y=y, x0=x0),
            lower_bounds=self.lower_bounds_theta,
            upper_bounds=self.upper_bounds_theta,
        )

    @property
    @abstractmethod
    def lower_bounds_theta(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def upper_bounds_theta(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def lower_bounds_x(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def upper_bounds_x(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
