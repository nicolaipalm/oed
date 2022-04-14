from abc import ABC, abstractmethod

import numpy as np

# TODO: dont need this class?
class BlackboxModel(ABC):

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        pass

    def calculate_fisher_information_matrix(self, x0: np.ndarray, theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def calculate_cramer_rao_lower_bound(self, x0: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return np.linalg.inv(
            self.calculate_fisher_information_matrix(x0=x0, theta=theta) +
            2e-7 * np.identity(len(theta)))

    def calculate_likelihood(self, x0: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def calculate_maximum_likelihood_estimation(self, x0: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError
