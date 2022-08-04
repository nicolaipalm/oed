import numpy as np

from src.uncertainty_quantification.interfaces.probability_measure import (
    ProbabilityMeasure,
)


class MultivariateGaussian(ProbabilityMeasure):
    def __init__(self, mean: np.ndarray, covariance_matrix: np.ndarray):
        self.mean = mean
        self.covariance_matrix = covariance_matrix

    def random(self) -> np.ndarray:
        return np.random.multivariate_normal(mean=self.mean, cov=self.covariance_matrix)
