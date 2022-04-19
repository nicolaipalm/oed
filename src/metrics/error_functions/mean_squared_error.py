import numpy as np

from src.metrics.interfaces.error_function import ErrorFunction


class MeanSquaredError(ErrorFunction):
    def __call__(self, y1: np.ndarray, y2: np.ndarray) -> float:
        if len(y1.shape) > 1 or len(y2.shape) > 1:
            raise ValueError("The arrays must be of dimension 1.")
        return np.average((y1 - y2) ** 2)

    @property
    def name(self) -> str:
        return "Mean squared error function"
