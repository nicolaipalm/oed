import numpy as np

from src.metrics.interfaces.error_function import ErrorFunction


class AverageError(ErrorFunction):
    """Average error calculated as mean over the absolute difference over the

    """
    def __call__(self, y1: np.ndarray, y2: np.ndarray) -> float:
        if len(y1.shape) > 2 or len(y2.shape) > 2:
            raise ValueError("The arrays must be of dimension 1 or 2.")
        return np.average(abs(y1.flatten() - y2.flatten()))

    @property
    def name(self) -> str:
        return "Average error function"
