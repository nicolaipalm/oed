import numpy as np

from src.metrics.interfaces.error_function import ErrorFunction


class MeanSquaredError(ErrorFunction):
    """TBA

    """
    def __call__(self, y1: np.ndarray, y2: np.ndarray) -> float:
        if len(y1.shape) > 2 or len(y2.shape) > 2:
            raise ValueError("The arrays must be of dimension 1 or 2.")
        return np.average((y1.flatten() - y2.flatten()) ** 2)

    @property
    def name(self) -> str:
        return "Mean squared error function"
