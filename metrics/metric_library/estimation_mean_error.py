import numpy as np

from metrics.interfaces.metric import Metric


class EstimationMeanError(Metric):
    # !!!! no
    def calculate(self, output_data: np.ndarray, output_model: np.ndarray) -> float:
        return np.average(abs(output_data - output_model))

    @property
    def name(self) -> str:
        return "Approximate mean error"
