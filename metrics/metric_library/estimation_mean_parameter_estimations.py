import numpy as np

from metrics.interfaces.metric import Metric


class EstimationMeanParameterEstimations(Metric):
    def calculate(self,
                  evaluations_blackbox_function: np.ndarray = None,
                  estimations_of_parameter: np.ndarray = None,
                  ) -> float:
        return np.average(estimations_of_parameter, axis=0)

    @property
    def name(self) -> str:
        return "Estimated mean of parameter estimations"
