import numpy as np

from designs_of_experiments.interfaces.design_of_experiment import DesignOfExperiment
from metrics.interfaces.metric import Metric


class EstimationVarianceParameterEstimations(Metric):
    def calculate(self,
                  evaluations_blackbox_function: np.ndarray,
                  estimations_of_parameter: np.ndarray,
                  design: DesignOfExperiment = None,
                  ) -> float:
        return np.average((estimations_of_parameter - np.average(estimations_of_parameter, axis=0)) ** 2, axis=0)

    @property
    def name(self) -> str:
        return "Estimated variance of parameter estimations"
