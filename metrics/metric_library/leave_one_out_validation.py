import numpy as np

from designs_of_experiments.interfaces.design_of_experiment import DesignOfExperiment
from metrics.interfaces.metric import Metric


class LeaveOneOutValidation(Metric):

    def calculate(self,
                  evaluations_blackbox_function: np.ndarray,
                  estimations_of_parameter: np.ndarray,
                  design: DesignOfExperiment = None,
                  ) -> float:
        pass

    @property
    def name(self) -> str:
        return "Leave one out validation"
