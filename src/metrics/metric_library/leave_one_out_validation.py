import numpy as np

from src.experiments.interfaces.design_of_experiment import (
    Experiment,
)
from src.metrics.interfaces.metric import Metric


class LeaveOneOutValidation(Metric):
    def calculate(
        self,
        evaluations_blackbox_function: np.ndarray,
        estimations_of_parameter: np.ndarray,
        design: Experiment = None,
    ) -> float:
        pass

    @property
    def name(self) -> str:
        return "Leave one out validation"
