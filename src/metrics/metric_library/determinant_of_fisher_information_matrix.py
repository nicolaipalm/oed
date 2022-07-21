import numpy as np

from src.experiments.interfaces.design_of_experiment import (
    Experiment,
)
from src.metrics.interfaces.metric import Metric
from src.statistical_models.interfaces.statistical_model import StatisticalModel


class DeterminantOfFisherInformationMatrix(Metric):
    """Determinant of Fisher information matrix implemented within the metric interface

    """
    def __init__(self, statistical_model: StatisticalModel, theta: np.ndarray):
        self._statistical_model = statistical_model
        self.theta = theta

    def calculate(
        self,
        experiment: Experiment,
        evaluations_blackbox_function: np.ndarray = None,
        estimations_of_parameter: np.ndarray = None,
    ) -> np.ndarray:
        return self._statistical_model.calculate_determinant_fisher_information_matrix(
            x0=experiment.designs, theta=self.theta
        ) * np.ones(1)

    @property
    def name(self) -> str:
        return "Determinant of Fisher information matrix"
