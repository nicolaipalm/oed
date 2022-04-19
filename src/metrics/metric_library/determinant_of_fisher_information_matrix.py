import numpy as np

from src.designs_of_experiments.interfaces.design_of_experiment import DesignOfExperiment
from src.metrics.interfaces.metric import Metric
from src.statistical_models.interfaces.statistical_model import StatisticalModel


class DeterminantOfFisherInformationMatrix(Metric):
    def __init__(self, statistical_model: StatisticalModel, theta: np.ndarray):
        self._statistical_model = statistical_model
        self.theta = theta

    def calculate(self,
                  design: DesignOfExperiment,
                  evaluations_blackbox_function: np.ndarray = None,
                  estimations_of_parameter: np.ndarray = None,
                  ) -> np.ndarray:
        self._statistical_model.calculate_determinant_fisher_information_matrix(x0=design.design,
                                                                                theta=self.theta) * np.ones(1)
        return self._statistical_model.calculate_determinant_fisher_information_matrix(x0=design.design,
                                                                                       theta=self.theta) * np.ones(1)

    @property
    def name(self) -> str:
        return "Determinant of Fisher information matrix"
