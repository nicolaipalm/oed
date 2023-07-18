import numpy as np

from ...experiments.interfaces.design_of_experiment import (
    Experiment,
)
from ...metrics.interfaces.metric import Metric
from ...statistical_models.interfaces.statistical_model import StatisticalModel


class DeterminantOfFisherInformationMatrix(Metric):
    """Determinant of Fisher information matrix implemented within the metric interface

    """
    def __init__(self, statistical_model: StatisticalModel, theta: np.ndarray):
        """TBA

        Parameters
        ----------
        statistical_model :
        theta :
        """
        self._statistical_model = statistical_model
        self.theta = theta

    def calculate(
        self,
        experiment: Experiment,
        evaluations_blackbox_function: np.ndarray = None,
        estimations_of_parameter: np.ndarray = None,
    ) -> np.ndarray:
        """TBA

        Parameters
        ----------
        experiment :
        evaluations_blackbox_function :
        estimations_of_parameter :

        Returns
        -------

        """
        return self._statistical_model.calculate_determinant_fisher_information_matrix(
            x0=experiment.experiment, theta=self.theta
        ) * np.ones(1)

    @property
    def name(self) -> str:
        return "Determinant of Fisher information matrix"
