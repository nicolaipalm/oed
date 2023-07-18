import numpy as np

from ...experiments.interfaces.design_of_experiment import (
    Experiment,
)
from ...metrics.interfaces.metric import Metric


class LeaveOneOutValidation(Metric):
    """TBA

    """
    def calculate(
        self,
        evaluations_blackbox_function: np.ndarray,
        estimations_of_parameter: np.ndarray,
        experiment: Experiment = None,
    ) -> float:
        """TBA

        Parameters
        ----------
        evaluations_blackbox_function :
        estimations_of_parameter :
        experiment :

        Returns
        -------

        """
        pass

    @property
    def name(self) -> str:
        return "Leave one out validation"
