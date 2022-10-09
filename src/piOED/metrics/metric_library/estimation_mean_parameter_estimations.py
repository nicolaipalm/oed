import numpy as np

from piOED.experiments.interfaces.design_of_experiment import (
    Experiment,
)
from piOED.metrics.interfaces.metric import Metric


class EstimationMeanParameterEstimations(Metric):
    """Mean of parameter estimations implemented within the metric interface

    ...estimated by the arithmetic mean also known as sample mean.
    """
    def calculate(
        self,
        estimations_of_parameter: np.ndarray,
        evaluations_blackbox_function: np.ndarray = None,
        experiment: Experiment = None,
    ) -> np.ndarray:
        """TBA

        Parameters
        ----------
        estimations_of_parameter :
        evaluations_blackbox_function :
        experiment :

        Returns
        -------

        """
        return np.average(estimations_of_parameter, axis=0)

    @property
    def name(self) -> str:
        return "Estimated mean of parameter estimations"
