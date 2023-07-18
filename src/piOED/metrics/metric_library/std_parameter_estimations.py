import numpy as np

from ...experiments.interfaces.design_of_experiment import (
    Experiment,
)
from ...metrics.interfaces.metric import Metric


class StdParameterEstimations(Metric):
    """Standard deviation of parameter estimations implemented within the metric interface

    ... estimated according to https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation.
    """

    def calculate(self,
                  estimations_of_parameter: np.ndarray,
                  evaluations_blackbox_function: np.ndarray = None,
                  experiment: Experiment = None) -> float:
        """TBA

        Parameters
        ----------
        estimations_of_parameter :
        evaluations_blackbox_function :
        experiment :

        Returns
        -------

        """
        return (np.sqrt(
            1
            / (len(estimations_of_parameter) - 1)
            * np.sum(
                (
                        estimations_of_parameter
                        - np.average(estimations_of_parameter, axis=0)
                )
                ** 2,
                axis=0,
            ))
        )

    @property
    def name(self) -> str:
        return "Estimated standard deviation of parameter estimations"
