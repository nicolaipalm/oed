import numpy as np

from src.designs_of_experiments.interfaces.design_of_experiment import (
    DesignOfExperiment,
)
from src.metrics.interfaces.metric import Metric


class StdParameterEstimations(Metric):
    def calculate(
        self,
        estimations_of_parameter: np.ndarray,
        evaluations_blackbox_function: np.ndarray = None,
        design: DesignOfExperiment = None,
    ) -> float:
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
