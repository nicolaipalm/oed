from typing import Callable

import numpy as np

from designs_of_experiments.design_library.latin_hypercube import LatinHypercube
from designs_of_experiments.interfaces.design_of_experiment import DesignOfExperiment
from metrics.interfaces.metric import Metric
from statistical_models.interfaces.statistical_model import StatisticalModel


class EstimationMeanError(Metric):
    def __init__(self, blackbox_function: Callable, statistical_model: StatisticalModel, theta: np.ndarray,
                 number_evaluations: int = 100):
        self._blackbox_function = blackbox_function
        self._statistical_model = statistical_model
        self._number_evaluations = number_evaluations
        self._theta = theta

    # !!!! no
    def calculate(self,
                  evaluations_blackbox_function: np.ndarray = None,
                  estimations_of_parameter: np.ndarray = None,
                  design: DesignOfExperiment = None, ) -> float:
        print(f"The number of evaluations is {self._number_evaluations}")
        lh = LatinHypercube(number_designs=self._number_evaluations,
                            lower_bounds_design=self._statistical_model.lower_bounds_x,
                            upper_bounds_design=self._statistical_model.upper_bounds_x)
        output_data = np.array([self._blackbox_function(x) for x in lh.design])
        output_model = np.array([self._statistical_model(theta=self._theta, x=x) for x in lh.design])

        return np.average(abs(output_data - output_model))

    @property
    def name(self) -> str:
        return "Approximate mean error"
