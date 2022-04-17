import numpy as np

from designs_of_experiments.design_library.latin_hypercube import LatinHypercube
from designs_of_experiments.interfaces.design_of_experiment import DesignOfExperiment
from metrics.error_functions.average_error import AverageError
from metrics.interfaces.error_function import ErrorFunction
from metrics.interfaces.metric import Metric
from statistical_models.interfaces.statistical_model import StatisticalModel


class EstimationMeanError(Metric):
    """
    This takes the mean over the estimations of the integral over the error function over the estimated parameters, i.e.
    error_function(real_function,model)
    -> better docu and how this is related to cross validation error
    """

    def __init__(self, theta: np.ndarray, statistical_model: StatisticalModel,
                 number_evaluations: int = 100, error_function: ErrorFunction = AverageError()):
        self._theta = theta
        self._statistical_model = statistical_model
        self._number_evaluations = number_evaluations
        self._error_function = error_function

    def calculate(self,
                  estimations_of_parameter: np.ndarray,
                  evaluations_blackbox_function: np.ndarray = None,
                  design: DesignOfExperiment = None, ) -> np.ndarray:
        lh = LatinHypercube(number_designs=self._number_evaluations,
                            lower_bounds_design=self._statistical_model.lower_bounds_x,
                            upper_bounds_design=self._statistical_model.upper_bounds_x)
        error = []
        output_data = np.array([self._statistical_model(theta=self._theta, x=x) for x in lh.design])
        for parameter in estimations_of_parameter:
            output_model = np.array([self._statistical_model(theta=parameter, x=x) for x in lh.design])
            error.append(self._error_function(output_data, output_model))

        return np.average(error) * np.ones(1)

    @property
    def name(self) -> str:
        return "Approximate mean error"
