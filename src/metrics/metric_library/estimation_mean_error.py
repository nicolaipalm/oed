import numpy as np

from src.experiments.experiment_library.latin_hypercube import LatinHypercube
from src.experiments.interfaces.design_of_experiment import (
    Experiment,
)
from src.metrics.error_functions.average_error import AverageError
from src.metrics.interfaces.error_function import ErrorFunction
from src.metrics.interfaces.metric import Metric
from src.statistical_models.interfaces.statistical_model import StatisticalModel


class EstimationMeanError(Metric):
    # TODO: correct?
    """Mean error estimation implemented within the metric interface


    Requires statistical model with underlying parametric function f which is called by calling the statistical model
    and the true parameter theta of the parametric function.
    Given an error function e, we approximate the integral of e applied to the underlying function f_theta
    and the estimated function f_estimated_theta for each estimated theta.
    Calculating the metric results in the mean taken over the above (approximated) integrals.
    """

    def __init__(
        self,
        theta: np.ndarray,
        statistical_model: StatisticalModel,
        number_evaluations: int = 100,
        error_function: ErrorFunction = AverageError(),
    ):
        self._theta = theta
        self._statistical_model = statistical_model
        self._number_evaluations = number_evaluations
        self._error_function = error_function

    def calculate(
        self,
        estimations_of_parameter: np.ndarray,
        evaluations_blackbox_function: np.ndarray = None,
        design: Experiment = None,
    ) -> np.ndarray:
        lh = LatinHypercube(
            number_designs=self._number_evaluations,
            lower_bounds_design=self._statistical_model.lower_bounds_x,
            upper_bounds_design=self._statistical_model.upper_bounds_x,
        )
        error = []
        output_data = np.array(
            [self._statistical_model(theta=self._theta, x=x) for x in lh.designs]
        )
        for parameter in estimations_of_parameter:
            output_model = np.array(
                [self._statistical_model(theta=parameter, x=x) for x in lh.designs]
            )
            error.append(self._error_function(output_data, output_model))

        return np.average(error) * np.ones(1)

    @property
    def name(self) -> str:
        return "Approximate mean error"
