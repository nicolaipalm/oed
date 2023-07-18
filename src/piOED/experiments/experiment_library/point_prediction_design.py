import numpy as np

from ...experiments.interfaces.design_of_experiment import (
    Experiment,
)
from ...minimizer.interfaces.minimizer import Minimizer
from ...uncertainty_quantification.parametric_function_with_uncertainty import (
    ParametricFunctionWithUncertainty,
)


class PointPredictionDesign(Experiment):
    """Point prediction design

    seeks the experimental design which has greatest (sum of components of)
    alpha-quantile minus alpha-fractile in predictions based on the parametric function with uncertainty.
    """

    def __init__(
        self,
        lower_bounds_design: np.ndarray,
        upper_bounds_design: np.ndarray,
        minimizer: Minimizer,
        parametric_function_with_uncertainty: ParametricFunctionWithUncertainty,
        alpha: float = 0.9,
    ):
        self._design = minimizer(
            function=self.objective_function,
            fcn_args=(parametric_function_with_uncertainty, alpha),
            upper_bounds=upper_bounds_design,
            lower_bounds=lower_bounds_design,
            constraints=(),
        ).reshape(1, len(lower_bounds_design))

    @staticmethod
    def objective_function(x: np.ndarray, args: tuple) -> float:
        """Objective function for point prediction design."""

        parametric_function_with_uncertainty, alpha = args
        return -np.sum(
            parametric_function_with_uncertainty.calculate_quantile(
                x=x, alpha=alpha
            )
            - parametric_function_with_uncertainty.calculate_quantile(
                x=x, alpha=1 - alpha
            )
        )

    @property
    def name(self) -> str:
        return "point prediction"

    @property
    def experiment(self) -> np.ndarray:
        return self._design
