import numpy as np

from src.parametric_function_library.interfaces.parametric_function import (
    ParametricFunction,
)
from src.uncertainty_quantification.interfaces.parametric_function_with_uncertainty import (
    ParametricFunctionWithUncertainty,
)
from src.uncertainty_quantification.interfaces.probability_measure import (
    ProbabilityMeasure,
)


class ParametricFunctionWithStd(ParametricFunctionWithUncertainty):
    def __init__(
        self,
        parametric_function: ParametricFunction,
        probability_measure_on_parameter_space: ProbabilityMeasure,
    ):
        self._parametric_function = parametric_function
        self._probability_measure_on_parameter_space = (
            probability_measure_on_parameter_space
        )

    @property
    def parametric_function(self) -> ParametricFunction:
        return self._parametric_function

    @property
    def probability_measure_on_parameter_space(self) -> ProbabilityMeasure:
        return self._probability_measure_on_parameter_space

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.calculate_mean(x=x)

    def calculate_uncertainty(self, x: np.ndarray) -> np.ndarray:
        return self.calculate_std(x=x)
