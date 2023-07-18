import numpy as np
from ..parametric_function_library.interfaces.parametric_function import (
    ParametricFunction,
)
from ..uncertainty_quantification.interfaces.probability_measure import (
    ProbabilityMeasure,
)

import plotly.graph_objects as go

from ..visualization.plotting_functions import styled_figure


class ParametricFunctionWithUncertainty:
    def __init__(
        self,
        parametric_function: ParametricFunction,
        probability_measure_on_parameter_space: ProbabilityMeasure,
        sample_size_parameters: int = 1000,
    ):
        self._parametric_function = parametric_function
        self._probability_measure_on_parameter_space = (
            probability_measure_on_parameter_space
        )
        self._sampled_parameters = np.array(
            [
                self._probability_measure_on_parameter_space.random()
                for _ in range(sample_size_parameters)
            ]
        )

    def __call__(
        self,
        theta: np.ndarray,
        x: np.ndarray,
    ) -> np.ndarray:
        return self.parametric_function(theta=theta, x=x)

    @property
    def sampled_parameters(self) -> np.ndarray:
        return self._sampled_parameters

    @property
    def parametric_function(self) -> ParametricFunction:
        return self._parametric_function

    @property
    def probability_measure_on_parameter_space(self) -> ProbabilityMeasure:
        return self._probability_measure_on_parameter_space

    def histo(self, x: np.ndarray):
        data = [
            go.Histogram(
                x=np.array(
                    [
                        self.parametric_function(theta=theta, x=x)[i]
                        for theta in self.sampled_parameters
                    ]
                ).flatten()
            )
            for i, _ in enumerate(
                self.parametric_function(theta=self.sampled_parameters[0], x=x)
            )
        ]
        fig = styled_figure(data=data, title=f"Histogram of predictions at x={x}")
        fig.show()

    def calculate_mean(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        return np.average(
            [
                self.parametric_function(x=x, theta=parameter)
                for parameter in self.sampled_parameters
            ],
            axis=0,
        )

    def calculate_std(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        # In many cases random variables preserve the normal distribution.
        # -> main class of (non-linear) functions preserving normal distribution
        evaluations = np.array(
            [
                self.parametric_function(x=x, theta=parameter)
                for parameter in self.sampled_parameters
            ]
        )
        return np.sqrt(
            1 / (len(evaluations) - 1) * np.sum((evaluations - np.average(evaluations, axis=0)) ** 2, axis=0,)
        )

    def calculate_quantile(self, x: np.ndarray, alpha: float = 0.95) -> np.ndarray:
        evaluations = np.array(
            [
                self.parametric_function(x=x, theta=parameter)
                for parameter in self.sampled_parameters
            ]
        )
        return np.quantile(a=evaluations, q=alpha, axis=0)
