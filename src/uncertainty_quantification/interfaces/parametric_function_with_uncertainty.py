from abc import abstractmethod

import numpy as np

from src.parametric_function_library.interfaces.parametric_function import (
    ParametricFunction,
)
from src.uncertainty_quantification.interfaces.probability_measure import (
    ProbabilityMeasure,
)


class ParametricFunctionWithUncertainty:
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the function f_{-} at theta and x

        Parameters
        ----------
        theta : np.ndarray
            Parameter of the parametric model
        x : np.ndarray
            Point on which function should be evaluated

        Returns
        -------
        np.ndarray
            Parametric function f_theta(x) evaluated at theta and x
        """
        pass

    @property
    def probability_measure_on_parameter_space(self) -> ProbabilityMeasure:
        pass

    @property
    def parametric_function(self) -> ParametricFunction:
        pass

    @abstractmethod
    def calculate_uncertainty(self, x: np.ndarray) -> np.ndarray:
        pass

    def calculate_mean(
        self,
        x: np.ndarray,
        number_evaluations: int = 1000,
    ) -> np.ndarray:
        return np.average(
            [
                self.parametric_function(
                    x=x, theta=self.probability_measure_on_parameter_space.random()
                )
                for _ in range(number_evaluations)
            ],
            axis=0,
        )

    def calculate_std(
        self,
        x: np.ndarray,
        number_evaluations: int = 1000,
    ) -> np.ndarray:
        evaluations = [
            self.parametric_function(
                x=x, theta=self.probability_measure_on_parameter_space.random()
            )
            for _ in range(number_evaluations)
        ]
        return np.sqrt(
            1
            / (len(evaluations) - 1)
            * np.sum(
                (evaluations - np.average(evaluations, axis=0)) ** 2,
                axis=0,
            )
        )
