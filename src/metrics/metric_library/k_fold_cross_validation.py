import numpy as np
from sklearn.model_selection import KFold

from src.experiments.interfaces.design_of_experiment import (
    Experiment,
)
from src.metrics.error_functions.average_error import AverageError
from src.metrics.interfaces.error_function import ErrorFunction
from src.metrics.interfaces.metric import Metric
from src.minimizer.interfaces.minimizer import Minimizer
from src.statistical_models.interfaces.statistical_model import StatisticalModel


class KFoldCrossValidation(Metric):
    """K-fold cross validation metric implemented within the metric interface


    """
    def __init__(
        self,
        statistical_model: StatisticalModel,
        minimizer: Minimizer,
        error_function: ErrorFunction = AverageError(),
        number_splits: int = None,
    ):
        """TBA

        Parameters
        ----------
        statistical_model :
        minimizer :
        error_function :
        number_splits :
        """
        self.number_splits = number_splits
        self._error_function = error_function
        self._statistical_model = statistical_model
        self._minimizer = minimizer

    def calculate(
        self,
        evaluations_blackbox_function: np.ndarray,
        experiment: Experiment,
        estimations_of_parameter: np.ndarray = None,
    ) -> np.ndarray:
        """TBA

        Parameters
        ----------
        evaluations_blackbox_function :
        experiment :
        estimations_of_parameter :

        Returns
        -------

        """
        if len(evaluations_blackbox_function.shape) > 2:
            raise ValueError("The evaluation array must be of dimension 1 or 2.")

        if self.number_splits is None:
            self.number_splits = len(evaluations_blackbox_function)

        k_fold = KFold(n_splits=self.number_splits, shuffle=True)
        error = []
        for train_index, test_index in k_fold.split(evaluations_blackbox_function):
            design_training, design_test = (
                experiment.experiment[train_index],
                experiment.experiment[test_index],
            )
            y_training, y_test = (
                evaluations_blackbox_function[train_index],
                evaluations_blackbox_function[test_index],
            )
            estimated_parameter = (
                self._statistical_model.calculate_maximum_likelihood_estimation(
                    x0=design_training, y=y_training, minimizer=self._minimizer
                )
            )

            error.append(
                self._error_function(
                    y1=y_test,
                    y2=np.array(
                        [
                            self._statistical_model(theta=estimated_parameter, x=x)
                            for x in design_test
                        ]
                    ),
                )
            )

        return np.average(np.array(error)) * np.ones(1)

    @property
    def name(self) -> str:
        return "K-fold cross validation"
