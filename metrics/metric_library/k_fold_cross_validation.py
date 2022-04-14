import numpy as np

from metrics.interfaces.metric import Metric


class KFoldCrossValidation(Metric):
    def calculate(self,
                  evaluations_blackbox_function: np.ndarray = None,
                  estimations_of_parameter: np.ndarray = None,
                  ) -> float:
        pass

    @property
    def name(self) -> str:
        return "K-fold cross validation"
