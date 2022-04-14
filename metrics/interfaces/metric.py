from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    @abstractmethod
    def calculate(self,
                  evaluations_blackbox_function: np.ndarray = None,
                  estimations_of_parameter: np.ndarray = None,
                  ) -> float:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
