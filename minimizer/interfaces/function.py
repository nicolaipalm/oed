from abc import abstractmethod, ABC

import numpy as np

# TODO: redundant
class Function(ABC):
    @abstractmethod
    def __call__(self,
                 x: np.ndarray) -> float:
        pass

    def partial_derivative(self,
                           x: np.ndarray,
                           parameter_index: int) -> float:
        pass

    def second_partial_derivative(self,
                                  x: np.ndarray,
                                  parameter1_index: int,
                                  parameter2_index: int) -> float:
        pass
