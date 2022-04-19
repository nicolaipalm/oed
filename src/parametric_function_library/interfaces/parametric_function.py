from abc import abstractmethod, ABC

import numpy as np


class ParametricFunction(ABC):
    """
    a class of one dimensional functions parametrized by real numbers (theta)
    """

    @abstractmethod
    def __call__(self,
                 theta: np.ndarray,
                 x: np.ndarray) -> float:
        """
        :returns f_{theta}(x)
        """
        pass

    @abstractmethod
    def partial_derivative(self,
                           theta: np.ndarray,
                           x: np.ndarray,
                           parameter_index: int) -> float:
        """
        :param theta:
        :type theta:
        :param x:
        :type x:
        :param parameter_index:
        :type parameter_index:
        :return: partial derivative at theta_{parameter_index}
        :rtype: float
        """
        pass

    def second_partial_derivative(self,
                                  theta: np.ndarray,
                                  x: np.ndarray,
                                  parameter1_index: int,
                                  parameter2_index: int) -> float:
        pass
