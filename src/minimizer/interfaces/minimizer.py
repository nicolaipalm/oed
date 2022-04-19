from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class Minimizer(ABC):
    @abstractmethod
    def __call__(
        self, function: Callable, upper_bounds: np.ndarray, lower_bounds: np.ndarray
    ) -> np.ndarray:
        pass
