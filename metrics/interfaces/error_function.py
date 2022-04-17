from abc import ABC, abstractmethod

import numpy as np


class ErrorFunction(ABC):
    @abstractmethod
    def __call__(self, y1: np.ndarray, y2: np.ndarray) -> float:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
