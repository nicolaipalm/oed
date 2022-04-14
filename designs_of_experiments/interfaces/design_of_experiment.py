from abc import ABC, abstractmethod

import numpy as np


class DesignOfExperiment(ABC):
    """
    This
    """
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def design(self) -> np.ndarray:
        pass
