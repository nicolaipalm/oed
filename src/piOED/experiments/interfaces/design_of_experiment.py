from abc import ABC, abstractmethod

import numpy as np


class Experiment(ABC):
    """Interface for an experiment

    We refer to an experiment as vector of experimental experiment.
    We store an experiment as a numpy array with each entry representing an experimental design.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the experiment
        Returns
        -------
        str
            the name of the experiment
        """
        pass

    @property
    @abstractmethod
    def experiment(self) -> np.ndarray:
        """Experiment consisting of experimental designs

        We refer to an experiment as vector of experimental experiment.
        We store an experiment as a numpy array with each entry representing an experimental design.

        Returns
        -------
        np.ndarray
            experiment
        """
        pass
