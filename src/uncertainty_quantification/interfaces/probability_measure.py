from abc import abstractmethod

import numpy as np


class ProbabilityMeasure:
    @abstractmethod
    def random(self) -> np.ndarray:
        """Draw a random sample of the measurement space

        Parameters
        ----------

        Returns
        -------
        np.ndarray
            random drawing from the probability measure P
        """
        pass
