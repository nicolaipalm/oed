import numpy as np
from scipy.stats import qmc

from src.designs_of_experiments.interfaces.design_of_experiment import DesignOfExperiment


class LatinHypercube(DesignOfExperiment):
    def __init__(self,
                 number_designs: int,
                 lower_bounds_design: np.ndarray,
                 upper_bounds_design: np.ndarray):
        self._design = qmc.scale(qmc.LatinHypercube(d=len(lower_bounds_design)).random(n=number_designs),
                                 lower_bounds_design, upper_bounds_design)

    @property
    def name(self) -> str:
        return "Latin Hypercube design"

    @property
    def design(self) -> np.ndarray:
        return self._design
