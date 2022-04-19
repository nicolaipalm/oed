from random import uniform

import numpy as np

from src.designs_of_experiments.interfaces.design_of_experiment import DesignOfExperiment


class Random(DesignOfExperiment):

    def __init__(self,
                 number_designs: int,
                 lower_bounds_design: np.ndarray,
                 upper_bounds_design: np.ndarray):
        self._design = np.array([[uniform(lower_bounds_design[index], upper_bounds_design[index]) for index in
                                  range(len(lower_bounds_design))] for _ in
                                 range(number_designs)])

    @property
    def name(self) -> str:
        return "Random design"

    @property
    def design(self) -> np.ndarray:
        return self._design
