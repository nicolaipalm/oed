import numpy as np

from designs_of_experiments.interfaces.design_of_experiment import DesignOfExperiment
from minimizer.interfaces.minimizer import Minimizer
from statistical_models.interfaces.statistical_model import StatisticalModel


class PiDesign(DesignOfExperiment):

    def __init__(self,
                 number_designs: int,
                 lower_bounds_design: np.ndarray,
                 upper_bounds_design: np.ndarray,
                 column: int,
                 row: int,
                 initial_theta: np.ndarray,
                 statistical_model: StatisticalModel,
                 minimizer: Minimizer):
        np.array([upper_bounds_design for _ in range(number_designs)])
        self._design = minimizer(function=lambda x: statistical_model.calculate_cramer_rao_lower_bound(theta=initial_theta,
                                                           x0=x.reshape(number_designs, len(lower_bounds_design)))[
            column, row], lower_bounds=np.array(lower_bounds_design.tolist()*number_designs),
                                 upper_bounds=np.array(upper_bounds_design.tolist()*number_designs), ).reshape(
            number_designs, len(lower_bounds_design))

    @property
    def name(self) -> str:
        return "Parameter individual design"

    @property
    def design(self) -> np.ndarray:
        return self._design