import numpy as np

from src.experiments.interfaces.design_of_experiment import (
    Experiment,
)
from src.minimizer.interfaces.minimizer import Minimizer
from src.statistical_models.interfaces.statistical_model import StatisticalModel


class DDesign(Experiment):
    """D-optimal design implementation within the experiment interface

    The D-optimal design is calculated by maximizing the determinant of the Fisher information matrix
    by changing experimental experiments.
    """

    def __init__(
            self,
            number_designs: int,
            lower_bounds_design: np.ndarray,
            upper_bounds_design: np.ndarray,
            initial_theta: np.ndarray,
            statistical_model: StatisticalModel,
            minimizer: Minimizer,
            previous_experiment: Experiment = None,
    ):
        """
        Parameters
        ----------
        number_designs : int
            The number of experimental experiments over which the maximization is taken

        lower_bounds_design : np.ndarray
            Lower bounds for an experimental experiments x
            with each entry representing the lower bound of the respective entry of x

        upper_bounds_design :  np.ndarray
            Lower bounds for an experimental experiments x
            with each entry representing the lower bound of the respective entry of x

        initial_theta : np.ndarray
            Parameter theta of the statistical models on which the Fisher information matrix is evaluated

        statistical_model : StatisticalModel
            Underlying statistical models implemented within the StatisticalModel interface

        minimizer : Minimizer
            Minimizer used to maximize the Fisher information matrix

        previous_experiment : Experiment
            Joint previously conducted experiments used within the maximization
            of the determinant of the Fisher information matrix

        """
        print(f"Calculating the {self.name}...")
        np.array([upper_bounds_design for _ in range(number_designs)])
        self._design = \
            np.concatenate(
                (
                    previous_experiment.designs,
                    minimizer(
                        function=lambda x: -statistical_model.calculate_determinant_fisher_information_matrix(
                            theta=initial_theta,
                            x0=np.concatenate((previous_experiment.designs,
                                               x.reshape(number_designs, len(lower_bounds_design)))),
                        ),
                        lower_bounds=np.array(lower_bounds_design.tolist() * number_designs),
                        upper_bounds=np.array(upper_bounds_design.tolist() * number_designs),
                    ).reshape(number_designs, len(lower_bounds_design))), axis=0)
        print("finished!\n")

    @property
    def name(self) -> str:
        return "D-opt"

    @property
    def designs(self) -> np.ndarray:
        return self._design
