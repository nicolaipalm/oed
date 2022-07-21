import numpy as np

from src.experiments.interfaces.design_of_experiment import (
    Experiment,
)
from src.minimizer.interfaces.minimizer import Minimizer
from src.statistical_models.interfaces.statistical_model import StatisticalModel


class PiDesign(Experiment):
    """parameter-individual experiment implemented within the experiment interface

    This experiment is calculated by minimizing a diagonal entry of the CRLB by changing experimental experiment.
    """
    def __init__(
            self,
            number_designs: int,
            lower_bounds_design: np.ndarray,
            upper_bounds_design: np.ndarray,
            index: int,
            initial_theta: np.ndarray,
            statistical_model: StatisticalModel,
            minimizer: Minimizer,
            previous_experiment: Experiment = None,
    ):
        """

        Parameters
        ----------
        number_designs : int
            The number of experimental experiment over which the maximization is taken

        lower_bounds_design : np.ndarray
            Lower bounds for an experimental experiment x
            with each entry representing the lower bound of the respective entry of x

        upper_bounds_design :  np.ndarray
            Lower bounds for an experimental experiment x
            with each entry representing the lower bound of the respective entry of x
        index : int
            Index, i.e. diagonal entry, which should be minimized. Starts at zero.

        initial_theta : np.ndarray
            Parameter theta of the statistical models on which the Fisher information matrix is evaluated

        statistical_model : StatisticalModel
            Underlying statistical models implemented within the StatisticalModel interface

        minimizer : Minimizer
            Minimizer used to maximize the Fisher information matrix

        previous_experiment : Experiment
            Joint previously conducted experiment used within the maximization
            of the determinant of the Fisher information matrix
        """
        print(f"Calculating the {self.name}...")
        np.array([upper_bounds_design for _ in range(number_designs)])
        if previous_experiment is None:
            self._design = minimizer(
                function=lambda x: statistical_model.calculate_cramer_rao_lower_bound(
                    theta=initial_theta,
                    x0=x.reshape(number_designs, len(lower_bounds_design)),
                )[index, index],
                lower_bounds=np.array(lower_bounds_design.tolist() * number_designs),
                upper_bounds=np.array(upper_bounds_design.tolist() * number_designs),
            ).reshape(number_designs, len(lower_bounds_design))
        else:
            # If we want to consider an initial experiment within our calculation of the CRLB.
            self._design = np.concatenate(
                (
                    previous_experiment.experiment,
                    minimizer(
                        function=lambda x: statistical_model.calculate_cramer_rao_lower_bound(
                            theta=initial_theta,
                            x0=np.concatenate(
                                (
                                    previous_experiment.experiment,
                                    x.reshape(number_designs, len(lower_bounds_design)),
                                ),
                                axis=0,
                            ),
                        )[
                            index, index
                        ],
                        lower_bounds=np.array(
                            lower_bounds_design.tolist() * number_designs
                        ),
                        upper_bounds=np.array(
                            upper_bounds_design.tolist() * number_designs
                        ),
                    ).reshape(number_designs, len(lower_bounds_design)),
                ),
                axis=0,
            )

        print("finished!\n")

    @property
    def name(self) -> str:
        return "pi"

    @property
    def experiment(self) -> np.ndarray:
        return self._design
