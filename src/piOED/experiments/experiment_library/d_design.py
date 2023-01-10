import numpy as np

from piOED.experiments.interfaces.design_of_experiment import (
    Experiment,
)
from piOED.minimizer.interfaces.minimizer import Minimizer
from piOED.statistical_models.interfaces.statistical_model import StatisticalModel
from scipy.optimize import LinearConstraint, NonlinearConstraint


class DDesign(Experiment):
    """D-optimal design implementation within the experiment interface

    The D-optimal design is calculated by maximizing the determinant of the Fisher information matrix
    by changing experimental experiment.
    """

    def __init__(
            self,
            number_designs: int,
            lower_bounds_design: np.ndarray,
            upper_bounds_design: np.ndarray,
            initial_theta: np.ndarray,
            statistical_model: StatisticalModel,
            minimizer: Minimizer,
            constraints: {LinearConstraint, NonlinearConstraint}=None,
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

        initial_theta : np.ndarray
            Parameter theta of the statistical models on which the Fisher information matrix is evaluated

        statistical_model : StatisticalModel
            Underlying statistical models implemented within the StatisticalModel interface

        minimizer : Minimizer
            Minimizer used to maximize the Fisher information matrix

        constraints : {LinearConstraint, NonlinearConstraint}
            Constraints used within the minimization

        previous_experiment : Experiment
            Joint previously conducted experiment used within the maximization
            of the determinant of the Fisher information matrix

        """
        print(f"Calculating the {self.name}...")
        np.array([upper_bounds_design for _ in range(number_designs)])

        if previous_experiment is None:
            self._design = minimizer(
                function=statistical_model.optimize_determinant_fisher_information_matrix,
                fcn_args=(previous_experiment, initial_theta, number_designs, len(lower_bounds_design)),
                lower_bounds=np.array(lower_bounds_design.tolist() * number_designs),
                upper_bounds=np.array(upper_bounds_design.tolist() * number_designs),
                constraints=constraints,
            ).reshape(number_designs, len(lower_bounds_design))
        else:
            # If we want to consider an initial experiment within our calculation of the CRLB.
            self._design = np.concatenate(
                (
                    previous_experiment.experiment,
                    minimizer(
                        function=statistical_model.optimize_determinant_fisher_information_matrix,
                        fcn_args=(previous_experiment, initial_theta, number_designs, len(lower_bounds_design)),
                        lower_bounds=np.array(lower_bounds_design.tolist() * number_designs),
                        upper_bounds=np.array(upper_bounds_design.tolist() * number_designs),
                        constraints=constraints,
                    ).reshape(number_designs, len(lower_bounds_design)),
                ),
                axis=0,
            )

    @property
    def name(self) -> str:
        return "D-opt"

    @property
    def experiment(self) -> np.ndarray:
        return self._design
