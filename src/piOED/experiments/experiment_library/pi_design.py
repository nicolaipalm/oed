import numpy as np

from piOED.experiments.interfaces.design_of_experiment import (
    Experiment,
)
from piOED.minimizer.interfaces.minimizer import Minimizer
from piOED.statistical_models.interfaces.statistical_model import StatisticalModel
from scipy.optimize import LinearConstraint, NonlinearConstraint


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
            constraints: {LinearConstraint, NonlinearConstraint} = None,
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
                function=statistical_model.optimize_cramer_rao_lower_bound,
                fcn_args=(previous_experiment, initial_theta, number_designs, len(lower_bounds_design), index),
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
                        function=statistical_model.optimize_cramer_rao_lower_bound,
                        fcn_args=(previous_experiment, initial_theta, number_designs, len(lower_bounds_design), index),
                        lower_bounds=np.array(lower_bounds_design.tolist() * number_designs),
                        upper_bounds=np.array(upper_bounds_design.tolist() * number_designs),
                        constraints=constraints,
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
