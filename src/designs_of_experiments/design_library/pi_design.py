import numpy as np

from src.designs_of_experiments.interfaces.design_of_experiment import (
    DesignOfExperiment,
)
from src.minimizer.interfaces.minimizer import Minimizer
from src.statistical_models.interfaces.statistical_model import StatisticalModel


class PiDesign(DesignOfExperiment):
    def __init__(
        self,
        number_designs: int,
        lower_bounds_design: np.ndarray,
        upper_bounds_design: np.ndarray,
        column: int,
        row: int,
        initial_theta: np.ndarray,
        statistical_model: StatisticalModel,
        minimizer: Minimizer,
        previous_design: DesignOfExperiment = None,
    ):
        print(f"Calculating the {self.name}...")
        np.array([upper_bounds_design for _ in range(number_designs)])
        if previous_design is None:
            self._design = minimizer(
                function=lambda x: statistical_model.calculate_cramer_rao_lower_bound(
                    theta=initial_theta,
                    x0=x.reshape(number_designs, len(lower_bounds_design)),
                )[column, row],
                lower_bounds=np.array(lower_bounds_design.tolist() * number_designs),
                upper_bounds=np.array(upper_bounds_design.tolist() * number_designs),
            ).reshape(number_designs, len(lower_bounds_design))
        else:
            # If we want to consider an initial design within our calculation of the CRLB.
            self._design = np.concatenate(
                (
                    previous_design.design,
                    minimizer(
                        function=lambda x: statistical_model.calculate_cramer_rao_lower_bound(
                            theta=initial_theta,
                            x0=np.concatenate(
                                (
                                    previous_design.design,
                                    x.reshape(number_designs, len(lower_bounds_design)),
                                ),
                                axis=0,
                            ),
                        )[
                            column, row
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
        return "Parameter individual design"

    @property
    def design(self) -> np.ndarray:
        """
        :return: the previous design and the pi design optimized at the given parameter concatenated.
        :rtype:
        """
        return self._design
