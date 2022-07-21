import unittest

import numpy as np

from src.experiments.experiment_library.latin_hypercube import LatinHypercube
from src.experiments.experiment_library.pi_design import PiDesign
from src.minimizer.minimizer_library.differential_evolution import DifferentialEvolution
from src.parametric_function_library.linear_function import LinearFunction
from src.statistical_models.statistical_model_library.gaussian_noise_model import GaussianNoiseModel

number_designs = 3
dimension = 2
function = LinearFunction()
theta = np.ones(dimension + 1)
x = np.ones(dimension)
index = 0

lower_bounds_theta = 0 * np.ones(dimension + 1)
upper_bounds_theta = 3 * np.ones(dimension + 1)

lower_bounds_x = 0 * np.ones(dimension + 1)
upper_bounds_x = 10 * np.ones(dimension + 1)

minimizer = DifferentialEvolution()

previous_experiment = LatinHypercube(number_designs=number_designs,
                                     lower_bounds_design=lower_bounds_x,
                                     upper_bounds_design=upper_bounds_x)

statistical_model = GaussianNoiseModel(function=function,
                                       lower_bounds_theta=lower_bounds_theta,
                                       upper_bounds_theta=upper_bounds_theta,
                                       lower_bounds_x=lower_bounds_x,
                                       upper_bounds_x=upper_bounds_x,
                                       sigma=1,
                                       )

experiment = PiDesign(number_designs=number_designs,
                      lower_bounds_design=lower_bounds_x,
                      upper_bounds_design=upper_bounds_x,
                      initial_theta=theta,
                      minimizer=minimizer,
                      statistical_model=statistical_model,
                      previous_experiment=previous_experiment,
                      index=index,
                      )


class TestDDesign(unittest.TestCase):
    def test_if_name_is_correct(self):
        self.assertEqual("pi", experiment.name)  # add assertion here

    def test_if_experiment_is_calculated_correctly(self):
        # Test if random experiments have higher determinant
        all_smaller = True
        index_pi = statistical_model.calculate_cramer_rao_lower_bound(x0=experiment.designs, theta=theta).diagonal()[
            index]

        for _ in range(100):
            test_experiment = LatinHypercube(number_designs=2 * number_designs,
                                             lower_bounds_design=lower_bounds_x,
                                             upper_bounds_design=upper_bounds_x)

            index_test = statistical_model.calculate_cramer_rao_lower_bound(x0=test_experiment.designs,
                                                                            theta=theta).diagonal()[index]

            if index_test < index_pi:
                all_smaller = False
                break

        self.assertEqual(True, all_smaller)


if __name__ == '__main__':
    unittest.main()
