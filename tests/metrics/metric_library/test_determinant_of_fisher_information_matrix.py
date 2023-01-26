import unittest

import numpy as np

from piOED.experiments.experiment_library.latin_hypercube import LatinHypercube
from piOED.metrics.metric_library.determinant_of_fisher_information_matrix import DeterminantOfFisherInformationMatrix
from piOED.parametric_function_library.linear_function import LinearFunction
from piOED.statistical_models.statistical_model_library.gaussian_noise_model import GaussianNoiseModel

dimension = 3
function = LinearFunction()
theta = np.ones(dimension + 1)

lower_bounds_theta = 0 * np.ones(dimension + 1)
upper_bounds_theta = 3 * np.ones(dimension + 1)

lower_bounds_x = 0 * np.ones(dimension)
upper_bounds_x = 10 * np.ones(dimension)

number_designs = 3

experiment = LatinHypercube(number_designs=number_designs,
                            lower_bounds_design=lower_bounds_x,
                            upper_bounds_design=upper_bounds_x)

model = GaussianNoiseModel(function=function,
                           lower_bounds_theta=lower_bounds_theta,
                           upper_bounds_theta=upper_bounds_theta,
                           lower_bounds_x=lower_bounds_x,
                           upper_bounds_x=upper_bounds_x,
                           sigma=1,
                           )

metric = DeterminantOfFisherInformationMatrix(theta=theta, statistical_model=model)


class TestStdParameterEstimation(unittest.TestCase):
    def test_if_correct_name_is_returned(self):
        self.assertEqual("Determinant of Fisher information matrix", metric.name)

    def test_if_determinant_of_fim_is_calculated_correctly(self):
        # The FIM is not invertible, i.e. the determinant must be (approximately) zero
        self.assertAlmostEqual(0, metric.calculate(experiment=experiment)[0], delta=1e-7)


if __name__ == '__main__':
    unittest.main()
