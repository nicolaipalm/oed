import unittest

import numpy as np

from src.minimizer.minimizer_library.differential_evolution import DifferentialEvolution
from src.parametric_function_library.linear_function import LinearFunction
from src.statistical_models.statistical_model_library.gaussian_noise_model import GaussianNoiseModel

dimension = 3
function = LinearFunction()
theta = np.ones(dimension + 1)
x = np.ones(dimension)
x0 = np.identity(dimension)

lower_bounds_theta = 0 * np.ones(dimension + 1)
upper_bounds_theta = 3 * np.ones(dimension + 1)

lower_bounds_x = 0 * np.ones(dimension + 1)
upper_bounds_x = 10 * np.ones(dimension + 1)

model = GaussianNoiseModel(function=function,
                           lower_bounds_theta=lower_bounds_theta,
                           upper_bounds_theta=upper_bounds_theta,
                           lower_bounds_x=lower_bounds_x,
                           upper_bounds_x=upper_bounds_x,
                           sigma=0.1,
                           )


class TestGaussianNoiseModel(unittest.TestCase):

    def test_if_model_can_be_called_correctly(self):
        self.assertEqual(np.array([dimension + 1]), model(theta=theta, x=x))

    def test_if_random_drawing_returns_random_drawing(self):
        self.assertAlmostEqual(4, model.random(x=x, theta=theta)[0], delta=1)

    def test_if_fim_is_calculated_correctly(self):
        matrix = 1 / model._var * np.array([[3, 1, 1, 1], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]])
        self.assertEqual(True, (matrix == model.calculate_fisher_information_matrix(x0=x0, theta=theta)).all())

    def test_if_likelihood_is_calculated_correctly(self):
        pass

    def test_if_mle_is_approximately_found(self):
        minimizer = DifferentialEvolution()
        y = np.array([function(theta=theta, x=x) for x in x0])
        mle = model.calculate_maximum_likelihood_estimation(x0=x0, y=y, minimizer=minimizer)
        self.assertAlmostEqual(0, float(np.mean(mle - theta)), delta=0.5)

    def test_if_determinant_of_fim_is_calculated_correctly(self):
        # The FIM is not invertible, i.e. the determinant must be (approximately) zero
        self.assertAlmostEqual(0, model.calculate_determinant_fisher_information_matrix(x0=x0, theta=theta), delta=1e-5)

    def test_if_bounds_are_returned_correctly(self):
        self.assertEqual(True, (lower_bounds_x == model.lower_bounds_x).all())
        self.assertEqual(True, (lower_bounds_theta == model.lower_bounds_theta).all())
        self.assertEqual(True, (upper_bounds_x == model.upper_bounds_x).all())
        self.assertEqual(True, (upper_bounds_theta == model.upper_bounds_theta).all())


if __name__ == '__main__':
    unittest.main()
