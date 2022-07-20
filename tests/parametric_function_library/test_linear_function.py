import unittest

import numpy as np

from src.parametric_function_library.linear_function import LinearFunction


class TestLinearFunction(unittest.TestCase):
    def test_if_function_returns_correct_output(self):
        dimension = 3
        function = LinearFunction()
        theta = np.ones(dimension + 1)
        x = np.ones(dimension)
        self.assertEqual(np.array([dimension + 1]), function(theta=theta, x=x))

    def test_if_partial_derivative_returns_correct_output(self):
        dimension = 3
        function = LinearFunction()
        theta = np.ones(dimension + 1)
        x = np.arange(dimension)
        self.assertEqual(np.array([1]), function.partial_derivative(theta=theta, x=x, parameter_index=0))
        self.assertEqual(np.array([x[0]]), function.partial_derivative(theta=theta, x=x, parameter_index=1))

    def test_if_second_partial_derivative_returns_correct_output(self):
        dimension = 3
        function = LinearFunction()
        theta = np.ones(dimension + 1)
        x = np.ones(dimension)
        self.assertEqual(np.array([0]),
                         function.second_partial_derivative(parameter1_index=1, parameter2_index=2, theta=theta, x=x))


if __name__ == '__main__':
    unittest.main()
