import unittest

import numpy as np

from src.minimizer.minimizer_library.differential_evolution import DifferentialEvolution

minimizer = DifferentialEvolution()


class TestDifferentialEvolution(unittest.TestCase):
    def test_if_minimizer_returns_minimum_of_function(self):
        self.assertAlmostEqual(0, float(
            minimizer(lambda x: x ** 2, upper_bounds=np.array([10]), lower_bounds=np.array([-3]))),delta=0.01)


if __name__ == '__main__':
    unittest.main()
