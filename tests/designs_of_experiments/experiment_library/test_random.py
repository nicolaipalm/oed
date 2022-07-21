import unittest

import numpy as np

from src.experiments.experiment_library.random import Random

lower_bounds_design = np.zeros(5)
upper_bounds_design = np.ones(5)
number_designs = 4



experiment = Random(number_designs=number_designs,
                    lower_bounds_design=lower_bounds_design,
                    upper_bounds_design=upper_bounds_design)


class TestRandom(unittest.TestCase):
    def test_if_name_is_correct(self):
        self.assertEqual("Random", experiment.name)

    def test_if_experiment_is_calculated_correctly(self):
        pass


if __name__ == '__main__':
    unittest.main()
