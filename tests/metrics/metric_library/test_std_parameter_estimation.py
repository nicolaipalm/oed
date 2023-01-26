import unittest

import numpy as np

from piOED.metrics.metric_library.std_parameter_estimations import StdParameterEstimations

metric = StdParameterEstimations()
estimations_of_parameter = np.array([0, 0, 0] + [2, 2, 2])


class TestStdParameterEstimation(unittest.TestCase):
    def test_if_correct_name_is_returned(self):
        self.assertEqual("Estimated standard deviation of parameter estimations", metric.name)

    def test_something(self):
        self.assertAlmostEqual(np.sqrt(6 / 5), metric.calculate(estimations_of_parameter=estimations_of_parameter),
                               delta=1e-5)


if __name__ == '__main__':
    unittest.main()
