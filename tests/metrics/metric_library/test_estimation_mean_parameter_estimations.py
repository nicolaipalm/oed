import unittest

import numpy as np

from src.metrics.metric_library.estimation_mean_parameter_estimations import EstimationMeanParameterEstimations

metric = EstimationMeanParameterEstimations()
estimations_of_parameter = np.arange(1, 2 * 10)


class TestStdParameterEstimation(unittest.TestCase):
    def test_if_correct_name_is_returned(self):
        self.assertEqual("Estimated mean of parameter estimations", metric.name)

    def test_if_mean_is_calculated_correctly(self):
        self.assertEqual(10, metric.calculate(estimations_of_parameter=estimations_of_parameter))


if __name__ == '__main__':
    unittest.main()
