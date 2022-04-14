from typing import List, Callable

import numpy as np
from tqdm import tqdm

from designs_of_experiments.interfaces.design_of_experiment import DesignOfExperiment
from metrics.interfaces.metric import Metric
from minimizer.interfaces.minimizer import Minimizer
from statistical_models.interfaces.statistical_model import StatisticalModel


class Benchmarking:
    def __init__(self,
                 blackbox_model: Callable,
                 statistical_model: StatisticalModel,
                 designs_of_experiments: List[DesignOfExperiment],
                 metrics: List[Metric],
                 ):
        self.blackbox_model = blackbox_model
        self.statistical_model = statistical_model
        self.designs_of_experiments = designs_of_experiments
        self.metrics = metrics
        self.evaluations_blackbox_function = {}
        self.maximum_likelihood_estimations = {}
        self.evaluations_of_metrics = {}

    @property
    def design_names(self) -> List[str]:
        return [doe.name for doe in self.designs_of_experiments]

    @property
    def metric_names(self) -> List[str]:
        return [metric.name for metric in self.metrics]

    def evaluate_designs(self, number_of_evaluations: int, minimizer: Minimizer):
        for doe in self.designs_of_experiments:
            evaluations = []
            estimations = []
            print(f'\n Evaluate the {doe.name}...')
            # TODO: prevent overflow
            for _ in tqdm(range(number_of_evaluations)):
                evaluation = np.array([self.blackbox_model(x) for x in doe.design])
                evaluations.append(evaluation)
                estimations.append(
                    self.statistical_model.calculate_maximum_likelihood_estimation(x0=doe.design, y=evaluation,
                                                                                   minimizer=minimizer))
            self.evaluations_blackbox_function[doe.name] = np.array(evaluations)
            self.maximum_likelihood_estimations[doe.name] = np.array(estimations)

    def calculate_metrics(self):
        for metric in self.metrics:
            self.evaluations_of_metrics[metric.name] = {}
            for doe in self.designs_of_experiments:
                self.evaluations_of_metrics[metric.name][doe.name] = metric.calculate(
                    evaluations_blackbox_function=self.evaluations_blackbox_function[doe.name],
                    estimations_of_parameter=self.maximum_likelihood_estimations[doe.name])

