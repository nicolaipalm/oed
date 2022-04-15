from typing import List, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from designs_of_experiments.interfaces.design_of_experiment import DesignOfExperiment
from minimizer.interfaces.minimizer import Minimizer
from statistical_models.interfaces.statistical_model import StatisticalModel


class Benchmarking:
    """
    This class provides an easy way to evaluate different designs
    (i.e. calculate the maximum likelihood parameter estimation) and to store and load the data.
    """

    def __init__(self,
                 blackbox_model: Callable,
                 statistical_model: StatisticalModel,
                 designs_of_experiments: List[DesignOfExperiment],
                 ):
        self._blackbox_model = blackbox_model
        self._statistical_model = statistical_model
        self._designs_of_experiments = designs_of_experiments
        self.evaluations_blackbox_function = {}
        self.maximum_likelihood_estimations = {}

    @property
    def design_names(self) -> List[str]:
        return [doe.name for doe in self._designs_of_experiments]

    @property
    def designs(self) -> List[DesignOfExperiment]:
        return self._designs_of_experiments

    @property
    def statistical_model(self) -> StatisticalModel:
        return self._statistical_model

    @property
    def blackbox_model(self) -> Callable:
        return self._blackbox_model

    def evaluate_designs(self, number_of_evaluations: int, minimizer: Minimizer):
        for doe in self._designs_of_experiments:
            evaluations = []
            estimations = []
            print(f'\n Evaluate the {doe.name}...')
            # TODO: prevent overflow
            for _ in tqdm(range(number_of_evaluations)):
                evaluation = np.array([self._blackbox_model(x) for x in doe.design])
                evaluations.append(evaluation)
                estimations.append(
                    self._statistical_model.calculate_maximum_likelihood_estimation(x0=doe.design, y=evaluation,
                                                                                    minimizer=minimizer))
            self.evaluations_blackbox_function[doe] = np.array(evaluations)
            self.maximum_likelihood_estimations[doe] = np.array(estimations)

    def save_to_csv(self, file_name: str = "benchmarking") -> bool:
        data = pd.DataFrame([[design.name] +
                             self.evaluations_blackbox_function[design][index].tolist(
                             ) + self.maximum_likelihood_estimations[design][index].tolist(
        )
                             for design in self.designs
                             for index in range(len(self.evaluations_blackbox_function[design]))

                             ])
        pd.DataFrame(data=data).to_csv(file_name + '.csv')
        return True

    def load_from_csv(self, file_name: str = "benchmarking.csv"):
        t = pd.read_csv(file_name, index_col=0)
        number_parameters = len(self.statistical_model.lower_bounds_theta)
        for design in self.designs:
            evaluations = []
            estimations = []
            for row in t.T:
                if design.name == t.T[row][0]:
                    evaluations.append(t.T[row][1:].to_list()[:-number_parameters])
                    estimations.append(t.T[row][1:].to_list()[number_parameters:])

            self.evaluations_blackbox_function[design] = np.array(evaluations)
            self.maximum_likelihood_estimations[design] = np.array(estimations)

        return True
