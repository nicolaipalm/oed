from typing import Callable, List, Dict

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from tqdm import tqdm

from piOED.experiments.interfaces.design_of_experiment import (
    Experiment,
)
from piOED.minimizer.interfaces.minimizer import Minimizer
from piOED.statistical_models.interfaces.statistical_model import StatisticalModel
from piOED.visualization.plotting_functions import dot_scatter, styled_figure


class Benchmarking:
    """Benchmarking class in order to compare different experiment against selected metrics

    ...based on statistical models and their maximum likelihood estimations.
    """

    def __init__(
            self,
            blackbox_model: Callable,
            statistical_model: StatisticalModel,
            experiments: List[Experiment],
    ):
        """

        Parameters
        ----------
        blackbox_model : Callable
            underlying black box function from which the results of the experiment are obtained
        statistical_model : StatisticalModel
            underlying statistical model
        designs_of_experiments : List[Experiment]
            list of experiment for benchmarking
        """
        self._blackbox_model = blackbox_model
        self._statistical_model = statistical_model
        self._designs_of_experiments = experiments
        self._evaluations_blackbox_function = {}
        self._maximum_likelihood_estimations = {}

    @property
    def evaluations_blackbox_function(self) -> Dict[Experiment, np.ndarray]:
        """
        Returns
        -------
        Dict[Experiment:np.ndarray]
            evaluations of blackbox function corresponding to the respective experiment
        """
        return self._evaluations_blackbox_function

    @property
    def maximum_likelihood_estimations(self) -> Dict[Experiment, np.ndarray]:
        """
        Returns
        -------
        Dict[Experiment:np.ndarray]
            maximum likelihood estimations corresponding to the respective evaluation of the blackbox function
        """
        return self._maximum_likelihood_estimations

    @property
    def experiments_names(self) -> List[str]:
        """
        Returns
        -------
        List[str]
            names of the underlying experiment in the same order as the list of experiment.
        """
        return [doe.name for doe in self._designs_of_experiments]

    @property
    def experiments(self) -> List[Experiment]:
        """
        Returns
        -------
        List[Experiment]
            underlying experiment used in benchmarking
        """
        return self._designs_of_experiments

    @property
    def statistical_model(self) -> StatisticalModel:
        """
        Returns
        -------
        StatisticalModel
            underlying statistical model
        """
        return self._statistical_model

    @property
    def blackbox_model(self) -> Callable:
        """
        Returns
        -------
        StatisticalModel
            underlying blackbox function
        """
        return self._blackbox_model

    def evaluate_experiments(self, number_of_evaluations: int, minimizer: Minimizer):
        """Evaluate the experiment, i.e. repeat conducting the experiment several times
        ... calculate their maximum likelihood estimate and store in the respective variable.

        Parameters
        ----------
        number_of_evaluations : int
            number of repetitions of each experiment
        minimizer : Minimizer
            minimizer used in the calculations of the maximum likelihood estimations

        """
        for doe in self._designs_of_experiments:
            evaluations = []
            estimations = []
            for _ in tqdm(
                    range(number_of_evaluations), desc=f"Evaluate the {doe.name}"
            ):
                evaluation = np.array([self._blackbox_model(x) for x in doe.experiment])
                evaluations.append(evaluation)
                estimations.append(
                    self._statistical_model.calculate_maximum_likelihood_estimation(
                        x0=doe.experiment, y=evaluation, minimizer=minimizer
                    )
                )
            self.evaluations_blackbox_function[doe] = np.array(evaluations)
            self.maximum_likelihood_estimations[doe] = np.array(estimations)

    def save_to_csv(self, file_name: str = "benchmarking_evaluations") -> bool:
        """TBA

        Parameters
        ----------
        file_name :

        Returns
        -------

        """
        data = pd.DataFrame(
            [
                [design.name]
                + self.evaluations_blackbox_function[design][index].tolist()
                + self.maximum_likelihood_estimations[design][index].tolist()
                for design in self.experiments
                for index in range(len(self.evaluations_blackbox_function[design]))
            ]
        )
        pd.DataFrame(data=data).to_csv(file_name + ".csv")
        return True

    def load_from_csv(self, file_name: str = "benchmarking_evaluations.csv") -> bool:
        """TBA

        Parameters
        ----------
        file_name :

        Returns
        -------

        """
        t = pd.read_csv(file_name, index_col=0)
        number_parameters = len(self.statistical_model.lower_bounds_theta)
        for design in self.experiments:
            evaluations = []
            estimations = []
            for row in t.T:
                if design.name == t.T[row][0]:
                    evaluations.append(t.T[row][1:].to_list()[:-number_parameters])
                    estimations.append(t.T[row][1:].to_list()[number_parameters:])

            self.evaluations_blackbox_function[design] = np.array(evaluations)
            self.maximum_likelihood_estimations[design] = np.array(estimations)

        return True

    def plot_estimations(self) -> go.Figure:
        """Plot the maximum likelihood estimations

        """
        data = []

        initial_design = list(self.maximum_likelihood_estimations.keys())[0]
        number_of_parameters = len(
            self.maximum_likelihood_estimations[initial_design][0]
        )
        x_dots = [
            design.name
            for design in self.maximum_likelihood_estimations.keys()
            for _ in self.maximum_likelihood_estimations[design].T[0]
        ]

        for index in range(number_of_parameters):
            y_dots = [
                point
                for design in self.maximum_likelihood_estimations.keys()
                for point in self.maximum_likelihood_estimations[design].T[index]
            ]

            data.append(
                dot_scatter(x_dots=np.array([x_dots]), y_dots=np.array([y_dots]), visible=False, fill=None)
            )

        data[0].visible = True
        fig = styled_figure(
            title="MLE estimations for each parameter", data=data
        )
        # Add dropdowns
        button_layer_1_height = 1.12
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list(
                        [
                            dict(
                                label=parameter,
                                method="update",
                                args=[
                                    {
                                        "visible": [
                                            parameter == other_parameter
                                            for other_parameter in range(
                                                number_of_parameters
                                            )
                                        ]
                                    },
                                    {"annotations": []},
                                ],
                            )
                            for parameter in range(number_of_parameters)
                        ]
                    ),
                    type="buttons",
                    direction="right",
                    pad={"r": 10, "t": 10, "b": 15},
                    showactive=True,
                    x=0.25,
                    xanchor="left",
                    y=button_layer_1_height,
                ),
            ]
        )
        fig.update_layout(
            annotations=[
                dict(
                    text="Parameter Index:",
                    showarrow=False,
                    x=0,
                    y=1.085,
                    yref="paper",
                    align="left",
                )
            ]
        )

        return fig
