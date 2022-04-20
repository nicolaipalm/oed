from typing import Callable, List

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from tqdm import tqdm

from src.designs_of_experiments.interfaces.design_of_experiment import (
    DesignOfExperiment,
)
from src.minimizer.interfaces.minimizer import Minimizer
from src.statistical_models.interfaces.statistical_model import StatisticalModel
from src.visualization.plotting_functions import dot_scatter, styled_figure


class Benchmarking:
    """
    This class provides an easy way to evaluate different designs
    (i.e. calculate the maximum likelihood parameter estimation) and to store and load the data.
    """

    def __init__(
        self,
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
            # TODO: prevent overflow
            for _ in tqdm(
                range(number_of_evaluations), desc=f"Evaluate the {doe.name}"
            ):
                evaluation = np.array([self._blackbox_model(x) for x in doe.design])
                evaluations.append(evaluation)
                estimations.append(
                    self._statistical_model.calculate_maximum_likelihood_estimation(
                        x0=doe.design, y=evaluation, minimizer=minimizer
                    )
                )
            self.evaluations_blackbox_function[doe] = np.array(evaluations)
            self.maximum_likelihood_estimations[doe] = np.array(estimations)

    def save_to_csv(self, file_name: str = "benchmarking") -> bool:
        data = pd.DataFrame(
            [
                [design.name]
                + self.evaluations_blackbox_function[design][index].tolist()
                + self.maximum_likelihood_estimations[design][index].tolist()
                for design in self.designs
                for index in range(len(self.evaluations_blackbox_function[design]))
            ]
        )
        pd.DataFrame(data=data).to_csv(file_name + ".csv")
        return True

    def load_from_csv(self, file_name: str = "benchmarking.csv") -> bool:
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

    def plot_estimations(self) -> go.Figure:

        # Add the data points for each parameter
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
                dot_scatter(x_dots=x_dots, y_dots=y_dots, visible=False, fill=None)
            )

        data[0].visible = True
        fig = styled_figure(
            title="Evaluations of blackbox function for each parameter", data=data
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
