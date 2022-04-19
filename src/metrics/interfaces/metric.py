from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np
import plotly.graph_objects as go

from src.designs_of_experiments.interfaces.design_of_experiment import (
    DesignOfExperiment,
)
from src.visualization.plotting_functions import (
    dot_scatter,
    styled_figure,
    line_scatter,
)


class Metric(ABC):
    @abstractmethod
    def calculate(
        self,
        evaluations_blackbox_function: np.ndarray,
        estimations_of_parameter: np.ndarray,
        design: DesignOfExperiment,
    ) -> np.ndarray:
        """
        Either one, evaluations of blackbox OR estimations of parameter may be no
        :return Numpy array of dimension 1.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def plot(
        self,
        evaluations_blackbox_function_for_each_design: Dict[
            DesignOfExperiment, np.ndarray
        ] = None,
        estimations_of_parameter_for_each_design: Dict[
            DesignOfExperiment, np.ndarray
        ] = None,
        baseline: Union[str, np.ndarray] = "min",
    ) -> go.Figure:
        """
        :param evaluations_blackbox_function_for_each_design:
        :type evaluations_blackbox_function_for_each_design:
        :param estimations_of_parameter_for_each_design:
        :type estimations_of_parameter_for_each_design:
        :param baseline:
        :type baseline:
        min,max,zero or a default parameter stored in a numpy array of length equal the length of parameters;
        by default min
        :return:
        :rtype:
        """

        # Add the data points for each parameter
        data = []
        design = list(evaluations_blackbox_function_for_each_design.keys())[0]

        # ToDo: should be independent of input given?
        if evaluations_blackbox_function_for_each_design is None:
            evaluations_blackbox_function_for_each_design = (
                estimations_of_parameter_for_each_design
            )

        if estimations_of_parameter_for_each_design is None:
            estimations_of_parameter_for_each_design = (
                evaluations_blackbox_function_for_each_design
            )

        number_of_parameters = len(
            self.calculate(
                evaluations_blackbox_function=evaluations_blackbox_function_for_each_design[
                    design
                ],
                estimations_of_parameter=estimations_of_parameter_for_each_design[
                    design
                ],
                design=design,
            )
        )
        x_dots = [
            design.name
            for design in evaluations_blackbox_function_for_each_design.keys()
        ]

        for index in range(number_of_parameters):
            y_dots = [
                self.calculate(
                    evaluations_blackbox_function=evaluations_blackbox_function_for_each_design[
                        design
                    ],
                    estimations_of_parameter=estimations_of_parameter_for_each_design[
                        design
                    ],
                    design=design,
                )[index]
                for design in evaluations_blackbox_function_for_each_design.keys()
            ]
            if type(baseline) == np.ndarray:
                if len(baseline.shape) == 1:
                    optimal_parameter = baseline[index] * np.ones(len(x_dots))
                elif len(baseline.shape) == 2:
                    optimal_parameter = baseline[index]
                else:
                    raise ValueError(
                        "The baseline needs to be an array of dimension 1 or 2"
                    )

            elif baseline == "min":
                optimal_parameter = min(y_dots) * np.ones(len(x_dots))

            elif baseline == "max":
                optimal_parameter = max(y_dots) * np.ones(len(x_dots))

            else:
                optimal_parameter = 0 * np.ones(len(x_dots))

            data.append(
                line_scatter(
                    x_lines=x_dots,
                    y_lines=optimal_parameter,
                    visible=False,
                    name_line="baseline",
                )
            )
            data.append(
                dot_scatter(
                    x_dots=x_dots,
                    y_dots=y_dots,
                    visible=False,
                    text=[round(value, 2) for value in y_dots],
                )
            )

            # Add the real parameters as lines

        data[0].visible = True
        data[1].visible = True
        fig = styled_figure(title=self.name, data=data)
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
                                            (
                                                2 * parameter == other_parameter
                                                or 2 * parameter + 1 == other_parameter
                                            )
                                            for other_parameter in range(
                                                2 * number_of_parameters
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
                dict(
                    pad={"r": 10, "t": 10, "b": 15},
                    y=button_layer_1_height,
                    direction="down",
                    yanchor="auto",
                    buttons=list(
                        [
                            dict(
                                args=["type", "scatter"],
                                label="Scatter Plot",
                                method="restyle",
                            ),
                            dict(
                                args=["type", "bar"],
                                label="Bar Chart",
                                method="restyle",
                            ),
                        ]
                    ),
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
