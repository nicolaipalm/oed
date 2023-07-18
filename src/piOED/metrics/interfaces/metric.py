from abc import ABC, abstractmethod
from typing import Dict, Union, Optional

import numpy as np
import plotly.graph_objects as go

from ...experiments.interfaces.design_of_experiment import (
    Experiment,
)
from ...visualization.plotting_functions import (
    dot_scatter,
    line_scatter,
    styled_figure,
)


class Metric(ABC):
    """Interface for metrics applied to the benchmarking results

    Implementing the calculate method, you can plot the results without further code.
    """
    @abstractmethod
    def calculate(
        self,
        evaluations_blackbox_function: Optional[np.ndarray],
        estimations_of_parameter: Optional[np.ndarray],
        experiment: Experiment,
    ) -> np.ndarray:
        """Evaluate the metric

        Either one of evaluations or estimations must be a numpy array.

        Parameters
        ----------
        evaluations_blackbox_function : np.ndarray ,optional
            Evaluations of the blackbox function. Each entry represents a single evaluation.

        estimations_of_parameter : np.ndarray ,optional
            Parameter estimations corresponding to the blackbox evaluations.
            Each entry represents a single parameter estimation.

        experiment : Experiment
            The underlying experiment on which the metric is applied.

        Returns
        -------
        np.ndarray
            Evaluation of the metric.

        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the metric
        Returns
        -------
        str
            the name of the metric
        """
        pass

    def plot(
        self,
        evaluations_blackbox_function_for_each_experiment: Optional[Dict[
            Experiment, np.ndarray
        ]] = None,
        estimations_of_parameter_for_each_experiment: Optional[Dict[
            Experiment, np.ndarray
        ]] = None,
        baseline: Union[str, np.ndarray] = "min",
    ) -> go.Figure:
        """Plot the results of the metric applied to multiple experiment and its evaluations

        Parameters
        ----------
        evaluations_blackbox_function_for_each_experiment : Dict[Experiment, np.ndarray], optional
             Evaluations of the blackbox function for each experiment . Each entry represents a single evaluation.
        estimations_of_parameter_for_each_experiment : Dict[Experiment, np.ndarray], optional
             Parameter estimations for each experiment corresponding to the blackbox evaluations.
             Each entry represents a single parameter estimation.
        baseline : Union[str, np.ndarray]
             Line which should be plotted in addition to the evaluations of the metric.
             The dimension needs to be Line[parameter dimension][number of experiment]
             Current options are:
             - "min" (default; i.e. the minimum of all evaluated metrics)
             - "max" (i.e. the maximum of all evaluated metrics)
             - self defined line in form of numpy array with consisting of entries corresponding to the experiment
             - None which results in a zero baseline


        Returns
        -------

        """

        # Add the data points for each parameter
        data = []
        design = list(evaluations_blackbox_function_for_each_experiment.keys())[0]

        # TODO: should be independent of input given?
        if evaluations_blackbox_function_for_each_experiment is None:
            evaluations_blackbox_function_for_each_experiment = (
                estimations_of_parameter_for_each_experiment
            )

        if estimations_of_parameter_for_each_experiment is None:
            estimations_of_parameter_for_each_experiment = (
                evaluations_blackbox_function_for_each_experiment
            )

        number_of_parameters = len(
            self.calculate(evaluations_blackbox_function=evaluations_blackbox_function_for_each_experiment[
                design
            ], estimations_of_parameter=estimations_of_parameter_for_each_experiment[
                design
            ], experiment=design)
        )
        x_dots = [
            design.name
            for design in evaluations_blackbox_function_for_each_experiment.keys()
        ]

        for index in range(number_of_parameters):
            y_dots = [
                self.calculate(evaluations_blackbox_function=evaluations_blackbox_function_for_each_experiment[
                    design
                ], estimations_of_parameter=estimations_of_parameter_for_each_experiment[
                    design
                ], experiment=design)[index]
                for design in evaluations_blackbox_function_for_each_experiment.keys()
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
