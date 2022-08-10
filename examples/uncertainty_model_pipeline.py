"""Aging model pipeline script

This script allows the user to execute an example workflow for designing and evaluating an experiment,
specifically the pi experiment.
Being more precisely, we
* use as statistical model the white Gaussian noise model with
* parametric function the aging_model provided in the parametric function library
* benchmark various experiments against each other and
* plot several metrics
"""

####
# Importing modules

import numpy as np

import plotly.graph_objects as go

from src.experiments.experiment_library.latin_hypercube import LatinHypercube

####
# Designs
from src.experiments.experiment_library.point_prediction_design import (
    PointPredictionDesign,
)

from src.minimizer.minimizer_library.differential_evolution import DifferentialEvolution
from src.parametric_function_library.interfaces.parametric_function import (
    ParametricFunction,
)
from src.statistical_models.statistical_model_library.gaussian_noise_model import (
    GaussianNoiseModel,
)

#################################
#################################
# SETUP

####
# statistical model
from src.uncertainty_quantification.parametric_function_with_uncertainty import (
    ParametricFunctionWithUncertainty,
)
from src.uncertainty_quantification.probability_measures.multivariate_gaussian import (
    MultivariateGaussian,
)
from src.visualization.plotting_functions import (
    line_scatter,
    styled_figure,
    uncertainty_area_scatter,
    dot_scatter,
)

theta = np.array([1, 2, 0.1])

number_designs = 10

# real noise
sigma = 0.3

#################################
#################################
# Pipeline

####
# bounds
lower_bounds_x = np.array([-10])
upper_bounds_x = np.array([10])

lower_bounds_theta = np.array([0, 0, -10])
upper_bounds_theta = np.array([np.pi, np.pi, 10])


# Setup a parametric function family
class TestFunction(ParametricFunction):
    def __call__(self, theta: np.ndarray, x: float) -> np.ndarray:
        return np.array([theta[0] * np.sin(x + theta[1]) + theta[2]])

    def partial_derivative(
        self, theta: np.ndarray, x: np.ndarray, parameter_index: int
    ) -> np.ndarray:
        if parameter_index == 0:
            return np.array([np.sin(x + theta[1])])
        if parameter_index == 1:
            return np.array([theta[0] * np.sin(x + theta[1])])
        if parameter_index == 2:
            return np.array([1])

    def second_partial_derivative(
        self,
        theta: np.ndarray,
        x: np.ndarray,
        parameter1_index: int,
        parameter2_index: int,
    ) -> float:
        pass


parametric_function = TestFunction()

####
# minimizer
minimizer = DifferentialEvolution(maxiter=10000)

statistical_model = GaussianNoiseModel(
    function=parametric_function,
    lower_bounds_x=lower_bounds_x,
    upper_bounds_x=upper_bounds_x,
    lower_bounds_theta=lower_bounds_theta,
    upper_bounds_theta=upper_bounds_theta,
    sigma=sigma,
)


####
# blackbox function
def blackbox_model(x):
    return statistical_model.random(theta=theta, x=x)


LH = LatinHypercube(
    lower_bounds_design=lower_bounds_x,
    upper_bounds_design=upper_bounds_x,
    number_designs=number_designs,
)

#######

experiment = LH

evaluations = np.array([blackbox_model(x) for x in experiment.experiment])

estimated_theta = statistical_model.calculate_maximum_likelihood_estimation(
    x0=experiment.experiment, y=evaluations, minimizer=minimizer
)

print(estimated_theta)

covariance_matrix = statistical_model.calculate_cramer_rao_lower_bound(
    x0=experiment.experiment, theta=estimated_theta
)

det_of_FI = statistical_model.calculate_determinant_fisher_information_matrix(
    x0=experiment.experiment, theta=estimated_theta
)

probability_measure_on_parameter_space = MultivariateGaussian(
    mean=estimated_theta, covariance_matrix=covariance_matrix
)

parametric_function_with_uncertainty = ParametricFunctionWithUncertainty(
    parametric_function=parametric_function,
    probability_measure_on_parameter_space=probability_measure_on_parameter_space,
    sample_size_parameters=1000,
)

print(covariance_matrix, det_of_FI)

#######

parametric_function_with_uncertainty.histo(x=np.array(3))

# Plotting the results
x_lines = np.arange(-10, 10, 0.1)

y_lines = np.array(
    [parametric_function(theta=estimated_theta, x=x) for x in x_lines]
).flatten()

y_lines_true = np.array(
    [parametric_function(theta=theta, x=x) for x in x_lines]
).flatten()

alpha = 0.9
y_confidence_upper = np.array(
    [
        parametric_function_with_uncertainty.calculate_quantile(x=x, alpha=alpha)
        for x in x_lines
    ]
).flatten()
y_confidence_lower = np.array(
    [
        parametric_function_with_uncertainty.calculate_quantile(x=x, alpha=1 - alpha)
        for x in x_lines
    ]
).flatten()

data = [
    go.Scatter(
        x=experiment.experiment.flatten(), y=evaluations.flatten(), mode="markers"
    ),
    line_scatter(x_lines=x_lines, y_lines=y_lines_true),
    line_scatter(x_lines=x_lines, y_lines=y_lines),
    uncertainty_area_scatter(
        x_lines=x_lines, y_upper=y_confidence_upper, y_lower=y_confidence_lower
    ),
]
fig = styled_figure(data=data, title="", title_y="", title_x="")

fig.show()

new_design = PointPredictionDesign(
    lower_bounds_design=lower_bounds_x,
    upper_bounds_design=upper_bounds_x,
    minimizer=minimizer,
    parametric_function_with_uncertainty=parametric_function_with_uncertainty,
    alpha=0.95,
)

print(new_design.experiment)
