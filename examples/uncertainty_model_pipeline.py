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

from src.experiments.experiment_library.d_design import DDesign
from src.experiments.experiment_library.latin_hypercube import LatinHypercube
from src.experiments.experiment_library.pi_design import PiDesign
from src.experiments.experiment_library.random import Random

####
# Designs

from src.minimizer.minimizer_library.differential_evolution import DifferentialEvolution
from src.parametric_function_library.aging_model import AgingModel
from src.statistical_models.statistical_model_library.gaussian_noise_model import (
    GaussianNoiseModel,
)

#################################
#################################
# SETUP

####
# statistical model
from src.uncertainty_quantification.parametric_function_with_std import (
    ParametricFunctionWithStd,
)
from src.uncertainty_quantification.probability_measures.multivariate_gaussian import (
    MultivariateGaussian,
)

theta = np.array([1.8, 402, 0.13])

number_designs = 5
number_of_evaluations = 100

# real noise
sigma = 0.002

#################################
#################################
# Pipeline

####
# bounds
lower_bounds_x = np.array([0.05, 279.15])
upper_bounds_x = np.array([1, 333.15])

lower_bounds_theta = np.array([0.01, 0, 0])
upper_bounds_theta = np.array([10, 10000, 1])

# Setup a parametric function family
parametric_function = AgingModel()

####
# minimizer
minimizer = DifferentialEvolution()

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
    number_designs=2 * number_designs,
)

#######

experiment = LH

evaluations = np.array([blackbox_model(x) for x in experiment.experiment])

theta = statistical_model.calculate_maximum_likelihood_estimation(
    x0=experiment.experiment, y=evaluations, minimizer=minimizer
)

probability_measure_on_parameter_space = MultivariateGaussian(
    mean=theta,
    covariance_matrix=statistical_model.calculate_cramer_rao_lower_bound(
        x0=experiment.experiment, theta=theta
    ),
)

parametric_function_with_std = ParametricFunctionWithStd(
    parametric_function=parametric_function,
    probability_measure_on_parameter_space=probability_measure_on_parameter_space,
)

x = experiment.experiment[0]
print(
    parametric_function_with_std(x=x),
    parametric_function_with_std.calculate_uncertainty(x=x),
)

input()
#######


# print(LH.experiment, LH.name)

random_design = Random(
    number_designs=2 * number_designs,
    lower_bounds_design=lower_bounds_x,
    upper_bounds_design=upper_bounds_x,
)

# We split the number of experiment in half and perform first a latin hypercube
LH_half = LatinHypercube(
    lower_bounds_design=lower_bounds_x,
    upper_bounds_design=upper_bounds_x,
    number_designs=number_designs,
)

initial_theta = statistical_model.calculate_maximum_likelihood_estimation(
    x0=LH_half.experiment,
    y=np.array([blackbox_model(x) for x in LH_half.experiment]),
    minimizer=minimizer,
)

min_entry = PiDesign(
    number_designs=number_designs,
    lower_bounds_design=lower_bounds_x,
    upper_bounds_design=upper_bounds_x,
    index=1,
    initial_theta=initial_theta,
    previous_experiment=LH_half,
    statistical_model=statistical_model,
    minimizer=minimizer,
)

max_det = DDesign(
    number_designs=number_designs,
    lower_bounds_design=lower_bounds_x,
    upper_bounds_design=upper_bounds_x,
    initial_theta=initial_theta,
    statistical_model=statistical_model,
    minimizer=minimizer,
    previous_experiment=LH_half,
)
