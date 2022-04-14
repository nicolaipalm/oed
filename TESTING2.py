from visualization.create_dashboard import create_dashboard
from designs_of_experiments.does_benchmark import *
import numpy as np

from functions.aging_model_Naumann import AgingModelNaumann

################
# Defining the parameters of the benchmarking
# the underlying deterministic function


function = AgingModelNaumann()

# the designs we want to compare
designs = ['full_practical',
           'random',
           'LH',
           'min entry',
           'max det']

# theta (real parameter)
theta = np.array([4, 2300, 0.8])

# real noise
sigma = 0.029 ** 2

# Initial guess for theta
theta_initial = np.array([0.8, 1.3, 0.4]) * theta

# bounds of input
# bound_x[i]=(lower bound of i-th parameter, upper bound of i-th parameter)
bound_x = [(0.001, 1), (279.15, 333.15)]

# bounds for latin Hypercube
# lower bounds
# (length equals length input designs;
# l_boundy[i] resp. u_bounds[i] is the lower resp. upper bound of the i-th parameter)
l_bounds = [0.001, 279.15]

# upper bounds
u_bounds = [1, 333.15]

# bounds for theta
bound_theta = ((0.001, 10), (0.001, 10000), (0.001, 1))

# number of overall evaluations/experiments
number_evaluations = 30

# number of single evaluations in the repeated evaluations design
number_single_evaluations = 6

# entry of the diagonal in the repeated single evaluation design
entry_diagonal = 0

# Number of experiments in estimator estimation
M = 100




################
# Execute benchmarking
statistical_model = GaussianNoiseModel(function=function, sigma=sigma)

does = DOEs(theta=theta,
            theta_guess=theta_initial,
            bound_theta=bound_theta,
            bound_single_object=bound_x,
            l_bounds=l_bounds,
            u_bounds=u_bounds,
            statistical_model=statistical_model,
            number_evaluations=number_evaluations,
            number_single_evaluations=number_single_evaluations,
            entry_diagonal=entry_diagonal)

does.calculate_designs(designs=designs)

does_benchmark = DOE_benchmark(does,
                               M=M,
                               theta_initial=theta_initial,
                               bound_design=bound_theta,
                               statistical_model=statistical_model)

################
# create a dashboard to visualize results

create_dashboard(does, does_benchmark)



