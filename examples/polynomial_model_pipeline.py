####
# Importing modules

import numpy as np

from src.benchmarking.benchmarking import Benchmarking
from src.designs_of_experiments.experiment_library.d_design import DDesign
from src.designs_of_experiments.experiment_library.latin_hypercube import LatinHypercube
from src.designs_of_experiments.experiment_library.pi_design import PiDesign
from src.designs_of_experiments.experiment_library.random import Random
####
# Designs
from src.metrics.metric_library.determinant_of_fisher_information_matrix import (
    DeterminantOfFisherInformationMatrix,
)
from src.metrics.metric_library.estimation_mean_error import EstimationMeanError
from src.metrics.metric_library.estimation_mean_parameter_estimations import (
    EstimationMeanParameterEstimations,
)
from src.metrics.metric_library.k_fold_cross_validation import KFoldCrossValidation
from src.metrics.metric_library.std_parameter_estimations import StdParameterEstimations
from src.minimizer.minimizer_library.differential_evolution import DifferentialEvolution
from src.parametric_function_library.interfaces.parametric_function import ParametricFunction
from src.statistical_models.statistical_model_library.gaussian_noise_model import (
    GaussianNoiseModel,
)


#################################
#################################
# SETUP

####
# statistical model

# f(x) = theta_0+theta_1*x+theta_2*x**2+...
class PolynomialFunction(ParametricFunction):
    def __call__(
            self,
            theta: np.ndarray,
            x: float) -> float:
        return np.sum(theta[1:] * (x ** np.arange(1, len(theta)))) + theta[0]

    def partial_derivative(
            self,
            theta: np.ndarray, x: np.ndarray,
            parameter_index: int) -> float:
        if parameter_index == 0:
            return 1
        else:
            return x ** parameter_index

    def second_partial_derivative(
            self,
            theta: np.ndarray,
            x: np.ndarray,
            parameter1_index: int,
            parameter2_index: int,
    ) -> float:
        return 0


# highest grade of polynomial
grade = 2

theta = np.array([0.01 for _ in range(grade + 1)])

number_designs = 5
number_of_evaluations_in_benchmarking = 100

# real noise
sigma = 1

#################################
#################################
# Pipeline

####
# bounds
lower_bounds_x = np.array([-10])
upper_bounds_x = np.array([10])

lower_bounds_theta = -2*np.ones(grade + 1)
upper_bounds_theta = 2 * np.ones(grade + 1)

# Setup a parametric function family
parametric_function = PolynomialFunction()

####
# minimizer
minimizer = DifferentialEvolution(maxiter=100)

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

# print(LH.designs, LH.name)

random_design = Random(
    number_designs=2 * number_designs,
    lower_bounds_design=lower_bounds_x,
    upper_bounds_design=upper_bounds_x,
)

# We split the number of experiments in half and perform first a latin hypercube
LH_half = LatinHypercube(lower_bounds_design=lower_bounds_x,
                         upper_bounds_design=upper_bounds_x,
                         number_designs=number_designs)

initial_theta = statistical_model.calculate_maximum_likelihood_estimation(
    x0=LH_half.designs, y=np.array([blackbox_model(x) for x in LH_half.designs]), minimizer=minimizer)

print(initial_theta)

min_entry = PiDesign(
    number_designs=number_designs,
    lower_bounds_design=lower_bounds_x,
    upper_bounds_design=upper_bounds_x,
    index=0,
    initial_theta=initial_theta,
    previous_design=LH_half,
    statistical_model=statistical_model,
    minimizer=minimizer,
)

print(min_entry.designs)

max_det = DDesign(
    number_designs=number_designs,
    lower_bounds_design=lower_bounds_x,
    upper_bounds_design=upper_bounds_x,
    initial_theta=initial_theta,
    statistical_model=statistical_model,
    minimizer=minimizer,
    previous_design=LH_half,
)

metrics = [
    DeterminantOfFisherInformationMatrix(
        theta=initial_theta,
        statistical_model=statistical_model
    ),
    EstimationMeanParameterEstimations(),
    StdParameterEstimations(),
    EstimationMeanError(
        number_evaluations=1000,
        theta=theta,
        statistical_model=statistical_model
    ),
    KFoldCrossValidation(
        statistical_model=statistical_model, minimizer=minimizer, number_splits=2
    ),
]

####
# benchmarking


benchmarking = Benchmarking(
    blackbox_model=blackbox_model,
    statistical_model=statistical_model,
    designs_of_experiments=[LH,
                            random_design,
                            min_entry,
                            max_det
                            ],
)

benchmarking.evaluate_designs(
    number_of_evaluations=number_of_evaluations_in_benchmarking, minimizer=minimizer
)

k_fold_data = {}
for design in benchmarking.evaluations_blackbox_function.keys():
    k_fold_data[design] = benchmarking.evaluations_blackbox_function[design][0]

#####
# saving the benchmarking results


#####
# plotting

baseline = np.sqrt(np.array([statistical_model.calculate_cramer_rao_lower_bound(
    x0=design.designs, theta=theta).diagonal() for design in benchmarking.designs]).T)

fig2 = metrics[2].plot(
    evaluations_blackbox_function_for_each_design=benchmarking.evaluations_blackbox_function,
    estimations_of_parameter_for_each_design=benchmarking.maximum_likelihood_estimations,
    baseline=baseline,
)
fig2.show()

fig0 = metrics[0].plot(
    evaluations_blackbox_function_for_each_design=benchmarking.evaluations_blackbox_function,
    estimations_of_parameter_for_each_design=benchmarking.maximum_likelihood_estimations,
    baseline="max",
)
fig0.show()

fig1 = metrics[1].plot(
    evaluations_blackbox_function_for_each_design=benchmarking.evaluations_blackbox_function,
    estimations_of_parameter_for_each_design=benchmarking.maximum_likelihood_estimations,
    baseline=theta,
)
fig1.show()

fig3 = metrics[3].plot(
    evaluations_blackbox_function_for_each_design=benchmarking.evaluations_blackbox_function,
    estimations_of_parameter_for_each_design=benchmarking.maximum_likelihood_estimations,
    baseline="min",
)
fig3.show()

fig4 = metrics[4].plot(
    evaluations_blackbox_function_for_each_design=k_fold_data,
    baseline="min",
)
fig4.show()

fig = benchmarking.plot_estimations()
fig.show()
