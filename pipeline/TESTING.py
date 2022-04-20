import numpy as np

from src.benchmarking.benchmarking import Benchmarking
from src.designs_of_experiments.design_library.d_design import DDesign
from src.designs_of_experiments.design_library.latin_hypercube import LatinHypercube
from src.designs_of_experiments.design_library.pi_design import PiDesign
from src.designs_of_experiments.design_library.random import Random

####
# Designs
from src.metrics.metric_library.determinant_of_fisher_information_matrix import (
    DeterminantOfFisherInformationMatrix,
)
from src.metrics.metric_library.estimation_mean_error import EstimationMeanError
from src.metrics.metric_library.estimation_mean_parameter_estimations import (
    EstimationMeanParameterEstimations,
)
from src.metrics.metric_library.estimation_variance_parameter_estimations import (
    EstimationVarianceParameterEstimations,
)
from src.metrics.metric_library.k_fold_cross_validation import KFoldCrossValidation
from src.minimizer.minimizer_library.differential_evolution import DifferentialEvolution
from src.parametric_function_library.aging_model_Naumann import AgingModelNaumann
from src.statistical_models.statistical_model_library.gaussian_noise_model import (
    GaussianNoiseModel,
)

####
# statistical model

theta = np.array([4, 2300, 0.8])

number_designs = 6
number_of_evaluations = 10

# real noise
sigma = 0.029**2

lower_bounds_x = np.array([0.1, 279.15])
upper_bounds_x = np.array([1, 333.15])

lower_bounds_theta = np.array([0.1, 0.001, 0.1])
upper_bounds_theta = np.array([10, 10000, 1])

# print(random_design.design)

parametric_function = AgingModelNaumann()

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
    number_designs=number_designs,
)

# print(LH.design, LH.name)

random_design = Random(
    number_designs=number_designs,
    lower_bounds_design=lower_bounds_x,
    upper_bounds_design=upper_bounds_x,
)

min_entry = PiDesign(
    number_designs=number_designs,
    lower_bounds_design=lower_bounds_x,
    upper_bounds_design=upper_bounds_x,
    column=0,
    row=0,
    initial_theta=theta,
    statistical_model=statistical_model,
    minimizer=minimizer,
)

max_det = DDesign(
    number_designs=number_designs,
    lower_bounds_design=lower_bounds_x,
    upper_bounds_design=upper_bounds_x,
    initial_theta=theta,
    statistical_model=statistical_model,
    minimizer=minimizer,
)

print(
    "LH: 1 entry CRLB",
    statistical_model.calculate_cramer_rao_lower_bound(x0=LH.design, theta=theta)[1, 1],
)
print(
    "Min entry",
    statistical_model.calculate_cramer_rao_lower_bound(
        x0=min_entry.design, theta=theta
    )[1, 1],
)
print(
    statistical_model.calculate_determinant_fisher_information_matrix(
        x0=LH.design, theta=theta
    )
)

####
# MLE
# print(LH.design[0], )
# for _ in range(10):
#    eval = np.array([_blackbox_model(x) for x in LH.design])

# print('The MLE is: ',
#      _statistical_model.calculate_maximum_likelihood_estimation(minimizer=minimizer, x0=LH.design, y=eval))

####
# metrics

metrics = [
    DeterminantOfFisherInformationMatrix(
        theta=theta, statistical_model=statistical_model
    ),
    EstimationMeanParameterEstimations(),
    EstimationVarianceParameterEstimations(),
    EstimationMeanError(
        number_evaluations=100, theta=theta, statistical_model=statistical_model
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
    designs_of_experiments=[LH, random_design, min_entry, max_det],
)

benchmarking2 = Benchmarking(
    blackbox_model=blackbox_model,
    statistical_model=statistical_model,
    designs_of_experiments=[LH, random_design, min_entry, max_det],
)

benchmarking.evaluate_designs(
    number_of_evaluations=number_of_evaluations, minimizer=minimizer
)

k_fold_data = {}
for design in benchmarking.evaluations_blackbox_function.keys():
    k_fold_data[design] = benchmarking.evaluations_blackbox_function[design][0]
#####

fig2 = metrics[2].plot(
    evaluations_blackbox_function_for_each_design=benchmarking.evaluations_blackbox_function,
    estimations_of_parameter_for_each_design=benchmarking.maximum_likelihood_estimations,
)
fig2.show()

fig = metrics[0].plot(
    evaluations_blackbox_function_for_each_design=benchmarking.evaluations_blackbox_function,
    estimations_of_parameter_for_each_design=benchmarking.maximum_likelihood_estimations,
    baseline="max",
)
fig.show()

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
