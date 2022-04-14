import numpy as np

from benchmarking.benchmarking import Benchmarking
from designs_of_experiments.design_library.d_design import DDesign
from designs_of_experiments.design_library.latin_hypercube import LatinHypercube
from designs_of_experiments.design_library.minimum_entry_of_CRLB_design import MinimumEntryOfCRLBDesign
from designs_of_experiments.design_library.pi_design import PiDesign
from designs_of_experiments.design_library.random import Random
####
# Designs
from metrics.metric_library.estimation_mean_parameter_estimations import EstimationMeanParameterEstimations
from metrics.metric_library.estimation_variance_parameter_estimations import EstimationVarianceParameterEstimations
from minimizer.minimizer_library.differential_evolution import DifferentialEvolution
from parametric_function_library.aging_model_Naumann import AgingModelNaumann
from statistical_models.statistical_model_library.gaussian_noise_model import GaussianNoiseModel

####
# statistical model

theta = np.array([4, 2300, 0.8])

number_designs = 12

# real noise
sigma = 0.029 ** 2

lower_bounds_x = np.array([0.001, 279.15])
upper_bounds_x = np.array([1, 333.15])

lower_bounds_theta = np.array([0.001, 0.001, 0.001])
upper_bounds_theta = np.array([10, 10000, 1])

LH = LatinHypercube(lower_bounds_design=lower_bounds_x, upper_bounds_design=upper_bounds_x,
                    number_designs=number_designs)

print(LH.design, LH.name)

random_design = Random(number_designs=number_designs, lower_bounds_design=lower_bounds_x,
                       upper_bounds_design=upper_bounds_x)
print(random_design.design)

parametric_function = AgingModelNaumann()

####
# minimizer
minimizer = DifferentialEvolution()

statistical_model = GaussianNoiseModel(function=parametric_function, lower_bounds_x=lower_bounds_x,
                                       upper_bounds_x=upper_bounds_x, lower_bounds_theta=lower_bounds_theta,
                                       upper_bounds_theta=upper_bounds_theta, sigma=sigma)

min_entry = PiDesign(number_designs=number_designs, lower_bounds_design=lower_bounds_x,
                                     upper_bounds_design=upper_bounds_x, column=1, row=1, initial_theta=theta,
                                     statistical_model=statistical_model, minimizer=minimizer)

max_det = DDesign(number_designs=number_designs, lower_bounds_design=lower_bounds_x,
                                     upper_bounds_design=upper_bounds_x, initial_theta=theta,
                                     statistical_model=statistical_model, minimizer=minimizer)

print('LH:',statistical_model.calculate_cramer_rao_lower_bound(x0=LH.design, theta=theta)[1, 1])
print('Min entry',statistical_model.calculate_cramer_rao_lower_bound(x0=min_entry.design, theta=theta)[1, 1])
print(
    statistical_model.calculate_determinant_fisher_information_matrix(x0=LH.design, theta=theta))


####
# blackbox function
def blackbox_model(x):
    return statistical_model.random(theta=theta, x=x)


####
# MLE
print(LH.design[0], )
for _ in range(10):
    eval = np.array([blackbox_model(x) for x in LH.design])

   # print('The MLE is: ',
    #      statistical_model.calculate_maximum_likelihood_estimation(minimizer=minimizer, x0=LH.design, y=eval))

####
# metrics


metrics = [EstimationMeanParameterEstimations(), EstimationVarianceParameterEstimations()]

####
# benchmarking

benchmarking = Benchmarking(blackbox_model=blackbox_model, statistical_model=statistical_model,
                            designs_of_experiments=[LH, random_design,min_entry,max_det], metrics=metrics)

benchmarking.evaluate_designs(number_of_evaluations=100, minimizer=minimizer)

benchmarking.calculate_metrics()

print(benchmarking.evaluations_of_metrics)
