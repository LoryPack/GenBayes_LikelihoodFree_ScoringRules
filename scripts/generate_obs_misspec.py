import os
import sys
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
from abcpy.statistics import Identity

sys.path.append(os.getcwd())  # add the root of this project to python path
from src.models import instantiate_model
from src.parsers import parser_generate_obs_fixed_par
from src.utils import define_default_folders, define_exact_param_values

parser = parser_generate_obs_fixed_par()
parser.add_argument('--epsilon', type=float, default=0, help='Fraction of outliers')
parser.add_argument('--outliers_location', type=float, default=1,
                    help='Location around where the outliers are located (for the normal location model)')
args = parser.parse_args()

"""This is used for generating data for the misspecified normal location model, with a given fraction of outliers 
epsilon at a given location"""

model = args.model
n_observations_per_param = args.n_observations_per_param
sleep_time = args.sleep
results_folder = args.root_folder
epsilon = args.epsilon
outliers_location = args.outliers_location

default_root_folder = define_default_folders()

exact_param_values_dict = define_exact_param_values()
exact_param_values = exact_param_values_dict[model]

if model not in ["normal_location_misspec", "Lorenz96", "RecruitmentBoomBust"]:
    raise NotImplementedError

if results_folder is None:
    results_folder = default_root_folder[model]

print("Model", model)
observation_folder = results_folder + '/' + args.observation_folder + '/'

if sleep_time > 0:
    print("Wait for {} minutes...".format(sleep_time))
    sleep(60 * sleep_time)
    print("Done waiting!")

seed = 1
save_observation = True

rng = np.random.RandomState(seed)

# instantiate the model here; do not use reparametrized version as we provide param values for original parametrization
model_abcpy, statistics, param_bounds = instantiate_model(model_name=model, reparametrized=False)
np.save(observation_folder + "theta_obs",
        exact_param_values[:4] if model == "univariate_g-and-k" else exact_param_values)
if model == "normal_location_misspec":
    exact_param_values = exact_param_values + [1]
    outlier_parameters = [outliers_location, 1]
elif "g-and-k" in model:
    pass
elif "Lorenz" in model:
    outlier_parameters = [1.41, 0.1, 2.4, 0.95]
else:
    outlier_parameters = [0.9, 70.0, 0.80, 0.05]

simulations = model_abcpy.forward_simulate(exact_param_values,
                                           int(np.rint(n_observations_per_param * (1 - epsilon))), rng=rng)
simulations = Identity().statistics(simulations)
n_outliers = int(np.rint(n_observations_per_param * epsilon))
if n_outliers > 0:
    if model == "normal_location_misspec" or model == "RecruitmentBoomBust":
        outliers = model_abcpy.forward_simulate(outlier_parameters, n_outliers, rng=rng)
        outliers = Identity().statistics(outliers)
    else:
        outliers = model_abcpy.forward_simulate(outlier_parameters, n_outliers, rng=rng)
        outliers = Identity().statistics(outliers)
        fig, axes = plt.subplots(2, 4, figsize=(12, 8), sharex="all")
        axes = axes.flatten()
        for i in range(8):
            axes[i].plot(simulations.reshape(n_observations_per_param - n_outliers, 8, -1)[:, i].transpose(1, 0),
                         alpha=0.5, color="blue")
            axes[i].plot(outliers.reshape(n_outliers, 8, -1)[:, i].transpose(1, 0), alpha=0.5, color="r")
            axes[i].set_title(r"$y_{}$".format(i + 1), size=14)
            axes[i].set_xlabel("Timestep")
        if 0 < epsilon < 0.1:
            plt.savefig(observation_folder + "x_obs_epsilon_{:.2f}.pdf".format(epsilon))
        else:
            plt.savefig(observation_folder + "x_obs_epsilon_{:.1f}.pdf".format(epsilon))
    simulations = np.concatenate((simulations, outliers))
else:
    print("No outliers have been generated")
print("Parameters shape", len(exact_param_values))
print("Simulations shape", simulations.shape)

if model == "normal_location_misspec":
    if 0 < epsilon < 0.1:
        np.save(observation_folder + "x_obs_epsilon_{:.2f}_outliers_loc_{:.1f}".format(epsilon, outliers_location),
                simulations)
    else:
        np.save(observation_folder + "x_obs_epsilon_{:.1f}_outliers_loc_{:.1f}".format(epsilon, outliers_location),
                simulations)
else:
    if 0 < epsilon < 0.1:
        np.save(observation_folder + "x_obs_epsilon_{:.2f}".format(epsilon), simulations)
    else:
        np.save(observation_folder + "x_obs_epsilon_{:.1f}".format(epsilon), simulations)
