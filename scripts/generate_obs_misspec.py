import numpy as np
from time import sleep

import os, sys
from abcpy.statistics import Identity

sys.path.append(os.getcwd())  # add the root of this project to python path
from src.models import instantiate_model
from src.parsers import parser_generate_obs_fixed_par
from src.utils import define_default_folders_scoring_rules, define_exact_param_values

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

default_root_folder = define_default_folders_scoring_rules()

exact_param_values_dict = define_exact_param_values()
exact_param_values = exact_param_values_dict[model]

if model not in ["normal_location_misspec"]:
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

simulations = model_abcpy.forward_simulate(exact_param_values + [1],
                                           int(np.rint(n_observations_per_param * (1 - epsilon))), rng=rng)
outliers = model_abcpy.forward_simulate([outliers_location, 1],
                                        int(np.rint(n_observations_per_param * epsilon)),
                                        rng=rng)
outliers = Identity().statistics(outliers)
simulations = Identity().statistics(simulations)
simulations = np.concatenate((simulations, outliers))
print("Parameters shape", len(exact_param_values))
print("Simulations shape", simulations.shape)

np.save(observation_folder + "x_obs_epsilon_{:.1f}_outliers_loc_{:.1f}".format(epsilon, outliers_location), simulations)
