import os
import sys
from time import sleep

import numpy as np
from abcpy.statistics import Identity

sys.path.append(os.getcwd())  # add the root of this project to python path
from src.models import instantiate_model
from src.parsers import parser_generate_obs_fixed_par
from src.utils import define_default_folders, define_exact_param_values

parser = parser_generate_obs_fixed_par()
args = parser.parse_args()

"""For a fixed parameter value, this generates n_observations files with n_observations_per_param datapoints each."""

model = args.model
n_observations_per_param = args.n_observations_per_param
sleep_time = args.sleep
results_folder = args.root_folder

default_root_folder = define_default_folders()

exact_param_values_dict = define_exact_param_values()
exact_param_values = exact_param_values_dict[model]

if model not in exact_param_values_dict.keys():
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

if model == "univariate_Cauchy_g-and-k":
    simulations = rng.standard_cauchy((n_observations_per_param, 1))
elif model == "Cauchy_g-and-k":
    simulations = rng.standard_cauchy((n_observations_per_param, 5))
else:
    simulations = model_abcpy.forward_simulate(exact_param_values, n_observations_per_param,
                                               rng=np.random.RandomState(seed))
    simulations = Identity().statistics(simulations)
print("Parameters shape", len(exact_param_values))
print("Simulations shape", simulations.shape)

np.save(observation_folder + "x_obs", simulations)
