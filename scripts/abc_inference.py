import argparse
import logging
from time import time

import matplotlib.pyplot as plt
import numpy as np
from abcpy.backends import BackendDummy, BackendMPI
from abcpy.distances import Euclidean, Wasserstein
from abcpy.inferences import SMCABC

from src.models import instantiate_model
from src.utils import define_default_folders, load_journal_if_flag

parser = argparse.ArgumentParser()
parser.add_argument('model', help="The statistical model to consider.")
list_ABC = ["WABC", "ABC"]
parser.add_argument('method', type=str, help='ABC approach to use; can be ' + ", ".join(list_ABC))
parser.add_argument('--root_folder', type=str, default=None)
parser.add_argument('--observation_folder', type=str, default="observations")
parser.add_argument('--inference_folder', type=str, default="ABC_inference")
parser.add_argument('--use_MPI', '-m', action="store_true")
parser.add_argument('--plot_post', action="store_true")
parser.add_argument('--load_journal_if_available', action="store_true", help="")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--steps', type=int, default=10, help="Steps for SMCABC.")
parser.add_argument('--n_samples', type=int, default=1000, help="Number of posterior samples wanted.")
parser.add_argument('--n_samples_in_obs', type=int, default=1)
parser.add_argument('--n_samples_per_param', type=int, default=100)
parser.add_argument('--epsilon', type=float, default=0, help='Fraction of outliers (for location normal experiment).')
parser.add_argument('--outliers_location', type=float, default=1,
                    help='Location around where the outliers are located (for the normal location experiment)')
parser.add_argument('--add_seed_in_filename', action="store_true", help='Adds the seed in the filename of journal.')
parser.add_argument('--no_full_output', action="store_true",
                    default="Whether to disable full output in journal files.")

args = parser.parse_args()

model = args.model
method = args.method
results_folder = args.root_folder
observation_folder = args.observation_folder
inference_folder = args.inference_folder
use_MPI = args.use_MPI
plot_post = args.plot_post
load_journal_if_available = args.load_journal_if_available
seed = args.seed
steps = args.steps
n_samples = args.n_samples
n_samples_in_obs = args.n_samples_in_obs
n_samples_per_param = args.n_samples_per_param
epsilon = args.epsilon
outliers_location = args.outliers_location
add_seed_in_filename = args.add_seed_in_filename
full_output = 0 if not args.no_full_output else 1

if method not in list_ABC:
    raise NotImplementedError

default_root_folder = define_default_folders()
if results_folder is None:
    results_folder = default_root_folder[model]

observation_folder = results_folder + '/' + args.observation_folder + '/'
inference_folder = results_folder + '/' + args.inference_folder + '/'

backend = BackendMPI() if use_MPI else BackendDummy()

model_abc, statistics, param_bounds = instantiate_model(model)

# load observation
theta_obs = np.load(observation_folder + "theta_obs.npy")
if 0 < epsilon < 0.1:
    if model == "normal_location_misspec":
        x_obs = np.load(
            observation_folder + f"x_obs_epsilon_{epsilon:.2f}_outliers_loc_{outliers_location:.1f}.npy")
    elif "outlier" in model:
        x_obs = np.load(observation_folder + f"x_obs_epsilon_{epsilon:.2f}.npy")
    else:
        x_obs = np.load(observation_folder + "x_obs.npy")
else:
    if model == "normal_location_misspec":
        x_obs = np.load(
            observation_folder + f"x_obs_epsilon_{epsilon:.1f}_outliers_loc_{outliers_location:.1f}.npy")
    elif "outlier" in model:
        x_obs = np.load(observation_folder + f"x_obs_epsilon_{epsilon:.1f}.npy")
    else:
        x_obs = np.load(observation_folder + "x_obs.npy")

x_obs = [x_obs[i] for i in
         range(n_samples_in_obs)]  # only keep the first n_samples_in_obs elements from the observation
# print("Observation shape", len(x_obs), x_obs[0].shape)

# set up filename
filename = inference_folder + f"{method}_n-steps_{steps}_n-samples_{n_samples}" \
                              f"_n-sam-per-param_{n_samples_per_param}"

if add_seed_in_filename:
    filename += f"_seed_{seed}"
if 0 < epsilon < 0.1:
    if model == "normal_location_misspec":
        filename += f"_epsilon_{epsilon:.2f}_outliers_loc_{outliers_location:.1f}"
    elif "outlier" in model:
        filename += f"_epsilon_{epsilon:.2f}"
else:
    if model == "normal_location_misspec":
        filename += f"_epsilon_{epsilon:.1f}_outliers_loc_{outliers_location:.1f}"
    elif "outlier" in model:
        filename += f"_epsilon_{epsilon:.1f}"

loaded, journal = load_journal_if_flag(load_journal_if_available, filename + ".jnl")

if not loaded:

    # setup sampler
    kwargs_abc = {}
    sampler_class = SMCABC
    kwargs_abc["steps"] = steps
    kwargs_abc["epsilon_final"] = 0
    kwargs_abc["path_to_save_journal"] = filename + '.jnl'  # to save inside the inference routine
    if method == "ABC":
        kwargs_abc["which_mcmc_kernel"] = 0
        kwargs_abc["alpha"] = 0.95
        distance = Euclidean(statistics)
        version = "DelMoral"

    else:  # WABC
        kwargs_abc["which_mcmc_kernel"] = 2
        kwargs_abc["r"] = 2
        kwargs_abc["alpha"] = 0.5
        distance = Wasserstein(statistics)
        version = "Bernton"

    # define sampler_class (need to redefine it every time as otherwise some counters are not cleaned up)
    sampler_instance = sampler_class(root_models=[model_abc], distances=[distance], backend=backend, seed=seed,
                                     version=version)

    # inference
    start = time()

    journal = sampler_instance.sample([x_obs], n_samples=n_samples, n_samples_per_param=n_samples_per_param,
                                      full_output=full_output, **kwargs_abc)
    print(f"It took {time() - start:.4f} for {journal.number_of_simulations[-1]} simulations")

    # in journal name: put obs number, abc alg, number ABC steps, number of samples and n samples per param
    journal.save(filename + '.jnl')

print("ESS", journal.ESS)
print("Epsilon", journal.configuration["epsilon_arr"])
print("Steps", journal.configuration["steps"])
print("Number simulations", journal.number_of_simulations)

if plot_post:
    parameters_to_show = None
    bounds_for_plot = param_bounds
    if model == "MA2":
        bounds_for_plot = {"theta1": [-1, 1], "theta2": [-1, 1]}
    elif model == "MG1":
        bounds_for_plot = {'theta1': [0, 10], 'theta2': [0, 20], 'theta3': [0, 1 / 3]}
        # correct plot order; otherwise the true parameter values are associated to the wrong one:
        parameters_to_show = ["theta1", "theta2", "theta3"]
    for l_name in logging.Logger.manager.loggerDict:
        logging.getLogger(l_name).disabled = True
    if "univariate" in model:
        theta_obs = theta_obs[0:4]
    journal.plot_posterior_distr(
        true_parameter_values=theta_obs if model not in ["univariate_Cauchy_g-and-k", "Cauchy_g-and-k"] else None,
        show_samples=False, parameters_to_show=parameters_to_show, path_to_save=filename + '.png',
        ranges_parameters=bounds_for_plot)

    # convergence plot:
    journal.Wass_convergence_plot()
    plt.savefig(filename + '_wass_conv.png')
