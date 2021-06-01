import os
import sys
from time import time, sleep

import numpy as np
from abcpy.backends import BackendDummy, BackendMPI
from abcpy.inferences import PMC, MCMCMetropoliHastings

sys.path.append(os.getcwd())  # add the root of this project to python path

from src.models import instantiate_model
from src.parsers import parser_inference
from src.utils import define_default_folders_scoring_rules, dict_implemented_scoring_rules, load_journal_if_flag, \
    heuristics_estimate_w, estimate_bandwidth, transform_journal
import logging
import matplotlib.pyplot as plt

parser = parser_inference()
parser.add_argument('--epsilon', type=float, default=0, help='Fraction of outliers (for location normal experiment).')
parser.add_argument('--outliers_location', type=float, default=1,
                    help='Location around where the outliers are located (for the normal location experiment)')
parser.add_argument('--inipoint', type=float, default=None, help='Initial MCMC point')
parser.add_argument('--sigma_kernel', type=float, default=None, help='If provided, use this as bandwidth for Gaussian '
                                                                     'kernel in Kernel Score posterior')
parser.add_argument('--add_seed_in_filename', action="store_true", help='Adds the seed in the filename of journal.')
parser.add_argument('--add_weight_in_filename', action="store_true", help='Adds the weight in the filename of journal.')

args = parser.parse_args()

# It uses the PMC or MCMC algorithm with the approximate likelihood to perform inference

model = args.model
method = args.method
algorithm = args.algorithm
sleep_time = args.sleep
start_observation_index = args.start_observation_index
n_observations = args.n_observations
results_folder = args.root_folder
use_MPI = args.use_MPI
plot_post = args.plot_post
plot_trace = args.plot_trace
load_journal_if_available = args.load_journal_if_available
seed = args.seed
steps = args.steps
n_samples = args.n_samples
burnin = args.burnin
n_samples_in_obs = args.n_samples_in_obs
n_samples_per_param = args.n_samples_per_param
prop_size = args.prop_size
full_output = 0 if not args.no_full_output else 1
estimate_w = args.estimate_w
weight = args.weight
weight_file = args.weight_file
reference_method = args.reference_method
adapt_proposal_cov_interval = args.adapt_proposal_cov_interval
epsilon = args.epsilon
outliers_location = args.outliers_location
inipoint = args.inipoint
sigma_kernel = args.sigma_kernel
add_seed_in_filename = args.add_seed_in_filename
add_weight_in_filename = args.add_weight_in_filename

if sleep_time > 0:
    print("Wait for {} minutes...".format(sleep_time))
    sleep(60 * sleep_time)
    print("Done waiting!")

if method not in dict_implemented_scoring_rules().keys():
    raise NotImplementedError

if algorithm not in ["PMC", "MCMC"]:
    raise NotImplementedError

default_root_folder = define_default_folders_scoring_rules()
if results_folder is None:
    results_folder = default_root_folder[model]

observation_folder = results_folder + '/' + args.observation_folder + '/'
inference_folder = results_folder + '/' + args.inference_folder + '/'

backend = BackendMPI() if use_MPI else BackendDummy()

model_abc, statistics, param_bounds = instantiate_model(model)

# load observation
theta_obs = np.load(observation_folder + "theta_obs.npy")
if "misspec" not in model:
    x_obs = np.load(observation_folder + "x_obs.npy")
else:
    x_obs = np.load(observation_folder + "x_obs_epsilon_{:.1f}_outliers_loc_{:.1f}.npy".format(epsilon, outliers_location))

x_obs = [x_obs[i] for i in range(n_samples_in_obs)]  # only keep the first observation element from the observation
# print("Observation shape", len(x_obs), x_obs[0].shape)

SR_kwargs = {}
if method == "KernelScore":
    if sigma_kernel is not None:
        SR_kwargs["sigma"] = sigma_kernel
    else:
        # need here to estimate the kernel bandwidth
        start = time()
        SR_kwargs["sigma"] = estimate_bandwidth(model_abc, backend, seed=seed + 1, n_theta=1000,
                                                n_samples_per_param=n_samples_per_param, return_values=["median"])
        print(f"Estimated sigma for kernel score {SR_kwargs['sigma']:.4f}; it took {time() - start:.4f} seconds")

if method not in ["SyntheticLikelihood", "semiBSL"]:
    if estimate_w:
        # instantiate SRs for finding w
        print("Estimate w...")
        scoring_rule = dict_implemented_scoring_rules()[method](statistics, **SR_kwargs)
        reference_scoring_rule = dict_implemented_scoring_rules()[reference_method](statistics)
        start = time()
        weight = heuristics_estimate_w(model_abc, x_obs, scoring_rule, reference_scoring_rule, backend, seed=seed,
                                       n_theta=1000, n_theta_prime=1000, n_samples_per_param=n_samples_per_param)
        if np.isnan(weight):
            raise RuntimeError("Estimated weight is nan")
        print(f"Estimated w {weight:.4f}; it took {time() - start:.4f} seconds")
    elif weight_file is not None:  # if you did not estimate it, use the provided file
        weight = np.load(inference_folder + weight_file)
        print("correctly loaded weight:", weight)
    # re-instantiate the scoring rule with correct weight
    scoring_rule = dict_implemented_scoring_rules()[method](statistics, weight=weight, **SR_kwargs)
else:
    scoring_rule = dict_implemented_scoring_rules()[method](statistics, **SR_kwargs)

# define sampler_class, then perform inference
start = time()
if algorithm == "PMC":
    filename = inference_folder + f"{method}_PMC_n-steps_{steps}_n-samples_{n_samples}_n-sam-per-param_" \
                                  f"{n_samples_per_param}_n-sam-in-obs_{n_samples_in_obs}"
    loaded, journal = load_journal_if_flag(load_journal_if_available, filename + ".jnl")

    if not loaded:
        sampler_instance = PMC(root_models=[model_abc], loglikfuns=[scoring_rule], backend=backend, seed=seed)
        journal = sampler_instance.sample([x_obs], n_samples=n_samples, n_samples_per_param=n_samples_per_param,
                                          full_output=full_output, steps=steps)
        journal = transform_journal(journal, model)  # transform (reparametrize) journal
        journal.save(filename + '.jnl')

    print("ESS", journal.ESS)
else:
    filename = inference_folder + f"{method}_MCMC_burnin_{burnin}_n-samples_{n_samples}_n-sam-per-param_" \
                                  f"{n_samples_per_param}_n-sam-in-obs_{n_samples_in_obs}"
    if add_seed_in_filename:
        filename += f"_seed_{seed}"
    if add_weight_in_filename:
        filename += f"_weight_{weight}"
    if "misspec" in model:
        filename += f"_epsilon_{epsilon:.1f}_outliers_loc_{outliers_location:.1f}"

    loaded, journal = load_journal_if_flag(load_journal_if_available, filename + ".jnl")

    if not loaded:
        sampler_instance = MCMCMetropoliHastings(root_models=[model_abc], loglikfuns=[scoring_rule],
                                                 backend=backend, seed=seed)
        cov_matrices = [np.eye(len(sampler_instance.parameter_names)) * prop_size]  # diagonal cov matrix
        inipoint = np.array(
            [inipoint]) if inipoint is not None else None  # use the scalar initial point otherwise
        journal = sampler_instance.sample([x_obs], n_samples=n_samples, n_samples_per_param=n_samples_per_param,
                                          burnin=burnin, bounds=param_bounds,
                                          adapt_proposal_cov_interval=adapt_proposal_cov_interval,
                                          cov_matrices=cov_matrices,
                                          iniPoint=inipoint)
        journal = transform_journal(journal, model)  # transform (reparametrize) journal
        journal.save(filename + '.jnl')

    print("Acc rate", journal.configuration["acceptance_rates"])
    print("configuration", journal.configuration)

    if plot_trace:
        journal.traceplot()
        plt.savefig(filename + "_traceplot.png")

print(f"It took {time() - start:.4f} for {journal.number_of_simulations[-1]} simulations")
print("Number simulations", journal.number_of_simulations)
print("Covariance", journal.posterior_cov())
try:
    print("Trace of covariance", np.trace(journal.posterior_cov()[0]))
except:
    pass

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
    journal.plot_posterior_distr(
        true_parameter_values=theta_obs if model not in ["univariate_Cauchy_g-and-k", "Cauchy_g-and-k"] else None,
        show_samples=False,
        parameters_to_show=parameters_to_show, path_to_save=filename + '.png',
        ranges_parameters=bounds_for_plot)
    # journal.plot_posterior_distr(true_parameter_values=theta_obs, show_samples=False,
    #                              parameters_to_show=parameters_to_show, path_to_save=filename + 'double_only.png',
    #                              ranges_parameters=bounds_for_plot, double_marginals_only=True)
