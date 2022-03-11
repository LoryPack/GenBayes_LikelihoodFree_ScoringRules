import os
import sys
from time import time, sleep

import numpy as np
from abcpy.backends import BackendDummy, BackendMPI

sys.path.append(os.getcwd())  # add the root of this project to python path

from src.models import instantiate_model
from src.utils import define_default_folders, dict_implemented_scoring_rules, estimate_bandwidth, \
    heuristics_estimate_w
from src.parsers import parser_estimate_w

parser = parser_estimate_w()
args = parser.parse_args()

model = args.model
method = args.method
sleep_time = args.sleep
results_folder = args.root_folder
use_MPI = args.use_MPI
seed = args.seed
n_theta = args.n_theta
n_samples_per_param = args.n_samples_per_param
reference_method = args.reference_method
sigma_kernel = args.sigma_kernel

if sleep_time > 0:
    print("Wait for {} minutes...".format(sleep_time))
    sleep(60 * sleep_time)
    print("Done waiting!")

if method not in dict_implemented_scoring_rules().keys():
    raise NotImplementedError
if method in ["SyntheticLikelihood", "semiBSL"]:
    raise RuntimeError("We do not implement the weight for SyntheticLikelihood or semiBSL methods")

default_root_folder = define_default_folders()
if results_folder is None:
    results_folder = default_root_folder[model]

observation_folder = results_folder + '/' + args.observation_folder + '/'
inference_folder = results_folder + '/' + args.inference_folder + '/'

backend = BackendMPI() if use_MPI else BackendDummy()

model_abc, statistics, param_bounds = instantiate_model(model)

# load observation
theta_obs = np.load(observation_folder + "theta_obs.npy")
x_obs = np.load(observation_folder + "x_obs.npy")

x_obs = [x_obs[0]]
print("Observation shape", len(x_obs), x_obs[0].shape)

SR_kwargs = {}
if method == "KernelScore":
    if sigma_kernel is None:
        # need here to estimate the kernel bandwidth
        start = time()
        sigma_kernel = estimate_bandwidth(model_abc, statistics, backend, seed=seed + 1, n_theta=n_theta,
                                          n_samples_per_param=n_samples_per_param, return_values=["median"])
        print(f"Estimated sigma for kernel score {sigma_kernel:.4f}; it took {time() - start:.4f} seconds")
    SR_kwargs["sigma"] = sigma_kernel

# instantiate SRs for finding w
print("Estimate w...")
scoring_rule = dict_implemented_scoring_rules()[method](statistics, **SR_kwargs)
reference_scoring_rule = dict_implemented_scoring_rules()[reference_method](statistics)
start = time()
weight = heuristics_estimate_w(model_abc, x_obs, scoring_rule, reference_scoring_rule, backend, seed=seed,
                               n_theta=n_theta, n_theta_prime=n_theta, n_samples_per_param=n_samples_per_param)
if np.isnan(weight):
    raise RuntimeError("Estimated weight is nan")

print(f"Estimated w {weight:.4f}; it took {time() - start:.4f} seconds")
np.save(inference_folder + f"weight_single_obs_" + method + "_wrt_" + reference_method + ".npy", weight)
