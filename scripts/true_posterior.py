from time import sleep
import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
from theano import tensor as tt
sys.path.append(os.getcwd())  # add the root of this project to python path
from src.models import ma2_log_lik_for_mcmc
from src.parsers import parser_true_posterior
from src.utils import LogLike, define_default_folders_scoring_rules, transform_R_to_theta_MA2

parser = parser_true_posterior()
parser.add_argument('--epsilon', type=float, default=0, help='Fraction of outliers (for location normal experiment).')
parser.add_argument('--outliers_location', type=float, default=1,
                    help='Location around where the outliers are located (for the normal location experiment)')
args = parser.parse_args()

model = args.model
sleep_time = args.sleep
start_observation_index = args.start_observation_index
n_observations = args.n_observations
results_folder = args.root_folder
plot_post = args.plot_post
seed = args.seed
n_samples = args.n_samples
burnin = args.burnin
n_samples_in_obs = args.n_samples_in_obs
cores = args.cores
epsilon = args.epsilon
outliers_location = args.outliers_location

if sleep_time > 0:
    print("Wait for {} minutes...".format(sleep_time))
    sleep(60 * sleep_time)
    print("Done waiting!")

if model not in ["normal_location_misspec", "MA2"]:
    raise NotImplementedError

default_root_folder = define_default_folders_scoring_rules()
if results_folder is None:
    results_folder = default_root_folder[model]

observation_folder = results_folder + '/' + args.observation_folder + '/'
true_posterior_folder = results_folder + '/' + args.true_posterior_folder + '/'

# load observation
theta_obs = np.load(observation_folder + "theta_obs.npy")
if "misspec" not in model:
    x_obs = np.load(observation_folder + "x_obs.npy")
else:
    x_obs = np.load(observation_folder + "x_obs_epsilon_{:.1f}_outliers_loc_{:.1f}.npy".format(epsilon, outliers_location))

x_obs = x_obs[0:n_samples_in_obs]  # only keep n_samples_in_obs from the observation

print("Exact parameter value:", theta_obs)

if "misspec" not in model:
    filename = true_posterior_folder + f"n-samples_{n_samples}_burnin_{burnin}" \
                                       f"_n-sam-per-param_{n_samples_in_obs}"
else:
    filename = true_posterior_folder + f"n-samples_{n_samples}_burnin_{burnin}" \
                                       f"_n-sam-per-param_{n_samples_in_obs}" \
                                       f"_epsilon_{epsilon:.1f}_outliers_loc_{outliers_location:.1f}"

# posterior inference
if model in ["MA2"]:

    logl = LogLike(ma2_log_lik_for_mcmc, x_obs)

    with pm.Model() as model:
        R1 = pm.Uniform('R1', 0, 1)
        R2 = pm.Uniform('R2', 0, 1)

        theta = tt.as_tensor_variable([R1, R2])

        pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})

        trace_true = pm.sample(n_samples, tune=burnin, cores=cores, random_seed=seed)

        # now transform to the exact parameter spaces:
        R1_arr = trace_true['R1'].reshape(-1, 1)
        R2_arr = trace_true['R2'].reshape(-1, 1)

        theta1_arr, theta2_arr = transform_R_to_theta_MA2(R1_arr, R2_arr)

        trace_true_array = np.concatenate((theta1_arr, theta2_arr), axis=1)

elif model == "normal_location_misspec":
    with pm.Model() as model:
        mu = pm.Normal('mu', 0, 1)
        # on department desktop:
        obs = pm.Normal('obs', mu=mu, tau=1, observed=x_obs)
        trace_true = pm.sample(n_samples, tune=burnin, cores=cores, random_seed=seed)
        trace_true_array = trace_true['mu'].reshape(-1, 1)

else:
    raise NotImplementedError

if plot_post:
    pm.plot_posterior(trace_true)
    plt.savefig(filename + ".png")

# save posterior trace:
np.save(filename, trace_true_array)
