import os
import sys

import numpy as np
from abcpy.output import Journal

sys.path.append(os.getcwd())  # add the root of this project to python path

from src.parsers import parser_plots_marginals_and_traces
from src.utils import define_default_folders_scoring_rules, extract_params_from_journal_gk, \
    extract_params_from_journal_multiv_gk
from src.models import instantiate_model
import matplotlib.pyplot as plt

parser = parser_plots_marginals_and_traces()
parser.add_argument('--n_samples_in_obs', type=int, default=20, help="N observations used.")

args = parser.parse_args()

model = args.model
results_folder = args.root_folder
n_samples = args.n_samples
burnin = args.burnin
thin = args.thin
n_samples_per_param = args.n_samples_per_param
n_samples_in_obs = args.n_samples_in_obs

if model not in ["g-and-k", "univariate_g-and-k"]:
    raise NotImplementedError

model_abc, statistics, param_bounds = instantiate_model(model)
param_names = list(param_bounds.keys())
param_names_latex = [r'$A$', r'$B$', r'$g$', r'$k$', r'$\rho$']

default_root_folder = define_default_folders_scoring_rules()
if results_folder is None:
    results_folder = default_root_folder[model]

observation_folder = results_folder + '/' + args.observation_folder + '/'
inference_folder = results_folder + '/' + args.inference_folder + '/'

if model == "univariate_g-and-k":
    methods_list = ["SyntheticLikelihood"]
    n_params = 4
    extract_par_fcn = extract_params_from_journal_gk
elif model == "g-and-k":
    methods_list = ["SyntheticLikelihood", "semiBSL"]
    n_params = 5
    extract_par_fcn = extract_params_from_journal_multiv_gk

# load observation
theta_obs = np.load(observation_folder + "theta_obs.npy")
assert theta_obs.shape[0] == n_params

# do plots:
# one single fig, ax (problematic for titles)
# fig, axes = plt.subplots(nrows=len(methods_list), ncols=n_params, figsize=(n_params * 3, len(methods_list) * 2), )
# sharex="row")

# make big subplots only for putting titles
fig, big_axes = plt.subplots(figsize=(n_params * 3, len(methods_list) * 2.5), nrows=len(methods_list), ncols=1)

for row, big_ax in enumerate(big_axes):
    big_ax.set_title(methods_list[row], fontsize=18)
    # Turn off axis lines and ticks of the big subplot
    # obs alpha is 0 in RGBA string!
    big_ax.axis('off')
    big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
    # removes the white frame
    big_ax._frameon = False

for method_idx, method in enumerate(methods_list):
    axes = [fig.add_subplot(len(methods_list), n_params, i + n_params * method_idx + 1) for i in range(n_params)]

    # plot exact par values:
    for j in range(n_params):
        axes[j].axhline(theta_obs[j], color="green", ls="dashed")
        axes[j].set_ylabel(param_names_latex[j])
        axes[j].set_xlabel("MCMC step")
        if j < 4:
            axes[j].set_yticks(range(5))

    for seed in range(10):
        if seed == 2:
            break
        # load journal
        filename = inference_folder + f"{method}_MCMC_burnin_{burnin}_n-samples_{n_samples}_n-sam-per-param_" \
                                      f"{n_samples_per_param}_n-sam-in-obs_{n_samples_in_obs}_seed_{seed}"
        journal = Journal.fromFile(filename + ".jnl")

        print(journal.configuration["acceptance_rates"][0])
        # extract posterior samples
        post_samples = extract_par_fcn(journal)

        # traceplot, different color corresponding to different seed
        for j in range(n_params):
            axes[j].plot(post_samples[:, j], color=f"C{seed}", linestyle='solid', lw="1",
                         alpha=0.8)

# fig.tight_layout()  # for spacing
fig.subplots_adjust(hspace=0.5, wspace=0.45)
plt.savefig(inference_folder + f"Traces_burnin_{burnin}_n-samples_{n_samples}_n-sam-per-param_"
                               f"{n_samples_per_param}_n-sam-in-obs_{n_samples_in_obs}.pdf")
