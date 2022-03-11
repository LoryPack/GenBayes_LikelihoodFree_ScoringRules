import os
import sys

import numpy as np
from abcpy.output import Journal

sys.path.append(os.getcwd())  # add the root of this project to python path

from src.parsers import parser_plots_marginals_and_traces
from src.utils import define_default_folders, extract_params_from_journal_gk, \
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
n_samples_in_obs = args.n_samples_in_obs

if model not in ["g-and-k", "univariate_g-and-k"]:
    raise NotImplementedError

model_abc, statistics, param_bounds = instantiate_model(model)
param_names = list(param_bounds.keys())
param_names_latex = [r'$A$', r'$B$', r'$g$', r'$k$', r'$\rho$']

default_root_folder = define_default_folders()
if results_folder is None:
    results_folder = default_root_folder[model]

observation_folder = results_folder + '/' + args.observation_folder + '/'
inference_folder = results_folder + '/' + args.inference_folder + '/'

if model == "univariate_g-and-k":
    methods_list = ["SyntheticLikelihood"]
    n_params = 4
    n_simulations_list_lists = [[500, 1000, 1500, 2000, 2500, 3000], [500, 1000, 1500, 2000, 2500, 3000, 30000]]
    extract_par_fcn = extract_params_from_journal_gk
elif model == "g-and-k":
    methods_list = ["SyntheticLikelihood", "semiBSL"]
    n_simulations_list_lists = [[500, 1000, 1500, 2000, 2500, 3000, 30000], [500, 1000, 1500, 2000, 2500, 3000, 30000]]
    n_params = 5
    extract_par_fcn = extract_params_from_journal_multiv_gk

# load observation
theta_obs = np.load(observation_folder + "theta_obs.npy")
assert theta_obs.shape[0] == n_params

# make big subplots only for putting titles
fig, big_axes = plt.subplots(figsize=(n_params * 3, len(methods_list) * 2.5), nrows=len(methods_list), ncols=1)
if len(methods_list) == 1:
    big_axes = [big_axes]

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
    n_simulations_list = n_simulations_list_lists[method_idx]
    acc_rates_list = []
    # plot exact par values:
    for j in range(n_params):
        axes[j].axhline(theta_obs[j], color="green", ls="dashed")
        axes[j].set_ylabel(param_names_latex[j])
        axes[j].set_xlabel("MCMC step")
        if j < 4:
            axes[j].set_yticks(range(5))

    for index, n_samples_per_param in enumerate(n_simulations_list):
        # load journal
        filename = inference_folder + f"{method}_MCMC_burnin_{burnin}_n-samples_{n_samples}_n-sam-per-param_" \
                                      f"{n_samples_per_param}_n-sam-in-obs_{n_samples_in_obs}"
        journal = Journal.fromFile(filename + ".jnl")

        # store acc rate:
        acc_rates_list.append(journal.configuration["acceptance_rates"][0])

        # extract posterior samples
        post_samples = extract_par_fcn(journal)

        # traceplot, different color corresponding to different n_simulations
        for j in range(n_params):
            axes[j].plot(post_samples[:, j], color=f"C{index}", linestyle='solid', lw="1",
                         alpha=0.6, label=str(n_samples_per_param))

    # print the acceptance rates in nice format for latex
    strings = [f"${acc_rate:.1e}$" for acc_rate in acc_rates_list]
    strings2 = [" & " for i in range(len(strings))]
    strings3 = strings2 + strings
    strings3[::2] = strings2
    strings3[1::2] = strings
    print("".join(strings3))

    handles, labels = axes[-1].get_legend_handles_labels()
    if method_idx == 0:
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.91, 0.73),
                   bbox_transform=plt.gcf().transFigure)
    elif method_idx == 1:
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.91, 0.26),
                   bbox_transform=plt.gcf().transFigure)

# fig.tight_layout()  # for spacing
fig.subplots_adjust(hspace=0.5, wspace=0.4)
plt.savefig(inference_folder + f"Traces_burnin_{burnin}_n-samples_{n_samples}_n-sam-in-obs_{n_samples_in_obs}.pdf")
