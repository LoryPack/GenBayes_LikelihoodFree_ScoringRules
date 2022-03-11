import os
import sys
from copy import deepcopy

import numpy as np
import seaborn as sns
from abcpy.output import Journal
from scipy.stats import gaussian_kde
from tabulate import tabulate

sys.path.append(os.getcwd())  # add the root of this project to python path

from src.parsers import parser_plots_marginals_and_traces
from src.utils import define_default_folders, extract_params_from_journal_gk, \
    extract_params_from_journal_multiv_gk
from src.models import instantiate_model
import matplotlib.pyplot as plt
import matplotlib

parser = parser_plots_marginals_and_traces()
args = parser.parse_args()

model = args.model
results_folder = args.root_folder
n_samples = args.n_samples
burnin = args.burnin
thin = args.thin
n_samples_per_param = args.n_samples_per_param  # unused

n_simulations_list = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

if model not in ["g-and-k", "univariate_g-and-k", "univariate_Cauchy_g-and-k", "Cauchy_g-and-k"]:
    raise NotImplementedError

model_abc, statistics, param_bounds = instantiate_model(model)
param_names = list(param_bounds.keys())
param_names_latex = [r'$A$', r'$B$', r'$g$', r'$k$', r'$\rho$']

default_root_folder = define_default_folders()
if results_folder is None:
    results_folder = default_root_folder[model]

observation_folder = results_folder + '/' + args.observation_folder + '/'
inference_folder = results_folder + '/' + args.inference_folder + '/'

if model in ["univariate_g-and-k", "univariate_Cauchy_g-and-k"]:
    methods_list = ["SyntheticLikelihood", "KernelScore", "EnergyScore"]
    n_params = 4
    extract_par_fcn = extract_params_from_journal_gk
elif model in ["g-and-k", "Cauchy_g-and-k"]:
    if model == "g-and-k":
        methods_list = ["SyntheticLikelihood", "semiBSL", "KernelScore", "EnergyScore"]
    else:
        methods_list = ["KernelScore", "EnergyScore"]
    n_params = 5
    extract_par_fcn = extract_params_from_journal_multiv_gk

# load observation
theta_obs = np.load(observation_folder + "theta_obs.npy")

# make big subplots only for putting titles
fig, big_axes = plt.subplots(figsize=(n_params * 3, len(methods_list) * 2.5), nrows=len(methods_list), ncols=1)
if len(methods_list) == 1:
    big_axes = np.atleast_1d(big_axes)

for row, big_ax in enumerate(big_axes):
    big_ax.set_title(methods_list[row], fontsize=18)
    # Turn off axis lines and ticks of the big subplot
    # obs alpha is 0 in RGBA string!
    big_ax.axis('off')
    big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
    # removes the white frame
    big_ax._frameon = False

cmap_name = "crest"
cmap = sns.color_palette(cmap_name, as_cmap=True)
norm = matplotlib.colors.Normalize(vmin=1, vmax=max(n_simulations_list))

acc_rates_matrix = np.zeros((len(methods_list), len(n_simulations_list)))

all_tables_list = []

for method_idx, method in enumerate(methods_list):
    n_simulations_table = ["m"]
    cov_trace_table = ["Trace of covariance"]
    acc_rate_table = ["Acc rate"]
    # prop_size_table = ["Prop size"]
    table_list = [["m", "Acc rate", "Trace of covariance"]]

    print(method)
    axes = [fig.add_subplot(len(methods_list), n_params, i + n_params * method_idx + 1) for i in range(n_params)]

    # plot exact par values:
    for j in range(n_params):
        if model not in ["univariate_Cauchy_g-and-k", "Cauchy_g-and-k"]:
            axes[j].axvline(theta_obs[j], color="green")
        axes[j].set_xlabel(param_names_latex[j])
        if j < 4:
            axes[j].set_xticks(range(5))

    for n_simulations_idx, n_simulations in enumerate(n_simulations_list):
        print(n_simulations)
        # set color for KDEs
        rgba = cmap(norm(n_simulations))

        # load journal
        filename = inference_folder + f"{method}_MCMC_burnin_{burnin}_n-samples_{n_samples}_n-sam-per-param_" \
                                      f"{n_simulations}_n-sam-in-obs_{10}"

        journal = Journal.fromFile(filename + ".jnl")

        # extract posterior samples
        post_samples = extract_par_fcn(journal)

        # thinning
        post_samples = post_samples[::thin, :]

        # store acc rate:
        acc_rates_matrix[method_idx, n_simulations_idx] = journal.configuration["acceptance_rates"][0]

        cov_matrix = np.cov(post_samples.transpose(1, 0))
        cov_trace = np.trace(cov_matrix)
        acc_rate = journal.configuration["acceptance_rates"][0]
        prop_size = journal.configuration["cov_matrices"][0][0, 0]

        # now need to make KDE and plot in the correct panel, with different color corresponding to different n_obs
        for j in range(n_params):
            xmin, xmax = param_bounds[param_names[j]]
            positions = np.linspace(xmin, xmax, 100)
            gaussian_kernel = gaussian_kde(post_samples[:, j])
            axes[j].plot(positions, gaussian_kernel(positions), color=rgba, linestyle='solid', lw="1",
                         alpha=1, label="Density")

        # add things to the table for latex:
        n_simulations_table += [n_simulations]
        cov_trace_table += [f"{cov_trace:.4f}"]
        acc_rate_table += [f"{acc_rate:.3f}"]
        # prop_size_table += [prop_size]

        # other table
        table_list.append([n_simulations, f"{acc_rate:.3f}", f"{cov_trace:.4f}"])

    print(method)
    print(tabulate([n_simulations_table, cov_trace_table, acc_rate_table], tablefmt="latex_booktabs"))
    print(tabulate(table_list, headers="firstrow", tablefmt="latex_booktabs"))
    print(tabulate(table_list, headers="firstrow"))
    all_tables_list.append(deepcopy(table_list))

# fig.tight_layout()  # for spacing
fig.subplots_adjust(hspace=0.5)
plt.savefig(inference_folder + f"Different_m_plot_burnin_{burnin}_n-samples_{n_samples}"
                               f"_thinning_{thin}_{cmap_name}.pdf")

# need to create legend and better colors.

for i in range(acc_rates_matrix.shape[1]):
    strings = [f"${acc_rate:.3f}$" for acc_rate in acc_rates_matrix[:, i]]
    strings2 = [" & " for i in range(len(strings))]
    strings3 = strings2 + strings
    strings3[::2] = strings2
    strings3[1::2] = strings
    print("".join(strings3))

# the following only works if the two lists in all_tables_list have same length
if len(methods_list) == 2:
    all_tables_list = [all_tables_list[0][i] + all_tables_list[1][i][1:] for i in range(len(all_tables_list[0]))]
elif len(methods_list) == 3:
    all_tables_list = [all_tables_list[0][i] + all_tables_list[1][i][1:] + all_tables_list[2][i][1:] for i in
                       range(len(all_tables_list[0]))]
elif len(methods_list) == 4:
    all_tables_list = [all_tables_list[0][i] + all_tables_list[1][i][1:] + all_tables_list[2][i][1:] + all_tables_list[3][i][1:] for
                       i in range(len(all_tables_list[0]))]
print(tabulate(all_tables_list, headers="firstrow"))
print(tabulate(all_tables_list, headers="firstrow", tablefmt="latex_booktabs"))
