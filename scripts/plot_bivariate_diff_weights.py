import os
import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from abcpy.output import Journal
from tabulate import tabulate

sys.path.append(os.getcwd())  # add the root of this project to python path

from src.utils import define_default_folders_scoring_rules, extract_params_from_journal_MG1, \
    extract_params_from_journal_MA2
from src.parsers import parser_bivariate_plots

parser = parser_bivariate_plots()

args = parser.parse_args()

model = args.model
results_folder = args.root_folder
n_samples = args.n_samples
burnin = args.burnin
thin = args.thin
n_samples_per_param = args.n_samples_per_param
cmap_name = args.cmap_name
fill = not args.no_fill

n_samples_in_obs = 1
show_samples = False

if model not in ["MG1", "MA2"]:
    raise NotImplementedError

if model == "MG1":
    ranges_parameters = param_bounds = {'theta1': [0, 4.5], 'theta2': [0, 11], 'theta3': [0.12, 1 / 3]}
    names = param_names = list(param_bounds.keys())
    param_names_latex = [r'$\theta^1$', r'$\theta^2$', r'$\theta^3$', ]
    n_params = 3
    extract_par_fcn = extract_params_from_journal_MG1
elif model == "MA2":
    ranges_parameters = param_bounds = {'theta1': [-2, 2], 'theta2': [-1, 1]}
    names = param_names = list(param_bounds.keys())
    param_names_latex = [r'$\theta^1$', r'$\theta^2$', ]
    n_params = 2
    extract_par_fcn = extract_params_from_journal_MA2

default_root_folder = define_default_folders_scoring_rules()
if results_folder is None:
    results_folder = default_root_folder[model]

observation_folder = results_folder + '/' + args.observation_folder + '/'
inference_folder = results_folder + '/' + args.inference_folder + '/'
true_posterior_folder = results_folder + '/' + args.true_posterior_folder + '/'

methods_list = ["KernelScore", "EnergyScore"]

# load observation
theta_obs = np.load(observation_folder + "theta_obs.npy")
true_parameter_values = theta_obs
assert theta_obs.shape[0] == n_params

all_tables_list = []

for method in methods_list:
    if method == "EnergyScore":
        if model == "MG1":
            weightlist = [11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56]
        elif model == "MA2":
            weightlist = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    elif method == "KernelScore":
        if model == "MG1":
            weightlist = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]
        elif model == "MA2":
            weightlist = [250, 300, 350, 400, 450, 500, 550, 600, 620, 640]

    weight_table = ["Weight"]
    cov_trace_table = ["Trace of covariance"]
    acc_rate_table = ["Acc rate"]
    prop_size_table = ["Prop size"]
    table_list = [["Weight", "Prop size", "Acc rate", "Trace of covariance"]]
    # we will do plot in the following way: we need to make two panels for the all possible parameter combinations, and we
    # have 5 methods (the true posterior + other 4). Thwn make a (n_param_combination)xlen(methods) plot.

    figsize_scale = 3
    n_param_combinations = int(n_params * (n_params - 1) / 2)
    figsize_actual = figsize_scale * n_param_combinations
    if model == "MA2":
        title_size = figsize_actual / n_param_combinations * 8
        label_size = figsize_actual / n_param_combinations * 7
        ticks_size = figsize_actual / n_param_combinations * 6
    elif model == "MG1":
        title_size = figsize_actual / n_param_combinations * 8 * 1.35
        label_size = figsize_actual / n_param_combinations * 7 * 1.35
        ticks_size = figsize_actual / n_param_combinations * 6 * 1.35

    figwidth = len(weightlist) * figsize_scale - (1 * (len(weightlist) - 1))
    figheigth = figsize_actual
    # do plots:
    fig, axes = plt.subplots(nrows=n_param_combinations, ncols=len(weightlist),
                             figsize=(figwidth, figheigth), sharex="row", sharey="row")
    # fig axes size:
    if n_param_combinations == 1:
        axes = axes.reshape(1, -1)
    if len(weightlist) == 1:
        axes = axes.reshape(-1, 1)

    for weight_idx, weight in enumerate(weightlist):
        print(weight)

        # load journal
        filename = inference_folder + f"{method}_MCMC_burnin_{burnin}_n-samples_{n_samples}_n-sam-per-param_" \
                                      f"{n_samples_per_param}_n-sam-in-obs_{n_samples_in_obs}_weight_{weight:.1f}"
        journal = Journal.fromFile(filename + ".jnl")

        # extract posterior samples
        post_samples = extract_par_fcn(journal)

        # thinning
        post_samples = post_samples[::thin, :]

        # create dataframe for seaborn
        df = pd.DataFrame(post_samples, columns=names)

        cov_matrix = np.cov(post_samples.transpose(1, 0))
        cov_trace = np.trace(cov_matrix)
        acc_rate = journal.configuration["acceptance_rates"][0]
        prop_size = journal.configuration["cov_matrices"][0][0, 0]

        # add things to the table for latex:
        weight_table += [weight]
        cov_trace_table += [f"{cov_trace:.4f}"]
        acc_rate_table += [f"{acc_rate:.2f}"]
        prop_size_table += [prop_size]

        # other table
        table_list.append([weight, prop_size, f"{acc_rate:.2f}", f"{cov_trace:.4f}"])

        # suptitle in axis:
        # axes[0, weight_idx].set_title(r"$w={}$, Tr $ \Sigma={:.2f}$".format(int(weight), cov_trace), size=title_size)
        axes[0, weight_idx].set_title(r"$w={}$".format(int(weight)), size=title_size)
        if model == "MG1" and method == "KernelScore":
            axes[0, weight_idx].set_title(r"$\tilde w={}$".format(int(weight / 100)), size=title_size)
        # axes[0, weight_idx].set_title(r"$w={}\cdot 10^{3}$".format(int(weight)), size=title_size)

        # now need to make KDE and plot in the correct panel, with different color corresponding to different n_obs
        ax_counter = 0

        for x in range(n_params):
            for y in range(0, x):
                # this plots the posterior samples
                if show_samples:
                    axes[ax_counter, weight_idx].plot(post_samples[y], post_samples[x], '.k', markersize='1')

                xmin, xmax = ranges_parameters[names[y]]
                ymin, ymax = ranges_parameters[names[x]]

                # use seaborn
                if fill:
                    try:
                        sns.kdeplot(data=df, x=names[y], y=names[x], fill=True, thresh=0.05, levels=20, cmap=cmap_name,
                                    ax=axes[ax_counter, weight_idx])
                    except np.linalg.LinAlgError:
                        pass
                else:
                    try:
                        sns.kdeplot(data=df, x=names[y], y=names[x], fill=False, thresh=0.05, levels=15, cmap=cmap_name,
                                    ax=axes[ax_counter, weight_idx])
                    except np.linalg.LinAlgError:
                        pass

                if true_parameter_values is not None:
                    axes[ax_counter, weight_idx].plot([xmin, xmax],
                                                      [true_parameter_values[x], true_parameter_values[x]],
                                                      "green", markersize='20', linestyle='dotted')
                    axes[ax_counter, weight_idx].plot([true_parameter_values[y], true_parameter_values[y]],
                                                      [ymin, ymax],
                                                      "green", markersize='20', linestyle='dotted')

                axes[ax_counter, weight_idx].set_xlim([xmin, xmax])
                axes[ax_counter, weight_idx].set_ylim([ymin, ymax])

                axes[ax_counter, weight_idx].set_ylabel(param_names_latex[x], size=label_size)
                axes[ax_counter, weight_idx].set_xlabel(param_names_latex[y], size=label_size)

                axes[ax_counter, weight_idx].tick_params(axis='both', which='major', labelsize=ticks_size)
                # axes[ax_counter, weight_idx].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                # axes[ax_counter, weight_idx].yaxis.offsetText.set_fontsize(ticks_size)
                axes[ax_counter, weight_idx].yaxis.set_visible(True)
                # axes[ax_counter, weight_idx].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                # axes[ax_counter, weight_idx].yaxis.offsetText.set_fontsize(ticks_size)
                axes[ax_counter, weight_idx].xaxis.set_visible(True)

                ax_counter += 1

        if model == "MA2":
            # fill region out of prior range
            for j in range(n_param_combinations):
                axes[j, weight_idx].fill_between(x=[-2, 0], y1=[-1, -1], y2=[1, -1], facecolor="red", alpha=.2)
                axes[j, weight_idx].fill_between(x=[0, 2], y1=[-1, -1], y2=[-1, 1], facecolor="red", alpha=.2)

    print(tabulate([weight_table, cov_trace_table, acc_rate_table, prop_size_table], tablefmt="latex_booktabs"))
    print(tabulate(table_list, headers="firstrow", tablefmt="latex_booktabs"))
    print(tabulate(table_list, headers="firstrow"))

    fig.tight_layout()  # for spacing

    plt.savefig(
        inference_folder + f"Different_weights_plot_{method}_burnin_{burnin}_n-samples_{n_samples}_n-sam-per-param_"
                           f"{n_samples_per_param}_thinning_{thin}_{cmap_name}_{'fill' if fill else 'nofill'}.pdf")
    plt.close()

    all_tables_list.append(deepcopy(table_list))

# the following only works if the two lists in all_tables_list have same length
all_tables_list = [all_tables_list[0][i] + all_tables_list[1][i] for i in range(len(all_tables_list[0]))]
print(tabulate(all_tables_list, headers="firstrow"))
print(tabulate(all_tables_list, headers="firstrow", tablefmt="latex_booktabs"))
