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

from src.utils import define_default_folders, extract_params_from_journal_MG1, \
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

default_root_folder = define_default_folders()
if results_folder is None:
    results_folder = default_root_folder[model]

observation_folder = results_folder + '/' + args.observation_folder + '/'
inference_folder = results_folder + '/' + args.inference_folder + '/'
# true_posterior_folder = results_folder + '/' + args.true_posterior_folder + '/'  # unused

methods_list = ["SyntheticLikelihood", "semiBSL", "KernelScore", "EnergyScore"]
n_simulations_list = [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# load observation
theta_obs = np.load(observation_folder + "theta_obs.npy")
true_parameter_values = theta_obs
assert theta_obs.shape[0] == n_params

all_tables_list = []

for method in methods_list:
    n_simulations_table = ["m"]
    cov_trace_table = ["Trace of covariance"]
    acc_rate_table = ["Acc rate"]
    # prop_size_table = ["Prop size"]
    table_list = [["m", "Acc rate", "Trace of covariance"]]

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

    figwidth = len(n_simulations_list) * figsize_scale - (1 * (len(n_simulations_list) - 1))
    figheigth = figsize_actual
    # do plots:
    fig, axes = plt.subplots(nrows=n_param_combinations, ncols=len(n_simulations_list),
                             figsize=(figwidth, figheigth), sharex="row", sharey="row")
    # fig axes size:
    if n_param_combinations == 1:
        axes = axes.reshape(1, -1)
    if len(n_simulations_list) == 1:
        axes = axes.reshape(-1, 1)

    for n_simulations_idx, n_simulations in enumerate(n_simulations_list):
        print(n_simulations)

        # load journal
        filename = inference_folder + f"{method}_MCMC_burnin_{burnin}_n-samples_{n_samples}_n-sam-per-param_" \
                                      f"{n_simulations}_n-sam-in-obs_{n_samples_in_obs}"
        try:
            journal = Journal.fromFile(filename + ".jnl")
            loaded = True
        except (FileNotFoundError, AttributeError) as e:
            loaded = False
            print(filename)

        if loaded:
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
        else:
            cov_trace = np.nan
            acc_rate = np.nan
            prop_size = np.nan

        # add things to the table for latex:
        n_simulations_table += [n_simulations]
        cov_trace_table += [f"{cov_trace:.4f}"]
        acc_rate_table += [f"{acc_rate:.3f}"]
        # prop_size_table += [prop_size]

        # other table
        table_list.append([n_simulations, f"{acc_rate:.3f}", f"{cov_trace:.4f}"])

        # suptitle in axis:
        axes[0, n_simulations_idx].set_title(r"$m={}$".format(int(n_simulations)), size=title_size)

        ax_counter = 0

        for x in range(n_params):
            for y in range(0, x):

                xmin, xmax = ranges_parameters[names[y]]
                ymin, ymax = ranges_parameters[names[x]]

                axes[ax_counter, n_simulations_idx].set_xlim([xmin, xmax])
                axes[ax_counter, n_simulations_idx].set_ylim([ymin, ymax])

                axes[ax_counter, n_simulations_idx].set_ylabel(param_names_latex[x], size=label_size)
                axes[ax_counter, n_simulations_idx].set_xlabel(param_names_latex[y], size=label_size)

                axes[ax_counter, n_simulations_idx].tick_params(axis='both', which='major', labelsize=ticks_size)
                # axes[ax_counter, n_simulations_idx].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                # axes[ax_counter, n_simulations_idx].yaxis.offsetText.set_fontsize(ticks_size)
                axes[ax_counter, n_simulations_idx].yaxis.set_visible(True)
                # axes[ax_counter, n_simulations_idx].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                # axes[ax_counter, n_simulations_idx].yaxis.offsetText.set_fontsize(ticks_size)
                axes[ax_counter, n_simulations_idx].xaxis.set_visible(True)

                if loaded:
                    # now need to make KDE and plot in the correct panel, with different color corresponding to
                    # different n_obs

                    # this plots the posterior samples
                    if show_samples:
                        axes[ax_counter, n_simulations_idx].plot(post_samples[y], post_samples[x], '.k', markersize='1')

                    if true_parameter_values is not None:
                        axes[ax_counter, n_simulations_idx].plot([xmin, xmax],
                                                                 [true_parameter_values[x], true_parameter_values[x]],
                                                                 "green", markersize='20', linestyle='dotted')
                        axes[ax_counter, n_simulations_idx].plot([true_parameter_values[y], true_parameter_values[y]],
                                                                 [ymin, ymax],
                                                                 "green", markersize='20', linestyle='dotted')

                    # use seaborn
                    if fill:
                        try:
                            sns.kdeplot(data=df, x=names[y], y=names[x], fill=True, thresh=0.05, levels=20,
                                        cmap=cmap_name, ax=axes[ax_counter, n_simulations_idx])
                        except (np.linalg.LinAlgError, ValueError):
                            pass
                    else:
                        try:
                            sns.kdeplot(data=df, x=names[y], y=names[x], fill=False, thresh=0.05, levels=15,
                                        cmap=cmap_name, ax=axes[ax_counter, n_simulations_idx])
                        except (np.linalg.LinAlgError, ValueError):
                            pass

                ax_counter += 1

        if model == "MA2":
            # fill region out of prior range
            for j in range(n_param_combinations):
                axes[j, n_simulations_idx].fill_between(x=[-2, 0], y1=[-1, -1], y2=[1, -1], facecolor="red", alpha=.2)
                axes[j, n_simulations_idx].fill_between(x=[0, 2], y1=[-1, -1], y2=[-1, 1], facecolor="red", alpha=.2)

    print(method)
    print(tabulate([n_simulations_table, cov_trace_table, acc_rate_table], tablefmt="latex_booktabs"))
    print(tabulate(table_list, headers="firstrow", tablefmt="latex_booktabs"))
    print(tabulate(table_list, headers="firstrow"))

    fig.tight_layout()  # for spacing

    plt.savefig(
        inference_folder + f"Different_n_simulations_plot_{method}_burnin_{burnin}_n-samples_{n_samples}_n-sam-per-param_"
                           f"{n_samples_per_param}_thinning_{thin}_{cmap_name}_{'fill' if fill else 'nofill'}.pdf")
    plt.close()

    all_tables_list.append(deepcopy(table_list))

# the following only works if the two lists in all_tables_list have same length
if len(methods_list) == 2:
    all_tables_list = [all_tables_list[0][i] + all_tables_list[1][i][1:] for i in range(len(all_tables_list[0]))]
elif len(methods_list) == 3:
    all_tables_list = [all_tables_list[0][i] + all_tables_list[1][i][1:] + all_tables_list[2][i][1:] for i in
                       range(len(all_tables_list[0]))]
elif len(methods_list) == 4:
    all_tables_list = [
        all_tables_list[0][i] + all_tables_list[1][i][1:] + all_tables_list[2][i][1:] + all_tables_list[3][i][1:] for
        i in range(len(all_tables_list[0]))]
print(tabulate(all_tables_list, headers="firstrow"))
print(tabulate(all_tables_list, headers="firstrow", tablefmt="latex_booktabs"))
