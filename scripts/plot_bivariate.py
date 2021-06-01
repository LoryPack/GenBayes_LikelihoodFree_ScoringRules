import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from abcpy.output import Journal

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

methods_list = ["SyntheticLikelihood", "semiBSL", "KernelScore", "EnergyScore", "True posterior"]

# load observation
theta_obs = np.load(observation_folder + "theta_obs.npy")
true_parameter_values = theta_obs
assert theta_obs.shape[0] == n_params

# we will do plot in the following way: we need to make two panels for the all possible parameter combinations, and we
# have 5 methods (the true posterior + other 4). Thwn make a (n_param_combination)xlen(methods) plot.

figsize_scale = 3
n_param_combinations = int(n_params * (n_params - 1) / 2)
figsize_actual = figsize_scale * n_param_combinations
title_size = figsize_actual / n_param_combinations * 4.25
label_size = figsize_actual / n_param_combinations * 4
ticks_size = figsize_actual / n_param_combinations * 3
# figwidth = len(methods_list) * figsize_scale + (0.4 * 4 if model == "MA2" else 0)
figwidth = len(methods_list) * figsize_scale - (0.8 * (len(methods_list) - 1))
figheigth = figsize_actual
# do plots:
fig, axes = plt.subplots(nrows=n_param_combinations, ncols=len(methods_list),
                         figsize=(figwidth, figheigth), sharex="row", sharey="row")
# fig axes size:
if n_param_combinations == 1:
    axes = axes.reshape(1, -1)
if len(methods_list) == 1:
    axes = axes.reshape(-1, 1)

for method_idx, method in enumerate(methods_list):
    print(method)
    # suptitle in axis:
    for j in range(n_param_combinations):
        axes[j, method_idx].set_title(method)

    # set color for KDEs
    # rgba = cmap(norm(n_samples_in_obs))

    if method != "True posterior":
        # load journal
        filename = inference_folder + f"{method}_MCMC_burnin_{burnin}_n-samples_{n_samples}_n-sam-per-param_" \
                                      f"{n_samples_per_param}_n-sam-in-obs_{n_samples_in_obs}"
        if model == "MA2":
            if method == "KernelScore":
                filename += f"_weight_{640.0}"
            elif method == "EnergyScore":
                filename += f"_weight_{30.0}"
        elif model == "MG1":
            if method == "KernelScore":
                filename += f"_weight_{7000.0}"
            elif method == "EnergyScore":
                filename += f"_weight_{50.0}"

        journal = Journal.fromFile(filename + ".jnl")

        # extract posterior samples
        post_samples = extract_par_fcn(journal)

        # thinning
        post_samples = post_samples[::thin, :]

        # create dataframe for seaborn
        df = pd.DataFrame(post_samples, columns=names)
    else:
        # load the true posterior data here
        if model == "MG1":
            post_samples = np.load(
                true_posterior_folder + "n-samples_2610000_burnin_290000_n-sam-per-param_1.npy")
            # thinning (notice that has more than 2 million samples -> thin a lot)
            post_samples = post_samples[::1000, :]
            # create dataframe for seaborn
            df = pd.DataFrame(post_samples, columns=names)
        elif model == "MA2":
            post_samples = np.load(
                true_posterior_folder + "n-samples_10000_burnin_10000_n-sam-per-param_1.npy")
            # thinning
            post_samples = post_samples[::thin, :]
            # create dataframe for seaborn
            df = pd.DataFrame(post_samples, columns=names)

    cov_matrix = np.cov(post_samples.transpose(1, 0))
    # print("Covariance matrix:", cov_matrix)
    print("Trace of covariance", np.trace(cov_matrix))

    # now need to make KDE and plot in the correct panel, with different color corresponding to different n_obs
    ax_counter = 0

    for x in range(n_params):
        for y in range(0, x):
            # this plots the posterior samples
            if show_samples:
                axes[ax_counter, method_idx].plot(post_samples[y], post_samples[x], '.k', markersize='1')

            xmin, xmax = ranges_parameters[names[y]]
            ymin, ymax = ranges_parameters[names[x]]

            # use seaborn
            if fill:
                try:
                    sns.kdeplot(data=df, x=names[y], y=names[x], fill=True, thresh=0.05, levels=20, cmap=cmap_name,
                                ax=axes[ax_counter, method_idx])
                except np.linalg.LinAlgError:
                    pass
            else:
                try:
                    sns.kdeplot(data=df, x=names[y], y=names[x], fill=False, thresh=0.05, levels=15, cmap=cmap_name,
                                ax=axes[ax_counter, method_idx])
                except np.linalg.LinAlgError:
                    pass

            if true_parameter_values is not None:
                axes[ax_counter, method_idx].plot([xmin, xmax], [true_parameter_values[x], true_parameter_values[x]],
                                                  "green", markersize='20', linestyle='dotted')
                axes[ax_counter, method_idx].plot([true_parameter_values[y], true_parameter_values[y]], [ymin, ymax],
                                                  "green", markersize='20', linestyle='dotted')

            axes[ax_counter, method_idx].set_xlim([xmin, xmax])
            axes[ax_counter, method_idx].set_ylim([ymin, ymax])

            axes[ax_counter, method_idx].set_ylabel(param_names_latex[x], size=label_size)
            axes[ax_counter, method_idx].set_xlabel(param_names_latex[y], size=label_size)

            axes[ax_counter, method_idx].tick_params(axis='both', which='major', labelsize=ticks_size)
            # axes[ax_counter, method_idx].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            # axes[ax_counter, method_idx].yaxis.offsetText.set_fontsize(ticks_size)
            axes[ax_counter, method_idx].yaxis.set_visible(True)
            # axes[ax_counter, method_idx].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            # axes[ax_counter, method_idx].yaxis.offsetText.set_fontsize(ticks_size)
            axes[ax_counter, method_idx].xaxis.set_visible(True)

            ax_counter += 1

    if model == "MA2":
        # fill region out of prior range
        for j in range(n_param_combinations):
            axes[j, method_idx].fill_between(x=[-2, 0], y1=[-1, -1], y2=[1, -1], facecolor="red", alpha=.2)
            axes[j, method_idx].fill_between(x=[0, 2], y1=[-1, -1], y2=[-1, 1], facecolor="red", alpha=.2)

fig.tight_layout()  # for spacing

plt.savefig(inference_folder + f"Overall_plot_burnin_{burnin}_n-samples_{n_samples}_n-sam-per-param_"
                               f"{n_samples_per_param}_thinning_{thin}_{cmap_name}_{'fill' if fill else 'nofill'}.pdf")
