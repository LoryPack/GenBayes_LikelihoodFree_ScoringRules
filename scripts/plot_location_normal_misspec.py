import numpy as np
import os
import sys

from abcpy.output import Journal

sys.path.append(os.getcwd())  # add the root of this project to python path

from src.parsers import parser_plots_marginals_and_traces
from src.utils import define_default_folders_scoring_rules, extract_params_from_journal_normal, kde_plot
from src.models import instantiate_model
import matplotlib.pyplot as plt

parser = parser_plots_marginals_and_traces()
parser.add_argument('--n_samples_in_obs', type=int, default=100)
parser.add_argument('--true_posterior_folder', type=str, default="true_posterior")
args = parser.parse_args()

model = args.model
results_folder = args.root_folder
n_samples = args.n_samples
burnin = args.burnin
thin = args.thin
n_samples_per_param = args.n_samples_per_param
n_samples_in_obs = args.n_samples_in_obs

if model not in ["normal_location_misspec"]:
    raise NotImplementedError

model_abc, statistics, param_bounds = instantiate_model(model)
param_names_latex = [r'$\theta$']

default_root_folder = define_default_folders_scoring_rules()
if results_folder is None:
    results_folder = default_root_folder[model]

observation_folder = results_folder + '/' + args.observation_folder + '/'
inference_folder = results_folder + '/' + args.inference_folder + '/'
true_posterior_folder = results_folder + '/' + args.true_posterior_folder + '/'

if model == "normal_location_misspec":
    methods_list = ["Standard Bayes", "KernelScore", "EnergyScore"]
    n_params = 1
    extract_par_fcn = extract_params_from_journal_normal
    outliers_list = [1, 3.0, 5.0, 7.0, 10.0, 20.0]
    epsilon_list = [0, 0.1, 0.2]

# load observation
theta_obs = np.load(observation_folder + "theta_obs.npy")
assert theta_obs.shape[0] == n_params

type_plots = 2
n_methods = len(methods_list)

fig, axes = plt.subplots(figsize=(n_methods * 3, type_plots * 2.5), nrows=type_plots, ncols=n_methods, sharex="all",
                         sharey="all")
axes[0, 0].set_ylim(ymin=0, ymax=5)

# PLOT TYPE 1:

# first plot: fixed outlier type and increasing epsilon:
outlier = 10.0
for method_idx, method in enumerate(methods_list):
    print(method)

    # plot exact par values:
    axes[0, method_idx].axvline(theta_obs[0], color="green", ls="dashed")
    axes[0, method_idx].set_title(method)

    for epsilon in epsilon_list:
        print(epsilon)
        if epsilon == 0:
            # if epsilon == 0 -> we have done simulations fixing outlier to 1 (as it does not matter)
            actual_outlier = 1
        else:
            actual_outlier = outlier
        if method != "Standard Bayes":
            # load journal
            filename = inference_folder + f"{method}_MCMC_burnin_{burnin}_n-samples_{n_samples}_n-sam-per-param_" \
                                          f"{n_samples_per_param}_n-sam-in-obs_{n_samples_in_obs}_epsilon_{epsilon:.1f}_outliers_loc_{actual_outlier:.1f}"
            journal = Journal.fromFile(filename + ".jnl")
            print("Std dev", np.sqrt(journal.posterior_cov()[0]))

            # extract posterior samples
            post_samples = extract_par_fcn(journal)
        else:
            post_samples = np.load(true_posterior_folder + f"n-samples_{10000}_burnin_{10000}" \
                                                           f"_n-sam-per-param_{n_samples_in_obs}" \
                                                           f"_epsilon_{epsilon:.1f}_outliers_loc_{actual_outlier:.1f}.npy")
            print("Std dev", np.std(post_samples.reshape(-1)))

        # thinning
        post_samples = post_samples[::thin, :]

        # now need to make KDE and plot in the correct panel, with different color corresponding to different n_obs
        try:
            kde_plot(axes[0, method_idx], post_samples, r"$\epsilon = {}$".format(epsilon))
        except np.linalg.LinAlgError:
            pass

axes[0, 0].set_ylabel(r"Generalized posteriors" "\n" "$z={}$".format(int(outlier)))
axes[0, -1].legend()  # add legend in one panel only

# second plot: fixed epsilon and different outlier types:
epsilon = 0.1
for method_idx, method in enumerate(methods_list):
    print(method)

    # plot exact par values:
    axes[1, method_idx].axvline(theta_obs[0], color="green", ls="dashed")
    # axes[1, method_idx].set_title(method)
    axes[1, method_idx].set_xlabel(param_names_latex[0])

    for outlier_idx, outlier in enumerate(outliers_list):
        if outlier == 1:
            # if outlier == 1 -> we have done simulations fixing epsilon to 0 (as it does not matter)
            actual_epsilon = 0
        else:
            actual_epsilon = epsilon
        if method != "Standard Bayes":
            # load journal
            filename = inference_folder + f"{method}_MCMC_burnin_{burnin}_n-samples_{n_samples}_n-sam-per-param_" \
                                          f"{n_samples_per_param}_n-sam-in-obs_{n_samples_in_obs}_epsilon_{actual_epsilon:.1f}_outliers_loc_{outlier:.1f}"
            journal = Journal.fromFile(filename + ".jnl")
            # print("Covariance", journal.configuration["acceptance_rates"][0])
            # extract posterior samples
            post_samples = extract_par_fcn(journal)
        else:
            post_samples = np.load(true_posterior_folder + f"n-samples_{10000}_burnin_{10000}" \
                                                           f"_n-sam-per-param_{n_samples_in_obs}" \
                                                           f"_epsilon_{actual_epsilon:.1f}_outliers_loc_{outlier:.1f}.npy")

        # thinning
        post_samples = post_samples[::thin, :]

        # now need to make KDE and plot in the correct panel, with different color corresponding to different n_obs
        try:
            kde_plot(axes[1, method_idx], post_samples,
                     r"$z =  {}$".format(int(outlier)),
                     color="C{}".format(0 if outlier_idx == 0 else outlier_idx + 2))
        except np.linalg.LinAlgError:
            pass

axes[1, 0].set_ylabel(r"Generalized posteriors " "\n" " $\epsilon={}$".format(epsilon))
axes[1, -1].legend()  # add legend in one panel only

# fig.tight_layout()  # for spacing
# fig.subplots_adjust(hspace=0.5)
plt.savefig(inference_folder + f"Overall_plot_burnin_{burnin}_n-samples_{n_samples}_n-sam-per-param_"
                               f"{n_samples_per_param}_thinning_{thin}.pdf")

# PLOT TYPE 2

fig, axes = plt.subplots(figsize=((len(outliers_list) - 1) * 3, (len(epsilon_list) - 1) * 2.5),
                         nrows=len(epsilon_list) - 1, ncols=len(outliers_list) - 1, sharex="col", sharey="col")

if len(outliers_list) - 1 == 1:
    axes = axes.reshape(-1, 1)
elif len(epsilon_list) - 1 == 1:
    axes = axes.reshape(1, -1)

axes[0, 0].set_ylim(ymin=0, ymax=5)

std_matrix = np.zeros((len(epsilon_list[1:]), len(outliers_list[1:]), len(methods_list)))

for eps_index, epsilon in enumerate(epsilon_list[1:]):
    # print(epsilon)
    for outlier_index, outlier in enumerate(outliers_list[1:]):
        # print(outlier)
        for method_idx, method in enumerate(methods_list):
            # print(method)
            if method != "Standard Bayes":
                # load journal
                filename = inference_folder + f"{method}_MCMC_burnin_{burnin}_n-samples_{n_samples}_n-sam-per-param_" \
                                              f"{n_samples_per_param}_n-sam-in-obs_{n_samples_in_obs}_epsilon_{epsilon:.1f}_outliers_loc_{outlier:.1f}"
                journal = Journal.fromFile(filename + ".jnl")
                # print("Std dev", np.sqrt(journal.posterior_cov()[0]))
                std_matrix[eps_index, outlier_index, method_idx] = np.sqrt(journal.posterior_cov()[0])

                # extract posterior samples
                post_samples = extract_par_fcn(journal)
            else:
                post_samples = np.load(true_posterior_folder + f"n-samples_{10000}_burnin_{10000}" \
                                                               f"_n-sam-per-param_{n_samples_in_obs}" \
                                                               f"_epsilon_{epsilon:.1f}_outliers_loc_{outlier:.1f}.npy")
                # print("Std dev", np.std(post_samples.reshape(-1)))
                std_matrix[eps_index, outlier_index, method_idx] = np.std(post_samples.reshape(-1))

            try:
                kde_plot(axes[eps_index, outlier_index], post_samples, method)
            except np.linalg.LinAlgError:
                pass

            axes[eps_index, outlier_index].axvline(theta_obs[0], color="green", ls="dashed")
            axes[eps_index, outlier_index].set_title(r"$\epsilon={}, z = {}$ ".format(epsilon, int(outlier)))

axes[-1, -1].legend()

for outlier_index, outlier in enumerate(outliers_list[1:]):
    axes[-1, outlier_index].set_xlabel(param_names_latex[0])

for eps_index, epsilon in enumerate(epsilon_list[1:]):
    axes[eps_index, 0].set_ylabel(r"Generalized posteriors")

plt.savefig(inference_folder + f"Overall_plot_burnin_{burnin}_n-samples_{n_samples}_n-sam-per-param_"
                               f"{n_samples_per_param}_thinning_{thin}_2.pdf")

for method_idx, method in enumerate(methods_list):
    print(method)
    for eps_index, epsilon in enumerate(epsilon_list[1:]):
        print(epsilon)
        strings = [f"${std:.3f}$" for std in std_matrix[eps_index, :, method_idx]]
        strings2 = [" & " for i in range(len(strings))]
        strings3 = strings2 + strings
        strings3[::2] = strings2
        strings3[1::2] = strings
        print("".join(strings3))
