import numpy as np
import os
import sys
from scipy.stats import gaussian_kde
import seaborn as sns
from abcpy.output import Journal

sys.path.append(os.getcwd())  # add the root of this project to python path

from src.parsers import parser_plots_marginals_and_traces
from src.utils import define_default_folders_scoring_rules, extract_params_from_journal_gk, \
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
n_samples_per_param = args.n_samples_per_param

# the maximum n_observation lists we have been able to obtain:
n_samples_in_obs_list_max = [1] + np.arange(5, 105, 5).tolist()

if model not in ["g-and-k", "univariate_g-and-k", "univariate_Cauchy_g-and-k", "Cauchy_g-and-k"]:
    raise NotImplementedError

model_abc, statistics, param_bounds = instantiate_model(model)
param_names = list(param_bounds.keys())
param_names_latex = [r'$A$', r'$B$', r'$g$', r'$k$', r'$\rho$']

default_root_folder = define_default_folders_scoring_rules()
if results_folder is None:
    results_folder = default_root_folder[model]

observation_folder = results_folder + '/' + args.observation_folder + '/'
inference_folder = results_folder + '/' + args.inference_folder + '/'

if model in ["univariate_g-and-k", "univariate_Cauchy_g-and-k"]:
    methods_list = ["SyntheticLikelihood", "KernelScore", "EnergyScore"]
    n_samples_in_obs_list_lists = [n_samples_in_obs_list_max] * 3
    n_params = 4
    extract_par_fcn = extract_params_from_journal_gk
elif model in ["g-and-k", "Cauchy_g-and-k"]:
    if model == "g-and-k":
        methods_list = ["SyntheticLikelihood", "semiBSL", "KernelScore", "EnergyScore"]
        n_samples_in_obs_list_lists = [[1, 5, 10]] + [[1]] + [n_samples_in_obs_list_max] * 2
    else:
        methods_list = ["KernelScore", "EnergyScore"]
        n_samples_in_obs_list_lists = [n_samples_in_obs_list_max] * 2
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
norm = matplotlib.colors.Normalize(vmin=1, vmax=max(n_samples_in_obs_list_max))

acc_rates_matrix = np.zeros((len(methods_list), len(n_samples_in_obs_list_max)))

for method_idx, method in enumerate(methods_list):
    print(method)
    axes = [fig.add_subplot(len(methods_list), n_params, i + n_params * method_idx + 1) for i in range(n_params)]

    n_samples_in_obs_list = n_samples_in_obs_list_lists[method_idx]
    # plot exact par values:
    for j in range(n_params):
        if model not in ["univariate_Cauchy_g-and-k", "Cauchy_g-and-k"]:
            axes[j].axvline(theta_obs[j], color="green")
        axes[j].set_xlabel(param_names_latex[j])
        if j < 4:
            axes[j].set_xticks(range(5))

    for n_samples_in_obs_idx, n_samples_in_obs in enumerate(n_samples_in_obs_list):
        print(n_samples_in_obs)
        # set color for KDEs
        rgba = cmap(norm(n_samples_in_obs))

        # load journal
        filename = inference_folder + f"{method}_MCMC_burnin_{burnin}_n-samples_{n_samples}_n-sam-per-param_" \
                                      f"{n_samples_per_param}_n-sam-in-obs_{n_samples_in_obs}"

        journal = Journal.fromFile(filename + ".jnl")

        # extract posterior samples
        post_samples = extract_par_fcn(journal)

        # thinning
        post_samples = post_samples[::thin, :]

        # store acc rate:
        acc_rates_matrix[method_idx, n_samples_in_obs_idx] = journal.configuration["acceptance_rates"][0]

        # now need to make KDE and plot in the correct panel, with different color corresponding to different n_obs
        for j in range(n_params):
            xmin, xmax = param_bounds[param_names[j]]
            positions = np.linspace(xmin, xmax, 100)
            gaussian_kernel = gaussian_kde(post_samples[:, j])
            axes[j].plot(positions, gaussian_kernel(positions), color=rgba, linestyle='solid', lw="1",
                         alpha=1, label="Density")

# fig.tight_layout()  # for spacing
fig.subplots_adjust(hspace=0.5)
plt.savefig(inference_folder + f"Overall_plot_burnin_{burnin}_n-samples_{n_samples}_n-sam-per-param_"
                               f"{n_samples_per_param}_thinning_{thin}_{cmap_name}.pdf")

for i in range(acc_rates_matrix.shape[1]):
    strings = [f"${acc_rate:.3f}$" for acc_rate in acc_rates_matrix[:, i]]
    strings2 = [" & " for i in range(len(strings))]
    strings3 = strings2 + strings
    strings3[::2] = strings2
    strings3[1::2] = strings
    print("".join(strings3))
