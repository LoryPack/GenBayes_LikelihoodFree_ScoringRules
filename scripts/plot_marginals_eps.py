import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from abcpy.output import Journal

sys.path.append(os.getcwd())  # add the root of this project to python path

from src.parsers import parser_plots_marginals_and_traces
from src.utils import define_default_folders, extract_params_from_journal_RBB, extract_params_from_journal_Lorenz96
from src.models import instantiate_model

parser = parser_plots_marginals_and_traces()

parser.add_argument('--n_samples_in_obs', type=int, default=100)
parser.add_argument('--ABC_inference_folder', type=str, default="ABC_inference")
list_ABC = ["WABC", "ABC"]
parser.add_argument('--ABC_method', type=str, help='ABC approach to use; can be ' + ", ".join(list_ABC), default=None)
parser.add_argument('--ABC_steps', type=int, default=10, help="Steps for SMCABC.")
parser.add_argument('--ABC_n_samples', type=int, default=1000, help="Number of posterior samples wanted.")
parser.add_argument('--ABC_n_samples_per_param', type=int, default=100)

args = parser.parse_args()

model = args.model
results_folder = args.root_folder
n_samples = args.n_samples
burnin = args.burnin
thin = args.thin
n_samples_per_param = args.n_samples_per_param
n_samples_in_obs = args.n_samples_in_obs
ABC_method = args.ABC_method
ABC_steps = args.ABC_steps
ABC_n_samples = args.ABC_n_samples
ABC_n_samples_per_param = args.ABC_n_samples_per_param

if "outliers" not in model:
    raise NotImplementedError

print(model)
model_abc, statistics, param_bounds = instantiate_model(model)
param_names = list(param_bounds.keys())

default_root_folder = define_default_folders()
if results_folder is None:
    results_folder = default_root_folder[model]

observation_folder = results_folder + '/' + args.observation_folder + '/'
inference_folder = results_folder + '/' + args.inference_folder + '/'
ABC_inference_folder = results_folder + '/' + args.ABC_inference_folder + '/'

if model == "RecruitmentBoomBust":
    n_params = 4
    # methods_list = ["KernelScore", "EnergyScore"]
    methods_list = ["KernelScore"]
    extract_par_fcn = extract_params_from_journal_RBB
    param_names_latex = [r'$r$', r'$\kappa$', r'$\alpha$', r'$\beta$']
    eps_list = [0.0, 0.1, 0.2]
elif "Lorenz96" in model:
    n_params = 4
    methods_list = ["KernelScore", "EnergyScore"]
    extract_par_fcn = extract_params_from_journal_Lorenz96
    param_names_latex = [r'$b_0$', r'$b_1$', r'$\sigma_e$', r'$\phi$']
    eps_list = [0.0, 0.1, 0.2]

n_MCMC_methods = len(methods_list)
if ABC_method is not None:
    methods_list.append(ABC_method)

# load observation
theta_obs = np.load(observation_folder + "theta_obs.npy")

# make big subplots only for putting titles
fig, big_axes = plt.subplots(figsize=(n_params * 3, len(methods_list) * 2.5), nrows=len(methods_list), ncols=1)
if len(methods_list) == 1:
    big_axes = np.atleast_1d(big_axes)

for row, big_ax in enumerate(big_axes):
    method_name_fig = "SMC ABC" if methods_list[row] == "ABC" else methods_list[row]
    big_ax.set_title(method_name_fig, fontsize=18)
    # Turn off axis lines and ticks of the big subplot
    # obs alpha is 0 in RGBA string!
    big_ax.axis('off')
    big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
    # removes the white frame
    big_ax._frameon = False

acc_rates_matrix = np.zeros((n_MCMC_methods, len(eps_list)))

axes = []
for method_idx, method in enumerate(methods_list):
    print(method)
    if "Lorenz96" in model and method_idx > 0:
        axes.append(
            [fig.add_subplot(len(methods_list), n_params, i + n_params * method_idx + 1, sharex=axes[0][i]) for i in
             range(n_params)])  # , sharey=axes[0][i]
    else:
        axes.append(
            [fig.add_subplot(len(methods_list), n_params, i + n_params * method_idx + 1) for i in range(n_params)])

for method_idx, method in enumerate(methods_list):
    print(method)

    # plot exact par values:
    for j in range(n_params):
        if model not in ["univariate_Cauchy_g-and-k", "Cauchy_g-and-k"]:
            axes[method_idx][j].axvline(theta_obs[j], color="green")
        axes[method_idx][j].set_xlabel(param_names_latex[j])

    for eps_idx, eps in enumerate(eps_list):
        print(eps)
        # set color for KDEs

        # load journal
        if "ABC" in method:
            filename = ABC_inference_folder + f"{method}_n-steps_{ABC_steps}_n-samples_{ABC_n_samples}" \
                                              f"_n-sam-per-param_{ABC_n_samples_per_param}"
        else:
            filename = inference_folder + f"{method}_MCMC_burnin_{burnin}_n-samples_{n_samples}_n-sam-per-param_" \
                                          f"{n_samples_per_param}_n-sam-in-obs_{n_samples_in_obs}"
        if 0 < eps < 0.1:
            filename += f"_epsilon_{eps:.2f}"
        else:
            filename += f"_epsilon_{eps:.1f}"

        journal = Journal.fromFile(filename + ".jnl")

        # extract posterior samples
        post_samples = extract_par_fcn(journal)

        if not "ABC" in method:
            # thinning
            post_samples = post_samples[::thin, :]

            # store acc rate:
            acc_rates_matrix[method_idx, eps_idx] = journal.configuration["acceptance_rates"][0]

        # create dataframe for seaborn
        df = pd.DataFrame(post_samples, columns=param_names)

        # now need to make KDE and plot in the correct panel, with different color corresponding to different n_obs
        for j in range(n_params):
            if 0 < eps < 0.1:
                label = r"$\epsilon={:.2f}$".format(eps)
            else:
                label = r"$\epsilon={:.1f}$".format(eps)
            # xmin, xmax = param_bounds[param_names[j]]
            # positions = np.linspace(xmin, xmax, 100)
            # gaussian_kernel = gaussian_kde(post_samples[:, j])
            # axes[method_idx][j].plot(positions, gaussian_kernel(positions), linestyle='solid', lw="1",
            #              alpha=1, label=label)
            sns.kdeplot(data=df, x=param_names[j], ax=axes[method_idx][j], label=label)
            if j > 0:
                axes[method_idx][j].set_ylabel("")

    if method_idx == 0:
        axes[method_idx][0].legend()

# fig.tight_layout()  # for spacing
fig.subplots_adjust(hspace=0.5)
plt.savefig(inference_folder + f"Different_eps_plot_burnin_{burnin}_n-samples_{n_samples}"
                               f"_thinning_{thin}.pdf")

# need to create legend and better colors.

for i in range(acc_rates_matrix.shape[1]):
    strings = [f"${acc_rate:.3f}$" for acc_rate in acc_rates_matrix[:, i]]
    strings2 = [" & " for i in range(len(strings))]
    strings3 = strings2 + strings
    strings3[::2] = strings2
    strings3[1::2] = strings
    print("".join(strings3))
