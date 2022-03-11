import argparse
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from abcpy.backends import BackendMPI, BackendDummy
from abcpy.output import Journal
from tabulate import tabulate

sys.path.append(os.getcwd())

from src.models import instantiate_model
from src.scoring_rules import estimate_kernel_score_timeseries, estimate_energy_score_timeseries
from src.utils import subsample_trace, DrawFromParamValues, extract_params_from_journal_Lorenz96, \
    estimate_bandwidth_timeseries, define_default_folders, extract_params_from_journal_RBB, \
    dict_implemented_scoring_rules, estimate_bandwidth, extract_params_from_journal_gk, \
    extract_params_from_journal_multiv_gk

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, help="The statistical model to consider.")
parser.add_argument('--root_folder', type=str, default=None)
parser.add_argument('--observation_folder', type=str, default="observations")
parser.add_argument('--inference_folder', type=str, default="inferences")
parser.add_argument('--n_samples', type=int, default=20000, help="Number of steps in MCMC.")
parser.add_argument('--burnin', type=int, default=10000, help="Burnin steps used in the saved MCMC output.")
parser.add_argument('--subsample_size', type=int, default=1000,
                    help='Number of samples to take from the MCMC results to generate the predictive distribution')
parser.add_argument('--n_samples_in_obs', type=int, default=1)
parser.add_argument('--n_samples_per_param', type=int, default=1000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--load_results_if_available', action="store_true")
parser.add_argument('--use_MPI', '-m', action="store_true", help='Use MPI to generate simulations')
parser.add_argument('--gamma_kernel_score', type=float, default=None,
                    help='The value of bandwidth used in the kernel SR. If not provided, it is determined by running '
                         'simulations from the prior.')
parser.add_argument('--no_fill', action="store_true", help="Do not fill between contourplot lines")
parser.add_argument('--CI_level', type=float, default=95,
                    help="The size of confidence interval (CI) to produce the plots. It represents the confidence "
                         "intervals in the plots vs iterations, and the position of the whiskers in the boxplots.")
parser.add_argument('--ABC_inference_folder', type=str, default="ABC_inference")
list_ABC = ["WABC", "ABC"]
parser.add_argument('--ABC_method', type=str, help='ABC approach to use; can be ' + ", ".join(list_ABC), default=None)
parser.add_argument('--ABC_steps', type=int, default=10, help="Steps for SMCABC.")
parser.add_argument('--ABC_n_samples', type=int, default=1000, help="Number of posterior samples wanted.")
parser.add_argument('--ABC_n_samples_per_param', type=int, default=100)
parser.add_argument('--sigma_kernel', type=float, default=None,
                    help='If provided, use this as bandwidth for Gaussian kernel in the overall Kernel Score.')

args = parser.parse_args()

model = args.model
results_folder = args.root_folder
n_samples = args.n_samples
burnin = args.burnin
subsample_size = args.subsample_size
n_samples_in_obs = args.n_samples_in_obs
n_samples_per_param = args.n_samples_per_param
seed = args.seed
load_results_if_available = args.load_results_if_available
use_MPI = args.use_MPI
gamma_kernel_score = args.gamma_kernel_score
no_fill = args.no_fill
CI_level = args.CI_level
ABC_method = args.ABC_method
ABC_steps = args.ABC_steps
ABC_n_samples = args.ABC_n_samples
ABC_n_samples_per_param = args.ABC_n_samples_per_param
sigma_kernel = args.sigma_kernel

np.random.seed(seed)
backend = BackendMPI() if use_MPI else BackendDummy()

# set up the default root folder and other values
default_root_folder = define_default_folders()
if results_folder is None:
    results_folder = default_root_folder[model]

observation_folder = results_folder + '/' + args.observation_folder + '/'
inference_folder = results_folder + '/' + args.inference_folder + '/'
ABC_inference_folder = results_folder + '/' + args.ABC_inference_folder + '/'

# define the model
model_abc, statistics, param_bounds = instantiate_model(model)

model_is_lorenz = "Lorenz96" in model
if "outlier" in model:
    if model_is_lorenz:
        methods_list = ["KernelScore", "EnergyScore"]
        extract_par_fcn = extract_params_from_journal_Lorenz96
        eps_list = [0.0, 0.1, 0.2]
        num_vars = 8
    elif "Recruitment" in model:
        methods_list = ["KernelScore", "EnergyScore"]
        extract_par_fcn = extract_params_from_journal_RBB
        eps_list = [0.0, 0.1, 0.2]
        num_vars = 1
else:
    eps_list = [None]
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

if ABC_method is not None:
    methods_list.append(ABC_method)

namefile_postfix_out_files = ""

if model_is_lorenz or "Recruitment" in model:
    if gamma_kernel_score is None:
        print("Set gamma from simulations from the model")
        gamma_kernel_score = estimate_bandwidth_timeseries(model_abc, backend=backend, n_theta=3, seed=seed + 1,
                                                           # n_theta=1000
                                                           num_vars=num_vars)
        print("Estimated gamma ", gamma_kernel_score)

fig_sr_timeseries, ax_sr_timeseries = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
if (model_is_lorenz or "Recruitment" in model) and ABC_method is not None:
    fig_sr_timeseries_diff, ax_sr_timeseries_diff = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
    fig_sr_timeseries_diff_energy, ax_sr_timeseries_diff_energy = plt.subplots(nrows=1, ncols=2, figsize=(12*0.7, 3*0.7))
    fig_sr_timeseries_diff_kernel, ax_sr_timeseries_diff_kernel = plt.subplots(nrows=1, ncols=2, figsize=(12*0.7, 3*0.7))
    for axes in [ax_sr_timeseries_diff, ax_sr_timeseries_diff_energy, ax_sr_timeseries_diff_kernel]:
        axes[0].axhline(0, lw=1, ls="solid", color="red")
        axes[1].axhline(0, lw=1, ls="solid", color="red")
        axes[0].set_xlabel("Timestep")
        axes[1].set_xlabel("Timestep")
        axes[0].set_title("Kernel SR")
        axes[1].set_title("Energy SR")
linestyles = ["solid", "dashed", "dashdot", "dotted"]
# linestyles = ["solid", "dashed", "dotted"]

# create tables
table_list_list = [[["eps", "Kernel Score", "Energy Score"]] for i in range(len(methods_list))]
overall_kernel_matrix = np.zeros((len(eps_list), len(methods_list)))
overall_energy_matrix = np.zeros((len(eps_list), len(methods_list)))

for epsilon_idx, epsilon in enumerate(eps_list):
    print(epsilon)

    # --- LOAD OBSERVATION --- 
    if epsilon is not None:
        if 0 < epsilon < 0.1:
            # load the actual observation
            x_obs = np.load(observation_folder + f"x_obs_epsilon_{epsilon:.2f}.npy")
        else:
            # load the actual observation
            x_obs = np.load(observation_folder + f"x_obs_epsilon_{epsilon:.1f}.npy")
    else:
        x_obs = np.load(observation_folder + "x_obs.npy")
    # reshape the observation:
    if model_is_lorenz:
        x_obs = x_obs.reshape(n_samples_in_obs, num_vars, -1)
    # print(x_obs.shape)

    if epsilon is not None:
        n_outliers = int(epsilon * n_samples_in_obs)
        true_observations = x_obs[0:n_samples_in_obs - n_outliers]
        # print(true_observations.shape)
        if n_outliers > 0:
            outliers = x_obs[n_samples_in_obs - n_outliers:]
            # print(outliers.shape)
    else:
        true_observations = x_obs

    # --- LOAD REFERENCE (ABC) RESULTS FOR PLOTTING THE DIFFERENCE OF SRS AT EACH TIMESTEP --- 
    if (model_is_lorenz or "Recruitment" in model) and ABC_method is not None:
        filename = ABC_inference_folder + f"{ABC_method}_n-steps_{ABC_steps}_n-samples_{ABC_n_samples}" \
                                          f"_n-sam-per-param_{ABC_n_samples_per_param}"
        if epsilon is not None:
            if 0 < epsilon < 0.1:
                filename += f"_epsilon_{epsilon:.2f}"
            else:
                filename += f"_epsilon_{epsilon:.1f}"

        # load now the posterior for that observation
        jrnl_ABC = Journal.fromFile(filename + ".jnl")
        params = extract_par_fcn(jrnl_ABC)
        params = subsample_trace(params, size=subsample_size)

        print("Reference results loaded correctly")

        # now simulate for all the different param values
        print("Simulate...")
        draw_from_params = DrawFromParamValues([model_abc], backend=backend, seed=seed)
        posterior_simulations_params, posterior_simulations = draw_from_params.sample(params)
        print("Done!")
        # print(posterior_simulations.shape)
        if model_is_lorenz:
            posterior_simulations_for_plots = posterior_simulations.reshape(subsample_size, num_vars, -1)
            # print(posterior_simulations_for_plots.shape)
        else:
            posterior_simulations_for_plots = posterior_simulations

        # attempt loading the results if required:
        compute_srs = True
        if load_results_if_available:
            try:
                reference_energy_sr_values_timestep = np.load(f"{filename}_energy_sr_values_timestep.npy")
                reference_energy_sr_values_cumulative = np.load(f"{filename}_energy_sr_values_cumulative.npy")
                reference_kernel_sr_values_timestep = np.load(f"{filename}_kernel_sr_values_timestep.npy")
                reference_kernel_sr_values_cumulative = np.load(f"{filename}_kernel_sr_values_cumulative.npy")
                print("Loaded previously computed scoring rule values.")
                compute_srs = False
            except FileNotFoundError:
                pass

        if compute_srs:  # compute_srs:
            # estimate the SR for that observation and cumulate over the timesteps
            energy_scores = estimate_energy_score_timeseries(posterior_simulations_for_plots, true_observations)
            reference_energy_sr_values_timestep = energy_scores[0]
            reference_energy_sr_values_cumulative = energy_scores[1]

            kernel_scores = estimate_kernel_score_timeseries(posterior_simulations_for_plots, true_observations,
                                                             sigma=gamma_kernel_score)
            reference_kernel_sr_values_timestep = kernel_scores[0]
            reference_kernel_sr_values_cumulative = kernel_scores[1]

    for method_idx, method in enumerate(methods_list):
        print(method)
        # load stuff
        if "ABC" in method:
            filename = ABC_inference_folder + f"{method}_n-steps_{ABC_steps}_n-samples_{ABC_n_samples}" \
                                              f"_n-sam-per-param_{ABC_n_samples_per_param}"
        else:
            filename = inference_folder + f"{method}_MCMC_burnin_{burnin}_n-samples_{n_samples}_n-sam-per-param_" \
                                          f"{n_samples_per_param}_n-sam-in-obs_{n_samples_in_obs}"
        if epsilon is not None:
            if 0 < epsilon < 0.1:
                filename += f"_epsilon_{epsilon:.2f}"
            else:
                filename += f"_epsilon_{epsilon:.1f}"

        # load now the posterior for that observation
        jrnl_ABC = Journal.fromFile(filename + ".jnl")
        params = extract_par_fcn(jrnl_ABC)
        params = subsample_trace(params, size=subsample_size)
        if model in ["univariate_g-and-k", "univariate_Cauchy_g-and-k"]:
            # add dummy rho values
            params = np.concatenate((params, np.zeros((params.shape[0], 1))), axis=1)

        print("Results loaded correctly")

        # now simulate for all the different param values
        print("Simulate...")
        draw_from_params = DrawFromParamValues([model_abc], backend=backend, seed=seed)
        posterior_simulations_params, posterior_simulations = draw_from_params.sample(params)
        print("Done!")
        # print(posterior_simulations.shape)
        if model_is_lorenz:
            posterior_simulations_for_plots = posterior_simulations.reshape(subsample_size, num_vars, -1)
            # print(posterior_simulations_for_plots.shape)
        else:
            posterior_simulations_for_plots = posterior_simulations

        if model_is_lorenz or "Recruitment" in model:
            # --- CREATE PREDICTIVE POSTERIOR PLOTS ---
            # comparing observations, outliers and the posterior predictive simulations

            if model_is_lorenz:
                fig, axes = plt.subplots(2, 4)
                axes = axes.flatten()
                for i in range(8):
                    axes[i].plot(
                        true_observations.reshape(n_samples_in_obs - n_outliers, num_vars, -1)[:, i].transpose(1,
                                                                                                               0),
                        alpha=0.5, color="blue")
                    if n_outliers > 0:
                        axes[i].plot(outliers.reshape(n_outliers, num_vars, -1)[:, i].transpose(1, 0), alpha=0.5,
                                     color="r")
                    # now plot the simulations from the posterior predictive:
                    axes[i].plot(posterior_simulations_for_plots[:, i].transpose(1, 0), alpha=0.1, color="green")
            else:
                fig, ax = plt.subplots(1, 1)
                ax.plot(
                    true_observations.transpose(1, 0), alpha=0.5, color="blue")
                if n_outliers > 0:
                    ax.plot(outliers.transpose(1, 0), alpha=0.5, color="r")
                # now plot the simulations from the posterior predictive:
                ax.plot(posterior_simulations_for_plots.transpose(1, 0), alpha=0.1, color="green")

            savefile_name = ABC_inference_folder if "ABC" in method else inference_folder
            if epsilon is not None:
                if 0 < epsilon < 0.1:
                    savefile_name += f"{method}_post_predictive_epsilon_{epsilon:.2f}.pdf"
                else:
                    savefile_name += f"{method}_post_predictive_epsilon_{epsilon:.1f}.pdf"
            # fig.savefig(savefile_name)

            # --- COMPUTE STEP-WISE SRS ---
            # attempt loading the results if required:
            compute_srs = True
            if load_results_if_available:
                try:
                    energy_sr_values_timestep = np.load(f"{filename}_energy_sr_values_timestep.npy")
                    energy_sr_values_cumulative = np.load(f"{filename}_energy_sr_values_cumulative.npy")
                    kernel_sr_values_timestep = np.load(f"{filename}_kernel_sr_values_timestep.npy")
                    kernel_sr_values_cumulative = np.load(f"{filename}_kernel_sr_values_cumulative.npy")
                    print("Loaded previously computed scoring rule values.")
                    compute_srs = False
                except FileNotFoundError:
                    pass

            if compute_srs:  # compute_srs:
                # estimate the SR for that observation and cumulate over the timesteps
                energy_scores = estimate_energy_score_timeseries(posterior_simulations_for_plots, true_observations)
                energy_sr_values_timestep = energy_scores[0]
                energy_sr_values_cumulative = energy_scores[1]

                kernel_scores = estimate_kernel_score_timeseries(posterior_simulations_for_plots, true_observations,
                                                                 sigma=gamma_kernel_score)
                kernel_sr_values_timestep = kernel_scores[0]
                kernel_sr_values_cumulative = kernel_scores[1]

                np.save(f"{filename}_energy_sr_values_timestep.npy", energy_sr_values_timestep)
                np.save(f"{filename}_energy_sr_values_cumulative.npy", energy_sr_values_cumulative)
                np.save(f"{filename}_kernel_sr_values_timestep.npy", kernel_sr_values_timestep)
                np.save(f"{filename}_kernel_sr_values_cumulative.npy", kernel_sr_values_cumulative)

            print(f"Cumulative time-step-wise Energy Score: {energy_sr_values_cumulative:.4f}")
            print(f"Cumulative time-step-wise Kernel Score: {kernel_sr_values_cumulative:.4f}")
            ax_sr_timeseries[0].plot(kernel_sr_values_timestep, label=f"{method}" + (
                r" $\epsilon$={:.2f}".format(epsilon) if 0 < epsilon < 0.1 else r" $\epsilon$={:.1f}".format(epsilon)),
                                     ls=linestyles[epsilon_idx], color=f"C{method_idx}", alpha=0.6)
            ax_sr_timeseries[1].plot(energy_sr_values_timestep, label=f"{method}" + (
                r" $\epsilon$={:.2f}".format(epsilon) if 0 < epsilon < 0.1 else r" $\epsilon$={:.1f}".format(epsilon)),
                                     ls=linestyles[epsilon_idx], color=f"C{method_idx}", alpha=0.6)
            ax_sr_timeseries[0].set_title("Kernel SR")
            ax_sr_timeseries[1].set_title("Energy SR")
            ax_sr_timeseries[0].legend()
            # ax_sr_timeseries[1].legend()

            if ABC_method is not None and "ABC" not in method:
                # we also create the plot for the difference SRs
                print(f"Difference cumulative time-step-wise Energy Score wrt reference:"
                      f" {reference_energy_sr_values_cumulative - energy_sr_values_cumulative:.4f}")
                print(f"Difference cumulative time-step-wise Kernel Score wrt reference:"
                      f" {reference_kernel_sr_values_cumulative - kernel_sr_values_cumulative:.4f}")
                if method == "KernelScore":
                    axes_list = [ax_sr_timeseries_diff, ax_sr_timeseries_diff_kernel]
                elif method == "EnergyScore":
                    axes_list = [ax_sr_timeseries_diff, ax_sr_timeseries_diff_energy]

                labels_list = [f"{method}", ""]
                colors_list = [f"C{method_idx}", f"C{epsilon_idx}"]
                linestyles_list = [linestyles[epsilon_idx], "solid"]

                label_eps = r" $\epsilon$={:.2f}".format(
                    epsilon) if 0 < epsilon < 0.1 else r" $\epsilon$={:.1f}".format(epsilon)
                for axes, label, color, ls in zip(axes_list, labels_list, colors_list, linestyles_list):
                    label_final = label + label_eps

                    axes[0].plot(reference_kernel_sr_values_timestep - kernel_sr_values_timestep,
                                 label=label_final, ls=ls, color=color, alpha=1)
                    axes[1].plot(reference_energy_sr_values_timestep - energy_sr_values_timestep,
                                 label=label_final, ls=ls, color=color, alpha=1)

        # --- OVERALL SCORING RULE VALUES ---
        if sigma_kernel is None:
            # need here to estimate the kernel bandwidth
            sigma_kernel = estimate_bandwidth(model_abc, statistics, backend, seed=seed + 1, n_theta=1000,
                                              n_samples_per_param=n_samples_per_param, return_values=["median"])
            print(f"Estimated sigma for kernel score {sigma_kernel['sigma']:.4f}")

        # instantiate
        energy_overall_scoring_rule = dict_implemented_scoring_rules()["EnergyScore"](statistics)
        kernel_overall_scoring_rule = dict_implemented_scoring_rules()["KernelScore"](statistics, sigma=sigma_kernel)

        # compute
        en_overall = energy_overall_scoring_rule.score([x for x in posterior_simulations],
                                                       [x for x in true_observations])
        kernel_overall = kernel_overall_scoring_rule.score([x for x in posterior_simulations],
                                                           [x for x in true_observations])

        overall_energy_matrix[epsilon_idx, method_idx] = en_overall
        overall_kernel_matrix[epsilon_idx, method_idx] = kernel_overall
        print(f"Overall Energy Score: {en_overall:.4f}")
        print(f"Overall Kernel Score: {kernel_overall:.4f}")

        table_list_list[method_idx].append([epsilon, f"{kernel_overall:.4f}", f"{en_overall:.4f}"])

if model_is_lorenz or "Recruitment" in model:
    savefile_name = inference_folder + "sr_timestep"
    fig_sr_timeseries.savefig(savefile_name + ".pdf")
    if ABC_method is not None:
        ax_sr_timeseries_diff[0].legend()
        ax_sr_timeseries_diff_energy[0].legend()
        ax_sr_timeseries_diff_kernel[0].legend()
        fig_sr_timeseries_diff.savefig(savefile_name + "diff.pdf", bbox_inches="tight")
        fig_sr_timeseries_diff_energy.savefig(savefile_name + "diff_energy.pdf", bbox_inches="tight")
        fig_sr_timeseries_diff_kernel.savefig(savefile_name + "diff_kernel.pdf", bbox_inches="tight")

# create plot for the overall SR values, for the Lorenz or RBB case
if model_is_lorenz or "Recruitment" in model:
    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    # color1 = 'tab:blue'
    ax.set_xticks(eps_list)
    # ax.tick_params(axis='y')  # , labelcolor=color1)
    ax.ticklabel_format(axis='y', style="sci", scilimits=(0, 0))  # , labelcolor=color1)
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(u"Energy Score (\u25a0)")  # , color=color2)

    # color2 = 'tab:red'
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2.tick_params(axis='y')  # , labelcolor=color2)
    ax2.ticklabel_format(axis='y', style="sci", scilimits=(0, 0))  # , labelcolor=color1)
    ax2.set_ylabel(u"Kernel Score (\u25c6)")  # , color=color2)

    markers_list = ['s', 'D', 'o', 'v']
    patch_list = []

    for method_idx, method in enumerate(methods_list):
        if method == "EnergyScore":
            # exclude from the plot
            continue
        method_name_fig = "SMC ABC" if method == "ABC" else method
        # ax.plot(eps_list, overall_energy_matrix[:, method_idx], label=method, marker=markers_list[method_idx],
        #         color=color1, lw=0)
        # ax2.plot(eps_list, overall_kernel_matrix[:, method_idx], label=method, marker=markers_list[method_idx],
        #          color=color2, lw=0)
        # patch_list.append(
        #     mlines.Line2D([], [], color='black', marker=markers_list[method_idx], label=method, linewidth=0,
        #                   markersize=10))
        ax.plot(eps_list, overall_energy_matrix[:, method_idx], label=method_name_fig, marker=markers_list[0],
                color=f"C{method_idx}", lw=1, ls="solid")
        ax2.plot(eps_list, overall_kernel_matrix[:, method_idx], label=method_name_fig, marker=markers_list[1],
                 color=f"C{method_idx}", lw=1, ls="solid")
        patch_list.append(mpatches.Patch(color=f"C{method_idx}", label=method_name_fig))

    # customize legend
    ax.legend(handles=patch_list)
    fig.savefig(inference_folder + "overall_SR_plot.pdf", bbox_inches="tight")

# print tables:
for method_idx, method in enumerate(methods_list):
    print(method)
    print(tabulate(table_list_list[method_idx], headers="firstrow", tablefmt="latex_booktabs"))

if len(methods_list) == 2:
    table_list_list = [table_list_list[0][i] + table_list_list[1][i][1:] for i in range(len(table_list_list[0]))]
elif len(methods_list) == 3:
    table_list_list = [table_list_list[0][i] + table_list_list[1][i][1:] + table_list_list[2][i][1:] for i in
                       range(len(table_list_list[0]))]
elif len(methods_list) == 4:
    table_list_list = [
        table_list_list[0][i] + table_list_list[1][i][1:] + table_list_list[2][i][1:] + table_list_list[3][i][1:] for
        i in range(len(table_list_list[0]))]
print(tabulate(table_list_list, headers="firstrow", tablefmt="latex_booktabs"))
