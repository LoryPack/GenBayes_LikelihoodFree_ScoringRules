import argparse

from src.utils import dict_implemented_scoring_rules


def parser_generate_obs_fixed_par():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        help="The statistical model to consider; can be 'MA2', 'gaussian_mixture', 'univariate_g-and-k',"
                             "'g-and-k', 'MG1', 'Lotka-Volterra', 'Ricker', 'Lorenz'")
    parser.add_argument('--root_folder', type=str, default=None)
    parser.add_argument('--observation_folder', type=str, default="observations")
    parser.add_argument('--n_observations_per_param', type=int, default=1, help='Total number of observations per '
                                                                                'parameter value')
    parser.add_argument('--sleep', type=float, default=0, help='Minutes to sleep before starting')

    return parser


def parser_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        help="The statistical model to consider.")
    list_sr = dict_implemented_scoring_rules().keys()
    parser.add_argument('method', type=str, help='Divergence to use; can be ' + ", ".join(list_sr))
    parser.add_argument('--algorithm', type=str, default="MCMC",
                        help='Inference algorithm; can be PMC or MCMC (default)')
    parser.add_argument('--sleep', type=float, default=0, help='Minutes to sleep before starting')
    parser.add_argument('--start_observation_index', type=int, default=0, help='Index to start from')
    parser.add_argument('--n_observations', type=int, default=10, help='Total number of observations.')
    parser.add_argument('--root_folder', type=str, default=None)
    parser.add_argument('--observation_folder', type=str, default="observations")
    parser.add_argument('--inference_folder', type=str, default="inferences")
    parser.add_argument('--use_MPI', '-m', action="store_true")
    parser.add_argument('--plot_post', action="store_true")
    parser.add_argument('--plot_trace', action="store_true", help="Unused with PMC")
    parser.add_argument('--load_journal_if_available', action="store_true", help="")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--steps', type=int, default=10, help="Steps for PMC; unused for MCMC.")
    parser.add_argument('--n_samples', type=int, default=1000, help="Number of posterior samples wanted; represents "
                                                                    "particles in PMC and steps in MCMC.")
    parser.add_argument('--burnin', type=int, default=1000, help="Burnin steps for MCMC; unused for PMC.")
    parser.add_argument('--n_samples_in_obs', type=int, default=1)
    parser.add_argument('--n_samples_per_param', type=int, default=100)
    parser.add_argument('--prop_size', type=float, default=1.0,
                        help="Value for the diagonal elements of the covariance matrix of the MCMC proposal.")
    parser.add_argument('--no_full_output', action="store_true",
                        help="Whether to disable full output in journal files.")
    parser.add_argument('--estimate_w', action="store_true",
                        help="Whether to estimate weight or not from a reference method.")
    parser.add_argument('--weight', type=float, default=1.0,
                        help="Weight value for SR; ignored for BSL and semiBSL; moreover, overwritten by the estimated "
                             "one if --estimate_w is used; also, overwritten by the --weight_file when that is used")
    parser.add_argument('--weight_file', type=str, default=None,
                        help="Weight file for SR; ignored for BSL and semiBSL; moreover, overwritten by the estimated "
                             "one if --estimate_w is used")
    parser.add_argument('--reference_method', type=str, default="SyntheticLikelihood")
    parser.add_argument('--adapt_proposal_cov_interval', type=int, default=None)
    return parser


def parser_plots_marginals_and_traces():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help="The statistical model to consider.")
    parser.add_argument('--root_folder', type=str, default=None)
    parser.add_argument('--observation_folder', type=str, default="observations")
    parser.add_argument('--inference_folder', type=str, default="inferences")
    parser.add_argument('--n_samples', type=int, default=100000, help="Number of steps in MCMC.")
    parser.add_argument('--burnin', type=int, default=10000, help="Burnin steps used in the saved MCMC output.")
    parser.add_argument('--thin', type=int, default=10, help="MCMC thinning parameter.")
    parser.add_argument('--n_samples_per_param', type=int, default=500)
    return parser


def parser_bivariate_plots():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help="The statistical model to consider.")
    parser.add_argument('--root_folder', type=str, default=None)
    parser.add_argument('--observation_folder', type=str, default="observations")
    parser.add_argument('--inference_folder', type=str, default="inferences")
    parser.add_argument('--true_posterior_folder', type=str, default="true_posterior")
    parser.add_argument('--n_samples', type=int, default=20000, help="Number of steps in MCMC.")
    parser.add_argument('--burnin', type=int, default=10000, help="Burnin steps used in the saved MCMC output.")
    parser.add_argument('--thin', type=int, default=10, help="MCMC thinning parameter.")
    parser.add_argument('--n_samples_per_param', type=int, default=1000)
    parser.add_argument('--no_fill', action="store_true", help="Do not fill between contourplot lines")
    parser.add_argument('--cmap_name', type=str, default="mako_r")
    return parser


def parser_estimate_w():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        help="The statistical model to consider.")
    list_sr = dict_implemented_scoring_rules().keys()
    parser.add_argument('method', type=str, help='Divergence to use; can be ' + ", ".join(list_sr))
    parser.add_argument('--sleep', type=float, default=0, help='Minutes to sleep before starting')
    parser.add_argument('--root_folder', type=str, default=None)
    parser.add_argument('--observation_folder', type=str, default="observations")
    parser.add_argument('--inference_folder', type=str, default="inferences")
    parser.add_argument('--use_MPI', '-m', action="store_true")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_theta', type=int, default=1000, help="Number of prior samples to generate to estimate w.")
    parser.add_argument('--n_samples_per_param', type=int, default=100)
    parser.add_argument('--reference_method', type=str, default="SyntheticLikelihood")
    parser.add_argument('--sigma_kernel', type=float, default=None,
                        help='If provided, use this as bandwidth for Gaussian '
                             'kernel in Kernel Score posterior')
    return parser


def parser_true_posterior():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        help="The statistical model to consider; can be 'MA2', 'gaussian_mixture', 'g-and-k', 'MG1', "
                             "'Lotka-Volterra', 'Ricker', 'Lorenz'")
    parser.add_argument('--sleep', type=float, default=0, help='Minutes to sleep before starting')
    parser.add_argument('--start_observation_index', type=int, default=0, help='Index to start from')
    parser.add_argument('--n_observations', type=int, default=10, help='Total number of observations.')
    parser.add_argument('--root_folder', type=str, default=None)
    parser.add_argument('--observation_folder', type=str, default="observations")
    parser.add_argument('--true_posterior_folder', type=str, default="true_posterior")
    parser.add_argument('--plot_post', action="store_true")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--burnin', type=int, default=10000)
    parser.add_argument('--n_samples_in_obs', type=int, default=100)
    parser.add_argument('--cores', type=int, default=2)
    return parser
