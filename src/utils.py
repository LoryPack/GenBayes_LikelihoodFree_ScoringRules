import abcpy.statistics
import matplotlib.pyplot as plt
import numpy as np
from abcpy.acceptedparametersmanager import AcceptedParametersManager
from abcpy.approx_lhd import SynLikelihood, SemiParametricSynLikelihood
from abcpy.inferences import InferenceMethod
from abcpy.output import Journal
from scipy.stats import gaussian_kde
from theano import tensor as tt

from src.scoring_rules import EnergyScore, KernelScore


def transform_R_to_theta_MA2(R1, R2):
    R1_sqrt = R1 ** 0.5
    theta1 = ((4 - 2 * R2) * R1_sqrt - 2)
    theta2 = (1 - 2 * R2 * R1_sqrt)
    return theta1, theta2


# the following is based on the RejectionABC; I use it to sample from prior.
class DrawFromPrior(InferenceMethod):
    model = None
    rng = None
    n_samples = None
    backend = None

    n_samples_per_param = None  # this needs to be there otherwise it does not instantiate correctly

    def __init__(self, root_models, backend, seed=None, discard_too_large_values=True):
        self.model = root_models
        self.backend = backend
        self.rng = np.random.RandomState(seed)
        self.discard_too_large_values = discard_too_large_values
        # An object managing the bds objects
        self.accepted_parameters_manager = AcceptedParametersManager(self.model)

    def sample(self, n_samples, n_samples_per_param):
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param
        self.accepted_parameters_manager.broadcast(self.backend, 1)

        # now generate an array of seeds that need to be different one from the other. One way to do it is the
        # following.
        # Moreover, you cannot use int64 as seeds need to be < 2**32 - 1. How to fix this?
        # Note that this is not perfect; you still have small possibility of having some seeds that are equal. Is there
        # a better way? This would likely not change much the performance
        # An idea would be to use rng.choice but that is too
        seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=n_samples, dtype=np.uint32)
        # check how many equal seeds there are and remove them:
        sorted_seed_arr = np.sort(seed_arr)
        indices = sorted_seed_arr[:-1] == sorted_seed_arr[1:]
        # print("Number of equal seeds:", np.sum(indices))
        if np.sum(indices) > 0:
            # the following can be used to remove the equal seeds in case there are some
            sorted_seed_arr[:-1][indices] = sorted_seed_arr[:-1][indices] + 1
        # print("Number of equal seeds after update:", np.sum(sorted_seed_arr[:-1] == sorted_seed_arr[1:]))
        rng_arr = np.array([np.random.RandomState(seed) for seed in sorted_seed_arr])
        rng_pds = self.backend.parallelize(rng_arr)

        parameters_simulations_pds = self.backend.map(self._sample_parameter, rng_pds)
        parameters_simulations = self.backend.collect(parameters_simulations_pds)
        parameters, simulations = [list(t) for t in zip(*parameters_simulations)]

        parameters = np.array(parameters)
        simulations = np.array(simulations)

        parameters = parameters.reshape((parameters.shape[0], parameters.shape[1]))
        if len(simulations.shape) == 4:
            simulations = simulations.reshape((simulations.shape[0], simulations.shape[2], simulations.shape[3],))
        elif len(simulations.shape) == 5:
            simulations = simulations.reshape((simulations.shape[0], simulations.shape[2], simulations.shape[4],))

        return parameters, simulations

    def sample_in_chunks(self, n_samples, n_samples_per_param, max_chunk_size=10 ** 4):
        """This splits the data generation in chunks. It is useful when generating large datasets with MPI backend,
        which gives an overflow error due to pickling very large objects."""
        parameters_list = []
        simulations_list = []
        samples_to_sample = n_samples
        while samples_to_sample > 0:
            parameters_part, simulations_part = self.sample(min(samples_to_sample, max_chunk_size), n_samples_per_param)
            samples_to_sample -= max_chunk_size
            parameters_list.append(parameters_part)
            simulations_list.append(simulations_part)
        parameters = np.concatenate(parameters_list)
        simulations = np.concatenate(simulations_list)
        return parameters, simulations

    def _sample_parameter(self, rng, npc=None):
        ok_flag = False

        while not ok_flag:
            self.sample_from_prior(rng=rng)
            theta = self.get_parameters(self.model)
            y_sim = self.simulate(self.n_samples_per_param, rng=rng, npc=npc)

            # if there are no potential infinities there (or if we do not check for those).
            # For instance, Lorenz model may give too large values sometimes (quite rarely).
            if np.sum(np.isinf(np.array(y_sim).astype("float32"))) > 0 and self.discard_too_large_values:
                print("y_sim contained too large values for float32; simulating again.")
            else:
                ok_flag = True

        return theta, y_sim


def define_default_folders():
    default_root_folder = {"MA2": "results/MA2",
                           "univariate_g-and-k": "results/univariate_g-and-k",
                           "g-and-k": "results/g-and-k",
                           "univariate_Cauchy_g-and-k": "results/univariate_Cauchy_g-and-k",
                           "Cauchy_g-and-k": "results/Cauchy_g-and-k",
                           "MG1": "results/MG1",
                           "normal_location_misspec": "results/normal_location_misspec",
                           "RecruitmentBoomBust": "results/RecruitmentBoomBust",
                           "Lorenz96": "results/Lorenz96"}

    return default_root_folder


def define_exact_param_values():
    """These are according to the original parametrization, for models which need to be reparametrized to obtain the
    right prior distribution."""
    exact_par_values = {"MA2": [0.6, 0.2],
                        "univariate_g-and-k": [3, 1.5, 0.5, 1.5, -0.3],  # notice rho (last one) not used here.
                        "g-and-k": [3, 1.5, 0.5, 1.5, -0.3],
                        "univariate_Cauchy_g-and-k": [],
                        "Cauchy_g-and-k": [],
                        "MG1": [1, 5, 0.2],
                        "normal_location_misspec": [1],
                        "RecruitmentBoomBust": [0.4, 50.0, 0.09, 0.05],
                        "Lorenz96": [2, 0.8, 1.7, 0.4]}
    return exact_par_values


def dict_implemented_scoring_rules():
    return {"SyntheticLikelihood": SynLikelihood,
            "semiBSL": SemiParametricSynLikelihood,
            "KernelScore": KernelScore,
            "EnergyScore": EnergyScore}


class LogLike(tt.Op):
    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """
    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, observation):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.observation = observation

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.observation)

        outputs[0][0] = np.array(logl)  # output the log-likelihood


def load_journal_if_flag(flag, filename):
    if flag:
        try:
            journal = Journal.fromFile(filename)
            print("Loaded journal")
            return True, journal
        except FileNotFoundError:
            return False, None
    else:
        return False, None


def transform_journal_MG1(journal):
    """transform journal with the reparametrized variable to the original parametrization"""
    # loop over the outer list;
    for i in range(len(journal.names_and_parameters)):
        journal.names_and_parameters[i]["theta2"] = []  # empty list
        for j in range(len(journal.names_and_parameters[i]["theta1"])):
            journal.names_and_parameters[i]["theta2"].append([journal.names_and_parameters[i]["theta1"][j][0] +
                                                              journal.names_and_parameters[i]["theta2_minus_theta1"][j][
                                                                  0]])
        # drop the original parameter:
        journal.names_and_parameters[i].pop("theta2_minus_theta1")

    # similar for the accepted_parameters:
    new_accepted_parameters = []
    for i in range(len(journal.accepted_parameters)):
        new_accepted_parameters.append([])
        for j in range(len(journal.accepted_parameters[i])):
            # order this as the dictionary above: theta1, theta3 and theta2
            new_accepted_parameters[i].append(
                [journal.accepted_parameters[i][j][0], journal.accepted_parameters[i][j][2],
                 journal.accepted_parameters[i][j][1] + journal.accepted_parameters[i][j][0]])

    journal.accepted_parameters = new_accepted_parameters
    return journal


def transform_journal_MA2(journal):
    """transform journal with the reparametrized variable to the original parametrization"""
    # loop over the outer list;
    new_accepted_parameters = []

    for i in range(len(journal.names_and_parameters)):
        new_accepted_parameters.append([])
        journal.names_and_parameters[i]["theta1"] = []  # empty list
        journal.names_and_parameters[i]["theta2"] = []  # empty list
        for j in range(len(journal.names_and_parameters[i]["R1"])):
            theta1, theta2 = transform_R_to_theta_MA2(journal.names_and_parameters[i]["R1"][j][0],
                                                      journal.names_and_parameters[i]["R2"][j][0])
            journal.names_and_parameters[i]["theta1"].append([theta1])
            journal.names_and_parameters[i]["theta2"].append([theta2])
            new_accepted_parameters[i].append([theta1, theta2])

        # drop the original parameter:
        journal.names_and_parameters[i].pop("R1")
        journal.names_and_parameters[i].pop("R2")

    journal.accepted_parameters = new_accepted_parameters
    return journal


def transform_journal(journal, model_name):
    if model_name == "MG1":
        journal = transform_journal_MG1(journal)
    elif model_name == "MA2":
        journal = transform_journal_MA2(journal)
    return journal


def extract_params_from_journal_normal(jrnl, step=None):
    params = jrnl.get_parameters(step)
    return np.array(params['theta']).reshape(-1, 1)


def extract_params_from_journal_gk(jrnl, step=None):
    params = jrnl.get_parameters(step)
    return np.concatenate((np.array(params['A']).reshape(-1, 1), np.array(params['B']).reshape(-1, 1),
                           np.array(params['g']).reshape(-1, 1), np.array(params['k']).reshape(-1, 1)), axis=1)


def extract_params_from_journal_multiv_gk(jrnl, step=None):
    params = jrnl.get_parameters(step)
    return np.concatenate((np.array(params['A']).reshape(-1, 1), np.array(params['B']).reshape(-1, 1),
                           np.array(params['g']).reshape(-1, 1), np.array(params['k']).reshape(-1, 1),
                           np.array(params['rho']).reshape(-1, 1)), axis=1)


def extract_params_from_journal_MG1(jrnl, step=None):
    params = jrnl.get_parameters(step)
    return np.concatenate((np.array(params['theta1']).reshape(-1, 1), np.array(params['theta2']).reshape(-1, 1),
                           np.array(params['theta3']).reshape(-1, 1)), axis=1)


def extract_params_from_journal_MA2(jrnl, step=None):
    params = jrnl.get_parameters(step)
    return np.concatenate((np.array(params['theta1']).reshape(-1, 1), np.array(params['theta2']).reshape(-1, 1)),
                          axis=1)


def extract_params_from_journal_RBB(jrnl, step=None):
    params = jrnl.get_parameters(step)
    return np.concatenate((np.array(params['r']).reshape(-1, 1), np.array(params['kappa']).reshape(-1, 1),
                           np.array(params['alpha']).reshape(-1, 1), np.array(params['beta']).reshape(-1, 1)),
                          axis=1)


def extract_params_from_journal_Lorenz96(jrnl, step=None):
    params = jrnl.get_parameters(step)
    return np.concatenate((np.array(params['theta1']).reshape(-1, 1), np.array(params['theta2']).reshape(-1, 1),
                           np.array(params['sigma_e']).reshape(-1, 1), np.array(params['phi']).reshape(-1, 1)),
                          axis=1)


def heuristics_estimate_w(model_abc, observation, target_SR, reference_SR, backend, n_theta=100,
                          n_theta_prime=100, n_samples_per_param=100, seed=42, return_values=["median"]):
    """Here observation is a list, and all of them are used at once in the SR. """
    # target_scoring_rule = dict_implemented_scoring_rules()[target_SR](statistics, **target_SR_kwargs)
    # reference_scoring_rule = dict_implemented_scoring_rules()[reference_SR](statistics)
    target_scoring_rule = target_SR
    reference_scoring_rule = reference_SR

    # generate the values of theta from prior
    theta_vect, simulations_theta_vect = DrawFromPrior([model_abc], backend, seed=seed).sample(n_theta,
                                                                                               n_samples_per_param)
    # generate the values of theta_prime from prior
    theta_prime_vect, simulations_theta_prime_vect = DrawFromPrior([model_abc], backend, seed=seed + 1).sample(
        n_theta_prime, n_samples_per_param)

    # now need to estimate w; here we assume the reference post
    # corresponding to the reference sr has no weight factor, so that log BF =
    w_estimates = np.zeros((n_theta, n_theta_prime))
    target_sr_1 = np.zeros((n_theta))
    reference_sr_1 = np.zeros((n_theta))
    target_sr_2 = np.zeros((n_theta_prime))
    reference_sr_2 = np.zeros((n_theta_prime))
    for i in range(n_theta):
        simulations_theta_i = simulations_theta_vect[i]
        simulations_theta_i = [data for data in simulations_theta_i]  # convert to list
        target_sr_1[i] = target_scoring_rule.loglikelihood(observation, simulations_theta_i)
        reference_sr_1[i] = reference_scoring_rule.loglikelihood(observation, simulations_theta_i)

    for j in range(n_theta_prime):
        simulations_theta_prime_j = simulations_theta_prime_vect[j]
        simulations_theta_prime_j = [data for data in simulations_theta_prime_j]  # convert to list
        target_sr_2[j] = target_scoring_rule.loglikelihood(observation, simulations_theta_prime_j)
        reference_sr_2[j] = reference_scoring_rule.loglikelihood(observation, simulations_theta_prime_j)

    # actually loglik is (- SR), but we have - factor both in numerator and denominator -> doesn't matter
    for i in range(n_theta):
        for j in range(n_theta_prime):
            w_estimates[i, j] = (reference_sr_1[i] - reference_sr_2[j]) / (
                    target_sr_1[i] - target_sr_2[j])

    w_estimates = w_estimates.flatten()
    print("There were ", np.sum(np.isnan(w_estimates)), " nan values out of ", n_theta * n_theta_prime)
    w_estimates = w_estimates[~np.isnan(w_estimates)]  # drop nan values

    return_list = []
    if "median" in return_values:
        return_list.append(np.median(w_estimates))
    if "mean" in return_values:
        return_list.append(np.mean(w_estimates))

    return return_list[0] if len(return_list) == 1 else return_list


def estimate_bandwidth(model_abc, statistics, backend, n_theta=100, n_samples_per_param=100, seed=42,
                       return_values=["median"]):
    """Estimate the bandwidth for the gaussian kernel in KernelSR. Specifically, it generates n_samples_per_param
    simulations for each theta, then computes the pairwise distances and takes the median of it. The returned value
    is the median (by default; you can also compute the mean if preferred) of the latter over all considered values
    of theta.  """

    # generate the values of theta from prior
    theta_vect, simulations_theta_vect = DrawFromPrior([model_abc], backend, seed=seed).sample(n_theta,
                                                                                               n_samples_per_param)
    if not isinstance(statistics, abcpy.statistics.Identity):
        simulations_theta_vect_list = [x for x in simulations_theta_vect.reshape(-1, simulations_theta_vect.shape[-1])]
        simulations_theta_vect = statistics.statistics(simulations_theta_vect_list)
        simulations_theta_vect = simulations_theta_vect.reshape(n_theta, n_samples_per_param,
                                                                simulations_theta_vect.shape[-1])

    print("Simulations shape for learning bandwidth", simulations_theta_vect.shape)

    distances = np.zeros((n_theta, n_samples_per_param * (n_samples_per_param - 1)))
    for theta_index in range(n_theta):
        simulations = simulations_theta_vect[theta_index]
        distances[theta_index] = np.linalg.norm(
            simulations.reshape(1, n_samples_per_param, -1) - simulations.reshape(n_samples_per_param, 1, -1), axis=-1)[
            ~np.eye(n_samples_per_param, dtype=bool)].reshape(-1)

    # distances = distances.reshape(n_theta, -1)  # reshape
    # take the median over the second index:
    distances_median = np.median(distances, axis=-1)

    return_list = []
    if "median" in return_values:
        return_list.append(np.median(distances_median.flatten()))
    if "mean" in return_values:
        return_list.append(np.mean(distances_median.flatten()))

    return return_list[0] if len(return_list) == 1 else return_list


def estimate_bandwidth_timeseries(model_abc, backend, num_vars, n_theta=100, seed=42, return_values=["median"]):
    """Estimate the bandwidth for the gaussian kernel in KernelSR. Specifically, it generates n_samples_per_param
    simulations for each theta, then computes the pairwise distances and takes the median of it. The returned value
    is the median (by default; you can also compute the mean if preferred) of the latter over all considered values
    of theta.  """

    # generate the values of theta from prior
    theta_vect, simulations_theta_vect = DrawFromPrior([model_abc], backend, seed=seed).sample(n_theta, 1)
    simulations_theta_vect = simulations_theta_vect.reshape(n_theta, num_vars, -1)  # last index is the timestep
    n_timestep = simulations_theta_vect.shape[2]

    distances_median = np.zeros(n_timestep)
    for timestep_index in range(n_timestep):
        simulations = simulations_theta_vect[:, :, timestep_index]
        distances = np.linalg.norm(
            simulations.reshape(1, n_theta, -1) - simulations.reshape(n_theta, 1, -1), axis=-1)[
            ~np.eye(n_theta, dtype=bool)].reshape(-1)
        # take the median over the second index:
        distances_median[timestep_index] = np.median(distances)

    return_list = []
    if "median" in return_values:
        return_list.append(np.median(distances_median.flatten()))
    if "mean" in return_values:
        return_list.append(np.mean(distances_median.flatten()))

    return return_list[0] if len(return_list) == 1 else return_list


def kde_plot(ax, post_samples, label, color=None):
    xmin, xmax = np.min(post_samples), np.max(post_samples)
    positions = np.linspace(xmin, xmax, 100)
    gaussian_kernel = gaussian_kde(post_samples[:, 0])
    ax.plot(positions, gaussian_kernel(positions), linestyle='solid', lw="1", alpha=1,
            label=label, color=color)


def subsample_trace(trace, size=1000):
    if len(trace) < size:
        return trace
    return trace[np.random.choice(range(len(trace)), size=size, replace=False)]


class DrawFromParamValues(InferenceMethod):
    model = None
    rng = None
    n_samples = None
    backend = None

    n_samples_per_param = None  # this needs to be there otherwise it does not instantiate correctly

    def __init__(self, root_models, backend, seed=None, discard_too_large_values=True):
        self.model = root_models
        self.backend = backend
        self.rng = np.random.RandomState(seed)
        self.discard_too_large_values = discard_too_large_values
        # An object managing the bds objects
        self.accepted_parameters_manager = AcceptedParametersManager(self.model)

    def sample(self, param_values):

        self.param_values = param_values  # list of parameter values
        self.n_samples = len(param_values)
        self.accepted_parameters_manager.broadcast(self.backend, 1)

        # now generate an array of seeds that need to be different one from the other. One way to do it is the
        # following.
        # Moreover, you cannot use int64 as seeds need to be < 2**32 - 1. How to fix this?
        # Note that this is not perfect; you still have small possibility of having some seeds that are equal. Is there
        # a better way? This would likely not change much the performance
        # An idea would be to use rng.choice but that is too
        seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=self.n_samples, dtype=np.uint32)
        # check how many equal seeds there are and remove them:
        sorted_seed_arr = np.sort(seed_arr)
        indices = sorted_seed_arr[:-1] == sorted_seed_arr[1:]
        # print("Number of equal seeds:", np.sum(indices))
        if np.sum(indices) > 0:
            # the following can be used to remove the equal seeds in case there are some
            sorted_seed_arr[:-1][indices] = sorted_seed_arr[:-1][indices] + 1
        # print("Number of equal seeds after update:", np.sum(sorted_seed_arr[:-1] == sorted_seed_arr[1:]))
        rng_arr = np.array([np.random.RandomState(seed) for seed in sorted_seed_arr])
        # zip with the param values:
        data_arr = list(zip(self.param_values, rng_arr))
        data_pds = self.backend.parallelize(data_arr)

        parameters_simulations_pds = self.backend.map(self._sample_parameter, data_pds)
        parameters_simulations = self.backend.collect(parameters_simulations_pds)
        parameters, simulations = [list(t) for t in zip(*parameters_simulations)]

        parameters = np.array(parameters).squeeze()
        simulations = np.array(simulations).squeeze()

        return parameters, simulations

    def _sample_parameter(self, data, npc=None):
        theta, rng = data[0], data[1]

        ok_flag = False

        while not ok_flag:
            # assume that we have one single model
            y_sim = self.model[0].forward_simulate(theta, 1, rng=rng)
            # self.sample_from_prior(rng=rng)
            # theta = self.get_parameters(self.model)
            # y_sim = self.simulate(1, rng=rng, npc=npc)

            # if there are no potential infinities there (or if we do not check for those).
            # For instance, Lorenz model may give too large values sometimes (quite rarely).
            if np.sum(np.isinf(np.array(y_sim).astype("float32"))) > 0 and self.discard_too_large_values:
                print("y_sim contained too large values for float32; simulating again.")
            else:
                ok_flag = True

        return theta, y_sim
