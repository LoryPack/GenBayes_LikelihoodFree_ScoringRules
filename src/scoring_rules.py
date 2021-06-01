import numpy as np
from abc import ABCMeta, abstractmethod

from abcpy.approx_lhd import Approx_likelihood


class ScoringRule(Approx_likelihood, metaclass=ABCMeta):
    """This is the abstract class for the ScoringRule which allows it to be used as an Approx_likelihood in ABCpy"""

    def __init__(self, statistics_calc, weight=1):
        """Needs to be called by each sub-class to correctly initialize the statistics_calc"""
        # call the super of Distance to initialize the statistics_calc stuff:
        Approx_likelihood.__init__(self, statistics_calc)
        self.weight = weight  # this is the weight used to multiply the scoring rule for the loglikelihood computation

    def loglikelihood(self, y_obs, y_sim):
        """Alias the score method to a loglikelihood method """
        return - self.weight * self.score(y_obs, y_sim)

    def _calculate_summary_stat(self, d1, d2):
        """This is called before computing the score to apply the summary statistics calculation."""
        # call the super method from Distance class.
        return Approx_likelihood._calculate_summary_stat(self, d1, d2)

    @abstractmethod
    def score(self, observations, simulations):
        """
        Notice: here the score is assumed to be a "penalty"; we use therefore the sign notation of Dawid, not the one
        in Gneiting and Raftery (2007).
        To be overwritten by any sub-class. Estimates the Continuous Ranked Probability Score. Here, I assume the
        observations and simulations are lists of length respectively n_obs and n_sim. Then,
        for each fixed observation the n_sim simulations are used to estimate the scoring rule. Subsequently, the
        values are summed over each of the n_obs observations.

        Parameters
        ----------
        observations: Python list
            Contains n1 data points.
        simulations: Python list
            Contains n2 data points.

        Returns
        -------
        numpy.ndarray
            The score between the simulations and the observations.

        Returns
        -------
        float
            Computed approximate loglikelihood.
        """

        raise NotImplementedError

    @abstractmethod
    def score_max(self):
        """To be overwritten by any sub-class"""
        raise NotImplementedError


class UnivariateContinuousRankedProbabilityScoreEstimate(ScoringRule):
    """Estimates the Continuous Ranked Probability Score. Here, I assume the observations and simulations are lists of
    length respectively n_obs and n_sim. Then, for each fixed observation the n_sim simulations are used to estimate the
    scoring rule. Subsequently, the values are averaged over each of the n_obs observations.
    """

    def __init__(self, statistics_calc):
        super(UnivariateContinuousRankedProbabilityScoreEstimate, self).__init__(statistics_calc)

    def score(self, observations, simulations):
        """Parameters
        ----------
        observations: Python list
            Contains n1 data points.
        simulations: Python list
            Contains n2 data points.

        Returns
        -------
        numpy.ndarray
            The score between the simulations and the observations.

        Notes
        -----
        When running an ABC algorithm, the observed dataset is always passed first to the distance. Therefore, you can
        save the statistics of the observed dataset inside this object, in order to not repeat computations.
        """

        s_observations, s_simulations = self._calculate_summary_stat(observations, simulations)

        scores = np.zeros(shape=(s_observations.shape[0]))
        # this for loop is not very efficient, can be improved; this is taken from the Euclidean distance.
        for ind1 in range(s_observations.shape[0]):
            scores[ind1] = self.estimate_CRPS_score(s_observations[ind1], s_simulations)
        return scores.sum()

    def score_max(self):
        # As the statistics are positive, the max possible value is 1
        return np.inf

    @staticmethod
    def estimate_CRPS_score(observation, simulations):
        """observation is a single value, while simulations is an array. We estimate this by building an empirical
         unbiased estimate of Eq. (1) in Ziel and Berk 2019"""
        diff_X_y = np.abs(observation - simulations)
        n_sim = simulations.shape[0]
        diff_X_tildeX = np.abs(simulations.reshape(1, -1) - simulations.reshape(-1, 1))

        return 2 * np.mean(diff_X_y) - np.sum(diff_X_tildeX) / (n_sim * (n_sim - 1))


class EnergyScore(ScoringRule):
    """ Estimates the EnergyScore. Here, I assume the observations and simulations are lists of
    length respectively n_obs and n_sim. Then, for each fixed observation the n_sim simulations are used to estimate the
    scoring rule. Subsequently, the values are summed over each of the n_obs observations.

    Note this scoring rule is connected to the energy distance between probability distributions.
    """

    def __init__(self, statistics_calc, beta=1, weight=1):
        """default value is beta=1"""
        self.beta = beta
        self.beta_over_2 = 0.5 * beta
        super(EnergyScore, self).__init__(statistics_calc, weight=weight)

    def score(self, observations, simulations):
        """Parameters
        ----------
        observations: Python list
            Contains n1 data points.
        simulations: Python list
            Contains n2 data points.

        Returns
        -------
        numpy.ndarray
            The score between the simulations and the observations.

        Notes
        -----
        When running an ABC algorithm, the observed dataset is always passed first to the distance. Therefore, you can
        save the statistics of the observed dataset inside this object, in order to not repeat computations.
        """

        s_observations, s_simulations = self._calculate_summary_stat(observations, simulations)

        score = self.estimate_energy_score_new(s_observations, s_simulations)
        return score

    def score_max(self):
        # As the statistics are positive, the max possible value is 1
        return np.inf

    def estimate_energy_score_new(self, observations, simulations):
        """observations is an array of size (n_obs, p) (p being the dimensionality), while simulations is an array
        of size (n_sim, p).
        We estimate this by building an empirical unbiased estimate of Eq. (2) in Ziel and Berk 2019"""
        n_obs = observations.shape[0]
        n_sim, p = simulations.shape
        diff_X_y = observations.reshape(n_obs, 1, -1) - simulations.reshape(1, n_sim, p)
        # check (specifically in case n_sim==p):
        # diff_X_y2 = np.zeros((observations.shape[0], *simulations.shape))
        # for i in range(observations.shape[0]):
        #     for j in range(n_sim):
        #         diff_X_y2[i, j] = observations[i] - simulations[j]
        # assert np.allclose(diff_X_y2, diff_X_y)
        diff_X_y = np.einsum('ijk, ijk -> ij', diff_X_y, diff_X_y)

        diff_X_tildeX = simulations.reshape(1, n_sim, p) - simulations.reshape(n_sim, 1, p)
        # check (specifically in case n_sim==p):
        # diff_X_tildeX2 = np.zeros((n_sim, n_sim, p))
        # for i in range(n_sim):
        #     for j in range(n_sim):
        #         diff_X_tildeX2[i, j] = simulations[j] - simulations[i]
        # assert np.allclose(diff_X_tildeX2, diff_X_tildeX)
        diff_X_tildeX = np.einsum('ijk, ijk -> ij', diff_X_tildeX, diff_X_tildeX)

        if self.beta_over_2 != 1:
            diff_X_y **= self.beta_over_2
            diff_X_tildeX **= self.beta_over_2

        return 2 * np.sum(np.mean(diff_X_y, axis=1)) - n_obs * np.sum(diff_X_tildeX) / (n_sim * (n_sim - 1))
        # here I am using an unbiased estimate; I could also use a biased estimate (dividing by n_sim**2). In the ABC
        # with energy distance, they use the biased estimate for energy distance as it is always positive; not sure this
        # is so important here however.


class KernelScore(ScoringRule):

    def __init__(self, statistics, kernel="gaussian", biased_estimator=False, weight=1, **kernel_kwargs):
        """
        Parameters
        ----------
        statistics_calc : abcpy.statistics.Statistics
            Statistics extractor object that conforms to the Statistics class.
        kernel : str or callable, optional
            Can be a string denoting the kernel, or a function. If a string, only gaussian is implemented for now; in
            that case, you can also provide an additional keyword parameter 'sigma' which is used as the sigma in the
            kernel.
        weight : int, optional.
        """

        super(KernelScore, self).__init__(statistics, weight=weight)

        self.kernel_vectorized = False
        if not isinstance(kernel, str) and not callable(kernel):
            raise RuntimeError("'kernel' must be either a string or a function of two variables returning a scalar.")
        if isinstance(kernel, str):
            if kernel == "gaussian":
                self.kernel = self.def_gaussian_kernel(**kernel_kwargs)
                self.kernel_vectorized = True  # the gaussian kernel is vectorized
            else:
                raise NotImplementedError("The required kernel is not implemented.")
        else:
            self.kernel = kernel  # if kernel is a callable already

        self.biased_estimator = biased_estimator

    def score(self, observations, simulations):
        """Parameters
        ----------
        observations: Python list
            Contains n1 data points.
        simulations: Python list
            Contains n2 data points.

        Returns
        -------
        numpy.ndarray
            The score between the simulations and the observations.

        Notes
        -----
        When running an ABC algorithm, the observed dataset is always passed first to the distance. Therefore, you can
        save the statistics of the observed dataset inside this object, in order to not repeat computations.
        """
        s_observations, s_simulations = self._calculate_summary_stat(observations, simulations)

        # compute the Gram matrix
        K_sim_sim, K_obs_sim = self.compute_Gram_matrix(s_observations, s_simulations)

        # Estimate MMD
        if self.biased_estimator:
            return self.MMD_V_estimator(K_sim_sim, K_obs_sim)
        else:
            return self.MMD_unbiased(K_sim_sim, K_obs_sim)

    def score_max(self):
        """
        Returns
        -------
        numpy.float
            The maximal possible value of the desired distance function.
        """

        # As the statistics are positive, the max possible value is 1
        return np.inf

    @staticmethod
    def def_gaussian_kernel(sigma=1):
        # notice in the MMD paper they set sigma to a median value over the observation; check that.
        sigma_2 = 2 * sigma ** 2

        # def Gaussian_kernel(x, y):
        #     xy = x - y
        #     # assert np.allclose(np.dot(xy, xy), np.linalg.norm(xy) ** 2)
        #     return np.exp(- np.dot(xy, xy) / sigma_2)

        def Gaussian_kernel_vectorized(X, Y):
            """Here X and Y have shape (n_samples_x, n_features) and (n_samples_y, n_features);
            this directly computes the kernel for all pairwise components"""
            XY = X.reshape(X.shape[0], 1, -1) - Y.reshape(1, Y.shape[0], -1)  # pairwise differences
            return np.exp(- np.einsum('xyi,xyi->xy', XY, XY) / sigma_2)

        return Gaussian_kernel_vectorized

    def compute_Gram_matrix(self, s_observations, s_simulations):

        if self.kernel_vectorized:
            K_sim_sim = self.kernel(s_simulations, s_simulations)
            K_obs_sim = self.kernel(s_observations, s_simulations)
        else:
            n_obs = s_observations.shape[0]
            n_sim = s_simulations.shape[0]

            K_sim_sim = np.zeros((n_sim, n_sim))
            K_obs_sim = np.zeros((n_obs, n_sim))

            for i in range(n_sim):
                # we assume the function to be symmetric; this saves some steps:
                for j in range(i, n_sim):
                    K_sim_sim[j, i] = K_sim_sim[i, j] = self.kernel(s_simulations[i], s_simulations[j])

            for i in range(n_obs):
                for j in range(n_sim):
                    K_obs_sim[i, j] = self.kernel(s_observations[i], s_simulations[j])

        return K_sim_sim, K_obs_sim

    @staticmethod
    def MMD_unbiased(K_sim_sim, K_obs_sim):
        # Adapted from https://github.com/eugenium/MMD/blob/2fe67cbc7378f10f3b273cfd8d8bbd2135db5798/mmd.py
        # The estimate when distribution of x is not equal to y
        n_obs, n_sim = K_obs_sim.shape

        t_obs_sim = (2. / n_sim) * np.sum(K_obs_sim)
        t_sim_sim = (1. / (n_sim * (n_sim - 1))) * np.sum(K_sim_sim - np.diag(np.diagonal(K_sim_sim)))

        return n_obs * t_sim_sim - t_obs_sim

    @staticmethod
    def MMD_V_estimator(K_sim_sim, K_obs_sim):
        # The estimate when distribution of x is not equal to y
        n_obs, n_sim = K_obs_sim.shape

        t_obs_sim = (2. / n_sim) * np.sum(K_obs_sim)
        t_sim_sim = (1. / (n_sim * n_sim)) * np.sum(K_sim_sim)

        return n_obs * t_sim_sim - t_obs_sim
