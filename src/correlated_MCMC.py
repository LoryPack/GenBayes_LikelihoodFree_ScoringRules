import copy
import logging
import warnings

from abcpy.acceptedparametersmanager import *
from abcpy.backends import BackendDummy
from abcpy.jointapprox_lhd import SumCombination
from abcpy.output import Journal
from abcpy.perturbationkernel import DefaultKernel, JointPerturbationKernel
from abcpy.probabilisticmodels import *
from abcpy.transformers import BoundedVarTransformer, DummyTransformer
from abcpy.inferences import BaseLikelihood, InferenceMethod
from tqdm import tqdm


class MCMCMetropoliHastings(BaseLikelihood, InferenceMethod):
    """
    Simple Metropolis-Hastings MCMC working with the approximate likelihood functions Approx_likelihood, with
    multivariate normal proposals.

    Parameters
    ----------
    root_models : list
        A list of the Probabilistic models corresponding to the observed datasets
    loglikfuns : list of abcpy.approx_lhd.Approx_likelihood
        List of Approx_loglikelihood object defining the approximated loglikelihood to be used; one for each model.
    backend : abcpy.backends.Backend
        Backend object defining the backend to be used.
    kernel : abcpy.perturbationkernel.PerturbationKernel, optional
        PerturbationKernel object defining the perturbation kernel needed for the sampling. If not provided, the
        DefaultKernel is used.
    seed : integer, optional
        Optional initial seed for the random number generator. The default value is generated randomly.

    """

    model = None
    likfun = None
    kernel = None
    rng = None

    n_samples = None
    n_samples_per_param = None

    backend = None

    def __init__(self, root_models, loglikfuns, backend, kernel=None, seed=None):
        self.model = root_models
        # We define the joint Sum of Loglikelihood functions using all the loglikelihoods for each individual models
        self.likfun = SumCombination(root_models, loglikfuns)

        mapping, garbage_index = self._get_mapping()
        models = []
        self.parameter_names_with_index = {}
        for mdl, mdl_index in mapping:
            models.append(mdl)
            self.parameter_names_with_index[mdl.name] = mdl_index  # dict storing param names with index

        self.parameter_names = [model.name for model in models]  # store parameter names

        if kernel is None:
            kernel = DefaultKernel(models)

        self.kernel = kernel
        self.backend = backend
        self.rng = np.random.RandomState(seed)
        self.logger = logging.getLogger(__name__)

        # these are usually big tables, so we broadcast them to have them once
        # per executor instead of once per task
        self.accepted_parameters_manager = AcceptedParametersManager(self.model)
        # this is used to handle the data for adapting the covariance:
        self.accepted_parameters_manager_adaptive_cov = AcceptedParametersManager(self.model)

        self.simulation_counter = 0

    def sample(self, observations, n_samples, n_samples_per_param=100, burnin=1000, cov_matrices=None, iniPoint=None,
               adapt_proposal_cov_interval=None, adapt_proposal_start_step=0, adapt_proposal_after_burnin=False,
               covFactor=None, bounds=None, speedup_dummy=True, n_groups_correlated_randomness=None, use_tqdm=True,
               journal_file=None, path_to_save_journal=None):
        """Samples from the posterior distribution of the model parameter given the observed
        data observations. The MCMC is run for burnin + n_samples steps, and n_samples_per_param are used at each step
        to estimate the approximate loglikelihood. The burnin steps are then discarded from the chain stored in the
        journal file.

        During burnin, the covariance matrix is adapted from the steps generated up to that point, in a way similar to
        what suggested in [1], after each adapt_proposal_cov_interval steps. Differently from the original algorithm in
        [1], here the proposal covariance matrix is fixed after the end of the burnin steps.

        In case the original parameter space is bounded (for instance with uniform prior on an interval), the MCMC can
        be optionally run on a transformed space. Therefore, the covariance matrix describes proposals on the
        transformed space; the acceptance rate then takes into account the Jacobian of the transformation. In order to
        use MCMC with transformed space, you need to specify lower and upper bounds in the corresponding parameters (see
        details in the description of `bounds`).

        The returned journal file contains also information on acceptance rates (in the configuration dictionary).

        [1] Haario, H., Saksman, E., & Tamminen, J. (2001). An adaptive Metropolis algorithm. Bernoulli, 7(2), 223-242.

        [2] Picchini, U., Simola, U., & Corander, J. (2022). Sequentially guided MCMC proposals for synthetic
        likelihoods and correlated synthetic likelihoods. Bayesian Analysis, 1(1), 1-31.

        Parameters
        ----------
        observations : list
            A list, containing lists describing the observed data sets; one for each model.
        n_samples : integer, optional
            number of samples to generate. The default value is 10000.
        n_samples_per_param : integer, optional
            number of data points in each simulated data set. The default value is 100.
        burnin : integer, optional
            Number of burnin steps to discard. Defaults to 1000.
        cov_matrices : list of matrices, optional
            list of initial covariance matrices for the proposals. If not provided, identity matrices are used. If the
            sample routine is restarting from a journal file and cov_matrices is not provided, cov_matrices is set to
            the value used for sampling after burnin in the previous journal file (ie what is stored in
            `journal.configuration["actual_cov_matrices"]`).
        iniPoint : numpy.ndarray, optional
            parameter value from where the sampling starts. By default sampled from the prior. Not used if journal_file
            is passed.
        adapt_proposal_cov_interval : integer, optional
            the proposal covariance matrix is adapted each adapt_cov_matrix steps during burnin, by using the chain up
            to that point. If None, no adaptation is done. Default value is None. Use with care as, if the likelihood
            estimate is very noisy, the adaptation may work pooly (see `covFactor` parameter).
        adapt_proposal_start_step: integer, optional
            the step after which to start adapting the covariance matrix following Haario's approach in [1]. See also
            adapt_proposal_cov_interval. Defaults to 0.
        adapt_proposal_after_burnin: boolean, optional
            If True, keep adapting the proposal after the end of burnin iterations following Haario's method in [1].
            Otherwise, stop adapting it after burnin, and use the last proposal matrix. Defaults to False.
        covFactor : float, optional
            the factor by which to scale the empirical covariance matrix in order to obtain the covariance matrix for
            the proposal, whenever that is updated during the burnin steps. If not provided, we use the default value
            2.4 ** 2 / dim_theta suggested in [1].
            Notice that this value was shown to be optimal (at least in some
            limit sense) in the case in which the likelihood is fully known. In the present case, in which the
            likelihood is estimated from data, that value may turn out to be too large; specifically, if
            the likelihood estimate is very noisy, that choice may lead to a very bad adaptation which may give rise
            to an MCMC which does not explore the space well (for instance, the obtained covariance matrix may turn out
            to be too small). If that happens, we suggest to set covFactor to a smaller value than the default one, in
            which case the acceptance rate of the chain will likely be smaller but the exploration will be better.
            Alternatively, it is possible to reduce the noise in the likelihood estimate by increasing
            `n_samples_per_param`.
        bounds : dictionary, optional
            dictionary containing the lower and upper bound for the transformation to be applied to the parameters. The
            key of each entry is the name of the parameter as defined in the model, while the value if a tuple (or list)
            with `(lower_bound, upper_bound)` content. If the parameter is bounded on one side only, the other bound
            should be set to 'None'. If a parameter is not in this dictionary, no transformation is applied to it.
            If a parameter is bounded on two sides, the used transformation is based on the logit. If conversely it is
            lower bounded, we apply instead a log transformation. Notice that we do not implement yet the transformation
            for upper bounded variables. If no value is provided, the default value is None, which means no
            transformation at all is applied.
        speedup_dummy: boolean, optional.
            If set to True, the map function is not used to parallelize simulations (for the new parameter value) when
            the backend is Dummy. This can improve performance as it can exploit potential vectorization in the model.
            However, this breaks reproducibility when using, for instance, BackendMPI with respect to BackendDummy, due
            to the different way the random seeds are used when speedup_dummy is set to True. Please set this to False
            if you are interested in preserving reproducibility across MPI and Dummy backend. Defaults to True.
        n_groups_correlated_randomness: integer, optional
            The number of groups to use to correlate the randomness in the correlated pseudo-marginal MCMC scheme.
            Specifically, if provided, the n_samples_per_param simulations from the model are split in
            n_groups_correlated_randomness groups. At each MCMC step, the random variables used in simulating the model
            are the same as in the previous step for all n_samples_per_param simulations except for a single group,
            for which fresh random variables are used.
            In practice, this is done by storing the random seeds. This approach should reduce stickiness of the chain
            and was discussed in [2] for the Bayesian Synthetic Likelihood framework. Notice that, when
            n_groups_correlated_randomness > 0 and speedup_dummy is True, you obtain different results for different
            values of n_groups_correlated_randomness due to different ways of handling random seeds.
            When None, we do not keep track of the random seeds. Default value is None.
        use_tqdm : boolean, optional
            Whether using tqdm or not to display progress. Defaults to True.
        journal_file: str, optional
            Filename of a journal file to read an already saved journal file, from which the first iteration will start.
            That's the only information used (it does not use the previous covariance matrix).
            The default value is None.
        path_to_save_journal: str, optional
            If provided, save the journal at the provided path at the end of the inference routine.

        Returns
        -------
        abcpy.output.Journal
            A journal containing simulation results, metadata and optionally intermediate results.
        """

        self.observations = observations
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param
        self.speedup_dummy = speedup_dummy
        # we use this in all places which require a backend but which are not parallelized in MCMC:
        self.dummy_backend = BackendDummy()

        # initializations for the correlated pseudo-marginal MCMC
        self.n_groups_correlated_randomness = n_groups_correlated_randomness
        if self.n_groups_correlated_randomness is not None:
            if self.n_groups_correlated_randomness > self.n_samples_per_param:
                raise RuntimeError("The number of random seeds groups need to be smaller with respect to the number "
                                   "of samples per parameters.")
            self.group_size = int(np.ceil(self.n_samples_per_param / self.n_groups_correlated_randomness))
            self.remainder = self.n_samples_per_param % self.group_size
            # initialize the random seeds
            if isinstance(self.backend, BackendDummy) and self.speedup_dummy:
                # use a single random seed for each group and simulate together, to have some speed up
                self.seed_arr_current = self.rng.randint(0, np.iinfo(np.uint32).max,
                                                         size=self.n_groups_correlated_randomness, dtype=np.uint32)
            else:
                self.seed_arr_current = self.rng.randint(0, np.iinfo(np.uint32).max, size=self.n_samples_per_param,
                                                         dtype=np.uint32)
        if self.n_groups_correlated_randomness is not None and isinstance(self.backend,
                                                                          BackendDummy) and self.speedup_dummy:
            # only case in which multiple simulations per map_step are done:
            self.n_simulations_per_map_step = self.group_size
        else:
            self.n_simulations_per_map_step = 1
        dim = len(self.parameter_names)

        if path_to_save_journal is not None:
            path_to_save_journal = path_to_save_journal if '.jnl' in path_to_save_journal else path_to_save_journal + '.jnl'

        if bounds is None:
            # no transformation is performed
            self.transformer = DummyTransformer()
        else:
            if not isinstance(bounds, dict):
                raise TypeError("Argument `bounds` need to be a dictionary")
            bounds_keys = bounds.keys()
            for key in bounds_keys:
                if key not in self.parameter_names:
                    raise KeyError("The keys in argument `bounds` need to correspond to the parameter names used "
                                   "in defining the model")
                if not hasattr(bounds[key], "__len__") or len(bounds[key]) != 2:
                    raise RuntimeError("Each entry in `bounds` need to be a tuple with 2 value, representing the lower "
                                       "and upper bound of the corresponding parameter. If the parameter is bounded on "
                                       "one side only, the other bound should be set to 'None'.")

            # create lower_bounds and upper_bounds_vector:
            lower_bound_transformer = np.array([None] * dim)
            upper_bound_transformer = np.array([None] * dim)

            for key in bounds_keys:
                lower_bound_transformer[self.parameter_names_with_index[key]] = bounds[key][0]
                upper_bound_transformer[self.parameter_names_with_index[key]] = bounds[key][1]

            # initialize transformer:
            self.transformer = BoundedVarTransformer(np.array(lower_bound_transformer),
                                                     np.array(upper_bound_transformer))

        accepted_parameters = []
        accepted_parameters_burnin = []
        if journal_file is None:
            journal = Journal(0)
            journal.configuration["type_model"] = [type(model).__name__ for model in self.model]
            journal.configuration["type_lhd_func"] = [type(likfun).__name__ for likfun in self.likfun.approx_lhds]
            journal.configuration["type_kernel_func"] = [type(kernel).__name__ for kernel in self.kernel.kernels] if \
                isinstance(self.kernel, JointPerturbationKernel) else type(self.kernel)
            journal.configuration["n_samples"] = self.n_samples
            journal.configuration["n_samples_per_param"] = self.n_samples_per_param
            journal.configuration["burnin"] = burnin
            journal.configuration["cov_matrices"] = cov_matrices
            journal.configuration["iniPoint"] = iniPoint
            journal.configuration["adapt_proposal_cov_interval"] = adapt_proposal_cov_interval
            journal.configuration["adapt_proposal_start_step"] = adapt_proposal_start_step
            journal.configuration["adapt_proposal_after_burnin"] = adapt_proposal_after_burnin
            journal.configuration["covFactor"] = covFactor
            journal.configuration["bounds"] = bounds
            journal.configuration["speedup_dummy"] = speedup_dummy
            journal.configuration["n_groups_correlated_randomness"] = n_groups_correlated_randomness
            journal.configuration["use_tqdm"] = use_tqdm
            journal.configuration["acceptance_rates"] = []
            # Initialize chain: when not supplied, randomly draw it from prior distribution
            # It is an MCMC chain: weights are always 1; forget about them
            # accepted_parameter will keep track of the chain position
            if iniPoint is None:
                self.sample_from_prior(rng=self.rng)
                accepted_parameter = self.get_parameters()
            else:
                accepted_parameter = iniPoint
                if isinstance(accepted_parameter, np.ndarray) and len(accepted_parameter.shape) == 1 or isinstance(
                        accepted_parameter, list) and not hasattr(accepted_parameter[0], "__len__"):
                    # reshape whether we pass a 1d array or list.
                    accepted_parameter = [np.array([x]) for x in accepted_parameter]  # give correct shape for later
            if burnin == 0:
                accepted_parameters_burnin.append(accepted_parameter)
            self.logger.info("Calculate approximate loglikelihood")
            approx_log_likelihood_accepted_parameter = self._simulate_and_compute_log_lik(accepted_parameter)
            # update the number of simulations (this tracks the number of parameters for which simulations are done;
            # the actual number of simulations is this times n_samples_per_param))
            self.simulation_counter += 1
            self.acceptance_rate = 0
        else:
            # check the following:
            self.logger.info("Restarting from previous journal")
            journal = Journal.fromFile(journal_file)
            # this is used to compute the overall acceptance rate:
            self.acceptance_rate = journal.configuration["acceptance_rates"][-1] * journal.configuration["n_samples"]
            accepted_parameter = journal.get_accepted_parameters(-1)[-1]  # go on from last MCMC step
            journal.configuration["n_samples"] += self.n_samples  # add the total number of samples
            journal.configuration["burnin"] = burnin
            if journal.configuration["n_samples_per_param"] != self.n_samples_per_param:
                warnings.warn("You specified a different n_samples_per_param from the one used in the passed "
                              "journal_file; the algorithm will still work fine.")
                journal.configuration["n_samples_per_param"] = self.n_samples_per_param
            journal.configuration["cov_matrices"] = cov_matrices
            journal.configuration["bounds"] = bounds  # overwrite
            if cov_matrices is None:  # use the previously stored one unless the user defines it
                cov_matrices = journal.configuration["actual_cov_matrices"]
            journal.configuration["speedup_dummy"] = speedup_dummy
            journal.configuration["n_groups_correlated_randomness"] = n_groups_correlated_randomness
            approx_log_likelihood_accepted_parameter = journal.final_step_loglik
            self.simulation_counter = journal.number_of_simulations[-1]  # update the number of simulations

        if covFactor is None:
            covFactor = 2.4 ** 2 / dim

        accepted_parameter_prior_pdf = self.pdf_of_prior(self.model, accepted_parameter)

        # set the accepted parameter in the kernel (in order to correctly generate next proposal)
        # do this on transformed parameter
        accepted_parameter_transformed = self.transformer.transform(accepted_parameter)
        self._update_kernel_parameters(accepted_parameter_transformed)

        # compute jacobian
        log_det_jac_accepted_param = self.transformer.jac_log_det(accepted_parameter)

        # 3: calculate covariance
        self.logger.info("Set kernel covariance matrix ")
        if cov_matrices is None:
            # need to set that to some value (we use identity matrices). Be careful, there need to be one
            # covariance matrix for each kernel; not sure that this works in case of multivariate parameters.

            # the kernel parameters are only used to get the exact shape of cov_matrices
            cov_matrices = [np.eye(len(self.accepted_parameters_manager.kernel_parameters_bds.value()[0][kernel_index]))
                            for kernel_index in range(len(self.kernel.kernels))]

        self.accepted_parameters_manager.update_broadcast(self.dummy_backend, accepted_cov_mats=cov_matrices)

        # main MCMC algorithm
        self.logger.info("Starting MCMC")
        for aStep in tqdm(range(burnin + n_samples), disable=not use_tqdm):

            self.logger.debug("Step {} of MCMC algorithm".format(aStep))

            # 1: Resample parameters
            self.logger.debug("Generate proposal")

            # perturb element 0 of accepted_parameters_manager.kernel_parameters_bds:
            # new_parameter = self.perturb(0, rng=self.rng)[1]  # do not use this as it leads to some weird error.
            # rather do:
            new_parameters_transformed = self.kernel.update(self.accepted_parameters_manager, 0, rng=self.rng)

            self._reset_flags()  # not sure whether this is needed, leave it anyway

            # Order the parameters provided by the kernel in depth-first search order
            new_parameter_transformed = self.get_correct_ordering(new_parameters_transformed)

            # transform back
            new_parameter = self.transformer.inverse_transform(new_parameter_transformed)

            # for now we are only using a simple MVN proposal. For bounded parameter values, this is not great; we
            # could also implement a proposal on transformed space, which would be better.
            new_parameter_prior_pdf = self.pdf_of_prior(self.model, new_parameter)
            if new_parameter_prior_pdf == 0:
                self.logger.debug("Proposal parameter at step {} is out of prior region.".format(aStep))
                if aStep >= burnin:
                    accepted_parameters.append(accepted_parameter)
                else:
                    accepted_parameters_burnin.append(accepted_parameter)
                continue

            # 2: calculate approximate likelihood for new parameter. If the backend is MPI, we distribute simulations
            # and then compute the approx likelihood locally
            self.logger.debug("Calculate approximate loglikelihood")
            approx_log_likelihood_new_parameter = self._simulate_and_compute_log_lik(new_parameter)
            self.simulation_counter += 1  # update the number of simulations

            log_det_jac_new_param = self.transformer.jac_log_det(new_parameter)
            # log_det_jac_accepted_param = self.transformer.jac_log_det(accepted_parameter)
            log_jac_term = log_det_jac_accepted_param - log_det_jac_new_param

            # compute acceptance rate:
            alpha = np.exp(
                log_jac_term + approx_log_likelihood_new_parameter - approx_log_likelihood_accepted_parameter) * (
                        new_parameter_prior_pdf) / (accepted_parameter_prior_pdf)  # assumes symmetric kernel

            # Metropolis-Hastings step:
            if self.rng.uniform() < alpha:
                # update param value and approx likelihood
                accepted_parameter_transformed = new_parameter_transformed
                accepted_parameter = new_parameter
                approx_log_likelihood_accepted_parameter = approx_log_likelihood_new_parameter
                accepted_parameter_prior_pdf = new_parameter_prior_pdf
                log_det_jac_accepted_param = log_det_jac_new_param
                # set the accepted parameter in the kernel (in order to correctly generate next proposal)
                self._update_kernel_parameters(accepted_parameter_transformed)
                # save the set of random seeds if we are keeping track of them:
                if self.n_groups_correlated_randomness is not None:
                    self.seed_arr_current = self.seed_arr_proposal
                if aStep >= burnin:
                    self.acceptance_rate += 1

            # save to the trace:
            if aStep >= burnin:
                accepted_parameters.append(accepted_parameter)
            else:
                accepted_parameters_burnin.append(accepted_parameter)

            if adapt_proposal_cov_interval is not None and (aStep < burnin or adapt_proposal_after_burnin):
                # adapt covariance of proposal:
                if aStep > adapt_proposal_start_step and (aStep + 1) % adapt_proposal_cov_interval == 0:
                    # store the accepted_parameters for adapting the covariance in the kernel.
                    # I use this piece of code as it formats the data in the right way
                    # for the sake of using them to compute the kernel cov:
                    self.accepted_parameters_manager_adaptive_cov.update_broadcast(
                        self.dummy_backend, accepted_parameters=accepted_parameters_burnin + accepted_parameters)
                    kernel_parameters = []
                    for kernel in self.kernel.kernels:
                        kernel_parameters.append(
                            self.accepted_parameters_manager_adaptive_cov.get_accepted_parameters_bds_values(
                                kernel.models))
                    self.accepted_parameters_manager_adaptive_cov.update_kernel_values(
                        self.dummy_backend, kernel_parameters=kernel_parameters)

                    self.logger.info("Updating covariance matrix")
                    cov_matrices = self.kernel.calculate_cov(self.accepted_parameters_manager_adaptive_cov)
                    # this scales with the cov_Factor:
                    cov_matrices = self._compute_accepted_cov_mats(covFactor, cov_matrices)
                    # store it in the main AcceptedParametersManager in order to perturb data with it in the following:
                    self.accepted_parameters_manager.update_broadcast(self.dummy_backend,
                                                                      accepted_cov_mats=cov_matrices)

        self.acceptance_rate /= journal.configuration["n_samples"]
        self.logger.info("Saving results to output journal")
        self.accepted_parameters_manager.update_broadcast(self.dummy_backend, accepted_parameters=accepted_parameters)
        names_and_parameters = self._get_names_and_parameters()
        if journal_file is not None:  # concatenate chains
            journal.add_accepted_parameters(journal.get_accepted_parameters() + copy.deepcopy(accepted_parameters))
            names_and_parameters = [(names_and_parameters[i][0],
                                     journal.get_parameters()[names_and_parameters[i][0]] + names_and_parameters[i][1])
                                    for i in range(len(names_and_parameters))]
            journal.add_user_parameters(names_and_parameters)
        else:
            journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
            journal.add_user_parameters(names_and_parameters)
        journal.number_of_simulations.append(self.simulation_counter)
        journal.configuration["acceptance_rates"].append(self.acceptance_rate)
        journal.add_weights(np.ones((journal.configuration['n_samples'], 1)))
        # store the final loglik to be able to restart the journal correctly
        journal.final_step_loglik = approx_log_likelihood_accepted_parameter
        # store the final actual cov_matrices, in order to use this when restarting from journal
        journal.configuration["actual_cov_matrices"] = cov_matrices

        if path_to_save_journal is not None:  # save journal
            journal.save(path_to_save_journal)

        return journal

    def _sample_parameter(self, rng, npc=None):
        """
        Generate a simulation from the model with the current value of accepted_parameter

        Parameters
        ----------
        rng: random number generator
            The random number generator to be used.
        Returns
        -------
        np.array
            accepted parameter
        """

        # get the new parameter value
        theta = self.new_parameter_bds.value()
        # Simulate the fake data from the model given the parameter value theta
        self.logger.debug("Simulate model for parameter " + str(theta))
        acc = self.set_parameters(theta)
        if acc is False:
            self.logger.debug("Parameter " + str(theta) + "has not been accepted")
        # Generate the correct number of simulations to generate in this instance of the method, using a single call
        # to the model.
        y_sim = self.simulate(self.n_simulations_per_map_step, rng=rng, npc=npc)

        return y_sim

    def _approx_log_lik_calc(self, y_sim, npc=None):
        """
        Compute likelihood for new parameters using approximate likelihood function

        Parameters
        ----------
        y_sim: list
            A list containing self.n_samples_per_param simulations for the new parameter value
        Returns
        -------
        float
            The approximated likelihood function
        """
        self.logger.debug("Extracting observation.")
        obs = self.observations

        self.logger.debug("Computing likelihood...")
        loglhd = self.likfun.loglikelihood(obs, y_sim)

        self.logger.debug("LogLikelihood is :" + str(loglhd))

        return loglhd

    def _simulate_and_compute_log_lik(self, new_parameter):
        """Helper function which simulates data from `new_parameter` and computes the approximate loglikelihood.
        In case the backend is not BackendDummy (ie parallelization is available) this parallelizes the different
        simulations (which are all for the same parameter value).

        Notice that, according to the used model, spreading the simulations in different tasks can be more inefficient
        than using one single call, according to the level of vectorization that the model uses and the overhead
        associated. For this reason, we do not split the simulations in different tasks when the backend is
        BackendDummy and self.speedup_dummy is True.

        Parameters
        ----------
        new_parameter
            Parameter value from which to generate data with which to compute the approximate loglikelihood.

        Returns
        -------
        float
            The approximated likelihood function
        """
        if isinstance(self.backend,
                      BackendDummy) and self.speedup_dummy and self.n_groups_correlated_randomness is None:
            # do all the required simulations here without parallellizing; however this gives different result
            # from the other option due to the way random seeds are handled.
            self.logger.debug('simulations')
            theta = new_parameter
            # Simulate the fake data from the model given the parameter value theta
            self.logger.debug("Simulate model for parameter " + str(theta))
            acc = self.set_parameters(theta)
            if acc is False:
                self.logger.debug("Parameter " + str(theta) + "has not been accepted")
            simulations_from_new_parameter = self.simulate(n_samples_per_param=self.n_samples_per_param, rng=self.rng)
        else:
            self.logger.debug('parallelize simulations for fixed parameter value')
            if self.n_groups_correlated_randomness is not None:
                if isinstance(self.backend, BackendDummy) and self.speedup_dummy:
                    # use a single random seed for each group and simulate together, to have some speed up
                    # reset the proposal seeds
                    self.seed_arr_proposal = self.seed_arr_current
                    # now: pick the position of random seeds to be modified:
                    index = self.rng.randint(0, self.n_groups_correlated_randomness)  # the number of correlated groups
                    # and replace it with a new seed.
                    self.seed_arr_proposal[index] = self.rng.randint(0, np.iinfo(np.uint32).max, size=1,
                                                                     dtype=np.uint32)
                    # todo in this case, this does not generate the correct number of simulations if
                    #  self.n_samples_per_param % self.n_groups_correlated_randomness != 0, but it generates
                    #  group_size * n_groups_correlated_randomness. Fix!
                else:
                    # reset the proposal seeds
                    self.seed_arr_proposal = self.seed_arr_current
                    # now: pick the position of random seeds to be modified:
                    index = self.rng.randint(0, self.n_groups_correlated_randomness)  # the number of correlated groups
                    # and replace it with a new seed.
                    if index == self.n_groups_correlated_randomness - 1 and self.remainder != 0:
                        self.seed_arr_proposal[
                        index * self.group_size: (index + 1) * self.group_size] = self.rng.randint(
                            0, np.iinfo(np.uint32).max, size=self.remainder, dtype=np.uint32)
                    else:
                        self.seed_arr_proposal[
                        index * self.group_size: (index + 1) * self.group_size] = self.rng.randint(
                            0, np.iinfo(np.uint32).max, size=self.group_size, dtype=np.uint32)
                seed_arr = self.seed_arr_proposal
            else:
                # do not keep track of seeds
                seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=self.n_samples_per_param, dtype=np.uint32)

            rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
            rng_pds = self.backend.parallelize(rng_arr)

            # need first to broadcast the new_parameter value:
            self.new_parameter_bds = self.backend.broadcast(new_parameter)

            # map step:
            simulations_from_new_parameter_pds = self.backend.map(self._sample_parameter, rng_pds)
            self.logger.debug("collect simulations from pds")
            simulations_from_new_parameter = self.backend.collect(simulations_from_new_parameter_pds)
            # now need to reshape that correctly. The first index has to be the model, then we need to have
            # n_samples_per_param and then the size of the simulation
            simulations_from_new_parameter = [
                [simulations_from_new_parameter[sample_index][model_index][i] for sample_index in
                 range(len(rng_arr)) for i in range(self.n_simulations_per_map_step)] for model_index in
                range(len(self.model))]
        approx_log_likelihood_new_parameter = self._approx_log_lik_calc(simulations_from_new_parameter)

        return approx_log_likelihood_new_parameter

    def _update_kernel_parameters(self, accepted_parameter):
        """This stores the last accepted parameter in the kernel so that it will be used to generate the new proposal
        with self.perturb.

        Parameters
        ----------
        accepted_parameter
            Parameter value from which you want to generate proposal at next iteration of MCMC.
        """
        # I use this piece of code as it formats the data in the right way for the sake of using it in the kernel
        self.accepted_parameters_manager.update_broadcast(self.dummy_backend, accepted_parameters=[accepted_parameter])

        kernel_parameters = []
        for kernel in self.kernel.kernels:
            kernel_parameters.append(
                self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))
        self.accepted_parameters_manager.update_kernel_values(self.dummy_backend, kernel_parameters=kernel_parameters)

    @staticmethod
    def _compute_accepted_cov_mats(covFactor, new_cov_mats):
        """
        Update the covariance matrices computed from data by multiplying them with covFactor and adding a small term in
        the diagonal for numerical stability.

        Parameters
        ----------
        covFactor : float
            factor to correct the covariance matrices
        new_cov_mats : list
            list of covariance matrices computed from data
        Returns
        -------
        list
            List of new accepted covariance matrices
        """
        # accepted_cov_mats = [covFactor * cov_mat for cov_mat in accepted_cov_mats]
        accepted_cov_mats = []
        for new_cov_mat in new_cov_mats:
            if not (new_cov_mat.size == 1):
                accepted_cov_mats.append(
                    covFactor * new_cov_mat + 1e-20 * np.trace(new_cov_mat) * np.eye(new_cov_mat.shape[0]))
            else:
                accepted_cov_mats.append((covFactor * new_cov_mat + 1e-20 * new_cov_mat).reshape(1, 1))
        return accepted_cov_mats
