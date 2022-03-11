import unittest

import numpy as np
from abcpy.continuousmodels import Normal
from abcpy.continuousmodels import Uniform
from abcpy.statistics import Identity

from src.models import instantiate_model
from src.scoring_rules import EnergyScore, UnivariateContinuousRankedProbabilityScoreEstimate, KernelScore


class InstantiateModelsTests(unittest.TestCase):
    def setUp(self):
        pass

    def test_initialization(self):
        model = instantiate_model("MA2")
        model = instantiate_model("MA2", reparametrized=False)
        model = instantiate_model("g-and-k")
        model = instantiate_model("Cauchy_g-and-k")
        model = instantiate_model("univariate_g-and-k")
        model = instantiate_model("univariate_Cauchy_g-and-k")
        model = instantiate_model("MG1")
        model = instantiate_model("MG1", reparametrized=False)
        model = instantiate_model("normal_location_misspec")


class EnergyScoreTests(unittest.TestCase):

    def setUp(self):
        self.mu = Uniform([[-5.0], [5.0]], name='mu')
        self.sigma = Uniform([[5.0], [10.0]], name='sigma')
        self.model = Normal([self.mu, self.sigma])
        self.statistics_calc = Identity(degree=1)
        self.scoring_rule = EnergyScore(self.statistics_calc, beta=2)
        self.scoring_rule_beta1 = EnergyScore(self.statistics_calc, beta=1)
        self.crps = UnivariateContinuousRankedProbabilityScoreEstimate(self.statistics_calc)
        self.statistics_calc_2 = Identity(degree=2)
        self.scoring_rule_2 = EnergyScore(self.statistics_calc_2, beta=2)
        # create fake simulated data
        self.mu._fixed_values = [1.1]
        self.sigma._fixed_values = [1.0]
        self.y_sim = self.model.forward_simulate(self.model.get_input_values(), 100, rng=np.random.RandomState(1))
        # create observed data
        self.y_obs = [1.8]
        self.y_obs_double = [1.8, 0.9]

    def test_score(self):
        # Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.scoring_rule.score, 3.4, [2, 1])
        self.assertRaises(TypeError, self.scoring_rule.score, [2, 4], 3.4)

        comp_score = self.scoring_rule.score(self.y_obs, self.y_sim)
        expected_score = 0.400940132262833
        # This checks whether it computes a correct value and dimension is right
        self.assertAlmostEqual(comp_score, expected_score * 2)

        comp_score = self.scoring_rule_2.score(self.y_obs, self.y_sim)
        expected_score = 1.57714326099926
        # This checks whether it computes a correct value and dimension is right
        self.assertAlmostEqual(comp_score, expected_score * 2)

    def test_match_crps(self):
        comp_score1 = self.scoring_rule_beta1.score(self.y_obs, self.y_sim)
        comp_score2 = self.crps.score(self.y_obs, self.y_sim)
        self.assertAlmostEqual(comp_score1, comp_score2)

    def test_alias(self):
        # test aliases for score
        comp_score = self.scoring_rule.score(self.y_obs, self.y_sim)
        comp_loglikelihood = self.scoring_rule.loglikelihood(self.y_obs, self.y_sim)
        comp_likelihood = self.scoring_rule.likelihood(self.y_obs, self.y_sim)
        self.assertEqual(comp_score, - comp_loglikelihood)
        self.assertAlmostEqual(comp_likelihood, np.exp(comp_loglikelihood))

    def test_score_additive(self):
        comp_loglikelihood_a = self.scoring_rule.score([self.y_obs_double[0]], self.y_sim)
        comp_loglikelihood_b = self.scoring_rule.score([self.y_obs_double[1]], self.y_sim)
        comp_loglikelihood_two = self.scoring_rule.score(self.y_obs_double, self.y_sim)

        self.assertAlmostEqual(comp_loglikelihood_two, comp_loglikelihood_a + comp_loglikelihood_b)

        comp_loglikelihood_a = self.scoring_rule_2.score([self.y_obs_double[0]], self.y_sim)
        comp_loglikelihood_b = self.scoring_rule_2.score([self.y_obs_double[1]], self.y_sim)
        comp_loglikelihood_two = self.scoring_rule_2.score(self.y_obs_double, self.y_sim)

        self.assertAlmostEqual(comp_loglikelihood_two, comp_loglikelihood_a + comp_loglikelihood_b)


class KernelScoreTests(unittest.TestCase):

    def setUp(self):
        self.mu = Uniform([[-5.0], [5.0]], name='mu')
        self.sigma = Uniform([[5.0], [10.0]], name='sigma')
        self.model = Normal([self.mu, self.sigma])
        self.statistics_calc = Identity(degree=1)
        self.scoring_rule = KernelScore(self.statistics_calc)
        self.statistics_calc_2 = Identity(degree=2)
        self.scoring_rule_2 = KernelScore(self.statistics_calc_2)

        def def_negative_Euclidean_distance(beta=1.0):
            if beta <= 0 or beta > 2:
                raise RuntimeError("'beta' not in the right range (0,2]")

            if beta == 1:
                def Euclidean_distance(x, y):
                    return - np.linalg.norm(x - y)
            else:
                def Euclidean_distance(x, y):
                    return - np.linalg.norm(x - y) ** beta

            return Euclidean_distance

        self.kernel_energy_SR = KernelScore(self.statistics_calc_2, kernel=def_negative_Euclidean_distance(beta=1.4))
        self.energy_SR = EnergyScore(self.statistics_calc_2, beta=1.4)

        # create fake simulated data
        self.mu._fixed_values = [1.1]
        self.sigma._fixed_values = [1.0]
        self.y_sim = self.model.forward_simulate(self.model.get_input_values(), 100, rng=np.random.RandomState(1))
        # create observed data
        self.y_obs = [1.8]
        self.y_obs_double = [1.8, 0.9]

    def test_score(self):
        # Checks whether wrong input type produces error message
        self.assertRaises(TypeError, self.scoring_rule.score, 3.4, [2, 1])
        self.assertRaises(TypeError, self.scoring_rule.score, [2, 4], 3.4)

        comp_score = self.scoring_rule.score(self.y_obs, self.y_sim)
        expected_score = -0.7045988787568286
        # This checks whether it computes a correct value and dimension is right
        self.assertAlmostEqual(comp_score, expected_score)

        comp_score = self.scoring_rule_2.score(self.y_obs, self.y_sim)
        expected_score = -0.13483814600999244
        # This checks whether it computes a correct value and dimension is right
        self.assertAlmostEqual(comp_score, expected_score)

    def test_match_energy_score(self):
        comp_score1 = self.kernel_energy_SR.score(self.y_obs_double, self.y_sim)
        comp_score2 = self.energy_SR.score(self.y_obs_double, self.y_sim)
        self.assertAlmostEqual(comp_score1, comp_score2)

    def test_alias(self):
        # test aliases for score
        comp_score = self.scoring_rule.score(self.y_obs, self.y_sim)
        comp_loglikelihood = self.scoring_rule.loglikelihood(self.y_obs, self.y_sim)
        comp_likelihood = self.scoring_rule.likelihood(self.y_obs, self.y_sim)
        self.assertEqual(comp_score, - comp_loglikelihood)
        self.assertAlmostEqual(comp_likelihood, np.exp(comp_loglikelihood))

    def test_score_additive(self):
        comp_loglikelihood_a = self.scoring_rule.score([self.y_obs_double[0]], self.y_sim)
        comp_loglikelihood_b = self.scoring_rule.score([self.y_obs_double[1]], self.y_sim)
        comp_loglikelihood_two = self.scoring_rule.score(self.y_obs_double, self.y_sim)

        self.assertAlmostEqual(comp_loglikelihood_two, comp_loglikelihood_a + comp_loglikelihood_b)

        comp_loglikelihood_a = self.scoring_rule_2.score([self.y_obs_double[0]], self.y_sim)
        comp_loglikelihood_b = self.scoring_rule_2.score([self.y_obs_double[1]], self.y_sim)
        comp_loglikelihood_two = self.scoring_rule_2.score(self.y_obs_double, self.y_sim)

        self.assertAlmostEqual(comp_loglikelihood_two, comp_loglikelihood_a + comp_loglikelihood_b)
