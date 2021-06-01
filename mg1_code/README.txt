Code to reproduce the results of ``On Bayesian inference for the M/G/1 queue with efficient MCMC sampling’’ by Shestopaloff and Neal.

The bash script `do_runs` creates matlab files which set the parameter values and call the queue_met matlab file, which performs the inference.

*** Producing the samples ***

We use three data sets generated from the queueing model for testing our proposed sampler. These data sets are stored in

Frequent arrivals: queue_frequent.mat
Intermediate case: queue_inter.mat
Rare arrivals: queue_rare.mat

We use five different MCMC samplers to make draws from the posterior of the parameters given each of these three datasets. Each sampler is run with five different random number generator seeds.

This is accomplished with the script ``do_runs’’. In this script, the following parameters can be varied.

data is set to one of ``frequent’’, ``inter’’, ``rare’’ to correspond to the data set used.
numiter is the number of iterations for a given sampler, as in Table 3.
shift / range_scale / rate_scale are set to 0 or 1 depending on whether these particular updates are used.
eta_1_std, eta_2_std, eta_3_std, eta_prop_scale, numupd_met, shift_std, c_range, c_rate are all set to values corresponding to a given sampler, as in in Table 2.

After all of the runs for all of the different samplers are complete, we can compute estimates of parameters along with standard errors using make_est_table.m, which is Table 4 in the paper.

*** Computing the autocorrelation time estimates ***

To compute the autocorrelation function estimates, we use the function acf_batch.m. The variable ``scen’’ is set to to one of ``frequent’’, ``inter’’, ``rare’’ and numupd_met is set to 1 for the ``frequent’’ scenario and to 16 for the ``inter’’ and ``rare’’ scenarios.

After the autocorrelation function estimates are computed, we need to estimate the cutoffs up to which to sum the autocorrelation function to compute an estimate of autocorrelation time.

This is done using plot_acf.m where we again set ``scen’’ to one of ``frequent’’, ``inter’’, ``rare’’ and ``numupd_met’’ to 1 for the ``frequent’’ scenario and to 16 for the ``inter’’ and ``rare’’ scenarios. We use the following autocorrelation function cutoffs for each of the three scenarios and five samplers.

Frequent

(shift, range, rate) = (0,0,0)
use_lags = [600, 600, 25000];

(shift, range, rate) = (1,0,0)
use_lags = [400, 400, 20000];

(shift, range, rate) = (0,1,0)
use_lags = [500, 500, 20000];

(shift, range, rate) = (0,0,1)
use_lags = [800, 800, 800];

(shift, range, rate) = (1,1,1)
use_lags = [400, 400, 400];

Intermediate

(shift, range, rate) = (0,0,0)
use_lags = [80, 80, 80];

(shift, range, rate) = (1,0,0)
use_lags = [80, 80, 80];

(shift, range, rate) = (0,1,0)
use_lags = [80, 80, 80];

(shift, range, rate) = (0,0,1)
use_lags = [80, 80, 80];

(shift, range, rate) = (1,1,1)
use_lags = [60, 60, 60];

Rare

(shift, range, rate) = (0,0,0)
use_lags = [32000, 32000, 500];

(shift, range, rate) = (1,0,0)
use_lags = [3000, 32000, 400];

(shift, range, rate) = (0,1,0)
use_lags = [1500, 1500, 400];

(shift, range, rate) = (0,0,1)
use_lags = [30000, 30000, 500];

(shift, range, rate) = (1,1,1)
use_lags = [300, 600, 300];

After all the autocorrelation function estimates have been computed, we use make_tau_table.m to create Table 5 of autocorrelation time estimates.

*** Plot the number of people in the queue ***

To produce the plots in Figure 1, we use the function queue_size_plots.m

*** Making the trace plots ***

This is done using the function trace_plots.m. We change the scenario to one of ``frequent’’, ``inter’’, ``rare’’ and the filenames to correspond to the scenario of interest. For the trace plots, we compare the basic method with the method that uses all of the additional updates. To produce plots for the various scenarios, we uncomment the corresponding sections.
