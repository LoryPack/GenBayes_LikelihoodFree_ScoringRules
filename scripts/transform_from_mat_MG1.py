import argparse
import os
import sys

import numpy as np
from scipy.io import loadmat

sys.path.append(os.getcwd())  # stupid but needed on my laptop for some weird reason

from src.utils import define_default_folders

"""Transforms the output from matlab to the numpy format"""

parser = argparse.ArgumentParser()
parser.add_argument('--n_steps', type=int, default=2900000, help='MCMC steps')
parser.add_argument('--burnin_frac', type=float, default=1 / 10, help='Fraction of steps to burn-in.')
parser.add_argument('--root_folder', type=str, default=None)
parser.add_argument('--true_posterior_folder', type=str, default="true_posterior")

args = parser.parse_args()

n_steps = args.n_steps
burnin_frac = args.burnin_frac
results_folder = args.root_folder

default_root_folder = define_default_folders()
if results_folder is None:
    results_folder = default_root_folder["MG1"]

true_posterior_folder = results_folder + '/' + args.true_posterior_folder + '/'

out = loadmat(true_posterior_folder + 'my_inference_steps_{}.mat'.format(n_steps))
# out is a dictionary containing a lot of stuff. Specifically, what is of interest to us is the trace, which is
# in out['par_mat']:
trace = out['par_mat'][int(burnin_frac * n_steps):]  # discard some burnin steps

# trace is given in terms of eta, so need to transform back:
# eta = [theta(1), theta(2)-theta(1), log(theta(3))];
trace[:, 1] = trace[:, 1] + trace[:, 0]
trace[:, 2] = np.exp(trace[:, 2])  # inverse of the log

np.save(true_posterior_folder + "n-samples_{}_burnin_{}_n-sam-per-param_1.npy".format(
    int((1 - burnin_frac) * n_steps), int(burnin_frac * n_steps)), trace)
