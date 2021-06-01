import argparse
import os
import sys

import numpy as np
from scipy.io import savemat

sys.path.append(os.getcwd())  # stupid but needed on my laptop for some weird reason
from src.utils import define_default_folders_scoring_rules

"""Transforms the observation to the matlab format"""

parser = argparse.ArgumentParser()
parser.add_argument('--root_folder', type=str, default=None)
parser.add_argument('--observation_folder', type=str, default="observations")

args = parser.parse_args()

results_folder = args.root_folder

default_root_folder = define_default_folders_scoring_rules()
if results_folder is None:
    results_folder = default_root_folder["MG1"]

observation_folder = results_folder + '/' + args.observation_folder + '/'

y = np.load(observation_folder + "x_obs.npy")

savemat('mg1_code/data/my_observation.mat', {'y': y[0]})
