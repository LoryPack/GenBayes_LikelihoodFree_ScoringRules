import numpy as np
import pyreadr

"""We extract here the observation used in An et al. 2020 and save it in our format."""

file = "results/RecruitmentBoomBust/data_bnb.rds"

obs = pyreadr.read_r(file)
obs = obs[None].to_numpy()

obs = obs.transpose(1, 0)
print(obs.shape)

# save the observation by An et al.
np.save("results/RecruitmentBoomBust/observations/x_obs.npy", obs)
