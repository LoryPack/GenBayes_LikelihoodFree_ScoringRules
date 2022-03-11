#!/bin/bash

# increasing number of samples in observation
model=univariate_g-and-k

# set up folders:
inference_folder=inferences
inference_folder=inferences
observation_folder=observations

mkdir results results/${model} results/${model}/${inference_folder} results/${model}/${observation_folder}

# generate the observations (we generate here 1 observed datasets with 100 iid observations)
echo Generate observations
python3 scripts/generate_obs.py $model --n_observations_per_param 100 --observation_folder $observation_folder

# now do inference with our methods
echo Inference

burnin=10000
n_samples=100000
n_samples_per_param=500
NGROUP=500

# estimate the weight for all my methods wrt SyntheticLikelihood, using one single observation; this stores the result
# in a file which we call later
METHODS=( EnergyScore KernelScore )
for ((k=0;k<${#METHODS[@]};++k)); do
    method=${METHODS[k]}
    echo $method

    python scripts/estimate_w_single_obs.py \
    $model  \
    $method  \
    --n_theta 1000  \
    --n_samples_per_param $n_samples_per_param \
    --inference_folder $inference_folder \
    --observation_folder $observation_folder

done

# loop over METHODS:

METHODS=( SyntheticLikelihood EnergyScore KernelScore )

N_SAMPLES_IN_OBS=( 1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 )

PROPSIZES_BSL=( 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 )
PROPSIZES_ENERGY=( 1 1 1 1 1 1 0.4 0.4 0.4 0.4 0.4 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 )
PROPSIZES_KERNEL=( 1 1 1 1 1 1 0.4 0.4 0.4 0.4 0.4 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 )

FOLDER=results/${model}/${inference_folder}/
for ((k=0;k<${#METHODS[@]};++k)); do
for ((k2=0;k2<${#N_SAMPLES_IN_OBS[@]};++k2)); do

        method=${METHODS[k]}
        n_samples_in_obs=${N_SAMPLES_IN_OBS[k2]}

        echo $method $n_samples_in_obs

     if [[ "$method" == "KernelScore" ]]; then
            PROPSIZE=${PROPSIZES_KERNEL[k2]}
     fi
     if [[ "$method" == "EnergyScore" ]]; then
            PROPSIZE=${PROPSIZES_ENERGY[k2]}
     fi
     if [[ "$method" == "SyntheticLikelihood" ]]; then
            PROPSIZE=${PROPSIZES_BSL[k2]}
     fi

        python scripts/inference.py \
        $model  \
        $method  \
        --n_samples $n_samples  \
        --burnin $burnin  \
        --n_samples_per_param $n_samples_per_param \
        --n_samples_in_obs $n_samples_in_obs \
        --inference_folder $inference_folder \
        --observation_folder $observation_folder \
        --load \
        --prop_size $PROPSIZE \
        --weight_file weight_obs_index_1_1_${method}_wrt_SyntheticLikelihood.npy \
        --n_groups $NGROUP \
         >${FOLDER}out_MCMC_${method}_${n_samples_in_obs}

    done
done

# Figure 1
python scripts/plot_marginals_n_obs.py $model \
    --inference_folder $inference_folder \
    --observation_folder $observation_folder \
    --thin 10 \
    --burnin $burnin \
    --n_samples $n_samples \
    --n_samples_per_param $n_samples_per_param > ${FOLDER}/acc_rates.txt
