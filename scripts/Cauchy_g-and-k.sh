#!/bin/bash

# increasing number of samples in observation

model=Cauchy_g-and-k

# set up folders:
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

# loop over METHODS:
METHODS=( EnergyScore KernelScore )

N_SAMPLES_IN_OBS=( 1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 )
PROPSIZES=( 1 1 1 1 0.4 0.4 0.4 0.4 0.2 0.2 0.2 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 )

FOLDER=results/${model}/${inference_folder}/
for ((k=0;k<${#METHODS[@]};++k)); do
for ((k2=0;k2<${#N_SAMPLES_IN_OBS[@]};++k2)); do

    method=${METHODS[k]}
    n_samples_in_obs=${N_SAMPLES_IN_OBS[k2]}

    echo $method $n_samples_in_obs

    PROPSIZE=${PROPSIZES[k2]}

    runcommand="python scripts/inference.py \
    $model  \
    $method  \
    --n_samples $n_samples  \
    --burnin $burnin  \
    --n_samples_per_param $n_samples_per_param \
    --n_samples_in_obs $n_samples_in_obs \
    --inference_folder $inference_folder \
    --observation_folder $observation_folder \
    --sigma 52.37 \
    --prop_size $PROPSIZE \
    --load \
     "

    if [[ "$method" == "KernelScore" ]]; then
            runcommand="$runcommand --weight 52.29"
    fi
    if [[ "$method" == "EnergyScore" ]]; then
            runcommand="$runcommand --weight 0.16"
    fi

     $runcommand >${FOLDER}out_MCMC_${method}_${n_samples_in_obs}

done
done

# Figure 5
python scripts/plot_marginals_n_obs.py $model \
    --inference_folder $inference_folder \
    --observation_folder $observation_folder \
    --thin 10 \
    --burnin $burnin \
    --n_samples $n_samples \
    --n_samples_per_param $n_samples_per_param

