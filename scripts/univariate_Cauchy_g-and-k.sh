#!/bin/bash

# increasing number of samples in observation
model=univariate_Cauchy_g-and-k

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
NGROUP=500

# loop over METHODS:

METHODS=( SyntheticLikelihood EnergyScore KernelScore )
N_SAMPLES_IN_OBS=( 1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 )

PROPSIZES_BSL=( 1 1 1 1 1 1 0.4 0.4 0.4 0.4 0.4 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 )
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
     echo prop_size $PROPSIZE

    runcommand="python scripts/inference.py \
    $model  \
    $method  \
    --n_samples $n_samples  \
    --burnin $burnin  \
    --n_samples_per_param $n_samples_per_param \
    --n_samples_in_obs $n_samples_in_obs \
    --inference_folder $inference_folder \
    --observation_folder $observation_folder \
    --sigma 5.5 \
    --prop_size $PROPSIZE \
    --load \
    --n_groups $NGROUP \
     "

    if [[ "$method" == "KernelScore" ]]; then
            runcommand="$runcommand --weight 18.30"
    fi
    if [[ "$method" == "EnergyScore" ]]; then
            runcommand="$runcommand --weight 0.35"
    fi

    $runcommand >${FOLDER}out_MCMC_${method}_${n_samples_in_obs}

done
done

# Figure 3
python scripts/plot_marginals_n_obs.py $model \
    --inference_folder $inference_folder \
    --observation_folder $observation_folder \
    --thin 10 \
    --burnin $burnin \
    --n_samples $n_samples \
    --n_samples_per_param $n_samples_per_param  > ${FOLDER}/acc_rates.txt


n_samples_in_obs=100  # consider the largest number of observations.
python scripts/predictive_validation_SRs.py \
    $model  \
    --n_samples $n_samples  \
    --burnin $burnin  \
    --n_samples_per_param $n_samples_per_param \
    --n_samples_in_obs $n_samples_in_obs \
    --inference_folder $inference_folder \
    --observation_folder $observation_folder \
    --seed 456 \
    --subsample 10000 \
    --load \
    --sigma 5.5 >  ${FOLDER}/predictive.txt
