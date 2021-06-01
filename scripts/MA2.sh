#!/bin/bash

# one single sample in observation repeated N_OBSERVATIONS times

model=MA2

# set up folders:
inference_folder=inferences
observation_folder=observations
true_posterior_folder=true_posterior

mkdir results/${model} results/${model}/${inference_folder} results/${model}/${observation_folder} results/${model}/${true_posterior_folder}

# generate the observations (we generate here N_OBSERVATIONS observed datasets with 1 observation each)
echo Generate observations
python3 scripts/generate_obs.py $model --n_observations_per_param 1 --observation_folder $observation_folder

# obtain true posterior (for one single observation for the moment).
python3 scripts/true_posterior.py $model --n_samples_in_obs 1 --cores 6 --observation_folder $observation_folder --true_posterior_folder $true_posterior_folder

# now do inference
echo Inference

burnin=10000
n_samples=20000
n_samples_per_param=500
n_samples_in_obs=1

# INFERENCE WITH BSL AND SEMIBSL
METHODS=( SyntheticLikelihood semiBSL )
FOLDER=results/${model}/${inference_folder}/
for ((k=0;k<${#METHODS[@]};++k)); do

    method=${METHODS[k]}
    echo $method

     if [[ "$method" == "SyntheticLikelihood" ]]; then
            PROPSIZE=1
     fi

     if [[ "$method" == "semiBSL" ]]; then
            PROPSIZE=0.2
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
     >${FOLDER}out_MCMC_${method}_steps_${n_samples}
done

# INFERENCE WITH SCORING RULES WITH DIFFERENT WEIGHTS
# estimate the weight for all my methods wrt SyntheticLikelihood; this however leads to broad posterior, so that we'll
# try with different weights.
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

# Perform inference with different weights
FOLDER=results/${model}/${inference_folder}/

for ((j=0;j<${#METHODS[@]};++j)); do
    method=${METHODS[j]}

     if [[ "$method" == "KernelScore" ]]; then
        WEIGHTLIST=( 250 300 350 400 450 500 550 600 620 640 )
        PROPSIZE=( 1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.15 0.15 )
     fi

     if [[ "$method" == "EnergyScore" ]]; then
        WEIGHTLIST=( 12 14 16 18 20 22 24 26 28 30 )
        PROPSIZES=( 0.3 0.3 0.3 0.3 0.15 0.15 0.15 0.15 0.1 0.1 )
     fi

for ((l=0;l<${#WEIGHTLIST[@]};++l)); do

    WEIGHT=${WEIGHTLIST[l]}
    PROPSIZE=${PROPSIZES[l]}

    echo $method $WEIGHT

    # we fix sigma=12.7715 for the kernel SR (which is what our strategy gives) to save computational time
    runcommand="python scripts/inference.py \
        $model \
        $method \
        --n_samples $n_samples  \
        --burnin $burnin  \
        --n_samples_per_param $n_samples_per_param \
        --n_samples_in_obs $n_samples_in_obs \
        --inference_folder $inference_folder \
        --observation_folder $observation_folder \
        --weight $WEIGHT \
        --prop_size $PROPSIZE \
        --load \
        --sigma 12.7715 \
        --add_weight_in_filename \
        --seed 123 "

    $runcommand >${FOLDER}out_MCMC_${method}_weight_${WEIGHT}

done
done

# Figure 7
python scripts/plot_bivariate.py $model \
    --inference_folder $inference_folder \
    --true_posterior_folder $true_posterior_folder \
    --n_samples $n_samples \
    --burnin $burnin \
    --n_samples_per_param $n_samples_per_param \
    --observation_folder $observation_folder

# Figure 13
python3 scripts/plot_bivariate_diff_weights.py $model \
        --inference_folder $inference_folder \
        --observation_folder $observation_folder \
        --n_samples $n_samples \
        --burnin $burnin \
        --n_samples_per_param $n_samples_per_param \
        --true_posterior_folder $true_posterior_folder > ${FOLDER}weights_results_out

