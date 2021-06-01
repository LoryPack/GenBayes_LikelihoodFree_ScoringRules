#!/bin/bash

# one single sample in observation repeated N_OBSERVATIONS times

model=MG1

# set up folders:
inference_folder=inferences
observation_folder=observations
true_posterior_folder=true_posterior

mkdir results/${model} results/${model}/${inference_folder} results/${model}/${observation_folder} results/${model}/${true_posterior_folder}

# generate the observations (we generate here N_OBSERVATIONS observed datasets with 1 observation each)
echo Generate observations
python3 scripts/generate_obs.py $model --n_observations_per_param 1 --observation_folder $observation_folder

# convert the observations in matlab format in order to be able to run the MCMC inference on true posterior
python3 scripts/transform_to_mat_MG1.py  --observation_folder $observation_folder

# true posterior using matlab code:
N_STEPS_TRUE_POST=2900000
chmod +x mg1_code/do_runs_mine.sh
cd mg1_code
./do_runs_mine.sh  --n_steps $N_STEPS_TRUE_POST --n_observations 1 --true_posterior_folder  results/${model}/${true_posterior_folder}
cd ..

# transform now the result to numpy data:
python3 scripts/transform_from_mat_MG1.py --n_steps $N_STEPS_TRUE_POST --true_posterior_folder $true_posterior_folder

# now do inference
echo Inference

burnin=10000
n_samples=20000
n_samples_per_param=1000
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
        WEIGHTLIST=( 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000 )
        PROPSIZE=( 1 0.8 0.5 0.3 0.2 0.1 0.1 0.05 0.05 0.04 0.02 0.02 0.01 0.005 0.005 0.002 )
     fi

     if [[ "$method" == "EnergyScore" ]]; then
        WEIGHTLIST=( 11 14 17 20 23 26 29 32 35 38 41 44 47 50 53 56 )
        PROPSIZES=( 0.9 0.8 0.6 0.5 0.4 0.3 0.2 0.05 0.05 0.05 0.05 0.04 0.04 0.04 0.04 0.01  )
     fi

for ((l=0;l<${#WEIGHTLIST[@]};++l)); do

    WEIGHT=${WEIGHTLIST[l]}
    PROPSIZE=${PROPSIZES[l]}

    echo $method $WEIGHT

    # we fix sigma=28.6955 for the kernel SR (which is what our strategy gives) to save computational time
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
        --sigma 28.6955 \
        --add_weight_in_filename \
        --seed 123 "

    $runcommand >${FOLDER}out_MCMC_${method}_weight_${WEIGHT}

done
done

# Figure 8
python scripts/plot_bivariate.py $model \
    --inference_folder $inference_folder \
    --true_posterior_folder $true_posterior_folder \
    --n_samples $n_samples \
    --burnin $burnin \
    --n_samples_per_param $n_samples_per_param \
    --observation_folder $observation_folder

# Figure 14
python3 scripts/plot_bivariate_diff_weights.py $model \
        --inference_folder $inference_folder \
        --observation_folder $observation_folder \
        --n_samples $n_samples \
        --burnin $burnin \
        --n_samples_per_param $n_samples_per_param \
        --true_posterior_folder $true_posterior_folder > ${FOLDER}weights_results_out

