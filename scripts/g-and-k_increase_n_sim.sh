#!/bin/bash

# increasing number of samples in observation

model=g-and-k

# set up folders:
inference_folder=inferences_increase_n_sim
observation_folder=observations

mkdir results results/${model} results/${model}/${inference_folder} results/${model}/${observation_folder}

# generate the observations (we generate here 1 observed datasets with 100 iid observations)
echo Generate observations
python3 scripts/generate_obs.py $model --n_observations_per_param 100 --observation_folder $observation_folder

# now do inference with our methods
echo Inference

burnin=0
n_samples=10000
NGROUP=500

# loop over METHODS:
METHODS=( SyntheticLikelihood semiBSL )
n_samples_in_obs=20

FOLDER=results/${model}/${inference_folder}/
for ((k=0;k<${#METHODS[@]};++k)); do
    method=${METHODS[k]}
    N_SAMPLES_PER_PARAM=( 500 1000 1500 2000 2500 3000 30000 )

    for ((k2=0;k2<${#N_SAMPLES_PER_PARAM[@]};++k2)); do

        n_samples_per_param=${N_SAMPLES_PER_PARAM[k2]}
        echo $method $n_samples_per_param

        runcommand="python scripts/inference.py \
        $model  \
        $method  \
        --n_samples $n_samples  \
        --burnin $burnin  \
        --n_samples_per_param $n_samples_per_param \
        --n_samples_in_obs $n_samples_in_obs \
        --inference_folder $inference_folder \
        --observation_folder $observation_folder \
        --load \
        --prop_size 0.4 \
        --sigma 52.37 \
        --n_group $NGROUP "


    $runcommand  >${FOLDER}out_MCMC_${method}_${n_samples_per_param}

    done
done

# Figure 10
python scripts/plot_traces_increase_n_sim.py $model \
    --inference_folder $inference_folder \
    --observation_folder $observation_folder \
    --burnin $burnin --n_samples $n_samples \
    --n_samples_per_param $n_samples_per_param > ${FOLDER}/acc_rates.txt
