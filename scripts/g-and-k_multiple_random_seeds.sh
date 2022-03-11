#!/bin/bash

# increasing number of samples in observation
model=g-and-k

# set up folders:
inference_folder=inferences_multiple_seeds
observation_folder=observations

mkdir results results/${model} results/${model}/${inference_folder} results/${model}/${observation_folder}

# generate the observations (we generate here 1 observed datasets with 100 iid observations)
echo Generate observations
python3 scripts/generate_obs.py $model --n_observations_per_param 100 --observation_folder $observation_folder

# now do inference with our methods
echo Inference

burnin=0
n_samples=10000
n_samples_per_param=500
NGROUP=500

# loop over METHODS:

METHODS=( SyntheticLikelihood semiBSL )

n_samples_in_obs=20
prop_size=0.4
SEEDLIST=( 0 1 2 3 4 5 6 7 8 9 )

FOLDER=results/${model}/${inference_folder}/
for ((k=0;k<${#METHODS[@]};++k)); do
for ((k2=0;k2<${#SEEDLIST[@]};++k2)); do

    method=${METHODS[k]}
    seed=${SEEDLIST[k2]}
    echo $method $seed

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
    --seed $seed \
    --prop_size $prop_size \
    --sigma 52.37 \
    --add_seed_in_filename \
    --n_group $NGROUP "

    $runcommand >${FOLDER}out_MCMC_${method}_${n_samples_in_obs}

done
done


# Figure 9
python scripts/plot_traces_multiple_seeds.py $model \
    --inference_folder $inference_folder \
    --observation_folder $observation_folder\
    --burnin $burnin --n_samples $n_samples \
    --n_samples_per_param $n_samples_per_param


