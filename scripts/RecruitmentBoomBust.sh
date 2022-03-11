#!/bin/bash

# increasing number of samples in observation

model=RecruitmentBoomBust

# set up folders:
inference_folder=inferences
observation_folder=observations

mkdir results results/${model} results/${model}/${inference_folder} results/${model}/${observation_folder}

# Generate observations with outliers; we generate 10 observations, and study what happens with 1 and 2 outliers.
echo Generate observations
EPSILON_VALUES=( 0 0.1 0.2 )

for ((k=0;k<${#EPSILON_VALUES[@]};++k)); do

    epsilon=${EPSILON_VALUES[k]}
    location=${LOCATION_VALUES[k2]}

    python3 scripts/generate_obs_misspec.py $model --n_observations_per_param 10 --observation_folder $observation_folder  --epsilon $epsilon

done


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

# now do inference with our methods
echo Inference

burnin=5000
n_samples=50000
n_samples_per_param=200  # should I increase this?
NGROUP=10

# experiment with multiple obs, some of them being outliers

# loop over METHODS:
METHODS=( EnergyScore KernelScore )
PROPSIZES=( 0.005 0.0125 )
n_samples_in_obs=10

FOLDER=results/${model}/${inference_folder}/
for ((k=0;k<${#METHODS[@]};++k)); do

    method=${METHODS[k]}
    PROPSIZE=${PROPSIZES[k]}

    echo $method $n_samples_in_obs

    runcommand="python scripts/inference.py \
    $model  \
    $method  \
    --n_samples $n_samples  \
    --burnin $burnin  \
    --n_samples_per_param $n_samples_per_param \
    --n_samples_in_obs $n_samples_in_obs \
    --inference_folder $inference_folder \
    --observation_folder $observation_folder \
    --sigma 23.3363 \
    --plot_trace \
    --seed 456 \
    --load \
    --add_weight_in_filename \
    --prop_size $PROPSIZE \
    --n_group $NGROUP "


    if [[ "$method" == "KernelScore" ]]; then
            runcommand="$runcommand --weight 20"
    fi
    if [[ "$method" == "EnergyScore" ]]; then
            runcommand="$runcommand --weight 1"
    fi

    echo No outliers
    $runcommand >${FOLDER}out_MCMC_${method}_no_outliers &

    # and with outliers
    for ((i=0;i<${#EPSILON_VALUES[@]};++i)); do

        epsilon=${EPSILON_VALUES[i]}

        echo eps=$epsilon

        $runcommand --epsilon $epsilon --seed $(( 2 * ${i}+42 )) \
         >${FOLDER}out_MCMC_${method}_epsilon_${epsilon}  &
    done
    wait


done

# ABC inference:
ABC_inference_folder=ABC_inference
FOLDER=results/${model}/${inference_folder}/
mkdir $FOLDER

ABC_method=ABC
ABC_n_samples=1000
ABC_n_samples_per_param=10
ABC_steps=25
for ((k2=0;k2<${#EPSILON_VALUES[@]};++k2)); do

        EPSILON=${EPSILON_VALUES[k2]}

        echo $EPSILON

        mpirun -n 8 \
        python scripts/abc_inference.py \
        $model  \
        $ABC_method  \
        --n_samples $ABC_n_samples  \
        --n_samples_per_param $ABC_n_samples_per_param \
        --n_samples_in_obs $n_samples_in_obs \
        --inference_folder $ABC_inference_folder \
        --observation_folder $observation_folder \
        --seed 1234 \
        --plot_post \
        --use_MPI \
        --steps $ABC_steps \
        --epsilon $EPSILON >${FOLDER}ABC_steps_${step}_eps_${EPSILON}_n_${n_samples}

done


# Figures 6b and 7b
mpirun -n 8 python scripts/predictive_validation_SRs.py \
    $model  \
    --n_samples $n_samples  \
    --burnin $burnin  \
    --n_samples_per_param $n_samples_per_param \
    --n_samples_in_obs $n_samples_in_obs \
    --inference_folder $inference_folder \
    --observation_folder $observation_folder \
    --seed 456 \
    --subsample 1000 \
    --use_MPI \
    --ABC_inference_folder $ABC_inference_folder \
    --ABC_method $ABC_method \
    --ABC_steps $ABC_steps \
    --ABC_n_samples $ABC_n_samples \
    --load \
    --sigma 23.3363 \
    --ABC_n_samples_per_param $ABC_n_samples_per_param   >  ${FOLDER}/predictive.txt


# Figure 17
python scripts/plot_marginals_eps.py $model \
    --inference_folder $inference_folder \
    --observation_folder $observation_folder \
    --thin 100 \
    --burnin $burnin \
    --n_samples $n_samples \
    --n_samples_in_obs $n_samples_in_obs \
    --n_samples_per_param $n_samples_per_param \
    --ABC_inference_folder $ABC_inference_folder \
    --ABC_method $ABC_method \
    --ABC_steps $ABC_steps \
    --ABC_n_samples $ABC_n_samples \
    --ABC_n_samples_per_param $ABC_n_samples_per_param   > ${FOLDER}/acc_rates.txt


