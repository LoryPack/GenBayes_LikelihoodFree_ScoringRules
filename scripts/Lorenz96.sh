#!/bin/bash

# increasing number of samples in observation

model=Lorenz96

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

    # Figure 14
    python3 scripts/generate_obs_misspec.py $model --n_observations_per_param 10 --observation_folder $observation_folder --epsilon $epsilon

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
# Energy Score
# Estimated w 3.4882; it took 265.9869 seconds
# KernelScore
# Estimated sigma for kernel score 0.8610; it took 121.0191 seconds
# Estimated w 20.7138; it took 272.0315 seconds

# now do inference with our methods
echo Inference

burnin=5000
n_samples=20000
n_samples_per_param=200
NGROUP=10

# loop over METHODS:
METHODS=( EnergyScore KernelScore )
n_samples_in_obs=10
PROPSIZES=( 0.25 0.1 )

FOLDER=results/${model}/${inference_folder}/
for ((k=0;k<${#METHODS[@]};++k)); do

    method=${METHODS[k]}

    echo $method $n_samples_in_obs

    PROPSIZE=${PROPSIZES[k]}

    runcommand="python scripts/inference.py \
    $model  \
    $method  \
    --n_samples $n_samples  \
    --burnin $burnin  \
    --n_samples_per_param $n_samples_per_param \
    --n_samples_in_obs $n_samples_in_obs \
    --inference_folder $inference_folder \
    --observation_folder $observation_folder \
    --sigma 0.8610 \
    --plot_trace \
    --plot_post \
    --prop_size $PROPSIZE \
    --seed 456 \
    --load \
    --n_group $NGROUP \
     "

    if [[ "$method" == "KernelScore" ]]; then
            runcommand="$runcommand --weight 20.7138"
    fi
    if [[ "$method" == "EnergyScore" ]]; then
            runcommand="$runcommand --weight 3.4882"
    fi

    echo No outliers
    $runcommand &  >${FOLDER}${method}_no_outliers &

    # and with outliers
    for ((i=0;i<${#EPSILON_VALUES[@]};++i)); do

        epsilon=${EPSILON_VALUES[i]}

        echo eps=$epsilon

        $runcommand --epsilon $epsilon & >${FOLDER}${method}_eps_${epsilon}  &

    done
    wait


done
wait


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
        --steps $ABC_steps \
        --epsilon $EPSILON >${FOLDER}ABC_steps_${step}_eps_${EPSILON}_n_${n_samples}

done


# Figures 6a, 7a and 16
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
    --sigma 0.8610 \
    --ABC_n_samples_per_param $ABC_n_samples_per_param  >  ${FOLDER}/predictive.txt

# Figure 15
python scripts/plot_marginals_eps.py $model \
    --inference_folder $inference_folder \
    --observation_folder $observation_folder \
    --thin 10 \
    --burnin $burnin \
    --n_samples $n_samples \
    --n_samples_in_obs $n_samples_in_obs \
    --n_samples_per_param $n_samples_per_param \
    --ABC_inference_folder $ABC_inference_folder \
    --ABC_method $ABC_method \
    --ABC_steps $ABC_steps \
    --ABC_n_samples $ABC_n_samples \
    --ABC_n_samples_per_param $ABC_n_samples_per_param > ${FOLDER}/acc_rates.txt

