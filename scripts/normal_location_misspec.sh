#!/bin/bash

# inference with different outlier proportion and location

model=normal_location_misspec

# set up folders:
inference_folder=inferences_start_from_prior
observation_folder=observations
true_posterior_folder=true_posterior

mkdir results/${model} results/${model}/${inference_folder} results/${model}/${observation_folder} results/${model}/${true_posterior_folder}

# GENERATE OBSERVATIONS (each observed dataset has 100 observations)
echo Generate observations
# well specified observation (no outliers):
python3 scripts/generate_obs_misspec.py $model --n_observations_per_param 100 --observation_folder $observation_folder --epsilon 0

EPSILON_VALUES=( 0.1 0.2 )
LOCATION_VALUES=( 3 5 7 10 20 )

for ((k=0;k<${#EPSILON_VALUES[@]};++k)); do
for ((k2=0;k2<${#LOCATION_VALUES[@]};++k2)); do

    epsilon=${EPSILON_VALUES[k]}
    location=${LOCATION_VALUES[k2]}

    python3 scripts/generate_obs_misspec.py $model --n_observations_per_param 100 --observation_folder $observation_folder \
     --epsilon $epsilon  --outliers_location $location

done
done

# TRUE POSTERIOR:
CORES=6
N_SAMPLES=10000
BURNIN=10000

# no outliers:
echo Sampling true posterior
echo No outliers
python3 scripts/true_posterior.py $model --n_samples_in_obs 100  --plot_post --cores $CORES --n_samples $N_SAMPLES --burnin $BURNIN --observation_folder $observation_folder --true_posterior_folder $true_posterior_folder

for ((k=0;k<${#EPSILON_VALUES[@]};++k)); do
for ((k2=0;k2<${#LOCATION_VALUES[@]};++k2)); do

    epsilon=${EPSILON_VALUES[k]}
    location=${LOCATION_VALUES[k2]}

    echo eps=$epsilon y=$location

    python3 scripts/true_posterior.py $model --n_samples_in_obs 100  --plot_post --cores $CORES --n_samples $N_SAMPLES --burnin $BURNIN  --observation_folder $observation_folder --true_posterior_folder $true_posterior_folder \
     --epsilon $epsilon \
     --outliers_location $location

done
done

# LFI
echo Inference with my methods

burnin=40000
n_samples=20000
n_samples_per_param=500
n_samples_in_obs=100
NGROUP=50
prop_size=2.0

# loop over METHODS:

METHODS=( SyntheticLikelihood EnergyScore KernelScore )
FOLDER=results/${model}/${inference_folder}/

for ((j=0;j<${#METHODS[@]};++j)); do

    method=${METHODS[j]}
    echo $method

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
        --n_groups $NGROUP \
        --plot_trace \
        --prop_size ${prop_size}"

    if [[ "$method" == "KernelScore" ]]; then
            runcommand="$runcommand --weight 2.8"
    fi

    # run with no outliers
    echo No outliers
    $runcommand >${FOLDER}out_MCMC_${method}_no_outliers

    # and with outliers
    for ((k=0;k<${#EPSILON_VALUES[@]};++k)); do
    for ((k2=0;k2<${#LOCATION_VALUES[@]};++k2)); do

        epsilon=${EPSILON_VALUES[k]}
        location=${LOCATION_VALUES[k2]}

        echo eps=$epsilon y=$location

        $runcommand --epsilon $epsilon --outliers_location $location --seed $(( 2 * ${k}+${k2}+42 )) \
         >${FOLDER}out_MCMC_${method}_epsilon_${epsilon}_location_${location}
    done
    done
done

# PLOTS
echo PLOTS
# Figures 5 and 11
python3 scripts/plot_location_normal_misspec.py $model \
        --inference_folder $inference_folder \
        --observation_folder $observation_folder \
        --n_samples $n_samples \
        --n_samples_per_param $n_samples_per_param \
        --burnin $burnin \
        --true_posterior_folder $true_posterior_folder

# # Figure 12 and 13
python3 scripts/plot_location_normal_misspec_SL.py $model \
        --inference_folder $inference_folder \
        --observation_folder $observation_folder \
        --n_samples $n_samples \
        --n_samples_per_param $n_samples_per_param \
        --burnin $burnin \
        --true_posterior_folder $true_posterior_folder


# REPEAT LFI INFERENCES STARTING FROM OUTLIER LOCATION RATHER THAN PRIOR (TO CHECK MCMC CONVERGENCE)
inference_folder2=inferences_start_outlier_location/
mkdir results/${model}/${inference_folder}
FOLDER2=results/${model}/${inference_folder2}/

for ((j=0;j<${#METHODS[@]};++j)); do

    method=${METHODS[j]}
    echo $method

    runcommand="python scripts/inference.py \
        $model  \
        $method  \
        --n_samples $n_samples  \
        --burnin $burnin  \
        --n_samples_per_param $n_samples_per_param \
        --n_samples_in_obs $n_samples_in_obs \
        --inference_folder $inference_folder2 \
        --observation_folder $observation_folder \
        --load \
        --n_groups $NGROUP \
        --prop_size ${prop_size}"

    if [[ "$method" == "KernelScore" ]]; then
            runcommand="$runcommand --weight 2.8"
    fi

    # copy the results without outliers from the previous simulation:
    cp ${FOLDER}/*epsilon_0.0*jnl ${FOLDER2}/.

    # run with outliers
    for ((k=0;k<${#EPSILON_VALUES[@]};++k)); do
    for ((k2=0;k2<${#LOCATION_VALUES[@]};++k2)); do

        epsilon=${EPSILON_VALUES[k]}
        location=${LOCATION_VALUES[k2]}

        runcommand2="$runcommand --inipoint ${location}"

        echo eps=$epsilon y=$location

        $runcommand2 --epsilon $epsilon --outliers_location $location --seed $(( 12 * ${k}+${k2}+42 )) \
         >${FOLDER2}out_MCMC_${method}_epsilon_${epsilon}_location_${location}
    done
    done
done

# PLOTS
python3 scripts/plot_location_normal_misspec.py $model \
        --inference_folder $inference_folder2 \
        --observation_folder $observation_folder \
        --n_samples $n_samples \
        --n_samples_per_param $n_samples_per_param \
        --burnin $burnin \
        --true_posterior_folder $true_posterior_folder