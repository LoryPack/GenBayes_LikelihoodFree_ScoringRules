#!/bin/bash
# this is to test performance with different values of m.
# this scripts produces Figures from 22 to 27 in the Appendix

# increasing number of samples in observation
MODELS=( g-and-k Cauchy_g-and-k univariate_g-and-k univariate_Cauchy_g-and-k MA2 MG1 )
N_SAMPLES_PER_PARAM=( 10 20 50 100 200 300 400 500 600 700 800 900 1000 )  # values of m
for ((i=0;i<${#MODELS[@]};++i)); do

    model=${MODELS[i]}

    # set up folders:
    inference_folder=inferences_different_n_sim
    observation_folder=observations

    mkdir results results/${model} results/${model}/${inference_folder} results/${model}/${observation_folder}

    # generate the observations (we generate here 1 observed datasets with 100 iid observations)
    echo Generate observations
    python3 scripts/generate_obs.py $model --n_observations_per_param 100 --observation_folder $observation_folder

    # now do inference with our methods
    echo Inference

    burnin=10000
    n_samples=100000
    n_samples_in_obs=10

    # loop over METHODS:
    if [[ "$model" == *"univariate"* ]]; then
        METHODS=( SyntheticLikelihood EnergyScore KernelScore )  # for univariate_g-and-k and univariate_Cauchy_g-and-k
    fi
    if [[ $model == "g-and-k" ]]; then
        METHODS=( SyntheticLikelihood EnergyScore KernelScore semiBSL )
    fi
    if [[ $model == "Cauchy_g-and-k" ]]; then
        METHODS=( EnergyScore KernelScore )
    fi
    if [[ "$model" == "M"* ]]; then
        n_samples=20000
        n_samples_in_obs=1
        METHODS=( SyntheticLikelihood EnergyScore KernelScore semiBSL )
    fi

    FOLDER=results/${model}/${inference_folder}/
    for ((k=0;k<${#METHODS[@]};++k)); do
            method=${METHODS[k]}
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
            --plot_trace"

        PROPSIZE=1 # we take prop_size=1 for all values of m except for some specific cases
        if [[ "$model" == *"univariate"* ]]; then
            if [[ "$method" == "KernelScore" ]]; then
                runcommand="$runcommand --weight 18.30"
                runcommand="$runcommand --sigma 5.5"
            fi
            if [[ "$method" == "EnergyScore" ]]; then
                runcommand="$runcommand --weight 0.35"
            fi
        fi
        if [[ $model == "g-and-k" ]] || [[ $model == "Cauchy_g-and-k" ]]; then
            if [[ "$method" == "KernelScore" ]]; then
                runcommand="$runcommand --weight 52.29"
                runcommand="$runcommand --sigma 52.37"
            fi
            if [[ "$method" == "EnergyScore" ]]; then
                runcommand="$runcommand --weight 0.16"
            fi
        fi
        if [[ $model == "MA2" ]]; then
            if [[ "$method" == "KernelScore" ]]; then
                runcommand="$runcommand --weight 207.6652"
                runcommand="$runcommand --sigma 12.7715"
            fi
            if [[ "$method" == "EnergyScore" ]]; then
                runcommand="$runcommand --weight 12.9689"
            fi
            if [[ "$method" == "semiBSL" ]]; then
                PROPSIZE=0.2
            fi
        fi
        if [[ $model == "MG1" ]]; then
            if [[ "$method" == "KernelScore" ]]; then
                runcommand="$runcommand --weight 200"
                runcommand="$runcommand --sigma 3.6439"
                PROPSIZE=0.3
            fi
            if [[ "$method" == "EnergyScore" ]]; then
                runcommand="$runcommand --weight 10.9802"
            fi
            if [[ "$method" == "semiBSL" ]]; then
                PROPSIZE=0.2
            fi
        fi

        runcommand="$runcommand --prop_size $PROPSIZE"

        $runcommand  >${FOLDER}out_MCMC_${method}_${n_samples_per_param} &

        done
        wait
    done

    # plots:

    # Figure ??
    if [[ $model == *"g-and-k"* ]]; then
        python scripts/plot_marginals_n_simulations.py $model \
        --inference_folder $inference_folder \
        --observation_folder $observation_folder \
        --thin 10 \
        --burnin $burnin \
        --n_samples $n_samples > ${FOLDER}m_results_out
    else
        python3 scripts/plot_bivariate_diff_n_simulations.py $model \
        --inference_folder $inference_folder \
        --observation_folder $observation_folder \
        --thin 10 \
        --burnin $burnin \
        --n_samples $n_samples  > ${FOLDER}m_results_out
    fi

done