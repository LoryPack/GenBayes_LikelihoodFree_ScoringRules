#!/bin/bash

# ARRAY_PARAMETERS=()
# ARRAY_VALUES=()

N_OBSERVATIONS=50
numiter=2900000
TRUE_POSTERIOR_FOLDER=results/MG1/true_posterior

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --n_observations)
    N_OBSERVATIONS="$2"
    shift # past argument
    shift # past value
    ;;
    --n_steps)
    numiter="$2"
    shift # past argument
    shift # past value
    ;;
    --true_posterior_folder)
    TRUE_POSTERIOR_FOLDER="$2"
    shift # past argument
    shift # past value
    ;;
    -h|--help)
    helpFunction
    exit 0
    ;;
    --default)
    DEFAULT=YES
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done

# echo post fold:
# echo ${TRUE_POSTERIOR_FOLDER}

# echo num iter
# echo $numiter

shift=1
range_scale=1
rate_scale=1

# Frequent

#eta_1_std=0.1701
#eta_2_std=0.2399
#eta_3_std=0.3051

#eta_prop_scale=0.7

#numupd_met=1

#shift_std=0.3
#c_range=1.008
#c_rate=1.7

# Intermediate

# not sure what these all are.

eta_1_std=0.0764
eta_2_std=0.1093
eta_3_std=0.1441

eta_prop_scale=1

numupd_met=16

shift_std=0.2
c_range=1.03
c_rate=1.004

# Rare

#eta_1_std=0.0655
#eta_2_std=0.2071
#eta_3_std=0.1403

#eta_prop_scale=1

#numupd_met=16

#shift_std=2
#c_range=1.4
#c_rate=1.00005

seed=0

for i in `seq 1 ${N_OBSERVATIONS}`

do
        #seed=`expr $i - 1`
        filename=run_mine_${shift}_${range_scale}_${rate_scale}_${numupd_met}_${i}.m
        echo "clear all;"> $filename
        echo "load data/my_observation_${i};">> $filename
        echo ''>> $filename
        echo "s = RandStream('mt19937ar','Seed',${seed});">> $filename
        echo 'RandStream.setGlobalStream(s);'>> $filename
        echo ''>> $filename
        echo "numiter = ${numiter};">> $filename
        echo "numupd_met = ${numupd_met};">> $filename
        echo ''>> $filename
        echo "shift=${shift};">> $filename
        echo "range_scale=${range_scale};">> $filename
        echo "rate_scale=${rate_scale};">> $filename
        echo ''>> $filename
        echo "shift_std = ${shift_std};">> $filename
        echo "c_range = ${c_range};">> $filename
        echo "c_rate = ${c_rate};">> $filename
        echo ''>> $filename
        echo "eta_prop_std = [${eta_1_std}, ${eta_2_std}, ${eta_3_std}]*${eta_prop_scale};">> $filename
        echo ''>> $filename
        echo "queue_met;">> $filename
        echo ''>> $filename
        echo "savefile = strcat('../${TRUE_POSTERIOR_FOLDER}/my_inference_${i}_steps_', num2str(numiter), '.mat');;">> $filename
        echo "save(savefile);">> $filename

        echo Submitting job $i
        # nohup
        matlab -nodesktop -nosplash < ${filename} >& logs/run_${shift}_${range_scale}_${rate_scale}_${numupd_met}_${i}.log; rm ${filename} &
        # octave version:
        # octave --no-gui < ${filename}; rm ${filename} &
done

wait  # wait for all to finish