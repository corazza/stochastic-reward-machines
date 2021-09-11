#!/bin/bash
today=`date '+%Y_%m_%d__%H_%M_%S'`
outer_results="../results/${exp_name}_${alg_name}"
results_dir="${outer_results}/${today}"

mkdir -p $outer_results
mkdir $results_dir

if [ "$exp_name" = "mining" ]; then
    env_name="MiningT3-v0"
    timesteps="3e6"
    printfreq="100000"
elif [ "$exp_name" = "mining_ns" ]; then
    env_name="MiningT5-v0."
    timesteps="3e6"
    printfreq="100000"
elif [ "$exp_name" = "harvest" ]; then
    env_name="Harvest4-v0"
    timesteps="0.25e5"
    printfreq="5000"
elif [ "$exp_name" = "harvest_ns" ]; then
    env_name="Harvest5-v0"
    timesteps="0.25e5"
    printfreq="5000"
else
    echo ERROR: wrong experiment name ${exp_name}
    exit 1
fi
