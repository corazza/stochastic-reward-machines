#!/bin/bash
exp_name="harvesting_jirp_noise"
today=`date '+%Y_%m_%d__%H_%M_%S'`
hourly=`date '+%Y_%m_%d__%H'`
outer_results="../mpi_results/${hourly}"
results_dir="${outer_results}/${exp_name}_${today}"

mkdir -p $outer_results
mkdir $results_dir

cd ../src
python run.py --alg=jirp_noise --env=Harvest-v0 --print_freq=5000 --num_timesteps=5e6 --gamma=0.9 --results_path=${results_dir}/Harvest-v0.json --rm_hidden
# python run.py --alg=jirp_noise --env=Harvest-v0 --print_freq=5000 --num_timesteps=1e5 --gamma=0.9 --results_path=${results_dir}/Harvest-v0.json --rm_hidden
