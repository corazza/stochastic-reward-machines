#!/bin/bash
exp_name="mining_jirp_traj"
today=`date '+%Y_%m_%d__%H_%M_%S'`
hourly=`date '+%Y_%m_%d__%H'`
outer_results="../mpi_results/${hourly}"
results_dir="${outer_results}/${exp_name}_${today}"

mkdir -p $outer_results
mkdir $results_dir

cd ../src
for i in `seq 1 4`; 
do
	python run.py --alg=jirp_traj --env=MiningT$i-v0 --print_freq=5000 --num_timesteps=5e6 --gamma=0.9 --results_path=${results_dir}/MiningT$i-v0.json --rm_hidden
	# python run.py --alg=jirp_traj --env=MiningT$i-v0 --print_freq=5000 --num_timesteps=1e5 --gamma=0.9 --results_path=${results_dir}/MiningT$i-v0.json --rm_hidden
done

for i in `seq 1 4`; 
do
	python run.py --alg=jirp_traj --env=MiningST$i-v0 --print_freq=5000 --num_timesteps=5e6 --gamma=0.9 --results_path=${results_dir}/MiningST$i-v0.json --rm_hidden
	# python run.py --alg=jirp_traj --env=MiningST$i-v0 --print_freq=5000 --num_timesteps=1e5 --gamma=0.9 --results_path=${results_dir}/MiningST$i-v0.json --rm_hidden
done
