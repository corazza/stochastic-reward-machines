#!/bin/bash
exp_name="mining"
. ./env.sh ${1}

cd ../src
for i in `seq 1 10`; 
do
	python run.py --alg=${1} --seed=${i} --env=MiningT3-v0 --print_freq=100000 --num_timesteps=3e6 --gamma=0.9 --results_path=${results_dir}/MiningT3-v0-${i}.json --rm_hidden &
done

# python run.py --alg=sjirp --seed=123 --env=MiningT3-v0 --print_freq=100000 --num_timesteps=3e6 --gamma=0.9 --results_path=AAAMINING.json --rm_hidden
