#!/bin/bash
exp_name=${1}
alg_name=${2}
. ./env.sh

cd ../src
for i in `seq 1 10`; 
do
	python run.py --alg=${alg_name} --seed=${i} --env=${env_name} --print_freq=${printfreq} --num_timesteps=${timesteps} --gamma=0.9 --results_path=${results_dir}/${env_name}-${i}.json --rm_hidden
done
