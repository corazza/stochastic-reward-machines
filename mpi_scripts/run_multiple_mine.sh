#!/bin/bash
envalg_name="mining_jirp_noise"
. ./env.sh ${1}

cd ../src
for i in `seq 1 10`; 
do
	python run.py --alg=jirp_noise --env=MiningT3-v0 --print_freq=5000 --num_timesteps=3e6 --gamma=0.9 --results_path=${results_dir}/MiningT3-v0-${i}.json --rm_hidden &
done
