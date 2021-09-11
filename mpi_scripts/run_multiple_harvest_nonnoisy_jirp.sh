#!/bin/bash
envalg_name="harvest_jirp"
. ./env.sh ${1}

cd ../src
for i in `seq 1 10`; 
do
	python run.py --alg=jirp --seed=${i} --env=Harvest5-v0 --print_freq=5000 --num_timesteps=0.25e5 --gamma=0.9 --results_path=${results_dir}/Harvest4-v0-${i}.json --rm_hidden &
done
