#!/bin/bash
envalg_name="mining_jirp_traj"
. ./env.sh ${1}

cd ../src
for i in `seq 3 3`; 
do
	python run.py --alg=jirp_traj --env=MiningT$i-v0 --print_freq=5000 --num_timesteps=5e6 --gamma=0.9 --results_path=${results_dir}/MiningT$i-v0.json --rm_hidden
done

# for i in `seq 1 4`; 
# do
# 	python run.py --alg=jirp_traj --env=MiningST$i-v0 --print_freq=5000 --num_timesteps=5e6 --gamma=0.9 --results_path=${results_dir}/MiningST$i-v0.json --rm_hidden
# done
