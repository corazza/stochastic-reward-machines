#!/bin/bash
cd ../src
for i in `seq 1 4`; 
do
	python run.py --alg=jirp_traj --env=MiningT$i-v0 --print_freq=5000 --num_timesteps=5e6 --gamma=0.9 --results_path=results/long4/jirp_traj_$i.json --rm_hidden
done


for i in `seq 1 4`; 
do
	python run.py --alg=jirp_traj --env=MiningST$i-v0 --print_freq=5000 --num_timesteps=5e6 --gamma=0.9 --results_path=results/long4/jirp_traj_slip_$i.json --rm_hidden
done
