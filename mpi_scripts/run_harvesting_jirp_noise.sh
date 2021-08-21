#!/bin/bash
envalg_name="harvesting_jirp_noise"
. ./env.sh ${1}

cd ../src
for i in `seq 1 4`;
do
    python run.py --alg=jirp_noise --env=Harvest$i-v0 --print_freq=5000 --num_timesteps=3e5 --gamma=0.9 --results_path=${results_dir}/Harvest$i-v0.json --rm_hidden
    # python run.py --alg=jirp_noise --env=Harvest$i-v0 --print_freq=5000 --num_timesteps=1e5 --gamma=0.9 --results_path=${results_dir}/Harvest$i-v0.json --rm_hidden
done
