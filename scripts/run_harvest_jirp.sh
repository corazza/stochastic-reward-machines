#!/bin/bash
envalg_name="harvest_jirp"
. ./env.sh ${1}

cd ../src
for i in 2 4;
do
    python run.py --alg=jirp --env=Harvest$i-v0 --print_freq=5000 --num_timesteps=3e6 --gamma=0.9 --results_path=${results_dir}/Harvest$i-v0.json --rm_hidden
done
