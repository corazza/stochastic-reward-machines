#!/bin/bash
cd ../src
python run.py --alg=jirp_noise --env=Garden-v0 --print_freq=5000 --num_timesteps=5e6 --gamma=0.9 --results_path=results/garden1/jirp_noise.json --no_rm

python run.py --alg=jirp_noise --env=GardenS-v0 --print_freq=5000 --num_timesteps=5e6 --gamma=0.9 --results_path=results/garden1/jirp_noise_slip.json --no_rm
