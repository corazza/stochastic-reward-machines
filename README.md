# Stochastic Reward Machines

To run the experiments go in the `mpi_scripts/` directory and run the appropriate script. The first and only argument is the experiment name. Results are saved in the `mpi_results/EXPERIMENT_NAME/` directory (which will be created).

1. For Mining environment with main algorithm: `./run_mining_jirp_noise.sh EXPERIMENT_NAME`
2. For Mining environment with naive algorithm: `./run_mining_jirp_traj.sh EXPERIMENT_NAME`
3. For Mining environment with JIRP: `./run_mining_jirp.sh EXPERIMENT_NAME`

Etc.

## Results processing

We have a tool for parsing experiment results and producing png, pgf, or tex files. `cd` into `src/` directory and run:

```bash
python rl_agents/jirp_noise/create_images.py ../mpi_results/EXPERIMENT_NAME tex
```

Where "tex" can be replaced with "png" or "pgf" as desired.
