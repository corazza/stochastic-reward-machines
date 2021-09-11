# Stochastic Reward Machines

## Building

You need to download and build the [Z3 theorem prover](https://github.com/Z3Prover/z3). Then in this repo `cd` into `src/rl_agents/sjirp/cpp` and build the constraint solver:

```bash
g++ -L/home/USER_NAME/REPOS/z3/build -lz3 -std=c++17 main.cpp
```

Where `/home/USER_NAME/REPOS/` is the directory where you cloned Z3. This repository includes the header file, one only needs to be able to use `-lz3`.

## Dependencies

We provide a full list of dependencies in `./environment.yml`.

## Running experiments

To run the experiments go in the `scripts/` directory and run the appropriate script with the name for the algorithm, e.g. `./run_mining_parallel sjirp`. The first and only argument can be `sjirp` (for S-JIRP), `baseline` (for the baseline algorithm), or `jirp` (for non-stochastic JIRP). The `parallel` versions of scripts run 10 experiments in forked shells, the `sequential` versions run 10 experiments one by one.

Results are saved in the `results/NAME/DATE` directory (which will be created).

## Results processing

We have a tool for parsing experiment results and producing png, pgf, or tex files. `cd` into `src/` directory and run:

To reproduce figures like the ones from the paper, run:

```bash
python src/process_results.py show results/EXPERIMENT1 results/EXPERIMENT2 results/EXPERIMENT3
```

Alternatively use `save` as the first argument to produce a tex file (saved as `combined.tex`). Order matters in the last three arguments because the legend is hardcoded to label the first as S-JIRP, second as the baseline algorithm, and third as JIRP.
