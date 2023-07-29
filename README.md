# Stochastic Reward Machines

## Building

You need to download and build the [Z3 theorem prover](https://github.com/Z3Prover/z3). Then in our repository `cd` into `./src/rl_agents/sjirp/cpp` and build the constraint solver:

```bash
g++ -L/home/jan/repos/z3/build -I/home/jan/repos/z3/src/api -std=c++17 main.cpp -lz3
```

Where `/home/USER_NAME/REPOS/z3` is the directory where you cloned and built Z3. Our repository includes the header file in the right place, one only needs to be able to use `-lz3`.

## Dependencies

We manually installed the [OpenAI Baselines from GitHub](https://github.com/openai/baselines), using commit `ea25b9` (Jan 31st 2020).

We provide a full list of dependencies in `./environment.yml`.

## Running experiments

We provide two scripts in the `./scripts/` directory:

- `run_parallel.sh` (runs 10 experiments in parallel)
- `run_sequential.sh` (runs 10 experiments one-by-one)

E.g. for 10 runs of S-JIRP on the Mining environment:

```bash
cd scripts
./run_parallel.sh mining sjirp
```

The first argument (`mining` in the above example) is one of:

- `mining` (regular Mining)
- `harvest` (regular Harvest)
- `mining_ns` (non-stochastic version of Mining)
- `harvest_ns` (non-stochastic version of Harvest)

The second argument (`sjirp` in the above example) is one of:

- `sjirp` (S-JIRP)
- `baseline` (the baseline algorithm)
- `jirp` (non-stochastic JIRP)

Results are saved in the `results/EXPERIMENT_ALGORITHM/DATE` directory (which will be created).

## Results processing

We provide a Python script for parsing and displaying experiment results. `cd` into `./src/` directory and run:

To reproduce figures like the ones from the paper, run:

```bash
python src/process_results.py show results/EXPERIMENT1 results/EXPERIMENT2 results/EXPERIMENT3
```

Alternatively use `save` as the first argument to produce a tex file (saved as `combined.tex`). Order matters in the last three arguments because the legend is hardcoded to label the first as S-JIRP, second as the baseline algorithm, and third as JIRP.
