import sys, os
import matplotlib
import matplotlib.pyplot as plt
import json
from types import SimpleNamespace
import IPython
import tikzplotlib
import random
import numpy as np
from os import walk

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "xelatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

# from rl_agents.jirp_noise.util import EvalResults

title=f"Mining (no noise)"
# DPI=600
# scale=1
# FIG_SIZE=(6.4*scale, 4.8*scale)
HORIZON=None # 0.3e5 # int(3e5)
PLOT_POINTS=250
output_image_filename = "combined.tex"

action = sys.argv[1]

results_paths = [sys.argv[2]]
if len(sys.argv) >= 4:
    results_paths.append(sys.argv[3])
if len(sys.argv) >= 5:
    results_paths.append(sys.argv[4])

example_result_is = [random.randint(0, 5), random.randint(0, 5), random.randint(0, 9)]
colors = ["blue", "green", "red"]
labels = ["main", "baseline", "JIRP"]
print(f"example_result_is: {example_result_is}")

def apply_horizon(results, horizon):
    results.step_rewards = list(filter(lambda x: x[1] <= horizon, results.step_rewards))
    results.step_rebuilding = list(filter(lambda x: x[1] <= horizon, results.step_rebuilding))
    results.step_corrupted = list(filter(lambda x: x <= horizon, results.step_corrupted))
    results.step_corrupted_end = list(filter(lambda x: x <= horizon, results.step_corrupted_end))            

def read_results(results_path):
    results = list()
    for root, dirs, files in os.walk(results_path, topdown=False):
        for name in files:
            input_filename = os.path.join(root, name)
            if not input_filename.endswith(".json"):
                print(f"skipping {input_filename}")
                continue
            print(f"using {input_filename}")

            with open(input_filename) as f:
                data = f.read()
                data = json.loads(data, object_hook=lambda d: SimpleNamespace(**d))
                # results = EvalResults.from_object(results)
                if HORIZON is not None:
                    apply_horizon(data, HORIZON)
                results.append(data)

    smallest = min(map(lambda x: x.step_rewards[-1][1], results))
    for result in results:
        apply_horizon(result, smallest)

    return results

def compute_lmh(results):
    lows = list()
    mids = list()
    highs = list()

    for i in range(0, len(results[0].step_rewards)):
        steps = [result.step_rewards[i][1] for result in results]
        assert len(set(steps)) == 1
        step = steps[0]
        datapoints = np.array([result.step_rewards[i][2] for result in results])

        low = np.percentile(datapoints, 25)
        mid = np.percentile(datapoints, 50)
        high = np.percentile(datapoints, 75)

        lows.append((step, low))
        mids.append((step, mid))
        highs.append((step, high))

    return (lows, mids, highs)

def thin_results(results, steps, lows, mids, highs, example_result_i):
    mean_rewards = list(map(lambda x: x[2], results[example_result_i].step_rewards))
    lows = list(map(lambda x: x[1], lows))
    mids = list(map(lambda x: x[1], mids))
    highs = list(map(lambda x: x[1], highs))
    plot_points = min(PLOT_POINTS, len(steps))
    idx = np.round(np.linspace(0, len(steps) - 1, plot_points)).astype(int)
    steps = np.array(steps)
    steps = list(steps[idx])

    mean_rewards = np.array(mean_rewards)
    mean_rewards = list(mean_rewards[idx])
    lows = np.array(lows)
    lows = list(lows[idx])
    mids = np.array(mids)
    mids = list(mids[idx])
    highs = np.array(highs)
    highs = list(highs[idx])

    return (steps, mean_rewards, lows, mids, highs)

def read(results_path, example_result_i):
    results = read_results(results_path)
    steps_rebuilding = list(map(lambda x: x[1], results[example_result_i].step_rebuilding))
    (lows, mids, highs) = compute_lmh(results)
    steps = list(map(lambda x: x[1], results[0].step_rewards))
    (steps, mean_rewards, lows, mids, highs) = thin_results(results, steps, lows, mids, highs, example_result_i)
    return (steps, mean_rewards, lows, mids, highs, steps_rebuilding)

to_plot = list()
for i in range(0, len(results_paths)):
    to_plot.append(read(results_paths[i], example_result_is[i]))

if action=="save":
    # plt.figure(dpi=DPI, figsize=FIG_SIZE)
    plt.figure()
elif action == "show":
    plt.figure()
else:
    raise ValueError("action must be save or show")

plt.ylabel("mean 100ep reward")
plt.xlabel("steps")
plt.title(title)

for i in range(0, len(to_plot)):
    (steps, mean_rewards, lows, mids, highs, steps_rebuilding) = to_plot[i]
    plt.plot(steps, mids, linewidth=1.0, color=colors[i], label=labels[i])
    plt.fill_between(steps, lows, highs, alpha=0.3, color=colors[i], lw=0)
    # for x in steps_rebuilding:
    #     plt.axvline(x, color=colors[i], linewidth=0.8, linestyle=':')

# if results_path_2 is not None:
plt.legend(loc="lower right")

if action=="save":
    # tikzplotlib.save(output_image_filename, dpi=DPI)
    tikzplotlib.save(output_image_filename)
    plt.clf()
    plt.close()
    print(f"saved {output_image_filename}")
else:
    plt.show()
