import sys, os
import matplotlib
import matplotlib.pyplot as plt
import json
from types import SimpleNamespace
import IPython
import tikzplotlib
import numpy as np

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "xelatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# from rl_agents.sjirp.util import EvalResults

DPI=100
scale=0.1
FIG_SIZE=(6.4*scale, 4.8*scale)
HORIZON=None # int(3e5)
PLOT_POINTS=1000

results_path = sys.argv[1]
file_type = sys.argv[2]

# sjirp
# description = { 
#     "env_name": env.unwrapped.spec.id,
#     "alg_name": "sjirp",
#     "alg_noise_epsilon": noise_epsilon,
#     "alg_noise_delta": noise_delta,
#     "total_timesteps": total_timesteps,
# }

# baseline
# description = { 
#     "env_name": env.unwrapped.spec.id,
#     "alg_name": "baseline",
#     "alg_noise_epsilon": noise_epsilon,
#     "alg_noise_delta": noise_delta,
#     "n_samples": n_samples,
#     "total_timesteps": total_timesteps
# }

for root, dirs, files in os.walk(results_path, topdown=False):
    for name in files:
        input_filename = os.path.join(root, name)
        output_image_filename = input_filename.replace(".json", f".{file_type}", 1)
        if output_image_filename.endswith(".tikz"):
            os.remove(output_image_filename)
        if not input_filename.endswith(".json") or (os.path.isfile(output_image_filename) and file_type == "png"):
            print(f"skipping {input_filename}")
            continue
        assert output_image_filename.endswith(f".{file_type}")

        with open(input_filename) as f:
            # data = json.load(f)
            data = f.read()
            results = json.loads(data, object_hook=lambda d: SimpleNamespace(**d))
            # results = EvalResults.from_object(results)
            if HORIZON is not None:
                # results = results.filter_horizon()
                results.step_rewards = list(filter(lambda x: x[1] <= HORIZON, results.step_rewards))
                results.step_rebuilding = list(filter(lambda x: x[1] <= HORIZON, results.step_rebuilding))
                results.step_corrupted = list(filter(lambda x: x <= HORIZON, results.step_corrupted))
                results.step_corrupted_end = list(filter(lambda x: x <= HORIZON, results.step_corrupted_end))            

            desc = results.description
            alg_name = desc.alg_name
            env_name = desc.env_name
            noise_epsilon = desc.alg_noise_epsilon
            noise_delta = desc.alg_noise_delta
            title=f"{env_name}"
            # title = f"{alg_name} | {env_name} (epsilon={noise_epsilon}"
            # if alg_name == 'baseline':
            #     n_samples = desc['n_samples']
            #     title += f", delta={noise_delta}, n_samples={n_samples})"
            # else:
            #     title += ")"

            steps = list(map(lambda x: x[1], results.step_rewards))
            mean_rewards = list(map(lambda x: x[2], results.step_rewards))
            steps_rebuilding = list(map(lambda x: x[1], results.step_rebuilding))
            n_states = list(map(lambda x: x[2], results.step_rebuilding))

            plot_points = min(PLOT_POINTS, len(steps))            
            idx = np.round(np.linspace(0, len(steps) - 1, plot_points)).astype(int)
            steps = np.array(steps)
            mean_rewards = np.array(mean_rewards)
            steps = list(steps[idx])
            mean_rewards = list(mean_rewards[idx])

            plt.figure(dpi=DPI, figsize=FIG_SIZE)
            plt.plot(steps, mean_rewards, linewidth=1.0)
            plt.ylabel("mean 100ep reward")
            plt.xlabel("steps")
            for x in steps_rebuilding:
                plt.axvline(x, color="red", linewidth=0.8, linestyle=':')
            plt.title(title)
            if file_type == "tex":
                tikzplotlib.save(output_image_filename, dpi=DPI)
            else:
                plt.savefig(output_image_filename, dpi=DPI)
            plt.clf()
            plt.close()
            print(f"saved {output_image_filename}")
