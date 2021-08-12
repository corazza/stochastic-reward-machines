import sys, os
import matplotlib.pyplot as plt
import json

DPI=600
scale=1.5
FIG_SIZE=(6.4*scale, 4.8*scale)

results_path = sys.argv[1]

# jirp_noise
# description = { 
#     "env_name": env.unwrapped.spec.id,
#     "alg_name": "jirp_noise",
#     "alg_noise_epsilon": noise_epsilon,
#     "alg_noise_delta": noise_delta,
#     "total_timesteps": total_timesteps,
# }

# jirp_traj
# description = { 
#     "env_name": env.unwrapped.spec.id,
#     "alg_name": "jirp_traj",
#     "alg_noise_epsilon": noise_epsilon,
#     "alg_noise_delta": noise_delta,
#     "n_samples": n_samples,
#     "total_timesteps": total_timesteps
# }


for root, dirs, files in os.walk(results_path, topdown=False):
    for name in files:
        input_filename = os.path.join(root, name)
        if not input_filename.endswith(".json"):
            print(f"skipping {input_filename}")
            continue
        output_image_filename = input_filename.replace(".json", ".png", 1)
        assert output_image_filename.endswith(".png")

        with open(input_filename) as f:
            data = json.load(f)
            # print(data.keys())
            # print(data['description'])

            desc = data['description']
            alg_name = desc['alg_name']
            env_name = desc['env_name']
            noise_epsilon = desc['alg_noise_epsilon']
            noise_delta = desc['alg_noise_delta']
            title = f"{alg_name} | {env_name} (epsilon={noise_epsilon}"

            if alg_name == 'jirp_traj':
                n_samples = desc['n_samples']
                title += f", delta={noise_delta}, n_samples={n_samples})"
            else:
                title += ")"

            steps = list(map(lambda x: x[1], data['step_rewards']))
            mean_rewards = list(map(lambda x: x[2], data['step_rewards']))
            steps_rebuilding = list(map(lambda x: x[1], data['step_rebuilding']))
            n_states = list(map(lambda x: x[2], data['step_rebuilding']))

            plt.figure(dpi=DPI, figsize=FIG_SIZE)
            plt.plot(steps, mean_rewards, linewidth=1.0)
            plt.ylabel("mean 100ep reward")
            plt.xlabel("steps")
            for x in steps_rebuilding:
                plt.axvline(x, color="red", linewidth=0.8, linestyle=':')
            plt.title(title)
            plt.savefig(output_image_filename, dpi=DPI)
            plt.clf()
            print(f"saved {output_image_filename}")
