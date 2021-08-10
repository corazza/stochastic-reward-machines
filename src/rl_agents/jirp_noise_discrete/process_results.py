import sys
import matplotlib.pyplot as plt
import json
import IPython

results_path = sys.argv[1]

print(f"Showing results for {results_path}")

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

with open(results_path) as f:
    data = json.load(f)
    print(data.keys())
    print(data['description'])

    desc = data['description']
    alg_name = desc['alg_name']
    env_name = desc['env_name']
    title = f"{alg_name} | {env_name}"

    steps = list(map(lambda x: x[1], data['step_rewards']))
    mean_rewards = list(map(lambda x: x[2], data['step_rewards']))
    steps_rebuilding = list(map(lambda x: x[1], data['step_rebuilding']))
    n_states = list(map(lambda x: x[2], data['step_rebuilding']))
    steps_corrupted = list(data["step_corrupted"])
    steps_corrupted_end = list(data["step_corrupted_end"])

    # IPython.embed()

    plt.plot(steps, mean_rewards)
    plt.ylabel("mean 100ep reward")
    plt.xlabel("steps")
    # for x in steps_rebuilding:
    #     plt.axvline(x, color="red", ls=":")
    for x in steps_corrupted:
        plt.axvline(x, color="green", ls=":")
    # for x in steps_corrupted_end:
    #     plt.axvline(x, color="blue", ls=":")
    plt.title(title)
    plt.show()

    IPython.embed()
