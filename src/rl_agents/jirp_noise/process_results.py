import sys
import matplotlib.pyplot as plt
import json

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

    plt.plot(steps, mean_rewards)
    plt.ylabel("mean 100ep reward")
    plt.xlabel("steps")
    for x in steps_rebuilding:
        plt.axvline(x, color="red")
    plt.title(title)
    plt.show()
