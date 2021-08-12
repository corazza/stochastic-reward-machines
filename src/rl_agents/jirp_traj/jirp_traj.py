"""
JIRP based method
"""
import math
import itertools
from profilehooks import profile
import random, time, copy
import IPython
from baselines import logger
import numpy as np
import os.path

from reward_machines.reward_machine import RewardMachine
from reward_machines.rm_environment import RewardMachineEnv, RewardMachineHidden
from rl_agents.jirp.util import sample_language, dnf_for_empty, rm_from_transitions, initial_Q, get_qmax, get_best_action, transfer_Q
from rl_agents.jirp.sat_hyp import sat_hyp
from rl_agents.jirp.consts import *
from rl_agents.jirp_noise.jirp_noise import detect_signal
from rl_agents.jirp_noise.consts import *
from rl_agents.jirp_noise.util import *



def consistent_hyp(noise_epsilon, X, X_tl, infer_termination, n_states_start=1, report=True):
    if len(X) == 0:
        transitions = dict()
        transitions[(0, tuple())] = [0, 0.0]
        return transitions, 1
    # TODO intercept empty X here
    for n_states in range(n_states_start, MAX_RM_STATES_N+1):
        if report:
            print(f"finding model with {n_states} states")
        # print("(SMT)")
        new_transitions = sat_hyp(noise_epsilon, X, X_tl, n_states, infer_termination)
        # print("(SAT)")
        # new_transitions_sat = sat_hyp(0.15, X, X_tl, n_states)
        if new_transitions is not None:
            # if new_transitions_sat is None:
            #     print(f"SAT couldn't find anything with n_states={n_states}")
            # display_transitions(new_transitions, "st")
            return new_transitions, n_states
        continue

    raise ValueError(f"Couldn't find machine with at most {MAX_RM_STATES_N} states")

def compute_n_samples(epsilon, delta):
    ci = delta/(2*4.0)
    sigma = (2*epsilon)**2 / 12
    return max(20, math.ceil(((Z_CONF * sigma) / ci)**2))

def add_to_bank(actions, labels, rewards, sequences_bank, actions_bank):
    if labels not in sequences_bank:
        sequences_bank[labels] = list()
        actions_bank[labels] = list()
    else:
        assert len(sequences_bank[labels][0]) == len(rewards)
        assert len(actions_bank[labels][0]) == len(actions)
    sequences_bank[labels].append(rewards)
    actions_bank[labels].append(actions)

def enough_in_bank(n_samples, sequences_bank, traj):
    if traj not in sequences_bank:
        return False
    return len(sequences_bank[traj]) >= n_samples

def recompute_X(n_samples, noise_delta, X, sequences_bank):
    X_result = set()
    means = set()
    for (labels, rewards) in X:
        rewards_average = list()
        samples = sequences_bank[labels]
        assert len(samples) >= n_samples
        for i in range(0, len(rewards)):
            rewards_average.append(list())
            for j in range(0, len(samples)):
                rewards_average[i].append(samples[j][i])
            rewards_average[i] = sorted(rewards_average[i])
            rewards_average[i] = (rewards_average[i][0] + rewards_average[i][-1]) / 2.0
            means.add(rewards_average[i])
        X_result.add((labels, tuple(rewards_average)))
    
    means = list(means)
    groups = list(range(0, len(means)))
    for i in range(0, len(means)):
        for j in range(i+1, len(means)):
            if abs(means[i] - means[j]) < noise_delta/2: # confidence intervals overlap
                groups[j] = groups[i]
    print("means", means)

    grouped_means = dict()
    for i in range(0, len(means)):
        if groups[i] not in grouped_means:
            grouped_means[groups[i]] = set()
        grouped_means[groups[i]].add(means[i])
    print("grouped_means", grouped_means)

    group_average_set = set()
    collapsed_means = dict()
    for group in grouped_means:
        group_average = sum(grouped_means[group])/len(grouped_means[group])
        group_average_set.add(group_average)
        for mean in grouped_means[group]:
            collapsed_means[mean] = group_average
    print("collapsed means", collapsed_means)
    print("group average set", group_average_set)

    X_final = set()
    for (labels, rewards) in X_result:
        rewards_collapsed = list()
        for reward in rewards:
            rewards_collapsed.append(collapsed_means[reward])
        X_final.add((labels, tuple(rewards_collapsed)))

    return X_final

# @profile
def learn(env,
          network=None,
          seed=None,
          lr=0.1,
          total_timesteps=100000,
          epsilon=0.1,
          print_freq=10000,
          gamma=0.9,
          q_init=1.0,
          use_crm=False,
          use_rs=False,
          results_path=None):
    assert env.is_hidden_rm() # JIRP doesn't work with explicit RM environments
    assert results_path is not None

    infer_termination = TERMINATION
    try:
        infer_termination = env.infer_termination_preference()
    except:
        pass
    print(f"(alg) INFERRING TERMINATION: {infer_termination}")

    noise_epsilon, noise_delta = extract_noise_params(env)
    
    n_samples = compute_n_samples(noise_epsilon, noise_delta)
    print("alg noise epsilon:", noise_epsilon)
    print("alg noise delta:", noise_delta)
    print(f"need {n_samples} samples for 99% confidence")

    description = { 
        "env_name": env.unwrapped.spec.id,
        "alg_name": "jirp_traj",
        "alg_noise_epsilon": noise_epsilon,
        "alg_noise_delta": noise_delta,
        "n_samples": n_samples,
        "total_timesteps": total_timesteps
    }
    results = EvalResults(description)

    sequences_bank = dict()
    actions_bank = dict()
    X = set()
    X_averaged = set()
    All = set()
    X_new = set()
    X_tl = set()

    done_actions_list = []
    labels = []
    rewards = []

    transitions, n_states_last = consistent_hyp(noise_epsilon, set(), set(), infer_termination)
    language = sample_language(X)
    empty_transition = dnf_for_empty(language)
    H = rm_from_transitions(transitions, empty_transition)
    # H = load("./envs/grids/reward_machines/office/t3.txt")
    actions = list(range(env.action_space.n))
    Q = initial_Q(H)

    episode_rewards = [0.0]
    evaluation_episode_rewards = [0.0]
    step = 0
    num_episodes = 0

    following_traj = None
    last_built = 0
    which_one = None
    started_following = 0

    while step < total_timesteps:
        s = tuple(env.reset())
        rm_state = H.reset()
        done_actions_list = []
        labels = []
        rewards = []
        next_random = False

        if s not in Q[rm_state]: Q[rm_state][s] = dict([(a, q_init) for a in actions])

        while True:
            # Selecting and executing the action
            if following_traj is None:
                a = random.choice(actions) if random.random() < epsilon or next_random else get_best_action(Q[rm_state],s,actions,q_init)
            else:
                try:
                    if len(done_actions_list) < len(actions_bank[following_traj][which_one]):
                        a = actions_bank[following_traj][which_one][len(done_actions_list)]
                    else:
                        a = random.choice(actions)
                except:
                    IPython.embed()            
            sn, r, done, info = env.step(a)

            sn = tuple(sn)
            true_props = env.get_events()
            labels.append(true_props) # L(s, a, s')
            done_actions_list.append(a)
            next_rm_state, _rm_reward, rm_done = H.step(rm_state, true_props, info)

            # update Q-function of current RM state
            if s not in Q[rm_state]: Q[rm_state][s] = dict([(b, q_init) for b in actions])
            if done: _delta = r - Q[rm_state][s][a]
            else:    _delta = r + gamma*get_qmax(Q[next_rm_state], sn, actions, q_init) - Q[rm_state][s][a]
            Q[rm_state][s][a] += lr*_delta

            # counterfactual updates
            for v in H.get_states():
                if v == rm_state:
                    continue
                v_next, h_r, _h_done = H.step(v, true_props, info)
                if s not in Q[v]: Q[v][s] = dict([(b, q_init) for b in actions])
                if done: _delta = h_r - Q[v][s][a]
                else:    _delta = h_r + gamma*get_qmax(Q[v_next], sn, actions, q_init) - Q[v][s][a]
                Q[v][s][a] += lr*_delta

            if not rm_done or not infer_termination:
                rm_state = next_rm_state
            else:
                next_random = True

            # HERE
            # Average every N episodes

            rewards.append(r)
            step += 1
            episode_rewards[-1] += r
    
            if following_traj is None:
                evaluation_episode_rewards[-1] += r

            num_episodes = len(episode_rewards)
            num_evaluation_episodes = len(evaluation_episode_rewards)
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            mean_100evalep_reward = np.mean(evaluation_episode_rewards[-101:-1])
            if step%print_freq == 0:
                logger.record_tabular("steps", step)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular(f"mean 100 episode reward", mean_100evalep_reward)
                logger.record_tabular("n states", len(H.U))
                logger.record_tabular("len(X_new)", len(X_new))
                found = 0
                if following_traj:
                    for trace in X_new:
                        if enough_in_bank(n_samples, sequences_bank, trace[0]):
                            found += 1
                logger.record_tabular("state", f"finding samples ({found})" if following_traj is not None else "exploring")
                logger.dump_tabular()
            if done:
                if os.path.isfile("signal.txt"):
                    print("detected signal")
                    IPython.embed()

                episode_rewards.append(0.0)
                if following_traj is None:
                    evaluation_episode_rewards.append(0.0)
                    results.register_mean_reward(step, mean_100evalep_reward)
                All.add((tuple(labels), tuple(rewards)))

                # TODO only add if looking for it (if following_traj == labels: ...)
                add_to_bank(tuple(done_actions_list), tuple(labels), tuple(rewards), sequences_bank, actions_bank)

                if following_traj is None and (num_episodes - last_built <= NOISE_UPDATE_X_EVERY_N or not X_new):
                    if not run_eqv_noise(noise_epsilon, rm_run(labels, H), rewards):
                        # (labels, rewards) = clean_trace(labels, rewards)
                        if "TimeLimit.truncated" in info: # could also see if RM is in a terminating state
                            if info["TimeLimit.truncated"]:
                                X_tl.add((tuple(labels), tuple(rewards)))

                        fixed = make_consistent(noise_epsilon, labels, rewards, X_averaged, H)
                        if fixed is not None: # we don't try to fix on first few counterexamples
                            X.add((tuple(labels), tuple(rewards)))
                            following_traj = tuple(labels)
                            which_one = random.randint(0, len(actions_bank[following_traj])-1)
                            H = fixed
                            print("FIXEDFIXEDFIXED")
                        else:
                            X_new.add((tuple(labels), tuple(rewards)))

                if following_traj is None and X_new and num_episodes - last_built >= NOISE_UPDATE_X_EVERY_N:
                    print(f"len(X)={len(X)}")
                    print(f"len(X_new)={len(X_new)}")

                    for trace in X.union(X_new):
                        if not enough_in_bank(n_samples, sequences_bank, trace[0]):
                            following_traj = trace[0]
                            which_one = random.randint(0, len(actions_bank[following_traj])-1)

                    if following_traj is None:
                        X_old = copy.deepcopy(X)
                        X.update(X_new)
                        X_new = set()
                        X_averaged = recompute_X(n_samples, noise_delta, X, sequences_bank)
                        X_tl_averaged = recompute_X(n_samples, noise_delta, X_tl, sequences_bank)

                        if detect_signal("xnew"):
                            print("ready to compute")
                            IPython.embed()

                        language = sample_language(X)
                        empty_transition = dnf_for_empty(language)
                        transitions_new, n_states_last = consistent_hyp(noise_epsilon, X_averaged, X_tl_averaged, infer_termination, n_states_last)
                        H_new = rm_from_transitions(transitions_new, empty_transition)
                        Q = transfer_Q(noise_epsilon, run_eqv_noise, H_new, H, Q, X_old)
                        H = H_new
                        transitions = transitions_new
                        last_built = num_episodes
                        results.register_rebuilding(step, list(transitions.items()))

                if following_traj:
                    following_traj = None
                    for trace in X.union(X_new):
                        if not enough_in_bank(n_samples, sequences_bank, trace[0]):
                            following_traj = trace[0]
                            which_one = random.randint(0, len(actions_bank[following_traj])-1)
                            break
                break
            s = sn

    results.save(results_path)
