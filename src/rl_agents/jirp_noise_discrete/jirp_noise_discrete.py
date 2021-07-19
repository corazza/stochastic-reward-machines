"""
JIRP based method
"""
import math
import itertools
import random, time, copy
import IPython
import numpy as np
from baselines import logger

from reward_machines.reward_machine import RewardMachine
from reward_machines.rm_environment import RewardMachineEnv, RewardMachineHidden
from rl_agents.jirp.maxsat_hyp import maxsat_hyp
from rl_agents.jirp.util import *
from rl_agents.jirp.consts import *
from rl_agents.jirp.test import *
from rl_agents.jirp.mip_approx import mip_approx
from rl_agents.jirp.smt_approx import smt_approx
from rl_agents.jirp.smt_hyp import smt_hyp
from rl_agents.jirp.mip_hyp import mip_hyp
from rl_agents.jirp.sat_hyp import sat_hyp
from rl_agents.jirp_noise.util import EvalResults


last_displayed_states = 0

def discrete_noise_hyp(X, X_tl, infer_termination, discrete_noise_p, n_states_start=2, report=True):
    """
    Returns set of corrupted traces to be removed later
    """
    if len(X) == 0:
        transitions = dict()
        transitions[(0, tuple())] = [0, 0.0]
        return transitions, 1, set()
    for n_states in range(n_states_start, MAX_RM_STATES_N+1):
        if report:
            print(f"finding model with {n_states} states")
        result = maxsat_hyp(0.0, X, X_tl, n_states, infer_termination) # epsilon does nothing in maxsat_hyp
        if result is not None:
            new_transitions, corrupted_traces = result
            cost = float(len(corrupted_traces))/float(len(X))
            # if cost <= discrete_noise_p:
            return new_transitions, n_states, corrupted_traces
        continue
    raise ValueError(f"Couldn't find machine with at most {MAX_RM_STATES_N} states")

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
    assert env.no_rm() or env.is_hidden_rm() # JIRP doesn't work with explicit RM environments

    try:
        discrete_noise_p = env.discrete_noise_p
    except:
        discrete_noise_p = 0
    assert discrete_noise_p >= 0

    infer_termination = TERMINATION
    try:
        infer_termination = env.infer_termination_preference()
    except:
        pass
    print(f"(alg) INFERRING TERMINATION: {infer_termination}")

    description = { 
        "env_name": env.unwrapped.spec.id,
        "alg_name": "jirp_noise_discrete",
        "discrete_noise_p": discrete_noise_p,
        "reward_flip_p": REWARD_FLIP_P,
        "total_timesteps": total_timesteps,
    }

    results = EvalResults(description)

    reward_total = 0
    total_episodes = 0
    step = 0
    num_episodes = 0
    episode_rewards = [0.0]

    A = set()
    A_tl = set()
    X = set()
    X_new = set()
    X_tl = set()
    labels = []
    rewards = []
    do_embed = True

    transitions, n_states_last, _corrupted_traces = discrete_noise_hyp(set(), set(), infer_termination, discrete_noise_p)
    language = sample_language(X)
    empty_transition = dnf_for_empty(language)
    H = rm_from_transitions(transitions, empty_transition)

    actions = list(range(env.action_space.n))
    Q = initial_Q(H)

    # rms = [load_c(i) for i in range(1, 11)]
    # language={"","a", "b","f", "c","e", "d"}
    # empty_transition=dnf_for_empty(language)
    # rm1 = load_c(3)
    # rm2 = load_c(8)
    # t = product_rm(language, rm1, rm2)
    # rm = rm_from_transitions(t, empty_transition)
    # IPython.embed()

    while step < total_timesteps:
        s = tuple(env.reset())
        true_props = env.get_events()
        rm_state = H.reset()
        labels = []
        rewards = []
        next_random = False

        if s not in Q[rm_state]: Q[rm_state][s] = dict([(a, q_init) for a in actions])

        while True:
            # Selecting and executing the action
            a = random.choice(actions) if random.random() < epsilon or next_random else get_best_action(Q[rm_state],s,actions,q_init)
            sn, r, done, info = env.step(a)

            sn = tuple(sn)
            true_props = env.get_events()
            labels.append(true_props) # L(s, a, s')
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
                rm_state = next_rm_state # TODO FIXME this entire loop, comment and organize
            else:
                next_random = True

            # moving to the next state
            reward_total += 1 if r > 2 else 0
            rewards.append(r)
            step += 1
            episode_rewards[-1] += r

            if step%print_freq == 0:
                mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
                num_episodes = len(episode_rewards)
                logger.record_tabular("steps", step)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("n states", len(H.U))
                logger.record_tabular("len(X_new)", len(X_new))
                logger.dump_tabular()
            if done:
                if os.path.isfile("signal.txt"):
                    print("detected signal")
                    IPython.embed()
                    
                num_episodes = len(episode_rewards)
                episode_rewards.append(0.0)

                time_limited = False
                if "TimeLimit.truncated" in info: # could also see if RM is in a terminating state
                    time_limited = info["TimeLimit.truncated"]

                trace = (tuple(labels), tuple(rewards))
                # A.add(trace)
                # if time_limited:
                #     A_tl.add(trace)
                if not run_eqv(EXACT_EPSILON, rm_run(labels, H), rewards):
                    X_new.add(trace)
                    if time_limited:
                        X_tl.add(trace)

                if X_new and num_episodes % UPDATE_X_EVERY_N == 0:
                    print(f"len(X)={len(X)}")
                    print(f"len(X_new)={len(X_new)}")
                    X.update(X_new)
                    X_new = set()
                    language = sample_language(X)
                    empty_transition = dnf_for_empty(language)
                    transitions_new, n_states_last, corrupted_traces = discrete_noise_hyp(X, X_tl, infer_termination, discrete_noise_p, n_states_start=n_states_last)
                    before_removal_n = len(X)
                    X -= corrupted_traces
                    after_removal_n = len(X)
                    print(f"removed {before_removal_n-after_removal_n}/{before_removal_n} corrupted traces")
                    H_new = rm_from_transitions(transitions_new, empty_transition)
                    Q = transfer_Q(EXACT_EPSILON, run_eqv, H_new, H, Q, X)
                    H = H_new
                    transitions = transitions_new
                break
            s = sn

    results.save(results_path)
