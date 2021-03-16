"""
JIRP based method
"""
import math
import itertools
import random, time, copy
import IPython
from profilehooks import profile
from baselines import logger
import numpy as np
import os.path

from reward_machines.reward_machine import RewardMachine
from reward_machines.rm_environment import RewardMachineEnv, RewardMachineHidden
from rl_agents.jirp.util import *
from rl_agents.jirp_noise.util import *
from rl_agents.jirp.consts import *
from rl_agents.jirp_noise.consts import *
from rl_agents.jirp_noise.smt_noise import *

def consistent_hyp(noise_epsilon, X, X_tl, n_states_start=2, report=True):
    """
    Finds a reward machine consistent with counterexample set X. Returns the RM
    and its number of states

    n_states_start makes the search start from machines with that number of states.
    Used to optimize succeeding search calls.
    """
    if len(X) == 0:
        transitions = dict()
        transitions[(0, tuple())] = [0, 0.0]
        return transitions, 2
    # TODO intercept empty X here
    for n_states in range(n_states_start, MAX_RM_STATES_N+1):
        if report:
            print(f"finding model with {n_states} states")
        # print("(SMT)")
        new_transitions = smt_noise(noise_epsilon, X, X_tl, n_states)
        # print("(SAT)")
        # new_transitions_sat = sat_hyp(0.15, X, X_tl, n_states)
        if new_transitions is not None:
            # if new_transitions_sat is None:
            #     print(f"SAT couldn't find anything with n_states={n_states}")
            # display_transitions(new_transitions, "st")
            return new_transitions, n_states
        continue

    raise ValueError(f"Couldn't find machine with at most {MAX_RM_STATES_N} states")

def start_stepping(H, env, Q, actions, q_init):
    rm_state = H.reset()
    s = tuple(env.reset())

    while True:
        env.show()
        a = get_best_action(Q[rm_state],s,actions,q_init)
        sn, r, done, info = env.step(a)
        sn = tuple(sn)
        true_props = env.get_events()
        next_rm_state, _rm_reward, rm_done = H.step(rm_state, true_props, info)
        rm_state = next_rm_state
        s = sn
        IPython.embed()

def clean_trace(labels, rewards):
    no_more_than = 10
    labels_new = list()
    rewards_new = list()
    last = None
    counter = 0
    for i in range(0, len(labels)):
        if labels[i] == last and counter > no_more_than:
            continue
        elif labels[i] == last:
            counter += 1
        else:
            counter = 0
        labels_new.append(labels[i])
        rewards_new.append(rewards[i])
        last = labels[i]
    return ((tuple(labels_new), tuple(rewards_new)))


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
          use_rs=False):
    assert env.is_hidden_rm() # JIRP doesn't work with explicit RM environments

    try:
        noise_epsilon = env.current_rm.epsilon_cont
    except:
        noise_epsilon = NOISE_EPSILON

    try:
        noise_delta = env.current_rm.noise_delta
        assert noise_delta is not None
    except:
        noise_delta = NOISE_DELTA
    
    print("alg noise epsilon:", noise_epsilon)
    print("alg noise delta:", noise_delta)

    X = set()
    All = set()
    X_new = set()
    X_tl = set()

    labels = []
    rewards = []

    transitions, n_states_last = consistent_hyp(noise_epsilon, set(), set())
    language = sample_language(X)
    empty_transition = dnf_for_empty(language)
    H = rm_from_transitions(transitions, empty_transition)
    # H = load("./envs/grids/reward_machines/office/t3.txt")
    actions = list(range(env.action_space.n))
    Q = initial_Q(H)

    episode_rewards = [0.0]
    step = 0
    step_scores = list()
    step_rebuilding = list()
    num_episodes = 0

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

            if not rm_done or not TERMINATION:
                rm_state = next_rm_state
            else:
                next_random = True

            # HERE
            # Average every N episodes

            rewards.append(r)
            step += 1
            episode_rewards[-1] += r

            num_episodes = len(episode_rewards)
            mean_100ep_reward = np.mean(episode_rewards[-101:-1])

            if step%5000 == 0:
                step_scores.append((step, mean_100ep_reward))

            if step%print_freq == 0:
                logger.record_tabular("steps", step)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular(f"mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("n states", len(H.U))
                logger.record_tabular("len(X_new)", len(X_new))
                logger.dump_tabular()
            if done:
                if os.path.isfile("signal.txt"):
                    print("detected signal")
                    IPython.embed()

                episode_rewards.append(0.0)

                All.add((tuple(labels), tuple(rewards)))
                if not run_eqv_noise(noise_epsilon, rm_run(labels, H), rewards):
                    # (labels, rewards) = clean_trace(labels, rewards)
                    if "TimeLimit.truncated" in info: # could also see if RM is in a terminating state
                        if info["TimeLimit.truncated"]:
                            X_tl.add((tuple(labels), tuple(rewards)))

                    fixed = make_consistent(noise_epsilon, labels, rewards, X, H)
                    if fixed is not None: # we don't try to fix on first few counterexamples
                        X.add((tuple(labels), tuple(rewards)))
                        H = fixed
                        print("FIXEDFIXEDFIXED")
                    else:
                        X_new.add((tuple(labels), tuple(rewards)))

                if X_new and num_episodes % NOISE_UPDATE_X_EVERY_N == 0:
                    print(f"len(X)={len(X)}")
                    print(f"len(X_new)={len(X_new)}")
                    if detect_signal("xnew"):
                        IPython.embed()
                    X.update(X_new)
                    X_new = set()
                    language = sample_language(X)
                    empty_transition = dnf_for_empty(language)
                    transitions_new, n_states_last = consistent_hyp(noise_epsilon, X, X_tl, n_states_last)
                    H_new = rm_from_transitions(transitions_new, empty_transition)
                    if not consistent_on_all(noise_epsilon, X, H_new):
                        print("NOT CONSISTENT IMMMEDIATELY")
                        IPython.embed()
                    H_old = copy.deepcopy(H)
                    H = H_new
                    average_on_X(noise_epsilon, H, All, X)
                    transitions = transitions_new
                    Q = transfer_Q(noise_epsilon, run_eqv_noise, H_new, H, Q, X)
                    step_rebuilding.append(step)
                break
            s = sn

    print("fin")
    IPython.embed()
