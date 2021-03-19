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
from rl_agents.jirp.util import *
from rl_agents.jirp.consts import *
from rl_agents.jirp.test import *
from rl_agents.jirp.mip_approx import mip_approx
from rl_agents.jirp.smt_approx import smt_approx
from rl_agents.jirp.smt_hyp import smt_hyp
from rl_agents.jirp.mip_hyp import mip_hyp
from rl_agents.jirp.sat_hyp import sat_hyp


last_displayed_states = 0

# @profile(sort="tottime")
def consistent_hyp(X, X_tl, infer_termination=True, n_states_start=2, report=True):
    """
    Finds a reward machine consistent with counterexample set X. Returns the RM
    and its number of states

    n_states_start makes the search start from machines with that number of states.
    Used to optimize succeeding search calls.
    """
    if len(X) == 0:
        transitions = dict()
        transitions[(0, tuple())] = [0, 0.0]
        return transitions, 1
    # TODO intercept empty X here
    for n_states in range(n_states_start, MAX_RM_STATES_N+1):
        if report:
            print(f"finding model with {n_states} states")
        # print("(SMT)")
        new_transitions = sat_hyp(0.15, X, X_tl, n_states, infer_termination)
        # print("(SAT)")
        # new_transitions_sat = sat_hyp(0.15, X, X_tl, n_states)
        if new_transitions is not None:
            # if new_transitions_sat is None:
            #     print(f"SAT couldn't find anything with n_states={n_states}")
            # display_transitions(new_transitions, "st")
            return new_transitions, n_states
        continue

    raise ValueError(f"Couldn't find machine with at most {MAX_RM_STATES_N} states")

def approximate_hyp(approximation_method, language, transitions, n_states):
    empty_transition = dnf_for_empty(language)
    if n_states >= 2:
        for i in range(2, n_states):
            minimized_rm = approximation_method(MINIMIZATION_EPSILON, language, i, n_states, transitions, empty_transition, report=True)
            if minimized_rm:
                print(f"FOUND MINIMIZED RM {i} < {n_states}")
                print(transitions)
                print(minimized_rm)
                return minimized_rm, n_states
    print("couldn't find minimized RM, returning exact")

    display = False
    global last_displayed_states
    if n_states > last_displayed_states or n_states <= 2:
        display = True
        last_displayed_states = n_states
    return approximation_method(EXACT_EPSILON, language, n_states, n_states, transitions, empty_transition, report=True, display=True), n_states

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

    reward_total = 0
    total_episodes = 0
    step = 0
    num_episodes = 0
    episode_rewards = [0.0]

    X = set()
    X_new = set()
    X_tl = set()
    labels = []
    rewards = []
    do_embed = True

    transitions, n_states_last = consistent_hyp(set(), set())
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

            if not rm_done or not TERMINATION:
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
                if step > 1e6 and do_embed:
                    IPython.embed()
                num_episodes = len(episode_rewards)
                episode_rewards.append(0.0)

                if not run_eqv(EXACT_EPSILON, rm_run(labels, H), rewards):
                    X_new.add((tuple(labels), tuple(rewards)))
                    if "TimeLimit.truncated" in info: # could also see if RM is in a terminating state
                        tl = info["TimeLimit.truncated"]
                        if tl:
                            X_tl.add((tuple(labels), tuple(rewards)))

                if X_new and num_episodes % UPDATE_X_EVERY_N == 0:
                    print(f"len(X)={len(X)}")
                    print(f"len(X_new)={len(X_new)}")
                    X.update(X_new)
                    X_new = set()
                    language = sample_language(X)
                    empty_transition = dnf_for_empty(language)
                    transitions_new, n_states_last = consistent_hyp(X, X_tl, n_states_last)
                    H_new = rm_from_transitions(transitions_new, empty_transition)
                    Q = transfer_Q(EXACT_EPSILON, run_eqv, H_new, H, Q, X)
                    H = H_new
                    transitions = transitions_new
                break
            s = sn
