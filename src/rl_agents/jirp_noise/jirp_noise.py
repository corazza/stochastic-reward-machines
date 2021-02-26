"""
JIRP based method
"""
import math
import itertools
import random, time, copy
from profilehooks import profile
import IPython
from baselines import logger

from reward_machines.reward_machine import RewardMachine
from reward_machines.rm_environment import RewardMachineEnv, RewardMachineHidden
from rl_agents.jirp.util import *
from rl_agents.jirp_noise.util import *
from rl_agents.jirp.consts import *
from rl_agents.jirp_noise.consts import *
from rl_agents.jirp.test import *
from rl_agents.jirp.mip_approx import mip_approx
from rl_agents.jirp.smt_approx import smt_approx
from rl_agents.jirp.smt_hyp import smt_hyp
from rl_agents.jirp_noise.smt_noise import smt_noise
from rl_agents.jirp.mip_hyp import mip_hyp
from rl_agents.jirp.sat_hyp import sat_hyp


last_displayed_states = 0

# @profile(sort="tottime")
def consistent_hyp(X, X_tl, n_states_start=2, report=True):
    """
    Finds a reward machine consistent with counterexample set X. Returns the RM
    and its number of states

    n_states_start makes the search start from machines with that number of states.
    Used to optimize succeeding search calls.
    """
    if len(X) == 0:
        transitions = dict()
        transitions[(0, tuple())] = [0, 0.0]
        transitions[(1, tuple())] = [0, 0.0]
        return transitions, 2
    # TODO intercept empty X here
    for n_states in range(n_states_start, MAX_RM_STATES_N+1):
        if report:
            print(f"finding model with {n_states} states")
        # print("(SMT)")
        new_transitions = smt_noise(0.15, X, X_tl, n_states)
        # print("(SAT)")
        # new_transitions_sat = sat_hyp(0.15, X, X_tl, n_states)
        if new_transitions is not None:
            # if new_transitions_sat is None:
            #     print(f"SAT couldn't find anything with n_states={n_states}")
            display_transitions(new_transitions, "st")
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

def equivalent_on_X(H1, v1, H2, v2, X):
    """
    Checks if state v1 from RM H1 is equivalent to v2 from H2
    """
    H1 = H1.with_initial(v1)
    H2 = H2.with_initial(v2)
    total = len(X)
    eqv = 0
    for (labels, _rewards) in X:
        output1 = rm_run(labels, H1)
        output2 = rm_run(labels, H2)
        if run_eqv_noise(NOISE_EPSILON+0.001, output1, output2):
            eqv += 1
        # if rm_run(labels, H1) == rm_run(labels, H2):
        #     eqv += 1
    if float(eqv)/total > EQV_THRESHOLD:
        print(f"H_new/{v1} ~ H_old/{v2} (p ~= {float(eqv)/total})")
        return True
    return False

def transfer_Q(H_new, H_old, Q_old, X = {}):
    """
    Returns new set of q-functions, indexed by states of H_new, where
    some of the q-functions may have been transfered from H_old if the
    respective states were determined to be equivalent
    
    Although the thm. requires the outputs be the same on _all_ label sequences,
    choosing probably equivalent states may be good enough.
    """
    Q = dict()
    Q[-1] = dict() # (-1 is the index of the terminal state) (TODO check if necessary)
    for v in H_new.get_states():
        Q[v] = dict()
        # find probably equivalent state u in H_old
        for u in H_old.get_states():
            if equivalent_on_X(H_new, v, H_old, u, X) and u in Q_old:
                Q[v] = copy.deepcopy(Q_old[u])
                break
    return Q

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

    # test2(env.current_rm)
    # exit()

    reward_total = 0
    total_episodes = 0
    step = 0
    num_episodes = 0

    X = set()
    X_new = set()
    X_tl = set()
    labels = []
    rewards = []

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

            if r > 0:
                r += random.uniform(-NOISE_EPSILON, NOISE_EPSILON)

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

            # if len(X_new) > 50:
            #     language = sample_language(X_new)
            #     rm = env.current_rm
            #     t_rm = rm_to_transitions(rm)
                # IPython.embed()

            # moving to the next state
            reward_total += r
            rewards.append(r)
            step += 1
            if step%print_freq == 0:
                logger.record_tabular("steps", step)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("n states", len(H.U))
                logger.record_tabular("len(X_new)", len(X_new))
                logger.record_tabular("total reward", reward_total)
                logger.record_tabular("positive / total", str(int(reward_total)) + "/" + str(total_episodes) + f" ({int(100*(reward_total/total_episodes))}%)")
                logger.dump_tabular()
                reward_total = 0
                total_episodes = 0
            if done:
                num_episodes += 1
                total_episodes += 1

                if not run_eqv_noise(NOISE_EPSILON+0.001, rm_run(labels, H), rewards):
                    X_new.add((tuple(labels), tuple(rewards)))
                    if "TimeLimit.truncated" in info: # could also see if RM is in a terminating state
                        tl = info["TimeLimit.truncated"]
                        if tl:
                            X_tl.add((tuple(labels), tuple(rewards)))

                if num_episodes % UPDATE_X_EVERY_N_EPISODES == 0 and X_new:
                    print(f"len(X)={len(X)}")
                    print(f"len(X_new)={len(X_new)}")
                    X.update(X_new)
                    X_new = set()
                    language = sample_language(X)
                    empty_transition = dnf_for_empty(language)
                    transitions_new, n_states_last = consistent_hyp(X, X_tl, n_states_last)
                    H_new = rm_from_transitions(transitions_new, empty_transition)
                    H = H_new
                    transitions = transitions_new
                    Q = transfer_Q(H_new, H, Q, X)
                break
            s = sn
