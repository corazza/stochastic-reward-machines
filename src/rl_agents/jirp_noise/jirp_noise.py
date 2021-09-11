"""
JIRP based method
"""
import random
import IPython
from baselines import logger
from baselines.common.misc_util import set_global_seeds
import numpy as np
import os.path

from rl_agents.jirp.util import *
from rl_agents.jirp_noise.util import *
from rl_agents.jirp.consts import *
from rl_agents.jirp_noise.consts import NOISE_UPDATE_X_EVERY_N, RERUN_ESTIMATES_EVERY_N
from rl_agents.jirp_noise.smt_noise import smt_noise_cpp


def consistent_hyp(noise_epsilon, X, X_tl, infer_termination, n_states_start=1, report=True, alg_name=None, seed=None):
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
        # new_transitions = smt_noise(noise_epsilon, X, X_tl, n_states)
        new_transitions = smt_noise_cpp(noise_epsilon, X, X_tl, n_states, infer_termination, report=False, alg_name=alg_name, seed=seed)

        # print("(SAT)")
        # new_transitions_sat = sat_hyp(0.15, X, X_tl, n_states)
        if new_transitions is not None:
            # if new_transitions_sat is None:
            #     print(f"SAT couldn't find anything with n_states={n_states}")
            # display_transitions(new_transitions, "st")
            return new_transitions, n_states
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
    ALG_NAME="jirp_noise"
    REPORT=True
    set_global_seeds(seed)
    print(f"set_global_seeds({seed})")
    assert env.no_rm() or env.is_hidden_rm() # JIRP doesn't work with explicit RM environments
    assert results_path is not None
    assert seed is not None

    infer_termination = TERMINATION
    try:
        infer_termination = env.infer_termination_preference()
    except:
        pass
    print(f"(alg) INFERRING TERMINATION: {infer_termination}")

    noise_epsilon, noise_delta = extract_noise_params(env)
    inference_epsilon = noise_epsilon
    checking_epsilon = inference_epsilon + 5*EXACT_EPSILON

    print("alg noise epsilon:", noise_epsilon)
    print("alg noise delta:", noise_delta)

    description = { 
        "env_name": env.unwrapped.spec.id,
        "alg_name": ALG_NAME,
        "reward_flip_p": REWARD_FLIP_P,
        "alg_noise_epsilon": noise_epsilon,
        "alg_noise_delta": noise_delta,
        "total_timesteps": total_timesteps,
        "slip_prob": env.slip_prob
    }

    results = EvalResults(description)

    All = set()
    X = set()
    X_new = set()
    X_tl = set()
    labels = []
    rewards = []

    episode_rewards = [0.0]
    step = 0
    num_episodes = 0

    transitions, n_states_last = consistent_hyp(inference_epsilon, set(), set(), infer_termination, report=REPORT, alg_name=ALG_NAME, seed=seed)
    language = sample_language(X)
    empty_transition = dnf_for_empty(language)
    H = rm_from_transitions(transitions, empty_transition)
    H_epsilon = rm_from_transitions(transitions, empty_transition)
    actions = list(range(env.action_space.n))
    Q = initial_Q(H)

    while step < total_timesteps:
        s = tuple(env.reset())
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

            if not rm_done:
                rm_state = next_rm_state
            else:
                next_random = True

            rewards.append(r)
            step += 1
            episode_rewards[-1] += r

            num_episodes = len(episode_rewards)
            mean_100ep_reward = np.mean(episode_rewards[-101:-1])

            if step % REGISTER_MEAN_REWARD_EVERY_N_STEP == 0:
                results.register_mean_reward(step, mean_100ep_reward)

            if step%print_freq == 0:
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

                episode_rewards.append(0.0)

                All.add((tuple(labels), tuple(rewards)))
                if not run_eqv_noise(inference_epsilon, rm_run(labels, H_epsilon), rewards):
                    if "TimeLimit.truncated" in info: # could also see if RM is in a terminating state
                        if info["TimeLimit.truncated"]:
                            X_tl.add((tuple(labels), tuple(rewards)))

                    fixed = make_consistent(checking_epsilon, inference_epsilon, labels, rewards, X, H_epsilon)
                    if fixed is not None: # we don't try to fix on first few counterexamples
                        X.add((tuple(labels), tuple(rewards)))
                        H_epsilon = fixed
                        H = average_on_X(checking_epsilon, H_epsilon, All, X, report=REPORT)
                        print("FIXEDFIXEDFIXED")
                    else:
                        X_new.add((tuple(labels), tuple(rewards)))

                if X_new and num_episodes % NOISE_UPDATE_X_EVERY_N == 0:
                    if REPORT:
                        print(f"len(X)={len(X)}")
                        print(f"len(X_new)={len(X_new)}")
                    if detect_signal("xnew"):
                        IPython.embed()
                    X.update(X_new)
                    X_new = set()
                    language = sample_language(X)
                    empty_transition = dnf_for_empty(language)
                    result = consistent_hyp(inference_epsilon, X, X_tl, infer_termination, n_states_start=n_states_last, report=REPORT, alg_name=ALG_NAME, seed=seed)
                    if result is not None:
                        transitions_new, n_states_last = result
                    else:
                        results.save(results_path)
                        raise ValueError(f"Couldn't find machine with at most {MAX_RM_STATES_N} states")
                    H_new_epsilon = rm_from_transitions(transitions_new, empty_transition)
                    if not consistent_on_all(checking_epsilon, X, H_new_epsilon):
                        print("NOT CONSISTENT IMMEDIATELY")
                        IPython.embed()
                    H_new = average_on_X(checking_epsilon, H_new_epsilon, All, X, report=REPORT)
                    Q = transfer_Q(checking_epsilon, run_eqv_noise, H_new_epsilon, H_epsilon, Q, X)
                    H = H_new
                    H_epsilon = H_new_epsilon
                    transitions = transitions_new
                    results.register_rebuilding(step, serializeable_rm(H))
                break
            s = sn
    results.save(results_path)
