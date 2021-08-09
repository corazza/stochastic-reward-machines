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
from reward_machines.rm_environment import RewardMachineDiscreteNoise, RewardMachineEnv, RewardMachineHidden
from rl_agents.jirp.maxsat_hyp import maxsat_hyp
from rl_agents.jirp.util import *
from rl_agents.jirp.consts import *
from rl_agents.jirp.test import *
from rl_agents.jirp.mip_approx import mip_approx
from rl_agents.jirp.smt_approx import smt_approx
from rl_agents.jirp.smt_hyp import smt_hyp
from rl_agents.jirp.mip_hyp import mip_hyp
from rl_agents.jirp.sat_hyp import sat_hyp
from rl_agents.jirp_noise.util import EvalResults, detect_signal


last_displayed_states = 0

def lookahead(X, X_tl, infer_termination, n_states_start=2, report=True):
    for n_states in range(1, 5):
        print(f"solving with {n_states} states")
        result = maxsat_hyp(0.0, X, X_tl, n_states, infer_termination) # epsilon does nothing in maxsat_hyp
        if result is not None:
            new_transitions, corrupted_traces = result
            n_traces = len(X)
            cost = float(len(corrupted_traces))/float(n_traces)
            z = (cost - DISCRETE_NOISE_P)*math.sqrt(float(n_traces)) / math.sqrt(DISCRETE_NOISE_P * (1 - DISCRETE_NOISE_P))
        continue
    IPython.embed()

def discrete_noise_hyp(num_episodes, X, X_tl, infer_termination, n_states_start=2, report=True):
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
            n_traces = len(X)
            cost = float(len(corrupted_traces))/float(num_episodes)
            z = (cost - DISCRETE_NOISE_P)*math.sqrt(float(num_episodes)) / math.sqrt(DISCRETE_NOISE_P * (1 - DISCRETE_NOISE_P))
            print(f"correct cost: {cost}, z={z} > 1.96")
            if z > 1.96 and num_episodes > 100:
                # if cost > 0:
                print("FAILEDFAILEDFAILED")
                print("FAILEDFAILEDFAILED")
                print("FAILEDFAILEDFAILED")
                print("FAILEDFAILEDFAILED")
                # IPython.embed()
                continue
            return new_transitions, n_states, corrupted_traces
        continue
    raise ValueError(f"Couldn't find machine with at most {MAX_RM_STATES_N} states")

def add_event(events, step, episode, name, info):
    events.append((step, episode, name, info))

def filter_events(events, name):
    return list(filter(lambda x: x[2] == name, events))

def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]

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
    assert env.is_discrete_noise()

    infer_termination = TERMINATION
    try:
        infer_termination = env.infer_termination_preference()
    except:
        pass
    infer_termination = False # TERMINATION
    print(f"(alg) INFERRING TERMINATION: {infer_termination}")

    description = { 
        "env_name": env.unwrapped.spec.id,
        "alg_name": "jirp_noise_discrete",
        "discrete_noise_p": DISCRETE_NOISE_P,
        "reward_flip_p": REWARD_FLIP_P,
        "total_timesteps": total_timesteps,
    }

    results = EvalResults(description)

    injected = False

    reward_total = 0
    total_episodes = 0
    step = 0
    num_episodes = 0
    episode_rewards = [0.0]

    A = set()
    A_tl = set()
    X = set()
    X_new = set()
    real_cx = 0
    X_real = set()
    X_noise = set()
    X_fake = set()
    X_tl = set()
    events = list()
    labels = []
    rewards = []
    do_embed = True

    transitions, n_states_last, _corrupted_traces = discrete_noise_hyp(0, set(), set(), infer_termination)
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

    current_corrupted = False

    # env.stop_noise()

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

            if not current_corrupted and info["discrete_noise"]: # TODO change to env.current_corrupted
                current_corrupted = True
                results.register_corruption(step)
            if current_corrupted and not info["discrete_noise"]:
                current_corrupted = False
                results.register_corruption_end(step)

            # moving to the next state
            reward_total += 1 if r > 2 else 0
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

                if len(X) > 160 and not env.stopped_noise:
                    print("stopping noise")
                    env.stop_noise()
                    # IPython.embed()

                # if env.stopped_noise and len(X_new) > 0:
                #     print("FOUND REAL")
                #     IPython.embed()

                if env.stopped_noise and num_episodes % 500 == 0:
                    a = list(corrupted_traces - X_noise)
                    faketraces = set()
                    for truetrace in a:
                        for i in range(10):
                            emptys = duplicates(truetrace[0], '')
                            throwouti = emptys[random.randint(0, len(emptys)-1)]
                            newtrace_labels = list(truetrace[0][:])
                            newtrace_rewards = list(truetrace[1 ][:])
                            del newtrace_labels[throwouti]
                            del newtrace_rewards[throwouti]
                            env_run = rm_run(newtrace_labels, env.current_rm)
                            hyp_run = rm_run(newtrace_labels, H)
                            if run_eqv(EXACT_EPSILON, env_run, newtrace_rewards) and not run_eqv(EXACT_EPSILON, hyp_run, newtrace_rewards):
                                faketraces.add((tuple(newtrace_labels), tuple(newtrace_rewards)))
                    print("generated faketraces")
                    a = len(X_new)
                    X_real.update(faketraces)
                    X_fake.update(faketraces)
                    X_new.update(faketraces)
                    print(f"increased X_new by {len(X_new) - a}")
                    IPython.embed()

                num_episodes = len(episode_rewards)
                episode_rewards.append(0.0)

                trace = (tuple(labels), tuple(rewards))
                time_limited = False
                if "TimeLimit.truncated" in info: # could also see if RM is in a terminating state
                    time_limited = info["TimeLimit.truncated"]

                if not run_eqv(EXACT_EPSILON, rm_run(labels, H), rewards):
                    X_new.add(trace)
                    if not env.current_corrupted:
                        real_cx += 1
                        X_real.add(trace)
                        add_event(events, step, num_episodes, "found real cx", trace)
                    else:
                        X_noise.add(trace)
                        add_event(events, step, num_episodes, "found noisy cx", trace)
                    if time_limited:
                        X_tl.add(trace)


                # Assertion below broken, reason why, then check if faketraces creates duplicates? DOESNT HAPPEN AGAIN
                # Add faketraces every rebuilding

                # RUN EXPERIMENTS WITH FAKETRACES AND WITHOUT
                # FIND OUT WHY THEY'RE NOT BEING FOUND NATURALLY

                if X_new and num_episodes % UPDATE_X_EVERY_N == 0:
                    print(f"len(X)={len(X)}")
                    print(f"len(X_new)={len(X_new)}")
                    print(f"real/total = {real_cx}/{len(X_new)}")
                    print(f"len(X_fake)={len(X_fake)}")

                    oldaaa = len(X)
                    addingaaa = len(X_new)
                    X.update(X_new)
                    newaaa = len(X)
                    if not oldaaa + addingaaa == newaaa:
                        print("ASSERTION BROKEN")
                        IPython.embed()
                    X_new = set()
                    real_cx = 0
                    language = sample_language(X)
                    empty_transition = dnf_for_empty(language)
                    transitions_new, n_states_last, corrupted_traces = discrete_noise_hyp(num_episodes, X, X_tl, infer_termination, n_states_start=n_states_last)
                    before_removal_n = len(X)
                    # X -= corrupted_traces
                    after_removal_n = len(X)
                    print(f"removed {before_removal_n-after_removal_n}/{before_removal_n} corrupted traces")
                    H_new = rm_from_transitions(transitions_new, empty_transition)
                    Q = transfer_Q(EXACT_EPSILON, run_eqv, H_new, H, Q, X)
                    H = H_new
                    transitions = transitions_new
                    results.register_rebuilding(step, serializeable_rm(H))
                break
            s = sn
    results.save(results_path)
