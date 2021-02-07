"""
JIRP based method
"""
import math
import itertools
import random, time, copy
from profilehooks import profile

from pysat.solvers import Glucose4
from baselines import logger

from reward_machines.reward_machine import RewardMachine
from reward_machines.rm_environment import RewardMachineEnv, RewardMachineHidden
from rl_agents.jirp.util import *
from rl_agents.jirp.consts import *
from rl_agents.jirp.mip_hyp import mlip_hyp
from rl_agents.jirp.smt_hyp import smt_hyp


# @profile
def consistent_hyp(X, n_states_start=2, report=True):
    """
    Finds a reward machine consistent with counterexample set X. Returns the RM
    and its number of states

    n_states_start makes the search start from machines with that number of states.
    Used to optimize succeeding search calls.
    """
    language = sample_language(X)
    empty_transition = dnf_for_empty(language)
    reward_alphabet = sample_reward_alphabet(X)

    for n_states in range(n_states_start, MAX_RM_STATES_N+1):
        if report:
            print(f"finding model with {n_states} states")

        prop_d = dict() # maps SAT's propvar (int) to (p: state, l: labels, q: state)
        prop_d_rev = dict()
        prop_o = dict() # maps SAT's propvar (int) to (p: state, l: labels, r: reward)
        prop_o_rev = dict()
        prop_x = dict() # maps SAT's propvar (int) to (l: labels, q: state)
        prop_x_rev = dict()
        used_pvars = [0] # p. var. counter
        g = Glucose4() # solver

        # convenience methods
        def add_pvar_d(d):
            nonlocal prop_d
            nonlocal prop_d_rev
            return add_pvar(prop_d, prop_d_rev, used_pvars, d)

        def add_pvar_o(o):
            nonlocal prop_o
            nonlocal prop_o_rev
            return add_pvar(prop_o, prop_o_rev, used_pvars, o)

        def add_pvar_x(x):
            nonlocal prop_x
            nonlocal prop_x_rev
            return add_pvar(prop_x, prop_x_rev, used_pvars, x)

        # Encoding reward machines
        # (1)
        for p in all_states(n_states):
            for l in language:
                g.add_clause([add_pvar_d((p, l, q)) for q in all_states(n_states)])
                for q1 in all_states(n_states):
                    for q2 in all_states(n_states):
                        if q1==q2:
                            continue
                        p_l_q1 = add_pvar_d((p, l, q1))
                        p_l_q2 = add_pvar_d((p, l, q2))
                        g.add_clause([-p_l_q1, -p_l_q2])

        # (2)
        for p in all_states(n_states):
            for l in language:
                g.add_clause([add_pvar_o((p, l, r)) for r in reward_alphabet])
                for r1 in reward_alphabet:
                    for r2 in reward_alphabet:
                        if r1 == r2:
                            continue
                        p_l_r1 = add_pvar_o((p, l, r1))
                        p_l_r2 = add_pvar_o((p, l, r2))
                        g.add_clause([-p_l_r1, -p_l_r2])

        # Consistency with sample
        # (3)
        g.add_clause([add_pvar_x((tuple(), INITIAL_STATE))]) # starts in the initial state
        for p in all_states(n_states):
            if p == INITIAL_STATE:
                continue
            g.add_clause([-add_pvar_x((tuple(), p))])

        # (4)
        for (labels, _rewards) in prefixes(X):
            if labels == ():
                continue
            lm = labels[0:-1]
            l = labels[-1]
            for p in all_states(n_states):
                for q in all_states(n_states):
                    x_1 = add_pvar_x((lm, p))
                    d = add_pvar_d((p, l, q))
                    x_2 = add_pvar_x((labels, q))
                    g.add_clause([-x_1, -d, x_2])

        # (5)
        for (labels, rewards) in prefixes(X):
            if labels == ():
                continue
            lm = labels[0:-1]
            l = labels[-1]
            r = rewards[-1]
            for p in all_states(n_states):
                x = add_pvar_x((lm, p))
                o = add_pvar_o((p, l, r))
                g.add_clause([-x, o])

        g.solve()
        if g.get_model() is None:
            continue
        
        if report:
            print("found")

        transitions = dict() #defaultdict(lambda: [None, None]) # maps (state, true_props) to (state, reward)

        for pvar in g.get_model():
            if abs(pvar) in prop_d:
                if pvar > 0:
                    (p, l, q) = prop_d[abs(pvar)]
                    # assert transitions[(p, tuple(l))][0] is None
                    if (p, tuple(l)) not in transitions:
                        transitions[(p, tuple(l))] = [None, None]
                    transitions[(p, tuple(l))][0] = q
                    # assert q is not None
            elif abs(pvar) in prop_o:
                if pvar > 0:
                    (p, l, r) = prop_o[abs(pvar)]
                    if (p, tuple(l)) not in transitions:
                        transitions[(p, tuple(l))] = [None, None]
                    # assert transitions[(p, tuple(l))][1] is None
                    transitions[(p, tuple(l))][1] = r
            elif abs(pvar) in prop_x:
                pass
            else:
                raise ValueError("Uknown p-var dict")

        # mlip_n_states = n_states - 1 if n_states >= 3 else n_states
        # return mlip_hyp(X, mlip_n_states, n_states, transitions, empty_transition), n_states

        # if n_states >= 3:
        #     for i in range(2, n_states):
        #         minimized_rm = smt_hyp(SMT_EPSILON, X, i, n_states, transitions, empty_transition)
        #         if minimized_rm:
        #             print(f"FOUND MINIMIZED RM {i} < {n_states} (epsilon={SMT_EPSILON})")
        #             # exit()
        #             return minimized_rm, n_states
            # test_transitions = dict()
            # test_transitions[(1, 'a')] = [2, 0]
            # test_transitions[(2, 'a')] = [3, 0]
            # test_transitions[(3, 'a')] = [1, 0]
            # minimized_rm = smt_hyp(SMT_EPSILON, X, 2, 3, test_transitions, empty_transition)
            # if minimized_rm:
            #     print(f"FOUND MINIMIZED RM")
            #     print(minimized_rm.delta_u)
            #     print()
            #     print(minimized_rm.delta_r)
            #     exit()
            #     return minimized_rm, n_states

        # print("couldn't find minimized RM, returning exact")

        g.delete()
        return smt_hyp(SMT_EPSILON, X, n_states, n_states, transitions, empty_transition, report), n_states
        # return rm_from_transitions(transitions, empty_transition), n_states

    raise ValueError(f"Couldn't find machine with at most {MAX_RM_STATES_N} states")

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
        if run_sum_approx_eqv(output1, output2):
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
            if equivalent_on_X(H_new, v, H_old, u, X):
                Q[v] = copy.deepcopy(Q_old[u])
                break
    return Q

def prune_X(X, transitions, n_states):
    print("pruning X")
    X_result = set(X)
    # n_states = len(H.U)
    found = True
    while found:
        found = False
        for i in range(0, len(X_result)):
            X_candidate = list(X_result)
            X_candidate.pop(i)
            X_candidate = set(X_candidate)
            transitions_new, _ = consistent_hyp(X_candidate, n_states, report=False)
            # equivalent = True
            # for (labels, rewards) in X:
            #     H_output = rm_run(labels, H)
            #     H_new_output = rm_run(labels, H_new)
            #     if H_output != H_new_output:
            #         equivalent = False
            if isomorphic(transitions, transitions_new, n_states):
                found = True
                X_result = X_candidate
                print(f"removed counterexample ({len(X_result)}/{len(X)})")
                # exit()
                break
        if len(X_result) <= X_PRUNE_MIN_SIZE:
            break
    print(f"new size is {len(X_result)}")
    return X_result

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

    X = set()
    X_new = set()
    labels = []
    rewards = []

    transitions, n_states_last = consistent_hyp(set())
    language = sample_language(X)
    empty_transition = dnf_for_empty(language)
    H = rm_from_transitions(transitions, empty_transition)

    actions = list(range(env.action_space.n))
    Q = initial_Q(H)

    while step < total_timesteps:
        s = tuple(env.reset())
        true_props = env.get_events()
        rm_state = H.reset()
        labels = []
        rewards = []

        if s not in Q[rm_state]: Q[rm_state][s] = dict([(a, q_init) for a in actions])

        while True:
            # Selecting and executing the action
            a = random.choice(actions) if random.random() < epsilon else get_best_action(Q[rm_state],s,actions,q_init)
            sn, r, done, info = env.step(a)

            sn = tuple(sn)
            true_props = env.get_events()
            labels.append(true_props) # L(s, a, s')
            next_rm_state, _rm_reward, _rm_done = H.step(rm_state, true_props, info)

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

            rm_state = next_rm_state # TODO FIXME this entire loop, comment and organize

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
                # print("DONE")
                num_episodes += 1
                total_episodes += 1
                # if rm_run(labels, H) != rewards:
                if not run_sum_approx_eqv(rm_run(labels, H), rewards):
                    X_new.add((tuple(labels), tuple(rewards)))

                if num_episodes % UPDATE_X_EVERY_N_EPISODES == 0 and X_new:
                    print(f"len(X)={len(X)}")
                    print(f"len(X_new)={len(X_new)}")
                    X.update(X_new)
                    X_new = set()
                    language = sample_language(X)
                    empty_transition = dnf_for_empty(language)
                    transitions_new, n_states_last = consistent_hyp(X, n_states_last)
                    # if n_states_last >= 3:
                    #     exit()
                    H_new = rm_from_transitions(transitions_new, empty_transition)
                    H = H_new
                    transitions = transitions_new
                    Q = transfer_Q(H_new, H, Q, X)
                    # if len(X) > X_PRUNE_MIN_SIZE:
                    #     X = prune_X(X, transitions, n_states_last)
                break
            s = sn
