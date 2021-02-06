"""
JIRP based method
"""
import math
import itertools
import random, time, copy
from profilehooks import profile

from mip import *
from baselines import logger
from reward_machines.reward_machine import RewardMachine
from reward_machines.rm_environment import RewardMachineEnv, RewardMachineHidden
from z3 import *


from rl_agents.jirp.util import *
from rl_agents.jirp.consts import *

def smt_hyp(epsilon, X, n_states, n_states_A, transitions, empty_transition):
    def delta_A(p_a, a):
        a = tuple(a)
        if (p_a, a) in transitions:
            return transitions[(p_a, a)][0]
        else:
            return TERMINAL_STATE
    def sigma_A(p_a, a):
        a = tuple(a)
        if (p_a, a) in transitions:
            return transitions[(p_a, a)][1]
        else:
            return 0.0

    def all_states_here(asdf):
        return all_states(asdf)

    language = sample_language(X)

    d_dict = dict()
    x_dict = dict()
    o_dict = dict()
    y_dict = dict()
    z_dict = dict()

    s = Solver()

    for p in all_states_here(n_states):
        for a in language:
            o_dict[(p, a)] = Real(f"o/{p}-{a}")

    for p in all_states_here(n_states):
        for a in language:
            for q in all_states_here(n_states):
                d_dict[(p, a, q)] = Bool(f"d/{p}-{a}-{q}")

    for p_A in all_states_here(n_states_A):
        for p in all_states_here(n_states):
            x_dict[(p_A, p)] = Bool(f"x/{p_A}-{p}")
            y_dict[(p_A, p)] = Real(f"y/{p_A}-{p}")
            z_dict[(p_A, p)] = Real(f"z/{p_A}-{p}")

    # (1)
    for p in all_states_here(n_states):
        for a in language:
            disj = []
            for q in all_states_here(n_states):
                disj.append(d_dict[(p, a, q)])
            disj = Or(*disj)
            s.add(disj)

    for p in all_states_here(n_states):
        for a in language:
            for q1 in all_states_here(n_states):
                for q2 in all_states_here(n_states):
                    if q1 == q2:
                        continue
                    s.add(Or(Not(d_dict[(p, a, q1)]), Not(d_dict[(p, a, q2)])))

    # (2)
    s.add(x_dict[(INITIAL_STATE, INITIAL_STATE)])

    for p in all_states_here(n_states):
        for q in all_states_here(n_states):
            for p_a in all_states_here(n_states_A):
                for a in language:
                    q_a = delta_A(p_a, a)
                    if q_a == TERMINAL_STATE:
                        continue
                    x_p = x_dict[(p_a, p)]
                    x_q = x_dict[(q_a, q)]
                    d = d_dict[(p, a, q)]
                    y_p = y_dict[(p_a, p)]
                    y_q = y_dict[(q_a, q)]
                    z_p = z_dict[(p_a, p)]
                    z_q = z_dict[(q_a, q)]
                    o = o_dict[(p, a)]
                    o_A = sigma_A(p_a, a)
                    # (3)
                    s.add(Implies(And(x_p, d), x_q))
                    # (4)
                    s.add(Implies(And(x_p, d), y_q >= y_p + (o_A - o)))
                    # (5)
                    s.add(Implies(And(x_p, d), z_q <= z_p + (o_A - o)))

    # (6)
    for p_A in all_states_here(n_states_A):
        for p in all_states_here(n_states):
            z_p = z_dict[(p_a, p)]
            y_p = y_dict[(p_a, p)]
            s.add(z_p <= y_p)

    # (7)
    z_qi = z_dict[(INITIAL_STATE, INITIAL_STATE)]
    y_qi = y_dict[(INITIAL_STATE, INITIAL_STATE)]
    s.add(z_qi <= 0, y_qi >= 0)

    # (8)
    for p_A in all_states_here(n_states_A):
        for p in all_states_here(n_states):
            y_p = y_dict[(p_a, p)]
            z_p = z_dict[(p_a, p)]
            s.add(z_p >= -epsilon, y_p <= epsilon)

    print(f"SMT SOLVING ({n_states}/{n_states_A}, epsilon={epsilon})")
    result = s.check()
    print(result)
    if result == sat:
        model = s.model()
        stransitions = dict()
        for (p, a, q) in d_dict:
            if p == TERMINAL_STATE or q == TERMINAL_STATE:
                continue
            if is_true(model[d_dict[(p, a, q)]]):
                print(f"{o_dict[(p, a)]} /// {model[o_dict[(p, a)]]}")
                o = model[o_dict[(p, a)]]
                if o is not None:
                    o = float(o.numerator_as_long())/float(o.denominator_as_long())
                else:
                    o = 0 # solver doesn't care (?)
                stransitions[(p, tuple(a))] = [q, o]

        display_transitions(transitions, f"original{n_states}-{n_states_A}")
        display_transitions(stransitions, f"approximation{n_states}-{n_states_A}")
        print(f"transitions for n_states={n_states}")
        print("transitions:", transitions)
        print("mtransitions:", stransitions)
        
        return rm_from_transitions(stransitions, empty_transition)
    else:
        return None

def mlip_hyp(X, n_states, n_states_A, transitions, empty_transition):
    def delta_A(p_a, a):
        a = tuple(a)
        if (p_a, a) in transitions:
            return transitions[(p_a, a)][0]
        else:
            return TERMINAL_STATE
    def sigma_A(p_a, a):
        a = tuple(a)
        if (p_a, a) in transitions:
            return transitions[(p_a, a)][1]
        else:
            return 0.0

    def all_states_here(asdf):
        return all_states(asdf)

    language = sample_language(X)

    reward_bound = 1000.0
    epsilon_bound = 1000.0
    interval_bound = 10000.0

    m = Model()
    # m.verbose = 0
    # m.emphasis = 1
    m.threads = -1
    m.pump_passes = 300

    epsilon = m.add_var(var_type=CONTINUOUS, ub=epsilon_bound)
    m.objective = minimize(epsilon)

    d_dict = dict()
    x_dict = dict()
    o_dict = dict()
    y_dict = dict()
    z_dict = dict()

    for p in all_states_here(n_states):
        for a in language:
            o_dict[(p, a)] = m.add_var(var_type=CONTINUOUS, lb=-reward_bound, ub=reward_bound)

    for p in all_states_here(n_states):
        for p_a in all_states_here(n_states_A):
            x_dict[(p_a, p)] = m.add_var(var_type=BINARY)
            y_dict[(p_a, p)] = m.add_var(var_type=CONTINUOUS, lb=-interval_bound, ub=interval_bound)
            z_dict[(p_a, p)] = m.add_var(var_type=CONTINUOUS, lb=-interval_bound, ub=interval_bound)

    for p in all_states_here(n_states):
        for q in all_states_here(n_states):
            for a in language:
                d_dict[(p, a, q)] = m.add_var(var_type=BINARY)

    # TODO remove
    # for q in all_states(n_states):
    #     x_dict[(TERMINAL_STATE, q)] = m.add_var(var_type=BINARY)
    #     y_dict[(TERMINAL_STATE, q)] = m.add_var(lb=-math.inf)
    #     z_dict[(TERMINAL_STATE, q)] = m.add_var(lb=-math.inf)

    # (i)
    m += epsilon >= 0

    # (ii)
    # for p in all_states_here(n_states):
    #     for q in all_states_here(n_states):
    #         for a in language:
    #             m += 0 <= d_dict[(p, a, q)] <= 1

    # (iii)
    for p in all_states_here(n_states):
        for a in language:
            m += xsum(d_dict[(p, a, q)] for q in all_states_here(n_states)) == 1

    # (iv)
    # for p_a in all_states_here(n_states_A):
    #     for p in all_states_here(n_states):
    #         m += 0 <= x_dict[(p_a, p)] <= 1

    # (v)
    m += x_dict[(INITIAL_STATE, INITIAL_STATE)] == 1
    
    for p in all_states_here(n_states):
        for q in all_states_here(n_states):
            for p_a in all_states_here(n_states_A):
                for a in language:
                    q_a = delta_A(p_a, a)
                    if q_a == TERMINAL_STATE:
                        continue
                    x_p = x_dict[(p_a, p)]
                    x_q = x_dict[(q_a, q)]
                    d = d_dict[(p, a, q)]
                    # (vi)
                    m += x_p + d - x_q <= 1
                    # (vii)
                    y_p = y_dict[(p_a, p)]
                    y_q = y_dict[(q_a, q)]
                    o = o_dict[(p, a)]
                    o_A = sigma_A(p_a, a)
                    m += M*x_p + M*d - (y_q - (y_p + (o_A - o))) <= 2*M
                    # (viii)
                    z_p = z_dict[(p_a, p)]
                    z_q = z_dict[(q_a, q)]
                    m += M*x_p + M*d - ((z_p + (o_A - o)) - z_q) <= 2*M
    
    # (ix)
    for p in all_states_here(n_states):
        for p_a in all_states_here(n_states_A):
            m += z_dict[(p_a, p)] <= y_dict[(p_a, p)]

    # (x)
    m += z_dict[(INITIAL_STATE, INITIAL_STATE)] <= 0
    m += y_dict[(INITIAL_STATE, INITIAL_STATE)] >= 0

    for p in all_states_here(n_states):
        for p_a in all_states_here(n_states_A):
            # (xi)
            m += y_dict[(p_a, p)] <= epsilon
            # (xii)
            m += -epsilon <= z_dict[(p_a, p)]

    print(f"Starting MLIP solving, n_states={n_states}, n_states_A={n_states_A}")
    m.optimize()
    print(f"Done, epsilon={epsilon.x}")

    if epsilon.x > 0:
        m.preprocess = 0
        m.optimize()
        print(f"Done new, epsilon={epsilon.x}")

    mtransitions = dict()
    for (p, a, q) in d_dict:
        if p == TERMINAL_STATE or q == TERMINAL_STATE or d_dict[(p, a, q)].x is None:
            continue
        if d_dict[(p, a, q)].x > 0:
            o = o_dict[(p, a)].x
            mtransitions[(p, tuple(a))] = [q, o]

    display_transitions(transitions, f"original{n_states}-{n_states_A}")
    display_transitions(mtransitions, f"approximation{n_states}-{n_states_A}")
    print(f"transitions for n_states={n_states}")
    print("transitions:", transitions)
    print("mtransitions:", mtransitions)
    return rm_from_transitions(mtransitions, empty_transition)

# @profile
def consistent_hyp(X, n_states_start=2):
    """
    Finds a reward machine consistent with counterexample set X. Returns the RM
    and its number of states

    n_states_start makes the search start from machines with that number of states.
    Used to optimize succeeding search calls.
    """
    language = sample_language(X)
    empty_transition = dnf_for_empty(language)
    reward_alphabet = sample_reward_alphabet(X)

    from pysat.solvers import Glucose4
    for n_states in range(n_states_start, MAX_RM_STATES_N+1):
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

        # mlip_n_states = n_states - 1 if n_states >= 4 else n_states
        # return mlip_hyp(X, mlip_n_states, n_states, transitions, empty_transition), n_states


        # if n_states >= 2:
        #     for i in range(2, n_states):
        #         minimized_rm = smt_hyp(SMT_EPSILON, X, i, n_states, transitions, empty_transition)
        #         if minimized_rm:
        #             print(f"FOUND MINIMIZED RM {i} < {n_states} (epsilon={SMT_EPSILON})")
        #             return minimized_rm, n_states

        # print("couldn't find minimized RM, returning exact")


        return smt_hyp(SMT_EPSILON, X, n_states, n_states, transitions, empty_transition), n_states
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

    H, n_states_last = consistent_hyp(set())
    actions = list(range(env.action_space.n))
    Q = initial_Q(H)
    X = set()
    X_new = set()
    labels = []
    rewards = []

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
                    H_new, n_states_last = consistent_hyp(X, n_states_last)
                    # if n_states_last >= 3:
                    #     exit()
                    Q = transfer_Q(H_new, H, Q, X)
                    H = H_new
                break
            s = sn
