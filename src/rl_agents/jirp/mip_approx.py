from mip import *

from rl_agents.jirp.util import *
from rl_agents.jirp.consts import *
from reward_machines.reward_machine import RewardMachine
from reward_machines.rm_environment import RewardMachineEnv, RewardMachineHidden


def mip_approx(epsilon_fixed, language, n_states, rm, report=True, inspect=False, display=False):
    n_states_A = len(rm.U)
    def delta_A(p_A, a):
        a = tuple(a)
        if p_A == rm.terminal_u:
            return rm.terminal_u
        u2, rew, _done = rm.step(p_A, a, {})
        return u2
    def sigma_A(p_A, a):
        a = tuple(a)
        if p_A == rm.terminal_u:
            return 0.0
        u2, rew, _done = rm.step(p_A, a, {})
        return rew

    reward_bound = 1.5
    epsilon_bound = 1.0
    interval_bound = 5.0

    m = Model()
    # m.verbose = 0
    m.emphasis = 1
    m.threads = -1
    m.pump_passes = 150

    epsilon = m.add_var(var_type=CONTINUOUS, ub=epsilon_bound)
    m.objective = minimize(epsilon) if epsilon_fixed is None else 1

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

    if epsilon_fixed:
        m += epsilon == epsilon_fixed

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
                    if q_a == TERMINAL_STATE and not TERMINATION:
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

    if TERMINATION:
        for p in all_states_here(n_states):
            if p == TERMINAL_STATE:
                continue
            for l in language:
                d = d_dict[(TERMINAL_STATE, l, p)]
                m += d == 0

        for p in all_states_here(n_states):
            for l in language:
                o = o_dict[(TERMINAL_STATE, l)]
                m += o == 0.0


    if report:
        print(f"Starting MIP solving, n_states={n_states}, n_states_A={n_states_A}")
    m.optimize()
    if report:
        print(f"Done, epsilon={epsilon.x}")

    if epsilon.x == None:
        return None

    # if epsilon.x > 0:
    #     m.preprocess = 0
    #     m.optimize()
    #     if report:
    #         print(f"Done new, epsilon={epsilon.x}")

    mtransitions = dict()
    for (p, a, q) in d_dict:
        if ((p == TERMINAL_STATE or q == TERMINAL_STATE) and not TERMINATION) or d_dict[(p, a, q)].x is None:
            continue
        if d_dict[(p, a, q)].x > 0:
            o = o_dict[(p, a)].x
            mtransitions[(p, tuple(a))] = [q, o]

    if display:
        display_transitions(transitions, f"mip_original{n_states}-{n_states_A}")
        display_transitions(mtransitions, f"mip_approximation{n_states}-{n_states_A}")

    return mtransitions
