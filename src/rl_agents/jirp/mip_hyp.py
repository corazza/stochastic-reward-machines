from mip import *

from rl_agents.jirp.util import *
from rl_agents.jirp.consts import *
from reward_machines.reward_machine import RewardMachine
from reward_machines.rm_environment import RewardMachineEnv, RewardMachineHidden


def mip_hyp(epsilon_bound, X, X_tl, n_states, report=True, inspect=False, display=False):
    language = sample_language(X)
    empty_transition = dnf_for_empty(language)
    reward_alphabet = sample_reward_alphabet(X)

    reward_bound = 2.0
    # epsilon_bound = 1000.0
    interval_bound = 3.0

    m = Model()
    # m.verbose = 0
    m.emphasis = 1
    m.threads = -1
    m.pump_passes = 500 if len(X) < 10 else 1500
    m.infeas_tol = 1e-5
    m.cutoff = epsilon_bound*2.0
    m.max_solutions = 1

    epsilon = m.add_var(var_type=CONTINUOUS, ub=epsilon_bound*3.0)
    m.objective = minimize(epsilon)

    d_dict = dict()
    o_dict = dict()
    x_dict = dict()
    z_dict = dict()
    y_dict = dict()
    
    for p in all_states_here(n_states):
        for a in language:
            for q in all_states_here(n_states):
                d_dict[(p, a, q)] = m.add_var(var_type=BINARY)

    for p in all_states_here(n_states):
        for a in language:
            o_dict[(p, a)] = m.add_var(var_type=CONTINUOUS, lb=-reward_bound, ub=reward_bound)

    for p in all_states_here(n_states):
        z_dict[p] = m.add_var(var_type=CONTINUOUS, lb=-interval_bound, ub=interval_bound)
        y_dict[p] = m.add_var(var_type=CONTINUOUS, lb=-interval_bound, ub=interval_bound)

    def add_x(ls, p):
        nonlocal x_dict
        if (ls, p) not in x_dict:
            x_dict[(ls, p)] = m.add_var(var_type=BINARY)
        return x_dict[(ls, p)]

    m += 0 <= epsilon <= epsilon_bound

    # Encoding reward machines
    # (1)
    for p in all_states_here(n_states):
        for l in language:
            disj = []
            for q in all_states_here(n_states):
                disj.append(d_dict[p, l, q])
            m += xsum(disj) == 1
            for q1 in all_states_here(n_states):
                for q2 in all_states_here(n_states):
                    if q1 == q2:
                        continue
                    p_l_q1 = d_dict[(p, l, q1)]
                    p_l_q2 = d_dict[(p, l, q2)]
                    m += p_l_q1 + p_l_q2 <= 1

    # got to 6 states, smoething wrong

    # (2)
    # for p in all_states_here(n_states):
    #     for l in language:
    #         s.add(Or(*[o_dict[(p, l, r)] for r in reward_alphabet]))
    #         for r1 in reward_alphabet:
    #             for r2 in reward_alphabet:
    #                 if r1 == r2:
    #                     continue
    #                 p_l_r1 = o_dict[(p, l, r1)]
    #                 p_l_r2 = o_dict[(p, l, r2)]
    #                 s.add(Or(Not(p_l_r1), Not(p_l_r2)))

    # Consistency with sample
    # (3)
    m += add_x(tuple(), INITIAL_STATE) == 1
    for p in all_states_here(n_states):
        if p == INITIAL_STATE:
            continue
        m += add_x(tuple(), p) == 0

    # (4)
    for (labels, _rewards) in prefixes(X, without_terminal=False):
        if labels == ():
            continue
        lm = labels[0:-1]
        l = labels[-1]
        for p in all_states_here(n_states):
            for q in all_states_here(n_states):
                x_1 = add_x(lm, p)
                d = d_dict[(p, l, q)]
                x_2 = add_x(labels, q)
                # s.add(Or(*[-x_1, -d, x_2])) # TODO use Implies and And
                # s.add(Implies(And(x_1, d), x_2))
                m += x_1 + d - x_2 <= 1

    # (5)
    for (labels, rewards) in prefixes(X, without_terminal=False):
        if labels == ():
            continue
        lm = labels[0:-1]
        l = labels[-1]
        r = rewards[-1]
        for p in all_states_here(n_states):
            x = add_x(lm, p)
            o = o_dict[(p, l)]
            z_p = z_dict[p]
            y_p = y_dict[p]
            for q in all_states_here(n_states):
                z_q = z_dict[q]
                y_q = y_dict[q]
                d = d_dict[(p, l, q)]
                m += M*x + M*d - (y_q - (y_p + (r - o))) <= 2*M
                m += M*x + M*d - ((z_p + (r - o)) - z_q) <= 2*M


    for p in all_states_here(n_states):
        z_p = z_dict[p]
        y_p = y_dict[p]
        m += z_p <= y_p
        m += z_p >= -epsilon
        m += y_p <= epsilon

    z_qi = z_dict[INITIAL_STATE]
    y_qi = y_dict[INITIAL_STATE]

    m += z_qi <= 0
    m += y_qi >= 0


    # (Termination)
    if TERMINATION:
        for (labels, _rewards) in prefixes(X, without_terminal=True):
            if labels == ():
                continue
            lm = labels[0:-1]
            l = labels[-1]
            x_2 = add_x(labels, TERMINAL_STATE) # TODO REMOVE unneeded
            for p in all_states_here(n_states):
                if p == TERMINAL_STATE:
                    continue
                x_1 = add_x(lm, p)
                d = d_dict[(p, l, TERMINAL_STATE)]
                m += x_1 - (1 - d) <= 0
                # s.add(Implies(x_1, Not(d)))

        for (labels, rewards) in X:
            if labels == ():
                continue
            lm = labels[0:-1]
            l = labels[-1]
            x_2 = add_x(labels, TERMINAL_STATE) # TODO REMOVE unneeded
            for p in all_states_here(n_states):
                if p == TERMINAL_STATE:
                    continue
                x_1 = add_x(lm, p)
                d = d_dict[(p, l, TERMINAL_STATE)]
                # d_t = -d if (labels, rewards) in X_tl else d
                # s.add(Implies(x_1, d_t))
                if (labels, rewards) in X_tl:
                    m += x_1 - (1 - d) <= 0
                else:
                    m += x_1 - d <= 0

        for p in all_states_here(n_states):
            if p == TERMINAL_STATE:
                continue
            for l in language:
                d = d_dict[(TERMINAL_STATE, l, p)]
                # s.add(Not(d))
                m += d == 0

        for p in all_states_here(n_states):
            for l in language:
                o = o_dict[(TERMINAL_STATE, l)]
                # s.add(o == 0.0)
                m += o == 0.0

    if report:
        print(f"Starting MIP solving, n_states={n_states}")
    # result = m.optimize(max_seconds=15*60)
    result = m.optimize()
    # IPython.embed()
    if report:
        print(f"Done, epsilon={epsilon.x}")

    if epsilon.x == None:
        return None
    elif epsilon.x >= epsilon_bound:
        return None

    mtransitions = dict()
    for (p, a, q) in d_dict:
        if ((p == TERMINAL_STATE or q == TERMINAL_STATE) and not TERMINATION) or d_dict[(p, a, q)].x is None:
            continue
        if d_dict[(p, a, q)].x > 0:
            o = o_dict[(p, a)].x
            mtransitions[(p, tuple(a))] = [q, o]

    return mtransitions
