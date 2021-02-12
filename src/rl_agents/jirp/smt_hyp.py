from z3 import *

from rl_agents.jirp.util import *
from rl_agents.jirp.consts import *
from reward_machines.reward_machine import RewardMachine
from reward_machines.rm_environment import RewardMachineEnv, RewardMachineHidden


def smt_hyp(epsilon, language, n_states, n_states_A, transitions, empty_transition, report=True, inspect=False, display=False):
    def delta_A(p_A, a):
        a = tuple(a)
        if (p_A, a) in transitions:
            return transitions[(p_A, a)][0]
        else:
            return TERMINAL_STATE
    def sigma_A(p_A, a):
        a = tuple(a)
        if (p_A, a) in transitions:
            return transitions[(p_A, a)][1]
        else:
            return 0.0

    d_dict = dict()
    x_dict = dict()
    o_dict = dict()
    y_dict = dict()
    z_dict = dict()

    s = Solver()
    # TODO REMOVE
    s.set(unsat_core=True)
    
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
            for p_A in all_states_here(n_states_A):
                for a in language:
                    q_A = delta_A(p_A, a)
                    if q_A == TERMINAL_STATE and not TERMINATION:
                        continue
                    x_p = x_dict[(p_A, p)]
                    x_q = x_dict[(q_A, q)]
                    d = d_dict[(p, a, q)]
                    y_p = y_dict[(p_A, p)]
                    y_q = y_dict[(q_A, q)]
                    z_p = z_dict[(p_A, p)]
                    z_q = z_dict[(q_A, q)]
                    o = o_dict[(p, a)]
                    o_A = sigma_A(p_A, a)
                    # (3)
                    s.add(Implies(And(x_p, d), x_q))
                    # (4)
                    s.add(Implies(And(x_p, d), y_q >= y_p + (o_A - o)))
                    # s.add(Implies(d, y_q >= y_p + (o_A - o)))
                    # (5)
                    s.add(Implies(And(x_p, d), z_q <= z_p + (o_A - o)))
                    # s.add(Implies(d, z_q <= z_p + (o_A - o)))

    # (6)
    for p_A in all_states_here(n_states_A):
        for p in all_states_here(n_states):
            z_p = z_dict[(p_A, p)]
            y_p = y_dict[(p_A, p)]
            s.add(z_p <= y_p)

    # (7)
    z_qi = z_dict[(INITIAL_STATE, INITIAL_STATE)]
    y_qi = y_dict[(INITIAL_STATE, INITIAL_STATE)]
    s.add(z_qi <= 0, y_qi >= 0)

    # (8)
    for p_A in all_states_here(n_states_A):
        for p in all_states_here(n_states):
            y_p = y_dict[(p_A, p)]
            z_p = z_dict[(p_A, p)]
            s.add(z_p >= -epsilon, y_p <= epsilon)

    # (Termination)
    if TERMINATION:
        for p in all_states_here(n_states):
            if p == TERMINAL_STATE:
                continue
            for l in language:
                d = d_dict[(TERMINAL_STATE, l, p)]
                s.add(Not(d))

        for p in all_states_here(n_states):
            for l in language:
                o = o_dict[(TERMINAL_STATE, l)]
                s.add(o == 0.0)

    if report:
        print(f"SMT SOLVING ({n_states}/{n_states_A}, epsilon={epsilon})")

    result = s.check()
    if report:
        print(result)
    # if result == unsat:
    #         import IPython
    #         IPython.embed()
    #         exit()
    if result == sat:
        model = s.model()
        stransitions = dict()
        for (p, a, q) in d_dict:
            if (p == TERMINAL_STATE or q == TERMINAL_STATE) and not TERMINATION:
                continue
            if is_true(model[d_dict[(p, a, q)]]):
                o = model[o_dict[(p, a)]]
                if o is not None:
                    o = float(o.numerator_as_long())/float(o.denominator_as_long())
                else:
                    o = 0 # solver doesn't care (?)
                stransitions[(p, tuple(a))] = [q, o]

        if display:
            display_transitions(transitions, f"smt_original{n_states}-{n_states_A}")
            display_transitions(stransitions, f"smt_approximation{n_states}-{n_states_A}")
        
        if inspect:
            display_transitions(transitions, "given")
            display_transitions(stransitions, "smt")

            def get_d(p, l, q):
                print(f"{d_dict[(p, l, q)]} = {model[d_dict[(p, l, q)]]}")

            def get_o(p, l):
                print(f"{o_dict[(p, l)]} = {model[o_dict[(p, l)]]}")

            def get_x(p_A, p):
                print(f"{x_dict[(p_A, p)]} = {model[x_dict[(p_A, p)]]}")

            def get_y(p_A, p):
                print(f"{y_dict[(p_A, p)]} = {model[y_dict[(p_A, p)]]}")

            def get_z(p_A, p):
                print(f"{z_dict[(p_A, p)]} = {model[z_dict[(p_A, p)]]}")

            for p_A in all_states_here(n_states_A):
                for p in all_states_here(n_states):
                    get_x(p_A, p)

            import IPython
            IPython.embed()
            exit()

        return stransitions
    else:
        return None
