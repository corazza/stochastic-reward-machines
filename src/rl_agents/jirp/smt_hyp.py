from z3 import *

from rl_agents.jirp.util import *
from rl_agents.jirp.consts import *
from reward_machines.reward_machine import RewardMachine
from reward_machines.rm_environment import RewardMachineEnv, RewardMachineHidden


last_displayed_states = 0

def smt_hyp(epsilon, X, n_states, n_states_A, transitions, empty_transition, report=True):
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
                    # s.add(Implies(And(x_p, d), x_q))
                    # (4)
                    s.add(Implies(And(x_p, d), y_q >= y_p + (o_A - o)))
                    # s.add(Implies(d, y_q >= y_p + (o_A - o)))
                    # (5)
                    s.add(Implies(And(x_p, d), z_q <= z_p + (o_A - o)))
                    # s.add(Implies(d, z_q <= z_p + (o_A - o)))

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
    if report:
        print(f"SMT SOLVING ({n_states}/{n_states_A}, epsilon={epsilon})")
    result = s.check()
    if report:
        print(result)
    if result == sat:
        model = s.model()
        stransitions = dict()
        for (p, a, q) in d_dict:
            if p == TERMINAL_STATE or q == TERMINAL_STATE:
                continue
            if is_true(model[d_dict[(p, a, q)]]):
                # print(f"{o_dict[(p, a)]} /// {model[o_dict[(p, a)]]}")
                o = model[o_dict[(p, a)]]
                if o is not None:
                    o = float(o.numerator_as_long())/float(o.denominator_as_long())
                else:
                    o = 0 # solver doesn't care (?)
                stransitions[(p, tuple(a))] = [q, o]

        global last_displayed_states
        if n_states != last_displayed_states and report:
            last_displayed_states = n_states
            display_transitions(transitions, f"original{n_states}-{n_states_A}")
            display_transitions(stransitions, f"approximation{n_states}-{n_states_A}")
            print(f"transitions for n_states={n_states}")
            print("transitions:", transitions)
            print("mtransitions:", stransitions)
        
        # return rm_from_transitions(stransitions, empty_transition)
        return stransitions
    else:
        return None
