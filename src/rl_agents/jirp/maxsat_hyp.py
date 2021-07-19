from rl_agents.jirp.util import *
from rl_agents.jirp.consts import *
from reward_machines.reward_machine import RewardMachine
from reward_machines.rm_environment import RewardMachineEnv, RewardMachineHidden
# from pysat.solvers import RC2
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

def maxsat_hyp(epsilon, X, X_tl, n_states, infer_termination, report=True, inspect=False, display=False):
    language = sample_language(X)
    empty_transition = dnf_for_empty(language)
    reward_alphabet = sample_reward_alphabet(X)

    prop_d = dict() # maps SAT's propvar (int) to (p: state, l: labels, q: state)
    prop_d_rev = dict()
    prop_o = dict() # maps SAT's propvar (int) to (p: state, l: labels, r: reward)
    prop_o_rev = dict()
    prop_x = dict() # maps SAT's propvar (int) to (l: labels, q: state)
    prop_x_rev = dict()
    prop_z = dict()
    prop_z_rev = dict()
    used_pvars = [0] # p. var. counter
    rc2 = RC2(WCNF()) # solver

    # convenience methods
    def add_pvar_d(d):
        nonlocal prop_d
        nonlocal prop_d_rev
        return add_pvar(prop_d, prop_d_rev, used_pvars, d)

    def add_pvar_o(o):
        nonlocal prop_o
        nonlocal prop_o_rev
        return add_pvar(prop_o, prop_o_rev, used_pvars, o)

    def add_pvar_x(trace, prefix, state):
        nonlocal prop_x
        nonlocal prop_x_rev
        subscript = (trace, prefix, state)
        return add_pvar(prop_x, prop_x_rev, used_pvars, subscript)
    
    def add_pvar_z(z):
        nonlocal prop_z
        nonlocal prop_z_rev
        return add_pvar(prop_z, prop_z_rev, used_pvars, z)

    # Encoding reward machines
    # (5)
    for p in all_states_here(n_states, infer_termination):
        for l in language:
            rc2.add_clause([add_pvar_d((p, l, q)) for q in all_states_here(n_states, infer_termination)])
            for q1 in all_states_here(n_states, infer_termination):
                for q2 in all_states_here(n_states, infer_termination):
                    if q1==q2:
                        continue
                    p_l_q1 = add_pvar_d((p, l, q1))
                    p_l_q2 = add_pvar_d((p, l, q2))
                    rc2.add_clause([-p_l_q1, -p_l_q2])

    # (6)
    for trace in X:
        rc2.add_clause([add_pvar_x(trace, tuple(), INITIAL_STATE)]) # starts in the initial state
        for p in all_states_here(n_states, infer_termination):
            if p == INITIAL_STATE:
                continue
            rc2.add_clause([-add_pvar_x(trace, tuple(), p)])

    # (7)
    for trace in X:
        for (labels, _rewards) in prefixes_trace(trace, without_terminal=False):
            if labels == ():
                continue
            lm = labels[0:-1]
            l = labels[-1]
            for p in all_states_here(n_states, infer_termination):
                for q in all_states_here(n_states, infer_termination):
                    x_1 = add_pvar_x(trace, lm, p)
                    d = add_pvar_d((p, l, q))
                    x_2 = add_pvar_x(trace, labels, q)
                    rc2.add_clause([-x_1, -d, x_2])

    # (8)
    for trace in X:
        for (labels, rewards) in prefixes_trace(trace, without_terminal=False):
            if labels == ():
                continue
            lm = labels[0:-1]
            l = labels[-1]
            r = rewards[-1]
            for p in all_states_here(n_states, infer_termination):
                for v in reward_alphabet:
                    if r == v:
                        continue
                    x = add_pvar_x(trace, lm, p)
                    o = add_pvar_o((p, l, v))
                    z = add_pvar_z(trace)
                    rc2.add_clause([-x, -o, z])

    # At least one output
    for p in all_states_here(n_states, infer_termination):
        for l in language:
            rc2.add_clause([add_pvar_o((p, l, r)) for r in reward_alphabet])

    # (Maximization)
    for trace in X:
        z = add_pvar_z(trace)
        rc2.add_clause([-z], weight=1)
    
    # (Termination)
    if infer_termination:
        for trace in X:
            for (labels, _rewards) in prefixes_trace(trace, without_terminal=True):
                if labels == ():
                    continue
                lm = labels[0:-1]
                l = labels[-1]
                for p in all_states_here(n_states, infer_termination):
                    if p == TERMINAL_STATE:
                        continue
                    x_1 = add_pvar_x(trace, lm, p)
                    d = add_pvar_d((p, l, TERMINAL_STATE))
                    rc2.add_clause([-x_1, -d])

        for trace in X:
            (labels, rewards) = trace
            if labels == ():
                continue
            lm = labels[0:-1]
            l = labels[-1]
            for p in all_states_here(n_states, infer_termination):
                if p == TERMINAL_STATE:
                    continue
                x_1 = add_pvar_x(trace, lm, p)
                d = add_pvar_d((p, l, TERMINAL_STATE))
                d_t = -d if (labels, rewards) in X_tl else d
                rc2.add_clause([-x_1, d_t])

        for p in all_states_here(n_states, infer_termination):
            if p == TERMINAL_STATE:
                continue
            for l in language:
                d = add_pvar_d((TERMINAL_STATE, l, p))
                rc2.add_clause([-d])

        for p in all_states_here(n_states, infer_termination):
            for l in language:
                o = add_pvar_o((TERMINAL_STATE, l, 0.0))
                rc2.add_clause([o])

    model = rc2.compute()
    if model is None:
        return None

    if report:
        cost = float(rc2.cost)/float(len(X))
        print(f"found, cost={rc2.cost}/{len(X)} ({cost})")

    corrupted_traces = set()
    transitions = dict() #defaultdict(lambda: [None, None]) # maps (state, true_props) to (state, reward)

    for pvar in model:
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
        elif abs(pvar) in prop_z:
            if pvar > 0:
                corrupted_traces.add(prop_z[abs(pvar)])
        else:
            raise ValueError("Uknown p-var dict")
    rc2.delete()
    return transitions, corrupted_traces
