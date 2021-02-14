
def prune_X(X, transitions, n_states):
    print("pruning X")
    X_result = set(X)
    # n_states = len(H.U)
    found = True
    while found:
        found = False
        for i in range(0, min(X_PRUNE_MAX, len(X_result))):
            i = random.randint(0, len(X_result)-1)
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

    # X_result = set()
    # first = True
    # for (labels, rewards) in X:
    #     labels_set = set(labels)
    #     if labels_set != set(('', 'b')) or first:
    #         X_result.add((labels, rewards))
    #         first = False

    print(f"new size is {len(X_result)}")

    return X_result





###############


from z3 import *

from rl_agents.jirp.util import *
from rl_agents.jirp.consts import *
from reward_machines.reward_machine import RewardMachine
from reward_machines.rm_environment import RewardMachineEnv, RewardMachineHidden


def smt_hyp(epsilon, X, X_tl, n_states, report=True, inspect=False, display=False):
    language = sample_language(X)
    empty_transition = dnf_for_empty(language)
    reward_alphabet = sample_reward_alphabet(X)

    d_dict = dict()
    o_dict = dict()
    x_dict = dict()
    y_dict = dict()
    z_dict = dict()
    
    s = Solver()

    for p in all_states_here(n_states):
        for a in language:
            for q in all_states_here(n_states):
                d_dict[(p, a, q)] = Bool(f"d/{p}-{a}-{q}")

    for p in all_states_here(n_states):
        for a in language:
            o_dict[(p, a)] = Real(f"o/{p}-{a}")

    for p in all_states_here(n_states):
        z_dict[p] = Real(f"z/{p}")
        y_dict[p] = Real(f"y/{p}")

    def add_x(ls, p):
        nonlocal x_dict
        if (ls, p) not in x_dict:
            x_dict[(ls, p)] = Bool(f"x/{len(x_dict)}")
        return x_dict[(ls, p)]

    # def add_z(ls, p):
    #     nonlocal z_dict
    #     if (ls, p) not in z_dict:
    #         z_dict[(ls, p)] = Real(f"z/{len(z_dict)}")
    #     return z_dict[(ls, p)]

    # def add_y(ls, p):
    #     nonlocal y_dict
    #     if (ls, p) not in y_dict:
    #         y_dict[(ls, p)] = Real(f"y/{len(y_dict)}")
    #     return y_dict[(ls, p)]

    # Encoding reward machines
    # (1)
    for p in all_states_here(n_states):
        for l in language:
            disj = []
            for q in all_states_here(n_states):
                disj.append(d_dict[p, l, q])
            s.add(Or(*disj))
            for q1 in all_states_here(n_states):
                for q2 in all_states_here(n_states):
                    if q1 == q2:
                        continue
                    p_l_q1 = d_dict[(p, l, q1)]
                    p_l_q2 = d_dict[(p, l, q2)]
                    s.add(Or(Not(p_l_q1), Not(p_l_q2)))

    # Consistency with sample
    # (3)
    s.add(add_x(tuple(), INITIAL_STATE)) # starts in the initial state
    for p in all_states_here(n_states):
        if p == INITIAL_STATE:
            continue
        s.add(Not(add_x(tuple(), p)))

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
                s.add(Implies(And(x_1, d), x_2))

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
            # s.add(Implies(x, -epsilon <= o - r <= epsilon))
            # s.add(x*M + (o - r) <= M + epsilon)
            # s.add(x*M - (o - r) <= M + epsilon)
            z_p = z_dict[p]
            y_p = y_dict[p]
            # z_p = add_z(lm, p)
            # y_p = add_y(lm, p)
            # s.add(z_p <= y_p)
            # s.add(z_p >= -epsilon, y_p <= epsilon)
            for q in all_states_here(n_states):
                z_q = z_dict[q]
                y_q = y_dict[q]
                # z_q = add_z(labels, q)
                # y_q = add_y(labels, q)
                # s.add(z_q <= y_q)
                # s.add(z_q >= -epsilon, y_q <= epsilon)
                d = d_dict[(p, l, q)]
                s.add(Implies(And(x, d), y_q >= y_p + (r - o)))
                s.add(Implies(And(x, d), z_q <= z_p + (r - o)))

    for p in all_states_here(n_states):
        z_p = z_dict[p]
        y_p = y_dict[p]
        s.add(z_p <= y_p)
        s.add(z_p >= -epsilon, y_p <= epsilon)

    z_qi = z_dict[INITIAL_STATE]
    y_qi = y_dict[INITIAL_STATE]
    # z_qi = add_z(tuple(), TERMINAL_STATE)
    # y_qi = add_y(tuple(), TERMINAL_STATE)

    s.add(z_qi <= 0, y_qi >= 0)

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
                s.add(Implies(x_1, Not(d)))

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
                d_t = Not(d) if (labels, rewards) in X_tl else d
                s.add(Implies(x_1, d_t))

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
        print(f"SMT SOLVING ({n_states}, epsilon={epsilon})")

    result = s.check()
    if report:
        print(result)

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
            display_transitions(stransitions, f"smttt{n_states}")

        return stransitions
    else:
        return None
        