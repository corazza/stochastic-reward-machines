from z3 import *
import IPython

from reward_machines.reward_machine import RewardMachine
from reward_machines.rm_environment import RewardMachineEnv, RewardMachineHidden
from rl_agents.jirp.consts import *
from rl_agents.jirp.util import *
import time

class Exporter:
    def __init__(self, epsilon, X, X_tl, n_states, language, empty_transition, infer_termination, seed):
        self.epsilon = epsilon
        self.X = sorted(list(X))
        self.X_tl = sorted(list(X_tl))
        self.n_states = n_states
        self.language = sorted(list(language))
        self.empty_transition = empty_transition
        self.infer_termination = infer_termination
        self.seed = seed
        # self.reward_alphabet = list(reward_alphabet)

# TODO termination inference
def smt_noise_cpp(epsilon, X, X_tl, n_states, infer_termination, report=True, inspect=False, display=False, alg_name = None, seed=None):
    import json, sys, os
    language = sample_language(X)
    empty_transition = dnf_for_empty(language)
    reward_alphabet = sample_reward_alphabet(X)
    exporter = Exporter(epsilon, X, X_tl, n_states, language, empty_transition, infer_termination, seed)

    timestamp = time.time_ns()
    filename=f"{alg_name}-{timestamp}tmp.json"
    data = json.dumps(exporter.__dict__)

    with open(filename, 'w') as f:
        f.write(data)

    if report:
        os.system(f"rl_agents/sjirp/cpp/a.out {filename} asdf")
    else:
        os.system(f"rl_agents/sjirp/cpp/a.out {filename}")

    json_file = open(filename, 'r')
    data = json.load(json_file)
    json_file.close()
    os.remove(filename)

    if data[0] == 'unsat':
        return None

    transitions = dict()        
    for transition in data:
        [p, a, q, o] = transition
        transitions[(p, tuple(a))] = [q, float(o)]
    return transitions

def smt_noise(epsilon, X, X_tl, n_states, infer_termination, report=True, inspect=False, display=False):
    language = sample_language(X)
    empty_transition = dnf_for_empty(language)
    reward_alphabet = sample_reward_alphabet(X)

    d_dict = dict()
    o_dict = dict() # represents guessed mean
    e_dict = dict() # exact
    n_dict = dict()
    x_dict = dict()
    
    s = Solver()

    for p in all_states_here(n_states):
        for a in language:
            for q in all_states_here(n_states):
                d_dict[(p, a, q)] = Bool(f"d/{p}-{a}-{q}")

    for p in all_states_here(n_states):
        for a in language:
            o_dict[(p, a)] = Real(f"o/{p}-{a}")
            e_dict[(p, a)] = Bool(f"e/{p}-{a}")

    def add_x(ls, p):
        nonlocal x_dict
        if (ls, p) not in x_dict:
            x_dict[(ls, p)] = Bool(f"x/{len(x_dict)}")
        return x_dict[(ls, p)]

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
    for (labels, rewards) in prefixes(X, without_terminal=False):
        if labels == ():
            continue
        lm = labels[0:-1]
        l = labels[-1]
        r = rewards[-1]
        for p in all_states_here(n_states):
            x_1 = add_x(lm, p)
            o = o_dict[(p, l)]
            s.add(Implies(x_1, And(r - o > -epsilon, r - o < epsilon)))
            for q in all_states_here(n_states):
                d = d_dict[(p, l, q)]
                x_2 = add_x(labels, q)
                s.add(Implies(And(x_1, d), x_2))

    # (Termination)
    if infer_termination:
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
            if (p == TERMINAL_STATE or q == TERMINAL_STATE) and not infer_termination:
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




# def smt_noise(epsilon, X, X_tl, n_states, report=True, inspect=False, display=False):
#     language = sample_language(X)
#     empty_transition = dnf_for_empty(language)
#     reward_alphabet = sample_reward_alphabet(X)

#     d_dict = dict()
#     o_dict = dict() # represents guessed mean
#     e_dict = dict() # exact
#     n_dict = dict()
#     x_dict = dict()
    
#     s = Solver()

#     print("here1")

#     for p in all_states_here(n_states):
#         for a in language:
#             for q in all_states_here(n_states):
#                 d_dict[(p, a, q)] = Bool(f"d/{p}-{a}-{q}")

#     print("here2")

#     for p in all_states_here(n_states):
#         for a in language:
#             o_dict[(p, a)] = Real(f"o/{p}-{a}")
#             e_dict[(p, a)] = Bool(f"e/{p}-{a}")

#     def add_x(ls, p):
#         nonlocal x_dict
#         if (ls, p) not in x_dict:
#             x_dict[(ls, p)] = Bool(f"x/{len(x_dict)}")
#         return x_dict[(ls, p)]

#     # Encoding reward machines
#     # (1)
#     for p in all_states_here(n_states):
#         for l in language:
#             disj = []
#             for q in all_states_here(n_states):
#                 disj.append(d_dict[p, l, q])
#             s.add(Or(*disj))
#             for q1 in all_states_here(n_states):
#                 for q2 in all_states_here(n_states):
#                     if q1 == q2:
#                         continue
#                     p_l_q1 = d_dict[(p, l, q1)]
#                     p_l_q2 = d_dict[(p, l, q2)]
#                     s.add(Or(Not(p_l_q1), Not(p_l_q2)))

#     print("here3")

#     # Consistency with sample
#     # (3)
#     s.add(add_x(tuple(), INITIAL_STATE)) # starts in the initial state
#     for p in all_states_here(n_states):
#         if p == INITIAL_STATE:
#             continue
#         s.add(Not(add_x(tuple(), p)))

#     print("here4")

#     # (4)
#     for (labels, rewards) in prefixes(X, without_terminal=False):
#         if labels == ():
#             continue
#         lm = labels[0:-1]
#         l = labels[-1]
#         r = rewards[-1]
#         for p in all_states_here(n_states):
#             for q in all_states_here(n_states):
#                 x_1 = add_x(lm, p)
#                 d = d_dict[(p, l, q)]
#                 x_2 = add_x(labels, q)
#                 s.add(Implies(And(x_1, d), x_2))

#     print("here5")

#     # (5)
#     for (labels, rewards) in prefixes(X, without_terminal=False):
#         if labels == ():
#             continue
#         lm = labels[0:-1]
#         l = labels[-1]
#         r = rewards[-1]
#         for p in all_states_here(n_states):
#             x = add_x(lm, p)
#             o = o_dict[(p, l)]
#             s.add(Implies(x, And(r - o > -epsilon, r - o < epsilon)))

#     print("here6")

#     # (Termination)
#     if TERMINATION:
#         for (labels, _rewards) in prefixes(X, without_terminal=True):
#             if labels == ():
#                 continue
#             lm = labels[0:-1]
#             l = labels[-1]
#             x_2 = add_x(labels, TERMINAL_STATE) # TODO REMOVE unneeded
#             for p in all_states_here(n_states):
#                 if p == TERMINAL_STATE:
#                     continue
#                 x_1 = add_x(lm, p)
#                 d = d_dict[(p, l, TERMINAL_STATE)]
#                 s.add(Implies(x_1, Not(d)))

#         print("here7")

#         for (labels, rewards) in X:
#             if labels == ():
#                 continue
#             lm = labels[0:-1]
#             l = labels[-1]
#             x_2 = add_x(labels, TERMINAL_STATE) # TODO REMOVE unneeded
#             for p in all_states_here(n_states):
#                 if p == TERMINAL_STATE:
#                     continue
#                 x_1 = add_x(lm, p)
#                 d = d_dict[(p, l, TERMINAL_STATE)]
#                 d_t = Not(d) if (labels, rewards) in X_tl else d
#                 s.add(Implies(x_1, d_t))

#         print("here8")

#         for p in all_states_here(n_states):
#             if p == TERMINAL_STATE:
#                 continue
#             for l in language:
#                 d = d_dict[(TERMINAL_STATE, l, p)]
#                 s.add(Not(d))

#         print("here9")

#         for p in all_states_here(n_states):
#             for l in language:
#                 o = o_dict[(TERMINAL_STATE, l)]
#                 s.add(o == 0.0)

#     if report:
#         print(f"SMT SOLVING ({n_states}, epsilon={epsilon})")

#     result = s.check()
#     if report:
#         print(result)

#     if result == sat:
#         model = s.model()
#         stransitions = dict()
#         for (p, a, q) in d_dict:
#             if (p == TERMINAL_STATE or q == TERMINAL_STATE) and not TERMINATION:
#                 continue
#             if is_true(model[d_dict[(p, a, q)]]):
#                 o = model[o_dict[(p, a)]]
#                 if o is not None:
#                     o = float(o.numerator_as_long())/float(o.denominator_as_long())
#                 else:
#                     o = 0 # solver doesn't care (?)
#                 stransitions[(p, tuple(a))] = [q, o]

#         if display:
#             display_transitions(stransitions, f"smttt{n_states}")

#         return stransitions
#     else:
#         return None
