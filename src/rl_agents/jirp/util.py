import random
import itertools
import os, tempfile
from collections import defaultdict
from graphviz import Digraph

from reward_machines.reward_machine import RewardMachine

from rl_agents.jirp.consts import *

def rm_run(labels, H):
    """
    Returns the output of H when labels is provided as input
    """
    current_state = H.reset()
    rewards = []
    for props in labels:
        current_state, reward, done = H.step(current_state, props, {"true_props": props})
        rewards.append(reward)
        if done: 
            break
    return rewards

def run_approx_eqv(output1, output2):
    """
    Returns True if outputs are approximately equivalent
    """
    if len(output1) != len(output2):
        return False
    for i in range(0, len(output1)):
        if abs(output1[i] - output2[i]) > SMT_EPSILON:
            return False
    return True

def sample_language(X):
    """
    Returns the set of all values for true_props strings for a given counter-example set X

    E.g. the sample language for {(("b", "ab"), (0.0, 1.0)), (("ab", "a", "f"), (0.0, 0.0, 1.0))}
    is {"", "f", "b", "a", "ab"}.
    """
    language = set()
    language.add("") # always contains empty string
    for (labels, _rewards) in X:
        language.update(labels)
    return language

def sample_reward_alphabet(X):
    """
    Returns the set of all reward values that appear in X
    """
    alphabet = set()
    alphabet.add(0.0) # always includes 0
    for (_labels, rewards) in X:
        alphabet.update(rewards)
    return alphabet

def dnf_for_empty(language):
    """
    Returns the "neutral" CNF for a given sample language corresponding
    to no events being true

    Convenience method. Works on the result of sample_language(X).
    Semantically equivalent to \\epsilon, but needed when forming DNFs
    """
    L = set()
    for labels in language:
        if labels == "":
            continue
        for label in labels:
            L.add("!" + str(label))
    return "&".join(L)

def prefixes(X):
    yield ((), ()) # (\epsilon, \epsilon) \in Pref(X)
    for (labels, rewards) in X:
        for i in range(1, len(labels)+1):
            yield (labels[0:i], rewards[0:i])

def all_pairs(xs):
    xs = list(xs)
    for i in range(0, len(xs)):
        for j in range(0, len(xs)):
            yield (xs[i], xs[j])

def different_pairs(xs):
    xs = list(xs)
    for i in range(0, len(xs)):
        for j in range(0, len(xs)):
            if i == j:
                continue
            yield (xs[i], xs[j])

def different_pairs_ordered(xs):
    xs = list(xs)
    for i in range(0, len(xs)):
        for j in range(i+1, len(xs)):
            yield (xs[i], xs[j])

def all_states(n_states):
    return range(INITIAL_STATE, n_states+1)

def all_states_terminal(n_states):
    return itertools.chain(all_states(n_states), [TERMINAL_STATE])

def add_pvar(storage, storage_rev, used_pvars, subscript):
    """
    Records a propositional variable indexed with the subscript by assigning it a unique
    index used by the solver. Returns this index

    If the variable indexed with that subscript was already recorded, no mutation is done,
    while the index is still returned.
    """
    key = subscript
    pvar = storage_rev.get(key)
    if pvar is not None:
        return pvar
    used_pvars[0] += 1
    storage[used_pvars[0]] = subscript
    storage_rev[key] = used_pvars[0]
    return used_pvars[0]

def rm_from_transitions(transitions, empty_transition):
    delta_u = defaultdict(dict)
    delta_r = defaultdict(dict)

    for (p, l) in transitions:
        (q, r) = transitions[(p, l)]
        conj = "&".join(l) or empty_transition
        if q not in delta_u[p]:
            delta_u[p][q] = conj
        else:
            delta_u[p][q] = delta_u[p][q] + "|" + conj
        if q not in delta_r[p]:
            delta_r[p][q] = [(conj, r)]
        else:
            delta_r[p][q].append((conj, r))
    
    rm_strings = [f"{INITIAL_STATE}", f"[]"]

    for p in delta_u:
        for q in delta_u[p]:
            rs = "{"
            for (label, reward) in delta_r[p][q]:
                rs += f"'{label}': {reward},"
            rs += "}"
            rs = f"LabelRewardFunction({rs})"
            s = "({},{},'{}',{})".format(p, q, delta_u[p][q], rs)
            rm_strings.append(s)

    rm_string = "\n".join(rm_strings)
    new_file, filename = tempfile.mkstemp()
    os.write(new_file, rm_string.encode())
    os.close(new_file)

    return RewardMachine(filename)

def initial_Q(H):
    """
    Returns a set of uninitialized q-functions indexed by states of RM H
    """
    Q = dict()
    Q[-1] = dict()
    for v in H.get_states():
        Q[v] = dict()
    return Q

def get_qmax(Q, s, actions, q_init):
    if s not in Q:
        Q[s] = dict([(a,q_init) for a in actions])
    return max(Q[s].values())

def get_best_action(Q, s, actions, q_init):
    qmax = get_qmax(Q,s,actions,q_init)
    best = [a for a in actions if Q[s][a] == qmax]
    return random.choice(best)

def display_transitions(transitions, name):
    dot = Digraph(comment=name, graph_attr={"fontsize":"6.0"}, edge_attr={"color": "#000000aa"})

    nodes = set()

    for (p, a) in transitions:
        if p not in nodes:
            nodes.add(p)
            dot.node(str(p))

    for (p, a) in transitions:
        [q, r] = transitions[(p, a)]
        r = 0.0 if r == 0.0 else r
        dot.edge(str(p), str(q), label=f"({''.join(a)}, {('%f' % r).rstrip('0').rstrip('.')})")
    dot = dot.unflatten()
    dot.render(f"graphviz/{name}.gv", view=True)
