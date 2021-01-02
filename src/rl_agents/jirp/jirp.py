"""
JIRP based method
"""
import os, tempfile
from collections import defaultdict
import random, time, copy
# from profilehooks import profile

from baselines import logger
from reward_machines.reward_machine import RewardMachine
from reward_machines.rm_environment import RewardMachineEnv, RewardMachineHidden

_MAX_RM_STATES_N = 50
_UPDATE_X_EVERY_N_EPISODES = 150
_EQV_THRESHOLD = 0.95 # probability threshold for concluding two states in two different RMs are equivalent


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
    Returns the "neutral" CNF for a given sample language

    Convenience method. Needed when forming DNFs. Works on the result of sample_language(X).
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

# @profile
def consistent_hyp(X, n_states_start=2):
    language = sample_language(X)
    empty_transition = dnf_for_empty(language)
    reward_alphabet = sample_reward_alphabet(X)

    from pysat.solvers import Glucose4
    for n_states in range(n_states_start, _MAX_RM_STATES_N+1):
        print(f"finding model with {n_states} states")
        def initial_state():
            return 1
        def all_states():
            return range(initial_state(), n_states+1)

        prop_d = dict() # maps SAT's propvar (int) to (p: state, l: labels, q: state)
        prop_d_rev = dict()
        prop_o = dict() # maps SAT's propvar (int) to (p: state, l: labels, r: reward)
        prop_o_rev = dict()
        prop_x = dict() # maps SAT's propvar (int) to (l: labels, q: state)
        prop_x_rev = dict()
        used_pvars = 0 # p. var. counter
        g = Glucose4() # solver

        def add_pvar(d, storage, storage_rev):
            nonlocal used_pvars
            key = d
            pvar = storage_rev.get(key)
            if pvar is not None:
                return pvar
            used_pvars += 1
            storage[used_pvars] = d
            storage_rev[key] = used_pvars
            return used_pvars

        def add_pvar_d(d):
            nonlocal prop_d
            nonlocal prop_d_rev
            return add_pvar(d, prop_d, prop_d_rev)

        def add_pvar_o(o):
            nonlocal prop_o
            nonlocal prop_o_rev
            return add_pvar(o, prop_o, prop_o_rev)

        def add_pvar_x(x):
            nonlocal prop_x
            nonlocal prop_x_rev
            return add_pvar(x, prop_x, prop_x_rev)

        # Encoding reward machines
        # (1)
        for p in all_states():
            for l in language:
                g.add_clause([add_pvar_d((p, l, q)) for q in all_states()])
                for q1 in all_states():
                    for q2 in all_states():
                        if q1==q2:
                            continue
                        p_l_q1 = add_pvar_d((p, l, q1))
                        p_l_q2 = add_pvar_d((p, l, q2))
                        g.add_clause([-p_l_q1, -p_l_q2])

        # (2)
        for p in all_states():
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
        g.add_clause([add_pvar_x((tuple(), initial_state()))]) # starts in the initial state
        for p in all_states():
            if p == initial_state():
                continue
            g.add_clause([-add_pvar_x((tuple(), p))])

        # (4)
        for (labels, _rewards) in prefixes(X):
            if labels == ():
                continue
            lm = labels[0:-1]
            l = labels[-1]
            for p in all_states():
                for q in all_states():
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
            for p in all_states():
                x = add_pvar_x((lm, p))
                o = add_pvar_o((p, l, r))
                g.add_clause([-x, o])

        g.solve()
        if g.get_model() is None:
            continue
        
        print("found")

        transitions = defaultdict(lambda: [None, None]) # maps (state, true_props) to (state, reward)
        for pvar in g.get_model():
            if abs(pvar) in prop_d:
                if pvar > 0:
                    (p, l, q) = prop_d[abs(pvar)]
                    assert transitions[(p, tuple(l))][0] is None
                    transitions[(p, tuple(l))][0] = q
                    assert q is not None
            elif abs(pvar) in prop_o:
                if pvar > 0:
                    (p, l, r) = prop_o[abs(pvar)]
                    assert transitions[(p, tuple(l))][1] is None
                    transitions[(p, tuple(l))][1] = r
            elif abs(pvar) in prop_x:
                pass
            else:
                raise ValueError("Uknown p-var dict")

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
        
        rm_strings = [f"{initial_state()}", f"[]"]
    
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

        return RewardMachine(filename), n_states

    raise ValueError(f"Couldn't find machine with at most {_MAX_RM_STATES_N} states")

def equivalent_on_X(H1, v1, H2, v2, X):
    """
    Checks if state v1 from RM H1 is equivalent to v2 from H2
    """
    H1 = H1.with_initial(v1)
    H2 = H2.with_initial(v2)
    total = len(X)
    eqv = 0
    for (labels, _rewards) in X:
        if rm_run(labels, H1) == rm_run(labels, H2):
            eqv += 1
    if float(eqv)/total > _EQV_THRESHOLD:
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
    # JIRP doesn't work with explicit RMs
    assert env.is_hidden_rm()

    # Running Q-Learning
    reward_total = 0
    total_episodes = 0
    step = 0
    num_episodes = 0
    actions = list(range(env.action_space.n))

    H, n_states_last = consistent_hyp(set())

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
                logger.record_tabular("total reward", reward_total)
                logger.record_tabular("positive / total", str(int(reward_total)) + "/" + str(total_episodes) + f" ({int(100*(reward_total/total_episodes))}%)")
                logger.dump_tabular()
                reward_total = 0
                total_episodes = 0
            if done:
                num_episodes += 1
                total_episodes += 1
                if rm_run(labels, H) != rewards:
                    X_new.add((tuple(labels), tuple(rewards)))
                if num_episodes % _UPDATE_X_EVERY_N_EPISODES == 0 and X_new:
                    print(f"len(X)={len(X)}")
                    print(f"len(X_new)={len(X_new)}")
                    X.update(X_new)
                    X_new = set()
                    H_new, n_states_last = consistent_hyp(X, n_states_last)
                    Q = transfer_Q(H_new, H, Q, X)
                    H = H_new
                break
            s = sn
