"""
JIRP based method
"""
import os, tempfile
from collections import defaultdict

import random, time
from baselines import logger
from reward_machines.reward_machine import RewardMachine
from reward_machines.rm_environment import RewardMachineEnv, RewardMachineHidden

_MAX_N_STATES = 50 # TODO FIXME guarantee?
_UPDATE_X_EVERY = 500 # episodes
_USE_TERMINAL_STATE = False


def get_qmax(Q,s,actions,q_init):
    if s not in Q:
        Q[s] = dict([(a,q_init) for a in actions])
    return max(Q[s].values())

def get_best_action(Q,s,actions,q_init):
    qmax = get_qmax(Q,s,actions,q_init)
    best = [a for a in actions if Q[s][a] == qmax]
    return random.choice(best)

def rm_run(labels, info, rm):
    current_state = rm.reset()
    rewards = []
    for props in labels:
        current_state, reward, done = rm.step(current_state, props, info)
        rewards.append(reward)
        if done: 
            break
    return rewards

def is_counterexample(rm_rewards, env_rewards):
    # d = len(env_rewards) - len(rm_rewards)
    # if d == 0:
    #     return rm_rewards != env_rewards
    # rm_rewards.extend([rm_rewards[-1]]*d)
    return rm_rewards != env_rewards
    # return True

def dnf_for_empty(language):
    L = set()
    for labels in language:
        if labels == "":
            continue
        for label in labels:
            L.add("!" + str(label))
    return "&".join(L)

def sample_language(X):
    language = set()
    language.add("")
    for (labels, _rewards) in X:
        language.update(labels)
    return language

def sample_reward_alphabet(X):
    alphabet = set()
    alphabet.add(0.0)
    for (_labels, rewards) in X:
        alphabet.update(rewards)
    return alphabet

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

def consistent_hyp(X, n_states_start=2):
    language = sample_language(X)
    empty_transition = dnf_for_empty(language)
    reward_alphabet = sample_reward_alphabet(X)

    from pysat.solvers import Glucose3
    for n_states in range(n_states_start, _MAX_N_STATES+1):
        def terminal_state():
            return 1 if _USE_TERMINAL_STATE else None
        def initial_state():
            return 1 if not _USE_TERMINAL_STATE else 2
        def all_states():
            return range(initial_state(), n_states+1)

        # maps SAT's propvar (int) to (p: state, l: labels, q: state)
        prop_d = dict()
        prop_d_rev = dict()
        # maps SAT's propvar (int) to (p: state, l: labels, r: reward)
        prop_o = dict()
        prop_o_rev = dict()
        # maps SAT's propvar (int) to (l: labels, q: state)
        prop_x = dict()
        prop_x_rev = dict()

        g = Glucose3()
        used_pvars = 0
        def add_pvar(d, storage, storage_rev):
            nonlocal used_pvars
            if str(d) in storage_rev:
                return storage_rev[str(d)]
            used_pvars += 1
            storage[used_pvars] = d
            storage_rev[str(d)] = used_pvars
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

        # Phi^{RM}
        for p in all_states():
            for l in language:
                g.add_clause([add_pvar((p, l, q), prop_d, prop_d_rev) for q in all_states()])
                for q1 in all_states():
                    for q2 in all_states():
                        if q1==q2:
                            continue
                        p_l_q1 = add_pvar((p, l, q1), prop_d, prop_d_rev)
                        p_l_q2 = add_pvar((p, l, q2), prop_d, prop_d_rev)
                        g.add_clause([-p_l_q1, -p_l_q2])

        for p in all_states():
            for l in language:
                g.add_clause([add_pvar((p, l, r), prop_o, prop_o_rev) for r in reward_alphabet])
                for r1 in reward_alphabet:
                    for r2 in reward_alphabet:
                        if r1 == r2:
                            continue
                        p_l_r1 = add_pvar((p, l, r1), prop_o, prop_o_rev)
                        p_l_r2 = add_pvar((p, l, r2), prop_o, prop_o_rev)
                        g.add_clause([-p_l_r1, -p_l_r2])

        # Consistency with sample
        # (3)
        g.add_clause([add_pvar((tuple(), initial_state()), prop_x, prop_x_rev)]) # starts in the initial state
        for p in all_states():
            if p == initial_state():
                continue
            g.add_clause([-add_pvar((tuple(), p), prop_x, prop_x_rev)])

        # (4)
        for (labels, _rewards) in prefixes(X):
            if labels == ():
                continue
            # print(f"prefix for {labels}")
            lm = labels[0:-1]
            l = labels[-1]
            for p in all_states():
                for q in all_states():
                    x_1 = add_pvar((lm, p), prop_x, prop_x_rev)
                    d = add_pvar((p, l, q), prop_d, prop_d_rev)
                    x_2 = add_pvar((labels, q), prop_x, prop_x_rev)
                    g.add_clause([-x_1, -d, x_2])

        # (5)
        for (labels, rewards) in prefixes(X):
            if labels == ():
                continue
            lm = labels[0:-1]
            l = labels[-1]
            r = rewards[-1]
            for p in all_states():
                x = add_pvar((lm, p), prop_x, prop_x_rev)
                o = add_pvar((p, l, r), prop_o, prop_o_rev)
                g.add_clause([-x, o])

        # MEALY vs. MOORE in Icarte's overview paper
        # fix by updating JIRP's definition of a reward machine
        # ie. d and o -> whatever Icarte uses
        for (p, q) in all_pairs(all_states()):
            for (l1, l2) in different_pairs(language):
                for (r1, r2) in different_pairs(reward_alphabet):
                    d1 = add_pvar_d((p, l1, q))
                    d2 = add_pvar_d((p, l2, q))
                    o1 = add_pvar_o((p, l1, r1))
                    o2 = add_pvar_o((p, l2, r2))
                    g.add_clause([-d1, -d2, -o1, -o2])

        g.solve()
        if g.get_model() is None:
            print("no model with {} states".format(n_states))
            continue

        transitions = defaultdict(lambda: [None, None])
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
                delta_r[p][q] = r
            else:
                if not delta_r[p][q] == r:
                    for transition in transitions:
                        print(transition, transitions[transition])
                    print(f"offending: {p} -> {q} (has {delta_r[p][q]} from {delta_u[p][q]}, new {r} from {l}")
                assert delta_r[p][q] == r
        
        if _USE_TERMINAL_STATE:
            rm_strings = [f"{initial_state()}", f"[{terminal_state()}]"]
        else:
            rm_strings = [f"{initial_state()}", f"[]"]
    
        for p in delta_u:
            for q in delta_u[p]:
                rs = "ConstantRewardFunction({})".format(delta_r[p][q])
                s = "({},{},'{}',{})".format(p, q, delta_u[p][q], rs)
                rm_strings.append(s)

        rm_string = "\n".join(rm_strings)

        print(delta_u)

        print(f"begin rm string, n_states={n_states}")
        print(rm_string)
        print("end rm string")

        new_file, filename = tempfile.mkstemp()
        os.write(new_file, rm_string.encode())
        os.close(new_file)

        return RewardMachine(filename), n_states

    raise ValueError("Couldn't find machine with at most {} states".format(_MAX_N_STATES))

def initial_hyp():
    return consistent_hyp(set())

def transfer_Q(H, prev_Q):
    Q = dict()
    Q[-1] = dict()
    for v in H.get_states():
        Q[v] = dict()
    return Q

def initial_Q(H):
    return transfer_Q(H, dict())

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

    H, n_states_last = initial_hyp()

    Q = initial_Q(H)
    X = set()
    X_new = set()
    labels = []
    rewards = []
    percentages = []

    while step < total_timesteps:
        s = tuple(env.reset())
        true_props = env.get_events()
        rm_state = H.reset()
        # print("reset")
        # print(labels)
        # print(rewards)
        labels = []
        rewards = []

        if s not in Q[rm_state]: Q[rm_state][s] = dict([(a, q_init) for a in actions])

        while True:
            # Selecting and executing the action
            a = random.choice(actions) if random.random() < epsilon else get_best_action(Q[rm_state],s,actions,q_init)
            sn, r, done, info = env.step(a)

            # if r > 0:
            #     print("GOT POSITIVE")

            sn = tuple(sn)
            true_props = env.get_events()
            labels.append(true_props) # L(s, a, s')
            next_rm_state, _rm_reward, rm_done = H.step(rm_state, true_props, info)

            # if true_props != '':
            #     print(f"JIRP: {rm_state} -> {next_rm_state} on {true_props}")
            #     print()

            # if not rm_done: # learn while possible
            # update Q-function of current RM state
            if s not in Q[rm_state]: Q[rm_state][s] = dict([(b, q_init) for b in actions])
            if done: _delta = r - Q[rm_state][s][a]
            else:    _delta = r + gamma*get_qmax(Q[next_rm_state], sn, actions, q_init) - Q[rm_state][s][a]
            Q[rm_state][s][a] += lr*_delta
            # if r > 0:
            #     print(f"Q[{rm_state}][{s}][{a}] += {lr*_delta} ({r})")

            # counterfactual updates
            for v in H.get_states():
                if v == rm_state:
                    continue
                v_next, h_r, h_done = H.step(v, true_props, info)
                if s not in Q[v]: Q[v][s] = dict([(b, q_init) for b in actions])
                if done: _delta = h_r - Q[v][s][a]
                else:    _delta = h_r + gamma*get_qmax(Q[v_next], sn, actions, q_init) - Q[v][s][a]
                Q[v][s][a] += lr*_delta
                # if h_r > 0:
                #     print(f"cf: Q[{v}][{s}][{a}] += {lr*_delta} ({h_r})")

            # if r > 0:
            #     print()
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
                print(f"total_episodes={total_episodes}")
                percentages.append(int(100*(reward_total/total_episodes)))
                reward_total = 0
                total_episodes = 0
            if done:
                num_episodes += 1
                total_episodes += 1
                if is_counterexample(rm_run(labels, info, H), rewards): # TODO test if this actually works (in terminal)
                    X_new.add((tuple(labels), tuple(rewards)))
                if num_episodes % _UPDATE_X_EVERY == 0 and X_new:
                    print(f"len(X)={len(X)}")
                    print(f"len(X_new)={len(X_new)}")
                    X.update(X_new)
                    X_new = set()
                    H, n_states_last = consistent_hyp(X, n_states_last)
                    Q = transfer_Q(H, Q)
                    print(H.delta_u)
                    print(H.delta_r)
                break
            s = sn
            # true_props = true_props_next

    print(float(sum(percentages[5:]))/len(percentages[5:]))

    with open("../jirp_data/X.txt", 'w') as x_file:
        for (ls, rs) in X:
            x_file.write(str(ls) + '\n')
            x_file.write(str(rs) + "\n\n")
