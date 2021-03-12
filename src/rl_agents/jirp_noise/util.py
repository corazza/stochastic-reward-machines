import copy
from typing import IO

import IPython

from rl_agents.jirp.util import run_eqv, rm_run


def run_eqv_noise(epsilon, output1, output2):
    """
    Returns True if all outputs are within epsilon of each other (output1 is a noise-distorted output2, eg.)
    """
    if len(output1) != len(output2):
        return False
    for i in range(0, len(output1)):
        if abs(output1[i] - output2[i]) > epsilon:
            return False
    return True

def consistent_on_all(epsilon, X, H):
    for (labelsx, rewardsx) in X:
        if not run_eqv_noise(epsilon, rm_run(labelsx, H), rewardsx):
            return False
    return True

def make_consistent(epsilon, labels, rewards, X, H):
    if not consistent_on_all(epsilon, X, H):
        print("not consistent!")
        IPython.embed()

    # assume H is consistent with X but not with (labels, rewards)
    H2 = copy.deepcopy(H)
    current_state = H2.reset()
    for i in range(0, len(labels)):
        if current_state == H2.terminal_u:
            return None
        props = labels[i]
        next_state, reward, done = H2.step(current_state, props, {"true_props": props})
        if abs(reward - rewards[i]) > epsilon:
            if reward < rewards[i]:
                new_mean = rewards[i] - epsilon
            else:
                new_mean = rewards[i] + epsilon
            H2.move_output(current_state, props, new_mean)
        current_state = next_state
    if not run_eqv_noise(epsilon, rm_run(labels, H2), rewards):
        return None
    if not consistent_on_all(epsilon, X, H2):
        return None
    return H2

def average_on_X(epsilon, H, X):
    outputs_dict = dict()
    for (labels, rewards) in X:
        current_state = H.reset()
        for i in range(0, len(labels)):
            props = labels[i]
            next_state, reward, done = H.step(current_state, props, {"true_props": props})
            if (current_state, props) not in outputs_dict:
                outputs_dict[(current_state, props)] = list()
            outputs_dict[(current_state, props)].append(rewards[i])
            current_state = next_state
    for statelabel in outputs_dict:
        rewards = outputs_dict[statelabel]
        average = sum(rewards)/len(rewards)
        H2 = copy.deepcopy(H)
        H2.move_output(statelabel[0], statelabel[1], average)
        if consistent_on_all(epsilon-0.0001, X, H2):
            H.move_output(statelabel[0], statelabel[1], average)
            print(f"average {statelabel[0]}-{statelabel[1]}: {average}")

def lower(H, language):
    for u in H.get_states():
        for l in language:
            _, rew, _ = H.step(u, l, {})
            if rew > 20:
                H.move_output(u, l, 0)
                