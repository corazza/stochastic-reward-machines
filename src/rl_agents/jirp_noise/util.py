import copy

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

def make_consistent(epsilon, labels, rewards, X, H):
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
    if not run_eqv_noise(epsilon+0.00001, rm_run(labels, H2), rewards):
        return None
    for (labelsx, rewardsx) in X:
        if not run_eqv_noise(epsilon+0.00001, rm_run(labelsx, H2), rewardsx):
            return None
    return H2
