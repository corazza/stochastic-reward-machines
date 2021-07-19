import copy
from typing import IO
import os
import time
import json
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

def average_on_X(epsilon, H, All, X):
    outputs_dict = dict()
    for (labels, rewards) in All:
        if not run_eqv_noise(epsilon, rm_run(labels, H), rewards):
            continue
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
        rewards = sorted(rewards)
        average = (rewards[0] + rewards[-1])/2.0
        H2 = copy.deepcopy(H)
        H2.move_output(statelabel[0], statelabel[1], average)
        if consistent_on_all(epsilon, X, H2):
            H.move_output(statelabel[0], statelabel[1], average)
            print(f"average {statelabel[0]}-{statelabel[1]}: {average}")
        else:
            print(f"inconsistent average {statelabel[0]}-{statelabel[1]}: {average}")


def extract_noise_params(env):
    noise_epsilon = None
    noise_delta = None

    try:
        noise_epsilon = env.current_rm.epsilon_cont
        assert noise_epsilon is not None
    except:
        noise_epsilon = env.noise_epsilon

    try:
        noise_delta = env.current_rm.noise_delta
        assert noise_delta is not None
    except:
        noise_delta = env.noise_delta
    
    return noise_epsilon, noise_delta

def detect_signal(a):
    if os.path.isfile(f"signals/{a}.txt"):
        print(f"detected signal signals/{a}.txt")
        return True
    return False

class EvalResults:
    def __init__(self, description):
        self.description = description
        self.step_rewards = list()
        self.step_rebuilding = list()

    def register_mean_reward(self, step, reward):
        self.step_rewards.append((time.time(), step, reward))
    
    def register_rebuilding(self, step, rm):
        self.step_rebuilding.append((time.time(), step, rm))
    
    def save(self, filename):
        data = json.dumps(self.__dict__)

        with open(filename, 'w') as f:
            f.write(data)
