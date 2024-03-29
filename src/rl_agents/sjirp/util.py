import copy
from typing import IO
import os
import time
import json
import IPython
from rl_agents.jirp.consts import EXACT_EPSILON

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

def run_eqv_noise_diff(epsilon, output1, output2):
    """
    Returns True if all outputs are within epsilon of each other (output1 is a noise-distorted output2, eg.)
    """
    if len(output1) != len(output2):
        return None
    for i in range(0, len(output1)):
        if abs(output1[i] - output2[i]) > epsilon:
            return output1[i] - output2[i]
    return None

def run_eqv_noise_i(epsilon, output1, output2):
    """
    Returns True if all outputs are within epsilon of each other (output1 is a noise-distorted output2, eg.)
    """
    if len(output1) != len(output2):
        return None
    for i in range(0, len(output1)):
        if abs(output1[i] - output2[i]) > epsilon:
            return i
    return None

def consistent_on_all_i(epsilon, X, H):
    for i in range(0, len(X)):
        (labelsx, rewardsx) = X[i]
        if not run_eqv_noise(epsilon, rm_run(labelsx, H), rewardsx):
            return i
    return None


def make_consistent(checking_epsilon, inference_epsilon, labels, rewards, X, H):
    # if not consistent_on_all(epsilon + 5*EXACT_EPSILON, X, H):
    if not consistent_on_all(checking_epsilon, X, H):
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
        if abs(reward - rewards[i]) > inference_epsilon:
            if reward < rewards[i]:
                new_mean = rewards[i] - inference_epsilon
            else:
                new_mean = rewards[i] + inference_epsilon
            H2.move_output(current_state, props, new_mean)
        current_state = next_state
    if not run_eqv_noise(checking_epsilon, rm_run(labels, H2), rewards):
        return None
    # if not consistent_on_all(epsilon + 5*EXACT_EPSILON, X, H2):
    if not consistent_on_all(checking_epsilon, X, H2):
        return None
    return H2

def average_on_X(epsilon, H, All, X, report=True):
    outputs_dict = dict()
    skipped = 0
    diffs = list()
    for (labels, rewards) in All:
        if not run_eqv_noise(epsilon, rm_run(labels, H), rewards):
            skipped += 1
            # diffs.append(run_eqv_noise_diff(epsilon, rm_run(labels, H), rewards))
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
        midrange = (rewards[0] + rewards[-1])/2.0
        average = sum(rewards) / len(rewards)
        H.move_output(statelabel[0], statelabel[1], midrange)

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
        self.step_corrupted = list()
        self.step_corrupted_end = list()

    def from_object(obj):
        new_results = EvalResults(obj.description)
        new_results.step_rewards = obj.step_rewards
        new_results.step_rebuilding = obj.step_rebuilding
        new_results.step_corrupted = obj.step_corrupted
        new_results.step_corrupted_end = obj.step_corrupted_end
        return new_results

    def filter_horizon(self, horizon):
        new_results = EvalResults(copy.deepcopy(self.description))
        new_results.step_rewards = list(filter(lambda x: x[1] <= horizon, self.step_rewards))
        new_results.step_rebuilding = list(filter(lambda x: x[1] <= horizon, self.step_rebuilding))
        new_results.step_corrupted = list(filter(lambda x: x <= horizon, self.step_corrupted))
        new_results.step_corrupted_end = list(filter(lambda x: x <= horizon, self.step_corrupted_end))            
        return new_results

    def register_mean_reward(self, step, reward):
        self.step_rewards.append((time.time(), step, reward))
    
    def register_rebuilding(self, step, rm):
        self.step_rebuilding.append((time.time(), step, {}))

    def register_corruption(self, step):
        self.step_corrupted.append(step)
    
    def register_corruption_end(self, step):
        self.step_corrupted_end.append(step)
    
    def save(self, filename):
        data = json.dumps(self.__dict__)

        with open(filename, 'w') as f:
            f.write(data)

def start_stepping(H, env, Q, actions, q_init):
    rm_state = H.reset()
    s = tuple(env.reset())

    while True:
        env.show()
        a = get_best_action(Q[rm_state],s,actions,q_init)
        sn, r, done, info = env.step(a)
        sn = tuple(sn)
        true_props = env.get_events()
        next_rm_state, _rm_reward, rm_done = H.step(rm_state, true_props, info)
        rm_state = next_rm_state
        s = sn
        IPython.embed()

def clean_trace(labels, rewards):
    no_more_than = 10
    labels_new = list()
    rewards_new = list()
    last = None
    counter = 0
    for i in range(0, len(labels)):
        if labels[i] == last and counter > no_more_than:
            continue
        elif labels[i] == last:
            counter += 1
        else:
            counter = 0
        labels_new.append(labels[i])
        rewards_new.append(rewards[i])
        last = labels[i]
    return ((tuple(labels_new), tuple(rewards_new)))
