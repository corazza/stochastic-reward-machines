from reward_machines.rm_environment import RewardMachineEnv
from gym.envs.atari.atari_env import AtariEnv
import gym

import numpy as np
import cv2

# see https://github.com/grockious/deepsynth/blob/main/training.py
def check(a, b, upper_left):
    ul_row = upper_left[0]
    ul_col = upper_left[1]
    b_rows, b_cols = b.shape
    a_slice = a[ul_row: ul_row + b_rows, :][:, ul_col: ul_col + b_cols]
    if a_slice.shape != b.shape:
        return False
    return (a_slice == b).all()

def subarray_detector(big_array, small_array):
    upper_left = np.argwhere(big_array == small_array[0, 0])
    for ul in upper_left:
        if check(big_array, small_array, ul):
            return True
    else:
        return False

def detected_in_frame(frame):
    agent_unique = [[478, 478, 478], [478, 478, 478], [344, 344, 344], [478, 478, 478]]
    frame = np.sum(frame, axis=2)

    if subarray_detector(frame[93:134, 76:83], np.array(agent_unique)):
        return 'l' # 'ladder_1'
    elif subarray_detector(frame[96:134, 110:115], np.array(agent_unique)):
        return 'r' # 'rope'
    elif subarray_detector(frame[136:179, 132:139], np.array(agent_unique)):
        return 'm' # 'ladder_2'
    elif subarray_detector(frame[136:179, 20:27], np.array(agent_unique)):
        return 'n' # 'ladder_3'
    elif subarray_detector(frame[99:106, 13:19], np.array(agent_unique)):
        return 'k' # 'key'
    elif subarray_detector(frame[50:92, 20:24], np.array(agent_unique)):
        return 'd' # 'door'
    elif subarray_detector(frame[50:92, 136:140], np.array(agent_unique)):
        return 'd' # 'door'
    else:
        return ''

# Wrapper for Atari environments that adds detected objects
# currently assumes Montezuma's Revenge
class AtariDetectionEnv(gym.Wrapper):
    def __init__(self, env):
        self.detected_objects = ""
        super().__init__(env)
    
    def step(self, action):
        next_obs, rew, env_done, info = self.env.step(action)
        self.detected_objects = detected_in_frame(next_obs)
        return next_obs, rew, env_done, info

    def get_events(self):
        return self.detected_objects
