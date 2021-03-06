from reward_machines.rm_environment import RewardMachineEnv
from gym.envs.atari.atari_env import AtariEnv
import gym
import random

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
    
    # and 'o' for life lost

def process_frame(frame, shape=(84, 84)):
    """Preprocesses a 210x160x3 frame to 84x84x1 grayscale
    Arguments:
        frame: The frame to process.  Must have values ranging from 0-255
    Returns:
        The processed frame
    """
    frame = frame.astype(np.uint8)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame[34:34 + 160, :160] 
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    frame = frame.reshape((*shape, 1))

    return frame

# Wrapper for Atari environments that adds detected objects and frame preprocessing
# currently assumes Montezuma's Revenge
class AtariDetectionEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.detected_objects = ""
        self.history_length = 4
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84,84,4),dtype=np.uint8)
    
    def get_events(self):
        return self.detected_objects
        

    def reset(self, evaluation=False):
        """Resets the environment
        Arguments:
            evaluation: Set to True when the agent is being evaluated. Takes a random number of no-op steps if True.
        """

        self.frame = self.env.reset()
        self.last_lives = 0

        # If evaluating, take a random number of no-op steps.
        # This adds an element of randomness, so that the each
        # evaluation is slightly different.
        if evaluation:
            for _ in range(random.randint(0, self.no_op_steps)):
                self.env.step(1)

        self.state = np.repeat(process_frame(self.frame), self.history_length, axis=2)

    # def step(self, action):
    #     next_obs, rew, env_done, info = self.env.step(action)
    #     self.detected_objects = detected_in_frame(next_obs)
    #     processed_frame = process_frame(next_obs)
    #     return processed_frame, rew, env_done, info

    def step(self, action, render_mode=None):
        """Performs an action and observes the result
        Arguments:
            action: An integer describe action the agent chose
            render_mode: None doesn't render anything, 'human' renders the screen in a new window, 'rgb_array' returns an np.array with rgb values
        Returns:
            processed_frame: The processed new frame as a result of that action
            reward: The reward for taking that action
            terminal: Whether the game has ended
            life_lost: Whether a life has been lost
            new_frame: The raw new frame as a result of that action
            If render_mode is set to 'rgb_array' this also returns the rendered rgb_array
        """
        new_frame, reward, terminal, info = self.env.step(action)
        # raw_frame = new_frame.copy()
        self.detected_objects = detected_in_frame(new_frame)

        if info['ale.lives'] < self.last_lives:
            life_lost = True
        else:
            life_lost = terminal
        self.last_lives = info['ale.lives']
        info['life_lost'] = life_lost

        if life_lost:
            self.detected_objects = f"{self.detected_objects}o"

        processed_frame = process_frame(new_frame)
        self.state = np.append(self.state[:, :, 1:], processed_frame, axis=2)

        if render_mode == 'rgb_array':
            return processed_frame, reward, terminal, life_lost, self.env.render(render_mode)
        elif render_mode == 'human':
            self.env.render()

        return processed_frame, reward, terminal, info
