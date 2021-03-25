import gym, random
from gym import spaces
import numpy as np
from reward_machines.rm_environment import RewardMachineEnv
from envs.grids.craft_world import CraftWorld
from envs.grids.office_world import OfficeWorld
from envs.grids.value_iteration import value_iteration
import random

STATE_NAMES = {
    0: 'initial',
    1: 'planted',
    2: 'watered',
    3: 'harvest_good',
    4: 'harvest_medium',
    5: 'harvest_bad'
}
STATE_NAMES_INV = dict(map(lambda x: (x[1], x[0]), STATE_NAMES.items()))

ACTION_NAMES = {
    0: 'plant',
    1: 'water',
    2: 'harvest',
    3: 'sell'
}

HARVEST_MEANS = {
    'harvest_good': 3,
    'harvest_medium': 2,
    'harvest_bad': 1
}
PUNISH_MEAN = -0.5

HARVEST_EPSILON = 0.2
PUNISH_EPSILON = 0.1

class HarvestEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(4) # 0=plant, 1=water, 2=harvest, 3=sell
        self.observation_space = spaces.Discrete(4) # 0=initial, 1,2,3 = harvests

        self.current_state = None
        self.labels = None
        self.noise_epsilon = max([HARVEST_EPSILON, PUNISH_EPSILON])
        self.noise_delta = 1.75

    def observation_from_state(self, state):
        if state >= 3:
            return (state - 2,)
        else:
            return (0,)

    def reset(self):
        self.current_state = 0
        self.labels = ''
        return self.observation_from_state(self.current_state)

    def step(self, action):
        current_state = STATE_NAMES[self.current_state]
        action = ACTION_NAMES[action]
        next_state = None
        reward = 0
        labels = ''

        if current_state == 'initial' and action == 'plant':
            next_state = 'planted'
        elif current_state == 'initial':
            next_state = 'initial'
            reward = PUNISH_MEAN + random.uniform(-PUNISH_EPSILON, PUNISH_EPSILON)
            # reward = -0.75
        if current_state == 'planted' and action == 'water':
            next_state = 'watered'
        elif current_state == 'planted':
            next_state = 'initial'
            reward = PUNISH_MEAN + random.uniform(-PUNISH_EPSILON, PUNISH_EPSILON)
            # reward = -0.75
        if current_state == 'watered' and action == 'harvest':
            a = random.random()
            if a < 0.1:
                next_state = 'harvest_good'
            elif a < 0.9:
                next_state = 'harvest_medium'
            else:
                next_state = 'harvest_bad'
        elif current_state == 'watered':
            next_state = 'initial'
            reward = PUNISH_MEAN + random.uniform(-PUNISH_EPSILON, PUNISH_EPSILON)
            # reward = -0.75
        if current_state in ['harvest_good', 'harvest_medium', 'harvest_bad'] and action == 'sell':
            next_state = 'initial'
            reward = HARVEST_MEANS[current_state] + random.uniform(-self.noise_epsilon, self.noise_epsilon)
        elif current_state in ['harvest_good', 'harvest_medium', 'harvest_bad']:
            next_state = 'initial'
            reward = PUNISH_MEAN + random.uniform(-PUNISH_EPSILON, PUNISH_EPSILON)
            # reward = -0.75

        if action == 'plant':
            labels = 'p'
        elif action == 'water':
            labels = 'w'
        elif action == 'harvest':
            if next_state == 'harvest_good':
                labels = 'g'
            elif next_state == 'harvest_medium':
                labels = 'm'
            elif next_state == 'harvest_bad':
                labels = 'b'
            else:
                labels = ''
        else:
            labels = 's'

        self.current_state = STATE_NAMES_INV[next_state]
        self.labels = labels
        observation = self.observation_from_state(self.current_state)
        done = False
        info = {}
        return observation, reward, done, info

    def get_events(self):
        return self.labels

    def is_hidden_rm(self):
        return False
    
    def no_rm(self):
        return True

    def infer_termination_preference(self):
        return False

    # def show(self):
    #     self.env.show()

    # def get_model(self):
    #     return self.env.get_model()
