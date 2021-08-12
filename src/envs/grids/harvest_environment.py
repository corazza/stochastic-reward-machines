import gym, random
from gym import spaces
import numpy as np
from reward_machines.rm_environment import RewardMachineEnv
from envs.grids.craft_world import CraftWorld
from envs.grids.office_world import OfficeWorld
from envs.grids.value_iteration import value_iteration
import numpy as np

ACTION_LABELS = {
    0: 'p', 1: 'w', 2: 'h', 3: 's'
}

class HarvestEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(4) # 0=plant, 1=water, 2=harvest, 3=sell
        self.observation_space = spaces.Box(low=0,high=2,shape=(1,),dtype=np.uint8) # bad, medium, good
        self.current_state = None

    def reset(self):
        self.current_state = 0
        self.last_action = None
        return self.current_state

    def new_harvest(self):
        a = random.random()
        if a < 0.1:
            return 0
        elif a < 0.9:
            return 1
        else:
            return 2

    def step(self, action):
        self.last_action = action
        if random.random() < 0.1:
            self.current_state = self.new_harvest()
        done = False
        info = {}
        return self.current_state, 0, done, info

    def get_events(self):
        action_label = ACTION_LABELS[self.last_action]
        return f'{action_label}{self.current_state}'

    def is_hidden_rm(self):
        return True
    
    def no_rm(self):
        return False

    def infer_termination_preference(self):
        return False

    # def show(self):
    #     self.env.show()

    # def get_model(self):
    #     return self.env.get_model()

class HarvestRMEnv1(RewardMachineEnv):
    def __init__(self):
        env = HarvestEnv()
        self.slip_prob = 0.00
        super().__init__(env, ['./envs/grids/reward_machines/harvest/t1.txt'])

class HarvestRMEnv2(RewardMachineEnv):
    def __init__(self):
        env = HarvestEnv()
        self.slip_prob = 0.00
        super().__init__(env, ['./envs/grids/reward_machines/harvest/t2.txt'])

class HarvestRMEnv3(RewardMachineEnv):
    def __init__(self):
        env = HarvestEnv()
        self.slip_prob = 0.00
        super().__init__(env, ['./envs/grids/reward_machines/harvest/t3.txt'])

class HarvestRMEnv4(RewardMachineEnv):
    def __init__(self):
        env = HarvestEnv()
        self.slip_prob = 0.00
        super().__init__(env, ['./envs/grids/reward_machines/harvest/t4.txt'])
