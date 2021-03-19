import IPython
import gym, random
from gym import spaces
import numpy as np
from reward_machines.rm_environment import RewardMachineEnv
from envs.grids.craft_world import CraftWorld
from envs.grids.office_world import OfficeWorld
from envs.grids.value_iteration import value_iteration
from envs.grids.game_objects import *
import random, math, os
import numpy as np
from rl_agents.jirp_noise.consts import NOISE_DELTA, NOISE_EPSILON


class GardenEnv(gym.Env):
    def __init__(self):
        self._load_map("./envs/grids/maps/map_12.txt")
        self.env_game_over = False
        self.state = 0 # "RM" implementation
        self.harvest = 0 # not hidden so as to not poison other states when doing QRM
        self.noise_epsilon = NOISE_EPSILON
        self.noise_delta = NOISE_DELTA
        self.slip_prob = None
        N,M      = self.map_height, self.map_width
        self.action_space = spaces.Discrete(4) # up, right, down, left
        self.observation_space = spaces.Box(low=0, high=max([N,M]), shape=(3,), dtype=np.uint8)
        print("garden noise epsilon:", self.noise_epsilon)

    def reset(self):
        self.state = 0
        self.harvest = 0
        self.agent.reset()
        return self.get_features()

    def get_events(self, internal=False):
        true_props = self.get_true_propositions()
        return true_props

    def step(self, action):
        self.execute_action(action)
        ni = self.agent.i
        nj = self.agent.j
        label = self.get_true_propositions()
        reward = 0
        done = False

        if label == 'w' and self.state == 0:
            self.state = 1
            reward = 0.1
        elif label == 'g' and self.state == 1:
            self.state = 2
            reward = 0.1
        if label == 'h' and self.state == 2:
            self.state = 3
            reward = 0.1
        elif label == 'g' and self.state == 3:
            self.state = 4
            reward = 0.1
            if random.random() < 0.01:
                self.harvest = 2
            else:
                self.harvest = 1
        elif label == 'm' and self.state == 4:
            if self.harvest == 1:
                reward = 3
            else:
                reward = 3 + self.noise_delta
            reward += random.uniform(-self.noise_epsilon, self.noise_epsilon)
            done = True
        elif label == 'e':
            reward = 0
            done = True

        obs = self.get_features()
        info = {"true_props": self.get_events(internal=True)}
        return obs, reward, done, info

    def execute_action(self, a):
        """
        We execute 'action' in the game
        """
        a = self._perturb_action(a)
        agent = self.agent
        ni,nj = agent.i, agent.j

        # Getting new position after executing action
        ni,nj = self._get_next_position(ni,nj,a)
        
        # Interacting with the objects that is in the next position (this doesn't include monsters)
        action_succeeded = self.map_array[ni][nj].interact(agent)

        # So far, an action can only fail if the new position is a wall
        if action_succeeded:
            agent.change_position(ni,nj)

    def _perturb_action(self, a):
        if random.random() < self.slip_prob:
            return random.randint(0, 3)
        return a

    def _get_next_position(self, ni, nj, a):
        """
        Returns the position where the agent would be if we execute action
        """
        action = Actions(a)
        
        # OBS: Invalid actions behave as NO-OP
        if action == Actions.up   : ni-=1
        if action == Actions.down : ni+=1
        if action == Actions.left : nj-=1
        if action == Actions.right: nj+=1
        
        return ni,nj


    def get_true_propositions(self):
        """
        Returns the string with the propositions that are True in this state
        """
        ret = str(self.map_array[self.agent.i][self.agent.j]).strip()
        if ret == 'z':
            ret = str(self.harvest)
        return ret

    def get_features(self):
        """
        Returns the features of the current state (i.e., the location of the agent)
        """
        return np.array([self.agent.i,self.agent.j, self.harvest])

    def show(self):    
        """
        Prints the current map
        """
        r = ""
        for i in range(self.map_height):
            s = ""
            for j in range(self.map_width):
                if self.agent.idem_position(i,j):
                    s += str(self.agent)
                else:
                    s += str(self.map_array[i][j])
            if(i > 0):
                r += "\n"
            r += s
        print(r)


    def get_model(self):
        """
        This method returns a model of the environment. 
        We use the model to compute optimal policies using value iteration.
        The optimal policies are used to set the average reward per step of each task to 1.
        """
        S = [(x,y) for x in range(1,40) for y in range(1,40)] # States
        A = self.actions.copy()                               # Actions
        L = dict([((x,y),str(self.map_array[x][y]).strip()) for x,y in S])  # Labeling function
        T = {}                  # Transitions (s,a) -> s' (they are deterministic)
        for s in S:
            x,y = s
            for a in A:
                x2,y2 = self._get_next_position(x,y,a)
                T[(s,a)] = s if str(self.map_array[x2][y2]) == "X" else (x2,y2)
        return S,A,L,T # SALT xD


    def _load_map(self,file_map):
        """
        This method adds the following attributes to the game:
            - self.map_array: array containing all the static objects in the map (no monsters and no agent)
                - e.g. self.map_array[i][j]: contains the object located on row 'i' and column 'j'
            - self.agent: is the agent!
            - self.map_height: number of rows in every room 
            - self.map_width: number of columns in every room
        The inputs:
            - file_map: path to the map file
        """
        # contains all the actions that the agent can perform
        self.actions = [Actions.up.value, Actions.right.value, Actions.down.value, Actions.left.value]
        # loading the map
        self.map_array = []
        self.class_ids = {} # I use the lower case letters to define the features
        f = open(file_map)
        i,j = 0,0
        for l in f:
            # I don't consider empty lines!
            if(len(l.rstrip()) == 0): continue
            
            # this is not an empty line!
            row = []
            j = 0
            for e in l.rstrip():
                if e in "abcdefghijklmnopqrstuvwxyzH":
                    entity = Empty(i,j,label=e)
                    if e not in self.class_ids:
                        self.class_ids[e] = len(self.class_ids)
                if e in " A":  entity = Empty(i,j)
                if e == "X":   entity = Obstacle(i,j)
                if e == "A":   self.agent = Agent(i,j,self.actions)
                row.append(entity)
                j += 1
            self.map_array.append(row)
            i += 1
        f.close()
        # height width
        self.map_height, self.map_width = len(self.map_array), len(self.map_array[0])

    def is_hidden_rm(self):
        return True

class GardenEnvNoSlip(GardenEnv):
    def __init__(self):
        super().__init__()
        self.slip_prob = 0

class GardenEnvSlip5(GardenEnv):
    def __init__(self):
        super().__init__()
        self.slip_prob = 0.05
