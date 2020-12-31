import math
from reward_machines.reward_machine_utils import evaluate_dnf

class RewardFunction:
    def __init__(self):
        pass

    # To implement...
    def get_reward(self, s_info):
        raise NotImplementedError("To be implemented")

    def get_type(self):
        raise NotImplementedError("To be implemented")


class ConstantRewardFunction(RewardFunction):
    """
    Defines a constant reward for a 'simple reward machine'
    """
    def __init__(self, c):
        super().__init__()
        self.c = c

    def get_type(self):
        return "constant"

    def get_reward(self, s_info):
        return self.c

    def __str__(self):
        return str(self.c)

    def __repr__(self):
        return str(self.c)

class LabelRewardFunction(RewardFunction):
    """
    Defines a reward function that depends on true_props/events/label
    """
    def __init__(self, label_rewards):
        super().__init__()
        self.label_rewards = label_rewards

    def get_type(self):
        return "label"

    def get_reward(self, s_info):
        true_props = s_info["true_props"]
        for dnf in self.label_rewards:
            if evaluate_dnf(dnf, true_props):
                return self.label_rewards[dnf]
        return 0

    def __str__(self):
        return str(self.label_rewards)

    def __repr__(self):
        return str(self.label_rewards)

class RewardControl(RewardFunction):
    """
    Gives a reward for moving forward
    """
    def __init__(self):
        super().__init__()

    def get_type(self):
        return "ctrl"

    def get_reward(self, s_info):
        return s_info['reward_ctrl']

class RewardForward(RewardFunction):
    """
    Gives a reward for moving forward
    """
    def __init__(self):
        super().__init__()

    def get_type(self):
        return "forward"

    def get_reward(self, s_info):
        #return s_info['reward_forward'] + s_info['reward_ctrl'] + s_info['reward_contact'] + s_info['reward_survive'] # ANT
        return s_info['reward_run'] + s_info['reward_ctrl']  #Cheetah


class RewardBackwards(RewardFunction):
    """
    Gives a reward for moving backwards
    """
    def __init__(self):
        super().__init__()

    def get_type(self):
        return "backwards"

    def get_reward(self, s_info):
        #return -s_info['reward_forward'] + s_info['reward_ctrl'] + s_info['reward_contact'] + s_info['reward_survive'] # ANT
        return -s_info['reward_run'] + s_info['reward_ctrl']  #Cheetah
