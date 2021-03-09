from reward_machines.reward_functions import RewardFunction
from rl_agents.jirp.dnf_compile import compile_dnf, evaluate_dnf_compiled
import random


class LabelRewardFunction(RewardFunction):
    """
    Defines a reward function that depends on true_props/events/label
    """
    def __init__(self, label_rewards):
        super().__init__()
        self.label_rewards = label_rewards
        self.compiled_dnfs = dict([(dnf, compile_dnf(dnf)) for dnf in label_rewards])

    def get_type(self):
        return "label"

    def get_reward(self, s_info):
        if "true_props" not in s_info:
            return 0.0
        true_props = s_info["true_props"]
        for dnf in self.label_rewards:
            if evaluate_dnf_compiled(self.compiled_dnfs[dnf], true_props):
                return self.label_rewards[dnf]
        return 0.0

    def change_for(self, true_props, to):
        for dnf in self.label_rewards:
            if evaluate_dnf_compiled(self.compiled_dnfs[dnf], true_props):
                self.label_rewards[dnf] = to
                return

    def __str__(self):
        return str(self.label_rewards)

    def __repr__(self):
        return str(self.label_rewards)

class NoisyContRewardFunction(RewardFunction):
    """
    Defines a constant reward for a 'simple reward machine'
    """
    def __init__(self, c, eps):
        super().__init__()
        self.c = c
        self.eps = eps

    def get_type(self):
        return "noisy_cont"

    def get_mean(self, s_info):
        return self.c

    def get_reward(self, s_info):
        return self.c + random.uniform(-self.eps, self.eps)

    def __str__(self):
        return f"{self.c} (eps={self.eps})"

    def __repr__(self):
        return f"{self.c} (eps={self.eps})"
