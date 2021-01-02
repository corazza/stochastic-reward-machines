from reward_machines.reward_functions import RewardFunction
from rl_agents.jirp.dnf_compile import compile_dnf, evaluate_dnf_compiled

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
        true_props = s_info["true_props"]
        for dnf in self.label_rewards:
            if evaluate_dnf_compiled(self.compiled_dnfs[dnf], true_props):
                return self.label_rewards[dnf]
        return 0

    def __str__(self):
        return str(self.label_rewards)

    def __repr__(self):
        return str(self.label_rewards)
