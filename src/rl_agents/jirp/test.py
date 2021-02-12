import IPython

from reward_machines.reward_machine import RewardMachine
from reward_machines.rm_environment import RewardMachineEnv, RewardMachineHidden
from rl_agents.jirp.util import *
from rl_agents.jirp.consts import *
from rl_agents.jirp.mip_hyp import mip_hyp
from rl_agents.jirp.smt_hyp import smt_hyp


def test1():
    empty_transition = "!a&!b"
    test_transitions = dict()
    test_transitions[(1, ('a',))] = [2, 0]
    test_transitions[(1, ('b',))] = [3, 0]
    test_transitions[(2, ('a',))] = [4, 0]
    test_transitions[(3, ('a',))] = [5, 0]
    test_transitions[(4, ('a',))] = [-1, 1.1]
    test_transitions[(5, ('a',))] = [-1, 0.9]
    mip_hyp(MINIMIZATION_EPSILON, {"a", "b"}, 3, 5, test_transitions, empty_transition, inspect=False, display=True, report=True)
    smt_hyp(MINIMIZATION_EPSILON, {"a", "b"}, 3, 5, test_transitions, empty_transition, inspect=False, display=True, report=True)

def test2(rm):
    # transitions = rm_to_transitions(rm)
    # display_transitions(transitions, "asdf")
    pass

def test3():
    empty_transition = "!a&!b"
    test_transitions = dict()
    test_transitions[(1, ('a',))] = [2, 0]
    test_transitions[(1, ('b',))] = [3, 0]
    test_transitions[(2, ('a',))] = [4, 0]
    test_transitions[(3, ('a',))] = [5, 0]
    test_transitions[(4, ('a',))] = [-1, 1.1]
    test_transitions[(5, ('a',))] = [-1, 0.9]
    mip_hyp(MINIMIZATION_EPSILON, {"a", "b"}, 3, 5, test_transitions, empty_transition, inspect=False, display=True, report=True)
    smt_hyp(MINIMIZATION_EPSILON, {"a", "b"}, 3, 5, test_transitions, empty_transition, inspect=False, display=True, report=True)
