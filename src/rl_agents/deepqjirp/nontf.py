import os
import tempfile
import IPython
import tracemalloc

import tensorflow as tf
import gym
import zipfile
import cloudpickle
import numpy as np

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func

from rl_agents.deepqjirp.util import *
from rl_agents.jirp.util import *
from rl_agents.jirp.consts import *
from rl_agents.jirp.jirp import consistent_hyp

class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None

    @staticmethod
    def load_act(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = deepq.build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def step(self, observation, **kwargs):
        # DQN doesn't use RNNs so we ignore states and masks
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        return self._act([observation], **kwargs), None, None, None

    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)

    def save(self, path):
        save_variables(path)


def load_act(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load_act(path)


def learn(env,
          network,
          seed=None,
          use_crm=False, # ignored
          use_rs=False,
          lr=5e-4,
          total_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          load_path=None,
          **network_kwargs
            ):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
        (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
        will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    use_crm: bool
        use counterfactual experience to train the policy (ignored)
    use_rs: bool
        use reward shaping
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
    batch_size: int
        size of a batch sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
    load_path: str
        path to load the model from. (default: None)
    **network_kwargs
        additional keyword arguments to pass to the network builder.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """

    # tracemalloc.start()
    assert use_crm

    # Adjusting hyper-parameters by considering the number of RM states for crm
    if use_crm:
        rm_states   = env.get_num_rm_states()
        buffer_size = rm_states*buffer_size
        batch_size  = rm_states*batch_size

    set_global_seeds(seed)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    observation_space = env.observation_space

    rm_obs = env.reset()
    obs = rm_obs['features']
    rm_state = rm_obs['rm-state']

    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    main_dqn = dict()
    target_dqn = dict()
    replay_buffer = dict()
    beta_schedule = dict()
    exploration = dict()

    for p in env.current_rm.U: # TODO currently assumes single RM
        main_dqn[p] = build_q_network(n_actions=env.action_space.n, learning_rate=lr)
        target_dqn[p] = build_q_network(n_actions=env.action_space.n, learning_rate=lr)

        # Create the replay buffer
        if prioritized_replay:
            replay_buffer[p] = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
            if prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = total_timesteps
            beta_schedule[p] = LinearSchedule(prioritized_replay_beta_iters,
                                        initial_p=prioritized_replay_beta0,
                                        final_p=1.0)
        else:
            replay_buffer[p] = ReplayBuffer(buffer_size)
            beta_schedule[p] = None

    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                initial_p=1.0,
                                final_p=exploration_final_eps)

    episode_rewards = [0.0]
    saved_mean_reward = None

    reset = True

    # JIRP
    X = set()
    X_new = set()
    X_tl = set()
    jirp_labels = []
    jirp_rewards = []

    transitions, n_states_last = consistent_hyp(set(), set())
    language = sample_language(X)
    empty_transition = dnf_for_empty(language)
    H = rm_from_transitions(transitions, empty_transition)    

    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "model")
        model_saved = False

        if tf.train.latest_checkpoint(td) is not None:
            load_variables(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True
        elif load_path is not None:
            load_variables(load_path)
            logger.log('Loaded model from {}'.format(load_path))

        for t in range(total_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True

            action = act[rm_state](np.array(obs)[None], update_eps=update_eps, **kwargs)[0]

            env_action = action
            reset = False
            rm_new_obs, rew, done, info = env.step(env_action)
            new_obs = rm_new_obs['features']
            new_rm_state = rm_new_obs['rm-state']
            true_props = env.get_events()

            # jirp_labels.append(true_props)
            # jirp_rewards.append(rew)

            # Store transition in the replay buffer.
            if use_crm:
                # Adding counterfactual experience (this will alrady include shaped rewards if use_rs=True)
                experiences = info["crm-experience"]
            elif use_rs:
                # Include only the current experince but shape the reward
                experiences = [info["rs-experience"]]
            else:
                # Include only the current experience (standard deepq)
                experiences = [(rm_obs, action, rew, new_obs, float(done))]
            # Adding the experiences to the replay buffer
            for _obs, _action, _r, _new_obs, _done in experiences:
                p = _obs['rm-state']
                with sess[p].as_default():
                    replay_buffer[p].add(_obs['features'], _action, _r, _new_obs, _done)
            
            obs = new_obs
            rm_state = new_rm_state

            episode_rewards[-1] += rew
            if done:
                rm_obs = env.reset()
                obs = rm_obs['features']
                rm_state = rm_obs['rm-state']
                episode_rewards.append(0.0)
                reset = True

                # if not run_eqv(3*EXACT_EPSILON, rm_run(jirp_labels,H), jirp_rewards):
                #     IPython.embed()

            if t > learning_starts and t % train_freq == 0:
                for p in env.current_rm.U:
                    with sess[p].as_default():
                        # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                        if prioritized_replay:
                            experience = replay_buffer[p].sample(batch_size, beta=beta_schedule[p].value(t))
                            (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                        else:
                            obses_t, actions, rewards, obses_tp1, dones = replay_buffer[p].sample(batch_size)
                            weights, batch_idxes = np.ones_like(rewards), None
                        td_errors = train[p](obses_t, actions, rewards, obses_tp1, dones, weights)
                        if prioritized_replay:
                            new_priorities = np.abs(td_errors) + prioritized_replay_eps
                            replay_buffer[p].update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t % target_network_update_freq == 0:
                for p in env.current_rm.U:
                    with sess[p].as_default():
                        # Update target network periodically.
                        update_target[p]()

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()
                # snapshot = tracemalloc.take_snapshot()
                # display_top(snapshot)

        #     if (checkpoint_freq is not None and t > learning_starts and
        #             num_episodes > 100 and t % checkpoint_freq == 0):
        #         if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
        #             if print_freq is not None:
        #                 logger.log("Saving model due to mean reward increase: {} -> {}".format(
        #                            saved_mean_reward, mean_100ep_reward))
        #             # save_variables(model_file)
        #             model_saved = True
        #             saved_mean_reward = mean_100ep_reward
        # if model_saved:
        #     if print_freq is not None:
        #         logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
        #     load_variables(model_file)

    return act
