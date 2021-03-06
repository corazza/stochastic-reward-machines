import os
import tempfile
import IPython
import tracemalloc

import tensorflow as tf
import gym
import zipfile
import cloudpickle
import numpy as np

# import baselines.common.tf_util as U
# from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
# from baselines.common.schedules import LinearSchedule
# from baselines.common import set_global_seeds

# from baselines import deepq
# from baselines.deepq.utils import ObservationInput

# from baselines.common.tf_util import get_session
# from baselines.deepq.models import build_q_func

from rl_agents.deepqjirp2.util import *
from rl_agents.deepqjirp2.consts import *
from rl_agents.jirp.util import *
from rl_agents.jirp.consts import *
from rl_agents.jirp.jirp import consistent_hyp


def learn2(env,
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
    # tracemalloc.start()
    assert use_crm

    # Adjusting hyper-parameters by considering the number of RM states for crm
    if use_crm:
        rm_states   = env.get_num_rm_states()
        buffer_size = rm_states*buffer_size
        batch_size  = rm_states*batch_size

    # sess = get_session()
    # set_global_seeds(seed)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    observation_space = env.observation_space

    rm_obs = env.reset()
    obs = rm_obs['features']
    rm_state = rm_obs['rm-state']

    main_dqn = dict()
    target_dqn = dict()
    replay_buffer = dict()
    exploration = dict()
    agent = dict()

    for p in env.current_rm.U: # TODO currently assumes single RM
        main_dqn[p] = build_q_network(n_actions=env.action_space.n, learning_rate=LEARNING_RATE)
        target_dqn[p] = build_q_network(n_actions=env.action_space.n, learning_rate=LEARNING_RATE)
        replay_buffer[p] = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE, use_per=USE_PER)
        agent[p] = Agent(main_dqn[p], target_dqn[p], replay_buffer[p], env.action_space.n,
                       input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, use_per=USE_PER)

    episode_rewards = [0.0]
    saved_mean_reward = None

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

        # if tf.train.latest_checkpoint(td) is not None:
        #     load_variables(model_file)
        #     logger.log('Loaded model from {}'.format(model_file))
        #     model_saved = True
        # elif load_path is not None:
        #     load_variables(load_path)
        #     logger.log('Loaded model from {}'.format(load_path))

        for t in range(TOTAL_TIMESTEPS):
            if callback is not None:
                if callback(locals(), globals()):
                    break

            action = agent[rm_state].get_action(t, env.state)

            env_action = action
            rm_new_obs, rew, done, info = env.step(env_action)
            # if rew > 0:
            #     IPython.embed()
            new_obs = rm_new_obs['features']
            new_rm_state = rm_new_obs['rm-state']
            true_props = env.get_events()

            # jirp_labels.append(true_props)
            # jirp_rewards.append(rew)

            experiences = info["crm-experience"]

            # HERE
            # replicate for non-RM-enviroment envs, into new folder

            # Adding the experiences to the replay buffer
            for _obs, _action, _r, _new_obs, _done in experiences:
                p = _obs['rm-state']
                agent[p].add_experience(action=_action, frame=_new_obs['features'][:, :, 0], reward=_r, terminal=_done or info['life_lost'])

            if t % UPDATE_FREQ == 0 and t > MIN_REPLAY_BUFFER_SIZE:
                for p in env.current_rm.get_states():
                    loss, _ = agent[p].learn(BATCH_SIZE, gamma=DISCOUNT_FACTOR,
                                                    frame_number=t,
                                                    priority_scale=PRIORITY_SCALE)
                # loss_list.append(loss)

            # Update target network
            if t % TARGET_UPDATE_FREQ == 0 and t > MIN_REPLAY_BUFFER_SIZE:
                for p in env.current_rm.get_states():
                    agent[p].update_target_network()

            obs = new_obs
            rm_state = new_rm_state

            episode_rewards[-1] += rew
            if done:
                rm_obs = env.reset()
                obs = rm_obs['features']
                rm_state = rm_obs['rm-state']
                episode_rewards.append(0.0)

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                # IPython.embed()
                # logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
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

    # return act
