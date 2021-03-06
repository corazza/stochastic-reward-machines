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
from rl_agents.jirp.jirp import consistent_hyp, equivalent_on_X


def transfer_Q(H_new, H_old, Q_old, X = {}):
    Q = dict()
    for v in H_new.get_states():
        Q[v] = dict()
        # find probably equivalent state u in H_old
        for u in H_old.get_states():
            if equivalent_on_X(H_new, v, H_old, u, X) and u in Q_old:
                Q[v] = Q_old[u]
                break
    return Q

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
            new_obs = rm_new_obs['features']
            next_rm_state = rm_new_obs['rm-state']
            true_props = env.get_events()

            experiences = info["crm-experience"]

            # Adding the experiences to the replay buffer
            for _obs, _action, _r, _new_obs, _done in experiences:
                p = _obs['rm-state']
                next_p = _new_obs['rm-state']
                agent[p].add_experience(action=_action, frame=_new_obs['features'][:, :, 0], reward=_r, next_rm_state=next_p, terminal=_done or info['life_lost'])

            if t % UPDATE_FREQ == 0 and t > MIN_REPLAY_BUFFER_SIZE:
                for p in env.current_rm.get_states():
                    loss, _ = agent[p].learn(BATCH_SIZE, gamma=DISCOUNT_FACTOR,
                                                    frame_number=t,
                                                    agent_dict=agent,
                                                    priority_scale=PRIORITY_SCALE)
                # loss_list.append(loss)

            # Update target network
            if t % TARGET_UPDATE_FREQ == 0 and t > MIN_REPLAY_BUFFER_SIZE:
                for p in env.current_rm.get_states():
                    agent[p].update_target_network()

            obs = new_obs
            rm_state = next_rm_state

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

    assert not use_crm

    observation_space = env.observation_space
    episode_rewards = [0.0]
    saved_mean_reward = None

    # JIRP
    X = set()
    X_new = set()
    X_tl = set()
    jirp_labels = []
    jirp_rewards = []
    next_random = False

    transitions, n_states_last = consistent_hyp(set(), set())
    language = sample_language(X)
    empty_transition = dnf_for_empty(language)
    H = rm_from_transitions(transitions, empty_transition)
    Q = initial_Q_nets(H, env)

    obs = env.reset()
    rm_state = H.reset()

    # intrinsic reward
    observed_events = set()

    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        for t in range(TOTAL_TIMESTEPS*6):
            if callback is not None:
                if callback(locals(), globals()):
                    break

            action = Q[rm_state].get_action(t, env.state)

            env_action = action
            new_obs, rew, done, info = env.step(env_action)
            true_props = env.get_events()
            next_rm_state, _rm_reward, rm_done = H.step(rm_state, true_props, info)

            jirp_labels.append(true_props)
            jirp_rewards.append(rew)

            if true_props not in observed_events:
                rew += 5
            observed_events.add(true_props)

            Q[rm_state].add_experience(action=action, frame=new_obs[:, :, 0], reward=rew, next_rm_state=next_rm_state, terminal=done or info['life_lost'])
            for p in H.get_states():
                if p == rm_state:
                    continue
                p_next, h_r, h_done = H.step(p, true_props, info)
                Q[p].add_experience(action=action, frame=new_obs[:, :, 0], reward=h_r, next_rm_state=p_next, terminal=done or info['life_lost'] or h_done)

            if not rm_done or not TERMINATION:
                rm_state = next_rm_state # TODO FIXME this entire loop, comment and organize
            else:
                next_random = True

            if t % UPDATE_FREQ == 0 and t > MIN_REPLAY_BUFFER_SIZE:
                for p in H.get_states():
                    loss, _ = Q[p].learn(BATCH_SIZE, gamma=DISCOUNT_FACTOR,
                                                    frame_number=t,
                                                    agent_dict=Q,
                                                    priority_scale=PRIORITY_SCALE)
                # loss_list.append(loss)

            # HERE
            # min_replay_buffer_size and q.replay_buffer_start size-- unalined in my current code?

            # Update target network
            if t % TARGET_UPDATE_FREQ == 0 and t > MIN_REPLAY_BUFFER_SIZE:
                for p in H.get_states():
                    Q[p].update_target_network()

            obs = new_obs
            rm_state = next_rm_state

            episode_rewards[-1] += rew
            if info['life_lost']:
                observed_events = set()
                rm_state = H.reset()
            if done:
                obs = env.reset()
                observed_events = set()
                rm_state = H.reset()
                episode_rewards.append(0.0)

                if not run_eqv(EXACT_EPSILON, rm_run(jirp_labels, H), jirp_rewards):
                    X_new.add((tuple(jirp_labels), tuple(jirp_rewards)))
                    if "TimeLimit.truncated" in info: # could also see if RM is in a terminating state
                        tl = info["TimeLimit.truncated"]
                        if tl:
                            X_tl.add((tuple(jirp_labels), tuple(jirp_rewards)))

                if num_episodes % UPDATE_X_EVERY_N_EPISODES == 0 and X_new:
                    print(f"len(X)={len(X)}")
                    print(f"len(X_new)={len(X_new)}")
                    X.update(X_new)
                    X_new = set()
                    language = sample_language(X)
                    empty_transition = dnf_for_empty(language)
                    transitions_new, n_states_last = consistent_hyp(X, X_tl, n_states_last)
                    H_new = rm_from_transitions(transitions_new, empty_transition)
                    Q = initial_Q_nets(H, env)
                    H = H_new
                    transitions = transitions_new

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("Q-nets", len(Q))
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
