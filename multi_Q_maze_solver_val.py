# %%
import copy
from itertools import count
import os
import pickle
import random
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import trange
from frozen_gridworld import FrozenLake
# from meeting_gridworld import MeetingGridworld

from utils import DuelingDQNLast, ReplayMemory, DQN, Transition, draw_history, draw_path, get_global_weights, moving_average, plot_rewards, sync_Agents_weights
import torch.nn.functional as F
import math
from IPython.display import clear_output

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
def val(Q_tables, floder_name=None, n_times=0, dynamic=False):
    # Hyperparameters
    EPISODES = 640
    EPSILON = 0.0  # Random rate
    MAX_STEPS_PER_EPISODE = 512  # Maximum steps per episode
    GAMMA = 0.99
    LR = 0.1  # Learning rate
    MAZE = np.loadtxt('maze17_0.2.txt')  # maze_cross_level4.txt  maze17_0.2.txt
    SIZE = MAZE.shape[0]

    FL_LOCAL_EPOCH = 8

    # Create environment
    n_agents = 3
    reward_dict=[{'step': -1, 'fall': -10, 'goal': 50, 'heuristic': False},
                # {'step': 0, 'fall': 0, 'goal': 0, 'heuristic': False},
                {'step': -1, 'fall': -10, 'goal': 50, 'heuristic': False},
                {'step': -1, 'fall': -10, 'goal': 50, 'heuristic': False}
                ]

    if dynamic:
        wall_positions=[(16, 9), (15, 9), (14, 12)]
        wall_odds=[0.9, 0.7, 0.6]
    else:
        wall_positions = wall_odds = None

    env = FrozenLake(size=SIZE, n_agents=n_agents, heuristic_reward=True, lake=MAZE,
                     reward_dict=reward_dict,
                     dynamic=dynamic, hole_positions=wall_positions, hole_odds=wall_odds)

    num_states = SIZE * SIZE
    num_actions = 4

    # Create Q-table
    # Q_tables = [np.zeros([num_states, num_actions]) for _ in range(env.n_agents)]
    # if share_params:
    #     Q_tables = [Q_tables[0] for _ in range(env.n_agents)]
    # global_Q = np.zeros([num_states, num_actions])
    
    # Initialize epsilon
    episode_durations = [[] for _ in range(env.n_agents)]
    agent_paths = [[] for _ in range(env.n_agents)]
    agent_rewards = [[] for _ in range(env.n_agents)]
    agent_paths_length = [[] for _ in range(env.n_agents)]
    agent_alives = [[] for _ in range(env.n_agents)]
    agent_wins = [[] for _ in range(env.n_agents)]

    train_history = {'episode_durations': episode_durations, \
                        'agent_paths': agent_paths, \
                        'agent_rewards': agent_rewards, \
                        'agent_paths_length': agent_paths_length, \
                        'agent_alives': agent_alives, \
                        'agent_wins': agent_wins}

    for i_episode in range(EPISODES):
        # print(len(train_history))
        # Initialize the environment and state
        env.reset()

        steps_done = [0 for _ in range(env.n_agents)]

        # Log intermediate variables
        rewards = [0.0 for _ in range(env.n_agents)]
        # Record the full history for each agent
        full_history_position = [[] for _ in range(env.n_agents)]

        # raise
        for t in count():
            # clear_output(wait=True)
            # env.render()
            # time.sleep(0.1)
            
            # 一人走一步
            # Break the loop if the maximum steps per episode is reached or all agents are done
            if t >= MAX_STEPS_PER_EPISODE or all([env.agents[i]['done'] for i in range(env.n_agents)]):
                # 当超出最大步数时，对于没有完成的agent，记录其历史
                for i in range(env.n_agents):
                    if not env.agents[i]['done']:  # Only record for agents that are not done
                        episode_durations[i].append(t + 1)
                        agent_paths[i].append(full_history_position[i])
                        agent_paths_length[i].append(len(full_history_position[i]))
                        agent_rewards[i].append(rewards[i])     # Log cumulative reward
                        agent_alives[i].append(1) if env.agents[i]['alive'] else agent_alives[i].append(0)
                        agent_wins[i].append(0)  # 超出最大步数，算失败
                break

            for idx in range(env.n_agents):
                # The agent might have been done
                if env.agents[idx]['done']:
                    continue

                # Select and perform an action
                state = env.get_state(idx)
                state_index = int(state[0] * SIZE + state[1])
                action = np.argmax(Q_tables[idx][state_index]) if np.random.uniform() > EPSILON else np.random.choice(env.action_space)
                
                # Perform action
                _, reward, done, status = env.step(idx, action)
                steps_done[idx] += 1  # Increment steps_done for this agent
                
                # Flatten next_state
                next_state = env.get_state(idx)
                next_state_index = int(next_state[0] * SIZE + next_state[1])

                # Record agent's path
                full_history_position[idx].append(env.agents[idx]['pos'])
                # Record agent's cumulative reward
                rewards[idx] += float(reward)

                # Update Q-table
                # 如果只更新A，而且不是第一个agent，那么就不更新
                # if onlyA and idx!=0:
                #     pass
                # else:
                # Q_tables[idx][state_index, action] = \
                #     Q_tables[idx][state_index, action] + LR * (reward + GAMMA * np.max(Q_tables[idx][next_state_index]) - Q_tables[idx][state_index, action])
                state = next_state

                if done:
                    episode_durations[idx].append(t + 1)
                    agent_paths[idx].append(full_history_position[idx])
                    agent_paths_length[idx].append(len(full_history_position[idx]))
                    agent_rewards[idx].append(rewards[idx])     # Log cumulative reward
                    agent_alives[idx].append(1) if env.agents[idx]['alive'] else agent_alives[idx].append(0)
                    agent_wins[idx].append(1) if env.agents[idx]['alive'] else agent_wins[idx].append(0)  # 完成时判断是否还活着，活着算成功
                    continue               

        # print(f'i_episode{i_episode}, steps_done{[agent_paths_length[k][-1] for k in range(2)]}, rewards{rewards}')
        # plot_rewards(agent_paths_length)
    
    #save models
    # name the save floder

    # if floder_name is None:
    #     prefix = 'Q_learning_M1720Forzen_noHRwd_5flod_'

    #     floder_name = prefix  # M17_20_all_delta_

    #     floder_name += 'dynamic_' if dynamic else ''
        
    #     floder_name += 'FL_' if FL else ''
    #     floder_name += 'FLMax_' if FLMax else ''
    #     floder_name += 'FLAll_' if FLAll else ''
    #     floder_name += 'FLDelta_' if FLdelta else ''

    #     str_decay_ratio = ''.join(map(str, [int(x * 10) for x in DECAY_RATIO]))
    #     floder_name += 'FLDynamicAvg_'+ str_decay_ratio +'_' if FLDynamicAvg else ''
    #     # floder_name += 'FLDynamicAvg_'+ str(EPS_DECAY) +'_' + str(EPS_START) + '_' + str(EPS_END) + '_' if FLDynamicAvg else ''
    #     floder_name += 'FLRewardShape_' if FLRewardShape else ''
    #     floder_name += 'FLGreedEpsilon_'+ str(EPS_DECAY) +'_' if FLGreedEpsilon else ''

    #     floder_name += 'FLdev_' if FLdev else ''

    #     floder_name += '{}LStep_'.format(FL_LOCAL_EPOCH) if 'FL' in floder_name else ''

    #     floder_name += 'Paramsshare_' if share_params else ''

    #     floder_name += 'onlyA_' if onlyA else ''
    #     # floder_name += 'role_' if role else ''
    #     # floder_name += 'Memoryshare_' if share_memory else ''


    #     if floder_name == prefix:
    #         floder_name += 'Independent_'
    #     floder_name += 'ep{}_'.format(EPISODES)
    #     floder_name += time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    # # create floder
    # if not os.path.exists('./logs/' + floder_name):
    #     print('create floder: ', floder_name)
    #     os.mkdir('./logs/' + floder_name)

    # for i in range(env.n_agents):
    #     # save Q_tables
    #     with open('./logs/{}/Q_table_{}_{}.pkl'.format(floder_name, i, n_times), 'wb') as f:
    #         pickle.dump(Q_tables[i], f)
    
    # with open('./logs/{}/train_history_{}.pkl'.format(floder_name, n_times), 'wb') as f:
    #     pickle.dump(train_history, f)

    return train_history, Q_tables, floder_name  # , deltas_dict

# dymnamic = False
# floder_name = None
# for n in trange(5):
#     train_history, policy_nets, floder_name = main(FLMax=True,
#                                                 dynamic=dymnamic,
#                                                 floder_name=floder_name, n_times=n)

# %%
# 读取Q表
log_paths = [
    'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_ep50000_2023-06-23-22-27-20'
    # 'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_FLDynamicAvg_262_8LStep_ep50000_2023-08-31-16-35-51',
]

for i, path in enumerate(log_paths):
    log_paths[i] = './logs/' + path + '/'

# 从pkl读取Q_tables
Q_tables = []
flods = 5
for fold in range(flods):
    Q_tables.append([])
    for i in range(3):
        with open(log_paths[0] + 'Q_table_{}_{}.pkl'.format(i, fold), 'rb') as f:
            Q_tables[fold].append(pickle.load(f))
Q_tables = np.array(Q_tables)
# %%
for fold in range(flods):
    history, _, _ = val(Q_tables[fold], floder_name=log_paths[0], n_times=1, dynamic=True)
    for n in range(3):
        print(np.mean(history['agent_wins'][n]))
    print('------------------')
    # break
# %%
np.array(history['agent_wins']).shape

# %%
for n in range(3):
    print(np.mean(history['agent_wins'][n]))
# %%
print(history.keys())
# %%

# %%
# %%
