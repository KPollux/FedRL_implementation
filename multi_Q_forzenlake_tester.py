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
def main(share_params=False, FL=False, role=False, share_memory=False, FLMax=False, FLdev=False, onlyA=False,
          FLAll=False, FLdelta=False, FLDynamicAvg=False, floder_name=None, n_times=0, dynamic=False):
    # Hyperparameters
    EPISODES = 1000
    EPSILON = 0  # Random rate
    MAX_STEPS_PER_EPISODE = 512  # Maximum steps per episode
    GAMMA = 0.99
    LR = 0.1  # Learning rate
    MAZE = np.loadtxt('maze17_0.2.txt')  # maze_cross_level4.txt  maze17_0.2.txt
    SIZE = MAZE.shape[0]

    # Create environment
    n_agents = 3
    reward_dict=[{'step': -1, 'fall': -10, 'goal': 50, 'heuristic': False},
                # {'step': 0, 'fall': 0, 'goal': 0, 'heuristic': False},
                {'step': -1, 'fall': -10, 'goal': 50, 'heuristic': False},
                {'step': -1, 'fall': -10, 'goal': 50, 'heuristic': False}
                ]

    # reward_dict=[{'step': 0, 'collision': 0, 'goal': 1, 'heuristic': True},
    #         # {'step': 0, 'collision': 0, 'goal': 0, 'heuristic': False},
    #         {'step': 0, 'collision': 0, 'goal': 1, 'heuristic': True},
    #         {'step': 0, 'collision': 0, 'goal': 1, 'heuristic': True}
    #         ]
    if dynamic:
        wall_positions=[(16, 9), (15, 9), (14, 12)]
        wall_odds=[0.9, 0.7, 0.6]
    else:
        wall_positions = wall_odds = None
    # env = MeetingGridworld(size=SIZE, n_agents=2, heuristic_reward=True, maze=MAZE)
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

    # load Q-Table
    # 遍历路径下的前缀为(Q_tables)的文件，找到最新的Q_tables
    Q_tables = []
    assert os.path.exists(floder_name), 'floder_name not exists'
    # 获取路径下所有文件名
    for client_idx in range(env.n_agents):
        Q_tables.append(pickle.load(open(os.path.join(floder_name, f'Q_table_{client_idx}_{n_times}.pkl'), 'rb')))
    # print(Q_tables)
    # raise
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

    deltas_dict = {'episodes': [], 'deltas_1': [], 'deltas_2': [],
                    'deltas_3': [], 'deltas_global': [], 'weights': []}

    for i_episode in range(EPISODES):
        # Initialize the environment and state
        env.reset()

        steps_done = [0 for _ in range(env.n_agents)]

        # Log intermediate variables
        rewards = [0.0 for _ in range(env.n_agents)]
        # Record the full history for each agent
        full_history_position = [[] for _ in range(env.n_agents)]

        # raise
        for t in count():
            
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
                action = np.argmax(Q_tables[idx][state_index])
                
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

                # # Update Q-table
                # # 如果只更新A，而且不是第一个agent，那么就不更新
                # if onlyA and idx!=0:
                #     pass
                # else:
                #     Q_tables[idx][state_index, action] = \
                #         Q_tables[idx][state_index, action] + LR * (reward + GAMMA * np.max(Q_tables[idx][next_state_index]) - Q_tables[idx][state_index, action])
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

    #     floder_name += 'FLDynamicAvg_' if FLDynamicAvg else ''

    #     floder_name += 'FLdev_' if FLdev else ''

    #     floder_name += '{}LStep_'.format(FL_LOCAL_EPOCH) if 'FL' in floder_name else ''

    #     floder_name += 'Paramsshare_' if share_params else ''

    #     floder_name += 'onlyA_' if onlyA else ''
    #     # floder_name += 'role_' if role else ''
    #     # floder_name += 'Memoryshare_' if share_memory else ''

    #     floder_name += 'ep{}_'.format(EPISODES)

    #     if floder_name == prefix:
    #         floder_name += 'Independent_'
    #     floder_name += time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    # # create floder
    # if not os.path.exists('./logs/' + floder_name):
    #     print('create floder: ', floder_name)
    #     os.mkdir('./logs/' + floder_name)

    # for i in range(env.n_agents):
    #     # save Q_tables
    #     with open('./logs/{}/Q_table_{}_{}.pkl'.format(floder_name, i, n_times), 'wb') as f:
    #         pickle.dump(Q_tables[i], f)
    
    with open('{}val_history_{}.pkl'.format(floder_name, n_times), 'wb') as f:
        pickle.dump(train_history, f)

    return train_history, Q_tables, floder_name, deltas_dict

# dymnamic = False
# floder_name = None
# for n in trange(5):
#     train_history, policy_nets, floder_name = main(FLMax=True,
#                                                 dynamic=dymnamic,
#                                                 floder_name=floder_name, n_times=n)

# %%
# 冰冻难度地图、三Agent、相同奖励(无启发式)、8ls（文章1）
# log_paths = [
#     'Q_learning_M1720Forzen_noHRwd_5flod_ep50000_2023-06-18-03-07-40',
#     'Q_learning_M1720Forzen_noHRwd_5flod_Paramsshare_ep50000_2023-06-18-03-22-03',
#     'Q_learning_M1720Forzen_noHRwd_5flod_FL_8LStep_ep50000_2023-06-18-03-36-13',
#     # 'Q_learning_M1720Forzen_noHRwd_FLDynamicAvg_8LStep_ep50000_2023-06-19-01-13-21',  # 跳变
#     'Q_learning_M1720Forzen_noHRwd_5flod_FLDynamicAvg_8LStep_ep50000_2023-06-20-02-51-11', # 'Q_learning_M1720Forzen_noHRwd_FLDynamicAvg_8LStep_ep50000_2023-06-19-01-43-33',  # 渐变1阶 5000轮
#     # 'Q_learning_M1720Forzen_noHRwd_FLDynamicAvg_8LStep_ep50000_2023-06-19-01-50-13', # 渐变2阶
#     # 'Q_learning_M1720Forzen_noHRwd_FLDynamicAvg_8LStep_ep50000_2023-06-19-18-25-35',  # 衰减10000轮
#     # 'Q_learning_M1720Forzen_noHRwd_5flod_FLDynamicAvg_8LStep_ep50000_2023-06-20-01-43-30', # 衰减50000轮
#     # 'Q_learning_M1720Forzen_noHRwd_5flod_FLMax_8LStep_ep50000_2023-06-18-04-04-30',
#     'Q_learning_M1720Forzen_noHRwd_5flod_FLAll_8LStep_ep50000_2023-06-18-04-19-31',
# ]


# # 冰冻动态地图、三Agent、相同奖励(无启发式)、8ls（文章1）
log_paths = [
    'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_ep50000_2023-06-23-22-27-20',
    'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_Paramsshare_ep50000_2023-06-23-22-45-30',
    'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_FL_8LStep_ep50000_2023-06-23-23-03-03',
    'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_FLDynamicAvg_8LStep_ep50000_2023-06-23-23-19-39',
    'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_FLAll_8LStep_ep50000_2023-06-23-23-36-20',
    # 'Q_learning_M1720Forzen_noHRwd_5flod_FLRewardShape_8LStep_ep50000_2023-06-24-00-12-15',
]

history_list = []

for i, path in enumerate(log_paths):
    log_paths[i] = './logs/' + path + '/'

    history_list_temp = []
    for n in trange(5):
        train_history, policy_nets, floder_name, deltas_dict = main(
                                                    dynamic=True,
                                                    floder_name=log_paths[i], n_times=n)
        history_list_temp.append(train_history)
    history_list.append(history_list_temp)
    # break
# %%
legends = ['INDL', 'SQ', 'QAvg', 'QGradual', 'QAll']
# legends = ['INDL', 'SQ', 'QAvg', 'QGradual', 'QAll', 'QRS']

alg_win_rates = []
alg_path_lengths = []

for history in history_list:
    alg_win_rates_flod_temp = []
    alg_path_lengths_flod_temp = []

    for n_flod in range(5):
        # initialize lists to store win rates and path lengths
        win_rates = []
        path_lengths = []

        # loop over each client
        for i in range(3):
            # get the agent win history for this client
            agent_wins = np.array(history[n_flod]['agent_wins'][i])
            agent_path_lengths = np.array(history[n_flod]['agent_paths_length'][i])

            # calculate win rate for this client and add to list
            win_rate = np.mean(agent_wins)
            win_rates.append(win_rate)

            # get the path lengths for wins only
            win_path_lengths = agent_path_lengths[agent_wins==1]

            # add to list
            path_lengths.append(win_path_lengths)

        # print results
        alg_win_rates_temp = []
        alg_path_lengths_temp = []
        for i in range(3):
            alg_win_rates_temp.append(win_rates[i])
            alg_path_lengths_temp.append(path_lengths[i])
            print(f"Client {i} win rate: {win_rates[i]}")
            print(f"Client {i} path lengths on win: {path_lengths[i]}")
        alg_win_rates_flod_temp.append(alg_win_rates_temp)
        alg_path_lengths_flod_temp.append(alg_path_lengths_temp)
    alg_win_rates.append(alg_win_rates_flod_temp)
    alg_path_lengths.append(alg_path_lengths_flod_temp)

# %% average win rates and errors
# Initialize lists to store average win rates and errors for each algorithm
avg_win_rates = []
errors = []

# Calculate average win rate and error for each algorithm
for alg_index, alg_win_rates_per_fold in enumerate(alg_win_rates):
    # Initialize lists to store average win rates and errors for each client
    avg_win_rates_per_alg = []
    errors_per_alg = []

    # Loop over each client
    for client_index in range(3):
        # Get the win rates for this client over all folds
        client_win_rates = [fold[client_index] for fold in alg_win_rates_per_fold]
        
        # Calculate the average win rate and error
        avg_win_rate = np.mean(client_win_rates)
        error = np.std(client_win_rates)
        
        # Store the results
        avg_win_rates_per_alg.append(avg_win_rate)
        errors_per_alg.append(error)

    # Store the results for this algorithm
    avg_win_rates.append(avg_win_rates_per_alg)
    errors.append(errors_per_alg)

# Print the results
for alg_index, legend in enumerate(legends):
    print(f"Algorithm: {legend}")
    for client_index in range(3):
        print(f"  Client {client_index} average win rate: {avg_win_rates[alg_index][client_index]:.2f} ± {errors[alg_index][client_index]:.2f}")

# %% 绘制箱型图
alg_path_lengths = []

for history in history_list:
    alg_path_lengths_fold_temp = []
    for n_flod in range(5):
        # Initialize lists to store path lengths
        path_lengths = []

        # Loop over each client
        for i in range(3):
            # Get the agent path length history for this client
            agent_wins = np.array(history[n_flod]['agent_wins'][i])
            agent_path_lengths = np.array(history[n_flod]['agent_paths_length'][i])

            # Get the path lengths for wins only
            win_path_lengths = agent_path_lengths[agent_wins==1]

            # Add to list
            path_lengths.append(win_path_lengths)

        # Store the results for this fold
        alg_path_lengths_fold_temp.append(path_lengths)
    alg_path_lengths.append(alg_path_lengths_fold_temp)

# Flatten the path lengths for each algorithm
flat_alg_path_lengths = [[length for fold in alg for client in fold for length in client] for alg in alg_path_lengths]

# Draw the box plot for path lengths
fig, ax = plt.subplots()

# 对数据进行对数转换
log_path_lengths = [np.log10(data) for data in flat_alg_path_lengths]

# Create the boxplot
bp = ax.boxplot(log_path_lengths, patch_artist=True)

colors = ['pink', 'lightblue', 'lightgreen', 'lightyellow', 'lightgrey']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Customize the labels
ax.set_xticklabels(legends)

# Adding title and labels
plt.title('Path lengths on win for each algorithm')
plt.xlabel('Algorithms')
plt.ylabel('Path Lengths')

# plt.ylim(0, 60)

# Show the plot
plt.show()

# %%
# Create a figure with subplots for each client
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)

# Loop over each client
for client_index in range(3):
    # Get the path lengths for this client for each algorithm
    client_path_lengths = [[fold[client_index] for fold in alg] for alg in alg_path_lengths]
    
    # Flatten the path lengths for each algorithm
    flat_client_path_lengths = [[length for fold in alg for length in fold] for alg in client_path_lengths]
    
    # 对数据进行对数转换
    log_client_path_lengths = [np.log10(data) for data in flat_client_path_lengths]

    # Create the boxplot for this client
    bp = axs[client_index].boxplot(log_client_path_lengths, patch_artist=True)
    
    colors = ['pink', 'lightblue', 'lightgreen', 'lightyellow', 'lightgrey']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Customize the labels
    axs[client_index].set_xticklabels(legends)
    
    # Adding title and labels
    axs[client_index].set_title(f'Client {client_index} Log Path lengths on win')
    axs[client_index].set_xlabel('Algorithms')
    if client_index == 0:
        axs[client_index].set_ylabel('Log Path Lengths')

# Show the plot
plt.tight_layout()
plt.show()

# %%
# Initialize the lists
alg_path_lengths = []
alg_names = ['INDL', 'SQ', 'QAvg', 'QGradual', 'QAll']

for alg_index, history in enumerate(history_list):
    alg_path_lengths_fold_temp = []
    for n_flod in range(5):
        path_lengths = []
        for i in range(3):
            agent_wins = np.array(history[n_flod]['agent_wins'][i])
            agent_path_lengths = np.array(history[n_flod]['agent_paths_length'][i])
            win_path_lengths = agent_path_lengths[agent_wins==1]
            path_lengths.append(win_path_lengths)
        alg_path_lengths_fold_temp.append(path_lengths)
    alg_path_lengths.append(alg_path_lengths_fold_temp)

# Now, `alg_path_lengths` has the structure [algorithm][fold][client][path_lengths]

# %%
for alg_index, alg in enumerate(alg_path_lengths):
    print(f"Algorithm: {alg_names[alg_index]}")
    print(alg)
# %%
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

fig, ax = plt.subplots()

width = 0.2
space_between_groups = 0.5
space_within_groups = 0.1

colors = ['pink', 'lightblue', 'lightgreen']

position = 0
for alg_index, alg in enumerate(alg_path_lengths):
    # For each client
    for client_index in range(3):
        client_path_lengths = []
        # For each fold
        for fold in alg:
            client_path_lengths.append(fold[client_index])

        flat_client_path_lengths = []
        # Flatten the path lengths for each algorithm
        for path_lengths in client_path_lengths:
            for length in path_lengths:
                flat_client_path_lengths.append(length)

        # calculate boxplot position
        box_position = position + client_index * (width + space_within_groups)
        # create boxplot
        bp = ax.boxplot(flat_client_path_lengths, positions=[box_position], widths=[width], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(colors[client_index])

    position += width * 3 + space_between_groups

# Customize the x-axis
ax.set_xticks([(width * 3 + space_between_groups) * i + 1.5 * width for i in range(len(legends))])
ax.set_xticklabels(legends)

# Add a custom legend
custom_lines = [Line2D([0], [0], color=color, lw=4) for color in colors]
ax.legend(custom_lines, [f'Client {i}' for i in range(3)])

plt.title('Path lengths on win for each client per algorithm')
plt.xlabel('Algorithms')
plt.ylabel('Path Lengths')

plt.show()





# %%
import matplotlib.pyplot as plt

# Create a figure instance
fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
bp = ax.boxplot(path_lengths)

# Customize the labels
ax.set_xticklabels(['Client 1', 'Client 2', 'Client 3'])

# Adding title and labels
plt.title('Path lengths on win for each client')
plt.xlabel('Clients')
plt.ylabel('Path Lengths')

# Show the plot
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

legends = ['INDL', 'SQ', 'QAvg', 'QGradual', 'QAll']

# Initialize lists to store win rates and path lengths for each algorithm
algorithm_win_rates = {legend: [] for legend in legends}
algorithm_path_lengths = {legend: [] for legend in legends}

for history, legend in zip(history_list, legends):
    # Initialize lists to store win rates and path lengths for each client in this algorithm
    win_rates = []
    path_lengths = []

    # Loop over each fold
    for n_flod in range(5):
        # Get the agent win history for this client
        agent_wins = np.array(history[n_flod]['agent_wins'])
        agent_path_lengths = np.array(history[n_flod]['agent_paths_length'])

        # Calculate win rate for this client and add to list
        win_rate = np.mean(agent_wins)
        win_rates.append(win_rate)

        # Get the path lengths for wins only
        win_path_lengths = agent_path_lengths[agent_wins == 1]

        # Add to list
        path_lengths.append(win_path_lengths)

    # Store the win rates and path lengths for this algorithm
    algorithm_win_rates[legend] = np.mean(win_rates)
    algorithm_path_lengths[legend] = path_lengths

# Print the average win rates for each algorithm
for legend in legends:
    print(f"Average win rate for {legend}: {algorithm_win_rates[legend]}")

# Draw the box plot for path lengths
fig, ax = plt.subplots()

# Collect all path lengths for box plot
all_path_lengths = [algorithm_path_lengths[legend] for legend in legends]

# Create the boxplot
bp = ax.boxplot(all_path_lengths, patch_artist=True)

colors = ['pink', 'lightblue', 'lightgreen', 'lightyellow', 'lightgrey']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Customize the labels
ax.set_xticklabels(legends)

# Adding title and labels
plt.title('Path lengths on win for each algorithm')
plt.xlabel('Algorithms')
plt.ylabel('Path Lengths')

# Show the plot
plt.show()

    


# %%
# %%
dymnamic = False

floder_name = None
for n in trange(5):
    train_history, policy_nets, floder_name = main(dynamic=dymnamic,
                                                floder_name=floder_name, n_times=n)
    
floder_name = None
for n in trange(5):
    train_history, policy_nets, floder_name = main(share_params=True,
                                                dynamic=dymnamic,
                                                floder_name=floder_name, n_times=n)

floder_name = None
for n in trange(5):
    train_history, policy_nets, floder_name = main(FL=True,
                                                dynamic=dymnamic,
                                                floder_name=floder_name, n_times=n)
    
floder_name = None
for n in trange(5):
    train_history, policy_nets, floder_name = main(FLdelta=True,
                                                dynamic=dymnamic,
                                                floder_name=floder_name, n_times=n)

floder_name = None
for n in trange(5):
    train_history, policy_nets, floder_name = main(FLMax=True,
                                                dynamic=dymnamic,
                                                floder_name=floder_name, n_times=n)

floder_name = None
for n in trange(5):
    train_history, policy_nets, floder_name = main(FLAll=True,
                                                dynamic=dymnamic,
                                                floder_name=floder_name, n_times=n)
    
# %%


floder_name = None
for n in range(5):
    train_history, policy_nets, floder_name = main(share_params=False, FL=False,
                                                role=False, share_memory=False, FLMax=False,
                                                FLdev=False, onlyA=False, FLAll=True, FLdelta=False,
                                                dynamic=True,
                                                floder_name=floder_name, n_times=n)

# %% 绘制参数
n_agents = 3
EPISODES = 50000
env_size = 17

agent_rewards = train_history['agent_rewards']
agent_paths_length = train_history['agent_paths_length']
agent_paths = train_history['agent_paths']
agent_alives = train_history['agent_alives']
agent_wins = train_history['agent_wins']

# %%
draw_history(agent_wins, agent_rewards, n_agents, EPISODES, window_size=1000, ylims=[-20, -20])
agent_paths_length

# %%
# deltas_dict
# plt.plot(deltas_dict['deltas_1'])
# plt.plot(deltas_dict['deltas_2'])
# plt.plot(deltas_dict['deltas_global'])

m1 = moving_average(deltas_dict['deltas_1'], 1000)
m2 = moving_average(deltas_dict['deltas_2'], 1000)
m3 = moving_average(deltas_dict['deltas_3'], 1000)
mg = moving_average(deltas_dict['deltas_global'], 1000)

# %%

mavg = np.array(mg)/3
# plt.ylim(0, 0.004)
plt.figure(figsize=(10, 7.5))
# 打开网格
# plt.grid()

plt.plot(m1, label='agent1')
plt.plot(m2, label='agent2')
plt.plot(m3, label='agent3')
plt.plot(mg, label='global(add_all)')
plt.plot(mavg, label='global(add_avg)')
plt.plot()
plt.legend()

plt.title('Delta Q with communication rounds')
plt.xlabel('Communication Rounds')
plt.ylabel('MAE of delta Q')

plt.tight_layout()
plt.show()


# %%
ws1s, ws2s, ws3s = [], [], []
for ws1, ws2, ws3 in deltas_dict['weights']:
    ws1s.append(ws1)
    ws2s.append(ws2)
    ws3s.append(ws3)
plt.plot(ws1s)
# plt.plot(ws2s)
# plt.plot(ws3s)




# %%


# %%
delta_Qs = [deltas_dict['deltas_1'][0], deltas_dict['deltas_2'][0], deltas_dict['deltas_3'][0]]
MAEs = [np.mean(np.abs(delta_Q)) for delta_Q in delta_Qs]
# Normalize MAEs to [0, 1] range
normalized_MAEs = (MAEs - np.min(MAEs)) / (np.max(MAEs) - np.min(MAEs))
print(normalized_MAEs)



# %%
episode_index = 900
draw_path(episode_index, agent_paths, n_agents, env_size)
print(agent_paths)
# %%
plt.plot(agent_paths_length[1])
# %%
# save train_history
# with open('./logs/{}/train_history.pkl'.format(floder_name), 'wb') as f:
#     train_history = pickle.dump(train_history, f)
# %%
# load train_history
with open('./logs/Memoryshare_level4_2023-05-23-00-17-26/train_history.pkl', 'rb') as f:
    train_history = pickle.load(f)
# %%

torch.eye(3)
# %%
plt.plot(agent_paths_length[1])

# %%
import pandas as pd

# 创建一个行名和列名为指定值的DataFrame，初始值全为0
rows = [(i, j) for i in range(17) for j in range(17)]
cols = ['↑', '→', '↓', '←']
df = pd.DataFrame(0, index=rows, columns=cols)

print(df)
# policy_nets[0]
# %%
df += policy_nets[0]
df
# %%
max_idx = df.idxmax(axis=1)
Q_policy = np.array(max_idx.to_list()).reshape(17, 17)
# %%
def val(Q_policy):
    MAZE = np.loadtxt('maze_cross_level4.txt')
    SIZE = MAZE.shape[0]
    MAX_STEPS_PER_EPISODE = 64
    VAL_EPISODES = 64
    
    env = Gridworld(size=SIZE, n_agents=2, heuristic_reward=True, maze=MAZE)
    num_states = SIZE * SIZE
    num_actions = 4

    # Initialize epsilon
    episode_durations = [[] for _ in range(env.n_agents)]
    agent_paths = [[] for _ in range(env.n_agents)]
    agent_rewards = [[] for _ in range(env.n_agents)]
    agent_paths_length = [[] for _ in range(env.n_agents)]

    val_history = {'episode_durations': episode_durations, \
                        'agent_paths': agent_paths, \
                        'agent_rewards': agent_rewards, \
                        'agent_paths_length': agent_paths_length}

    for i_episode in trange(VAL_EPISODES):
        # Initialize the environment and state
        env.reset()

        steps_done = [0 for _ in range(env.n_agents)]

        # Log intermediate variables
        rewards = [0.0 for _ in range(env.n_agents)]
        # Record the full history for each agent
        full_history_position = [[] for _ in range(env.n_agents)]

        # raise
        for t in count():
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
                break

            for idx in range(env.n_agents):
                # The agent might have been done
                if env.agents[idx]['done']:
                    continue

                # Select and perform an action
                state = env.get_state(idx)
                state_index = int(state[0] * SIZE + state[1])
                action = np.argmax(Q_policy[idx][state_index])
                
                # Perform action
                _, reward, done = env.step(idx, action)
                steps_done[idx] += 1  # Increment steps_done for this agent
                
                # Flatten next_state
                next_state = env.get_state(idx)
                next_state_index = int(next_state[0] * SIZE + next_state[1])

                # Record agent's path
                full_history_position[idx].append(env.agents[idx]['pos'])
                # Record agent's cumulative reward
                rewards[idx] += float(reward)

                state = next_state

                if done:
                    episode_durations[idx].append(t + 1)
                    agent_paths[idx].append(full_history_position[idx])
                    agent_paths_length[idx].append(len(full_history_position[idx]))
                    agent_rewards[idx].append(rewards[idx])     # Log cumulative reward
                    continue
    return val_history

val_history = val(policy_nets)
# %%
val_history
# %%
import numpy as np

# 假设 'Q1', 'Q2', 'Q3' 和 'Q_global' 是你的Q表
Q1 = np.random.rand(289, 4)
Q2 = np.random.rand(289, 4)
Q3 = np.random.rand(289, 4)
Q_global = np.random.rand(289, 4)

# 分别计算每个Agent的Q表与全局Q表的差值，得到DeltaQ
DeltaQ1 = Q1 - Q_global
DeltaQ2 = Q2 - Q_global
DeltaQ3 = Q3 - Q_global

# 将三张DeltaQ表堆叠到新的维度，形成一个新的数组，它的shape是(3, 289, 4)
DeltaQ = np.stack((DeltaQ1, DeltaQ2, DeltaQ3))

# 使用np.abs(DeltaQ)获取绝对值数组，然后在第0维上应用argmax函数获取绝对值最大的元素的索引
indices = np.abs(DeltaQ).argmax(axis=0)

# 使用np.take_along_axis获取绝对值最大的元素
max_DeltaQ = np.take_along_axis(DeltaQ, indices[np.newaxis, ...], axis=0)

# 打印结果数组的shape，它应该是(289, 4)
print(DeltaQ1)
print('-----------------')
print(DeltaQ2)
print('-----------------')
print(DeltaQ3)
print('-----------------')
print(max_DeltaQ)
print('-----------------')
print(max_DeltaQ.shape)
# %%
MAZE_ = np.loadtxt('maze17_0.2.txt')
plt.imshow(MAZE_)
# %%
a = 9
b = 20

for i in range(100):
    a = 0.9*a + 0.1*b + 5
    print(a)
# %%
EPS_START = 1
EPS_END = 1/3
EPS_DECAY = 5000
progress = []
total_rounds = 50000
for current_round in range(total_rounds):
    if current_round < EPS_DECAY:
        epsilon = EPS_START - ((EPS_START - EPS_END) * current_round / EPS_DECAY)
    else:
        epsilon = EPS_END
    progress.append(epsilon)

plt.plot(progress)

