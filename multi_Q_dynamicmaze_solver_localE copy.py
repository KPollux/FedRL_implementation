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
from cross_gridworld import Gridworld
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
          FLAll=False, FLdelta=False, FLDynamicAvg=False, FLGreedEpsilon=False, floder_name=None, n_times=0, dynamic=False):
    # Hyperparameters
    EPISODES = 1000
    EPSILON = 0.1  # Random rate
    MAX_STEPS_PER_EPISODE = 512  # Maximum steps per episode
    GAMMA = 0.99
    LR = 0.1  # Learning rate
    MAZE = np.loadtxt('maze17_0.2.txt')  # maze_cross_level4.txt  maze17_0.2.txt
    SIZE = MAZE.shape[0]

    FL_LOCAL_EPOCH = 8

    # Create environment
    n_agents = 3
    reward_dict=[{'step': -1, 'collision': -10, 'goal': 50, 'heuristic': False},
                # {'step': 0, 'collision': 0, 'goal': 0, 'heuristic': False},
                {'step': -1, 'collision': -10, 'goal': 50, 'heuristic': False},
                {'step': -1, 'collision': -10, 'goal': 50, 'heuristic': False}
                ]

    # reward_dict=[{'step': 0, 'collision': -1, 'goal': 10, 'heuristic': False},
    #         # {'step': 0, 'collision': 0, 'goal': 0, 'heuristic': False},
    #         {'step': 0, 'collision': -1, 'goal': 10, 'heuristic': False},
    #         {'step': 0, 'collision': -1, 'goal': 10, 'heuristic': False}
    #         ]
    if dynamic:
        wall_positions=[(16, 9), (15, 9), (14, 12)]
        wall_odds=[0.9, 0.7, 0.6]
    else:
        wall_positions = wall_odds = None
    # env = MeetingGridworld(size=SIZE, n_agents=2, heuristic_reward=True, maze=MAZE)
    env = Gridworld(size=SIZE, n_agents=n_agents, heuristic_reward=True, maze=MAZE,
                     reward_dict=reward_dict,
                     dynamic=dynamic, wall_positions=wall_positions, wall_odds=wall_odds)

    num_states = SIZE * SIZE
    num_actions = 4

    # Create Q-table
    Q_tables = [np.zeros([num_states, num_actions]) for _ in range(env.n_agents)]
    if share_params:
        Q_tables = [Q_tables[0] for _ in range(env.n_agents)]
    global_Q = np.zeros([num_states, num_actions])
    
    # Initialize epsilon
    episode_durations = [[] for _ in range(env.n_agents)]
    agent_paths = [[] for _ in range(env.n_agents)]
    agent_rewards = [[] for _ in range(env.n_agents)]
    agent_paths_length = [[] for _ in range(env.n_agents)]

    train_history = {'episode_durations': episode_durations, \
                        'agent_paths': agent_paths, \
                        'agent_rewards': agent_rewards, \
                        'agent_paths_length': agent_paths_length}

    deltas_dict = {'episodes': [], 'deltas_1': [], 'deltas_2': [], 'deltas_3': [], 'deltas_global': []}

    FL_steps = 0
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
                _, reward, done = env.step(idx, action)
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
                if onlyA and idx!=0:
                    pass
                else:
                    Q_tables[idx][state_index, action] = \
                        Q_tables[idx][state_index, action] + LR * (reward + GAMMA * np.max(Q_tables[idx][next_state_index]) - Q_tables[idx][state_index, action])
                state = next_state

                if done:
                    episode_durations[idx].append(t + 1)
                    agent_paths[idx].append(full_history_position[idx])
                    agent_paths_length[idx].append(len(full_history_position[idx]))
                    agent_rewards[idx].append(rewards[idx])     # Log cumulative reward
                    continue
        
            if FL:
                if t % FL_LOCAL_EPOCH == 0:
                    global_Q = copy.deepcopy(Q_tables[0])
                    for idx in range(env.n_agents):
                        if idx == 0:
                            global_Q = copy.deepcopy(Q_tables[0]) / env.n_agents
                        else:
                            global_Q += copy.deepcopy(Q_tables[idx]) / env.n_agents

                    for idx in range(env.n_agents):
                        Q_tables[idx] = copy.deepcopy(global_Q)
            
            elif FLdelta:
                if t % FL_LOCAL_EPOCH == 0:
                    # 分别计算每个Agent的Q表与全局Q表的差值，得到DeltaQ
                    delta_Qs = [Q_tables[i] - global_Q for i in range(env.n_agents)]             

                    for dQ in delta_Qs:
                        global_Q += dQ / env.n_agents

                    # 更新每个Q表
                    for idx in range(env.n_agents):
                        Q_tables[idx] = copy.deepcopy(global_Q)

            elif FLMax:
                if t % FL_LOCAL_EPOCH == 0:
                    # 分别计算每个Agent的Q表与全局Q表的差值，得到DeltaQ
                    delta_Qs = [Q_tables[i] - global_Q for i in range(env.n_agents)]
                    delta_Qs = np.array(delta_Qs)

                    # 使用np.abs(DeltaQ)获取绝对值数组，然后在第0维上应用argmax函数获取绝对值最大的元素的索引
                    indices = np.abs(delta_Qs).argmax(axis=0)

                    # 使用np.take_along_axis获取绝对值最大的元素
                    max_DeltaQs = np.take_along_axis(delta_Qs, indices[np.newaxis, ...], axis=0)                 

                    global_Q += max_DeltaQs[0]

                    # 更新每个Q表
                    for idx in range(env.n_agents):
                        Q_tables[idx] = copy.deepcopy(global_Q)

            elif FLAll:
                if t % FL_LOCAL_EPOCH == 0:
                    # 分别计算每个Agent的Q表与全局Q表的差值，得到DeltaQ
                    delta_Qs = [Q_tables[i] - global_Q for i in range(env.n_agents)]
                    # delta_Qs = np.array(delta_Qs)

                    # 使用np.abs(DeltaQ)获取绝对值数组，然后在第0维上应用argmax函数获取绝对值最大的元素的索引
                    # indices = np.abs(delta_Qs).argmax(axis=0)

                    # 使用np.take_along_axis获取绝对值最大的元素
                    # max_DeltaQs = np.take_along_axis(delta_Qs, indices[np.newaxis, ...], axis=0)                 

                    for dQ in delta_Qs:
                        global_Q += dQ

                    # 更新每个Q表
                    for idx in range(env.n_agents):
                        Q_tables[idx] = copy.deepcopy(global_Q)

            elif FLDynamicAvg:
                if i_episode % FL_LOCAL_EPOCH == 0:
                    # Initialize each agent's weight as 1
                    weights = np.ones(env.n_agents)

                    # Define the starting and ending epsilon values
                    EPS_START = 1.0
                    EPS_END = 1.0 / env.n_agents

                    # Define the decay rate
                    EPS_DECAY = 1000

                    # Assuming that `current_round` is the current training round
                    if i_episode < EPS_DECAY:
                        epsilon = EPS_START - ((EPS_START - EPS_END) * (i_episode / EPS_DECAY)**2)
                        # epsilon = EPS_START
                    else:
                        epsilon = EPS_END

                    # Adjust the weight of each agent
                    weights = weights * epsilon
                    # deltas_dict['weights'].append(weights)

                    # Before updating the global Q table, calculate the change rate of each agent's Q table
                    delta_Qs = [Q_tables[i] - global_Q for i in range(env.n_agents)] 

                    # Then use the adjusted weights to update the global Q table
                    for i in range(env.n_agents):
                        global_Q += delta_Qs[i] * weights[i]

                    # Finally, update each Q table
                    for idx in range(env.n_agents):
                        Q_tables[idx] = copy.deepcopy(global_Q)

            elif FLGreedEpsilon:
                if i_episode % FL_LOCAL_EPOCH == 0:

                    # 权重衰减
                    # Initialize each agent's weight as 1
                    weights = np.ones(env.n_agents)

                    # Define the starting and ending epsilon values
                    EPS_START = 1.0
                    EPS_END = 0.0 # 1.0 / env.n_agents

                    # Define the decay rate
                    EPS_DECAY = 200

                    # Assuming that `current_round` is the current training round
                    if i_episode < EPS_DECAY:
                        epsilon = EPS_START - ((EPS_START - EPS_END) * ((i_episode + 1) / EPS_DECAY))
                        # epsilon = EPS_START
                    else:
                        epsilon = EPS_END

                    # Adjust the weight of each agent
                    weights = weights * epsilon
                    # deltas_dict['weights'].append(weights)

                    # 计算MaxQ
                    # 分别计算每个Agent的Q表与全局Q表的差值，得到DeltaQ
                    delta_Qs = [Q_tables[i] - global_Q for i in range(env.n_agents)]
                    delta_Qs = np.array(delta_Qs)

                    # 使用np.abs(DeltaQ)获取绝对值数组，然后在第0维上应用argmax函数获取绝对值最大的元素的索引
                    indices = np.abs(delta_Qs).argmax(axis=0)

                    # 使用np.take_along_axis获取绝对值最大的元素
                    max_DeltaQs = np.take_along_axis(delta_Qs, indices[np.newaxis, ...], axis=0)  
                              
                    # 更新每个Q表
                    for idx in range(env.n_agents):
                        Q_tables[idx] = copy.deepcopy(global_Q)

                    # Then use the adjusted weights to update the global Q table
                    for i in range(env.n_agents):
                        global_Q += delta_Qs[i] / env.n_agents * (1 - weights[i]) + max_DeltaQs[0] * weights[i]

                    # Finally, update each Q table
                    for idx in range(env.n_agents):
                        Q_tables[idx] = copy.deepcopy(global_Q)

            
            elif FLdev:
                if t % FL_LOCAL_EPOCH == 0 and i_episode != 0:

                    # deltas_dict
                    # 分别计算每个Agent的Q表与全局Q表的差值，得到DeltaQ
                    delta_Qs = [Q_tables[i] - global_Q for i in range(env.n_agents)]
                    delta_Qs = np.array(delta_Qs)

                    # 使用np.abs(DeltaQ)获取绝对值数组，然后在第0维上应用argmax函数获取绝对值最大的元素的索引
                    indices = np.abs(delta_Qs).argmax(axis=0)

                    # 使用np.take_along_axis获取绝对值最大的元素
                    max_DeltaQs = np.take_along_axis(delta_Qs, indices[np.newaxis, ...], axis=0)                 

                    global_Q += max_DeltaQs[0]
            
                    # 全局Q表更新，根据DeltaQ的本地更新差
                    dQs = 0
                    for dQ in delta_Qs:
                        dQs += np.abs(dQ)
                    dQs = np.average(dQs)
                    
                    deltas_dict['episodes'].append(i_episode)
                    deltas_dict['deltas_1'].append(np.average(np.abs(delta_Qs[0])))
                    deltas_dict['deltas_2'].append(np.average(np.abs(delta_Qs[1])))
                    deltas_dict['deltas_3'].append(np.average(np.abs(delta_Qs[2])))
                    deltas_dict['deltas_global'].append(dQs)

                    # 更新每个Q表
                    for idx in range(env.n_agents):
                        Q_tables[idx] = copy.deepcopy(global_Q)


                    

        # print(f'i_episode{i_episode}, steps_done{[agent_paths_length[k][-1] for k in range(2)]}, rewards{rewards}')
        # plot_rewards(agent_paths_length)
    
    #save models
    # name the save floder

    if floder_name is None:
        prefix = 'Q_learning_M1720_noHRwd_5flod_'

        floder_name = prefix  # M17_20_all_delta_

        floder_name += 'dynamic_' if dynamic else ''
        
        floder_name += 'FL_' if FL else ''
        floder_name += 'FLMax_' if FLMax else ''
        floder_name += 'FLAll_' if FLAll else ''
        floder_name += 'FLDelta_' if FLdelta else ''
        floder_name += 'FLdev_' if FLdev else ''
        floder_name += 'FLDynamicAvg_' if FLDynamicAvg else ''
        floder_name += 'FLGreedEpsilon_' if FLGreedEpsilon else ''

        floder_name += '{}LStep_'.format(FL_LOCAL_EPOCH) if 'FL' in floder_name else ''

        floder_name += 'Paramsshare_' if share_params else ''

        floder_name += 'onlyA_' if onlyA else ''
        # floder_name += 'role_' if role else ''
        # floder_name += 'Memoryshare_' if share_memory else ''

        floder_name += 'ep{}_'.format(EPISODES)

        if floder_name == prefix:
            floder_name += 'Independent_'
        floder_name += time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    # create floder
    if not os.path.exists('./logs/' + floder_name):
        print('create floder: ', floder_name)
        os.mkdir('./logs/' + floder_name)

    for i in range(env.n_agents):
        # save Q_tables
        with open('./logs/{}/Q_table_{}_{}.pkl'.format(floder_name, i, n_times), 'wb') as f:
            pickle.dump(Q_tables[i], f)
    
    with open('./logs/{}/train_history_{}.pkl'.format(floder_name, n_times), 'wb') as f:
        pickle.dump(train_history, f)

    return train_history, Q_tables, floder_name  # , deltas_dict

# dymnamic = False
# floder_name = None
# for n in trange(5):
#     train_history, policy_nets, floder_name = main(FLMax=True,
#                                                 dynamic=dymnamic,
#                                                 floder_name=floder_name, n_times=n)

# %%
dymnamic = False

floder_name = None
for n in trange(5):
    train_history, policy_nets, floder_name = main(FLGreedEpsilon=True,
                                                dynamic=True,
                                                floder_name=floder_name, n_times=n)

# %%
dymnamic = True

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
EPISODES = 1000
env_size = 17

agent_rewards = train_history['agent_rewards']
agent_paths_length = train_history['agent_paths_length']
agent_paths = train_history['agent_paths']

# %%
draw_history(agent_rewards, agent_paths_length, n_agents, EPISODES, window_size=64)
agent_paths_length


# %%
# print(np.array(policy_nets).shape)

q_tables = np.array(policy_nets)

# 假设你的Q表存在一个名为q_tables的np数组中
q_tables = np.random.random((3, 289, 4)) # 这只是一个例子，你应该使用你的真实数据

# 对每个Q表中的最后一个维度（动作）进行argmax操作
max_actions = np.argmax(q_tables, axis=-1)

# 然后，我们将结果重塑为(3, 17, 17)
reshaped_max_actions = max_actions.reshape((3, 17, 17))

print(reshaped_max_actions)


# %%
# deltas_dict
# plt.plot(deltas_dict['deltas_1'])
# plt.plot(deltas_dict['deltas_2'])
# plt.plot(deltas_dict['deltas_global'])

m1 = moving_average(deltas_dict['deltas_1'], 1)
m2 = moving_average(deltas_dict['deltas_2'], 1)
m3 = moving_average(deltas_dict['deltas_3'], 1)
mg = moving_average(deltas_dict['deltas_global'], 1)
# plt.ylim(0, 0.004)
plt.figure(figsize=(10, 7.5))
# 打开网格
# plt.grid()

plt.plot(m1, label='agent1')
plt.plot(m2, label='agent2')
plt.plot(m3, label='agent3')
plt.plot(mg, label='global(add_all)')
plt.legend()

plt.title('Delta Q with communication rounds')
plt.xlabel('Communication Rounds')
plt.ylabel('MAE of delta Q')

plt.tight_layout()
plt.show()


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
import numpy as np

EPS_START = 1.0
EPS_END = 1.0 / 3

# Define the decay rate
EPS_DECAYs = [0, 200, 2500, 5000, 10000, 15000, 50000, np.inf]
legends = ['Decay-0(FedAvg)', 'Decay-200', 'Decay-2500', 'Decay-5000', 'Decay-10000', 'Decay-15000', 'Decay-50000', 'NoDecay']

EPISODES = 50000
for i, EPS_DECAY in enumerate(EPS_DECAYs):
    e_list = []
    for i_episode in range(EPISODES):
        # Assuming that `current_round` is the current training round
        if i_episode < EPS_DECAY:
            epsilon = EPS_START - ((EPS_START - EPS_END) * (i_episode / EPS_DECAY)**2)
            # epsilon = EPS_START
        else:
            epsilon = EPS_END
        e_list.append(epsilon)

    plt.plot(e_list, label=legends[i])
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Epsilon')
# %%
import numpy as np

EPS_START = 1.0
EPS_END = 1.0 / 3

# Define the decay rate

legends = ['Decay-0(FedAvg)', 'Decay-100', 'Decay-500', 'Decay-1000', 'NoDecay']
EPS_DECAYs = [0, 100, 500, 1000, np.inf]

EPISODES = 1000
for i, EPS_DECAY in enumerate(EPS_DECAYs):
    e_list = []
    for i_episode in range(EPISODES):
        # Assuming that `current_round` is the current training round
        if i_episode < EPS_DECAY:
            epsilon = EPS_START - ((EPS_START - EPS_END) * (i_episode / EPS_DECAY)**2)
            # epsilon = EPS_START
        else:
            epsilon = EPS_END
        e_list.append(epsilon)

    plt.plot(e_list, label=legends[i])
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Epsilon')
# %%
EPISODES = 1000
e_list = []

for i_episode in range(EPISODES):
    DECAY_RATIO = [0.1, 0.7, 0.2]
    one_third = 1 / 3

    # Define the episode thresholds for each decay phase
    first_phase_end = EPISODES * DECAY_RATIO[0]
    second_phase_end = EPISODES * (DECAY_RATIO[0] + DECAY_RATIO[1])

    if i_episode < first_phase_end:
        # First phase of episodes
        epsilon = 1 - ((1 - one_third) * (i_episode / first_phase_end))
    elif i_episode < second_phase_end:
        # Second phase of episodes
        epsilon = one_third
    else:
        # Last phase of episodes
        epsilon = one_third - ((one_third) * ((i_episode - second_phase_end) / (EPISODES - second_phase_end)))
    
    e_list.append(epsilon)

plt.plot(e_list)

# %%
DECAY_RATIO = [0.1, 0.7, 0.2]
str_decay_ratio = ''.join(map(str, [int(x * 10) for x in DECAY_RATIO]))
print(str_decay_ratio)
