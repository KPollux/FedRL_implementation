# %%
import copy
from itertools import count
import os
import pickle
import time
import gym
import numpy
import random
from os import system, name
from time import sleep

import numpy as np
import random
import gym

from tqdm import trange
from taxi_grid_world import TaxiEnv

class FLServer:
    def __init__(self, env, aggregation_method='QAvg'):
        self.agent_q_tables = []

        # not_pr_acts = 2 + 1 + 1 + 1   # gotoS,D + put + get + root (非原子动作数目)
        nA = env.action_space_n  # 动作总数 6+5=11      
        nS = env.observation_space_n  # 状态空间大小 500

        self.global_q_table = np.zeros([nS, nA])
        self.aggregation_method = aggregation_method
        self.i_episode = 0

    def update_q_table(self, agent_q_table, i_episode):
        self.i_episode = i_episode
        self.agent_q_tables.append(agent_q_table)
        if len(self.agent_q_tables) == NUMBER_OF_AGENTS: 
            self.aggregate_q_tables()

    def aggregate_q_tables(self):
        if self.aggregation_method == 'QAvg':
            self.global_q_table = np.mean(self.agent_q_tables, axis=0)  # Aggregates the Q-tables by taking the mean.

        elif self.aggregation_method == 'QAll':
            # 分别计算每个Agent的Q表与全局Q表的差值，得到DeltaQ
            delta_Qs = [agent_q_table - self.global_q_table for agent_q_table in self.agent_q_tables]
            # 将所有DeltaQ相加，得到新的全局Q表
            for dQ in delta_Qs:
                self.global_q_table += dQ
            
        elif self.aggregation_method == 'QDynamicAvg':
            # Initialize each agent's weight as 1
            weights = np.ones(NUMBER_OF_AGENTS)

            # Define the starting and ending epsilon values
            EPS_START = 1.0
            EPS_END = 1.0 / NUMBER_OF_AGENTS

            # Define the decay rate
            EPS_DECAY = 15000

            # Assuming that `current_round` is the current training round
            if self.i_episode < EPS_DECAY:
                epsilon = EPS_START - ((EPS_START - EPS_END) * (self.i_episode / EPS_DECAY)**2)
                # epsilon = EPS_START
            else:
                epsilon = EPS_END

            # Adjust the weight of each agent
            weights = weights * epsilon
            # deltas_dict['weights'].append(weights)

            # Before updating the global Q table, calculate the change rate of each agent's Q table
            delta_Qs = [agent_q_table - self.global_q_table for agent_q_table in self.agent_q_tables]

            # Then use the adjusted weights to update the global Q table
            for i in range(NUMBER_OF_AGENTS):
                self.global_q_table += delta_Qs[i] * weights[i]
        else:
            raise ValueError('Invalid aggregation method.')
            
        self.agent_q_tables = []  # Clears the list of agent Q-tables.

    def distribute_q_table(self):
        return self.global_q_table

class Agent:
    def __init__(self, env, server, alpha=0.1, gamma=0.6, epsilon=0.1):
        self.env = env
        self.server = server
        self.q_table = np.zeros([env.observation_space_n, env.action_space_n])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_count = 0
        self.done = False
        self.state = self.env.reset()
        self.log_rewards = 0

    def step(self):
        if self.done:
            return

        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.env.action_space)
        else:
            action = np.argmax(self.q_table[self.state])

        next_state, reward, done, info = self.env.step(action) 
        self.done = done

        old_value = self.q_table[self.state, action]
        next_max = np.max(self.q_table[next_state])

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[self.state, action] = new_value

        self.state = next_state

        self.log_rewards += reward

        # self.update_count += 1
        # if self.update_count % FL_LOCAL_STEP == 0:
        #     self.server.update_q_table(self.q_table)
    
    def reset(self):
        self.state = self.env.reset()
        self.done = False
        self.log_rewards = 0

# %% 
# maze_cross = np.loadtxt('maze17_0.2.txt')
# env = TaxiEnv(maze_cross)
# env.reset()

# env.step(0)
# random.choice(env.action_space)
# %%
maze_cross = np.loadtxt('maze17_0.2.txt')
# maze_cross = np.loadtxt('mazeTaxi5.txt')
env = TaxiEnv
FL = True
ShareQ = False
FL_LOCAL_STEP = 8
MAX_LOCAL_STEP = 512
NUMBER_OF_AGENTS = 3
training_episodes = 15000

server = FLServer(env(maze_cross), aggregation_method='QAll')
agents = [Agent(env(maze_cross), server) for _ in range(NUMBER_OF_AGENTS)] 

agent_rewards = []
for agent in agents:
    server.update_q_table(agent.q_table, 0)
for agent in agents:
    agent.q_table = copy.deepcopy(server.distribute_q_table())

if ShareQ:
    for agent in agents:
        agent.q_table = agents[0].q_table

# agents[0].q_table
# server.agent.q_table
# %%

for i in trange(training_episodes):

    for t in count():
        # 所有agent运行一步
        for agent in agents:
            # # 如果距离MAX_LOCAL_STEP还足够运行FL_LOCAL_STEP
            # if t + FL_LOCAL_STEP <= MAX_LOCAL_STEP:
            #     agent.step(FL_LOCAL_STEP)
            # else:
            #     agent.step(MAX_LOCAL_STEP - t)
            agent.step()

        # if all(agent.update_count % FL_LOCAL_STEP == 0 for agent in agents):
        if t % FL_LOCAL_STEP == 0 and FL:
            # 聚合所有Agent
            for agent in agents:
                server.update_q_table(agent.q_table, i)
            # if len(server.agent_q_tables) == NUMBER_OF_AGENTS:
            #     server.aggregate_q_tables()
            for agent in agents:
                agent.q_table = copy.deepcopy(server.distribute_q_table())

        if all(agent.done for agent in agents) or t > MAX_LOCAL_STEP:
            reward_temp = []
            for agent in agents:
                reward_temp.append(agent.log_rewards)
                agent.reset()
            agent_rewards.append(reward_temp)
            break

print("Training finished.\n")




# %%
import matplotlib.pyplot as plt
import seaborn as sns

def moving_average(data_list, window_size=100):
    moving_averages = []
    cumsum = [0]
    for i, x in enumerate(data_list, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=window_size:
            moving_aver = (cumsum[i] - cumsum[i-window_size])/window_size
            moving_averages.append(moving_aver)
    return moving_averages

agent_rewards = np.array(agent_rewards)
sum_list = agent_rewards[..., 1]

sns.set(style='darkgrid')
sns.set(font_scale=1.5)
plt.figure(figsize=(15, 7.5))
sum_list = moving_average(sum_list)
plt.plot(sum_list)
plt.xlabel('episode num')
plt.ylabel('points')
plt.show()

train_history = {'agent_rewards': agent_rewards}
# %%
floder_name = None
n_times = 0
if floder_name is None:
    prefix = 'Taxi17_Q_learning_'

    floder_name = prefix  # M17_20_all_delta_

    # floder_name += 'dynamic_' if dynamic else ''
    
    # floder_name += 'FL_' if FL else ''
    # floder_name += 'FLMax_' if FLMax else ''
    # floder_name += 'FLAll_' if FLAll else ''
    # floder_name += 'FLDelta_' if FLdelta else ''

    # floder_name += 'FLDynamicAvg_' if FLDynamicAvg else ''
    # floder_name += 'FLRewardShape_' if FLRewardShape else ''

    # floder_name += 'FLdev_' if FLdev else ''

    # floder_name += '{}LStep_'.format(FL_LOCAL_EPOCH) if 'FL' in floder_name else ''

    # floder_name += 'Paramsshare_' if share_params else ''

    # floder_name += 'onlyA_' if onlyA else ''
    # floder_name += 'role_' if role else ''
    # floder_name += 'Memoryshare_' if share_memory else ''

    # floder_name += 'ep{}_'.format(EPISODES)

    floder_name += 'QAll_'

    # if floder_name == prefix:
    #     floder_name += 'Independent_'
    floder_name += time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

# create floder
if not os.path.exists('./logs/' + floder_name):
    print('create floder: ', floder_name)
    os.mkdir('./logs/' + floder_name)

for i in range(NUMBER_OF_AGENTS):
    # save Q_tables
    with open('./logs/{}/Q_table_{}_{}.pkl'.format(floder_name, i, n_times), 'wb') as f:
        pickle.dump(agents[i].q_table, f)

with open('./logs/{}/train_history_{}.pkl'.format(floder_name, n_times), 'wb') as f:
    pickle.dump(train_history, f)

# %%
e = env(maze_cross)
e.reset()
print(e.render())

# %%
