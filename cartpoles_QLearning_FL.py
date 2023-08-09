# %%

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from easydict import EasyDict as edict
from tqdm import trange
import numpy as np

import asyncio

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
# %%
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
# %%
class FLServer:
    def __init__(self, env, agents, cfg, condition):
        self.cfg = cfg
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = env.action_space.n
        state, _ = env.reset()
        self.n_observations = len(state)
        self.global_model = DQN(self.n_observations, self.n_actions).to(self.device)
        self.agents = agents
        self.condition = condition
        self.i_episode = 0
        self.aggregate_method = cfg.aggregation_method

        self.weight_list = []

    def update_model(self, agent, i_episode):
        self.i_episode = i_episode
        self.weight_list.append(agent.policy_net.state_dict())

    def aggregate(self):
        new_weights = {}
        if self.aggregate_method == 'FedAvg':
            for key in self.weight_list[0].keys():
                new_weights[key] = torch.stack([model_weights[key] for model_weights in self.weight_list], dim=0).mean(dim=0)

            self.global_model.load_state_dict(new_weights)

        # elif self.aggregate_method == 'FedAll':
        #     for key in self.weight_list[0].keys():
        #         # 

        #     self.global_model.load_state_dict(new_weights)

        elif self.aggregate_method == 'FedGradual':
            weights = np.ones(cfg.num_agent)

            # Define the starting and ending epsilon values
            EPS_START = 1.0
            EPS_END = 1.0 / cfg.num_agent

            # Define the decay rate
            EPS_DECAY = 500

            # Assuming that `current_round` is the current training round
            if self.i_episode < EPS_DECAY:
                epsilon = EPS_START - ((EPS_START - EPS_END) * (self.i_episode / EPS_DECAY))
                # epsilon = EPS_START
            else:
                epsilon = EPS_END

            # Adjust the weight of each agent
            weights = weights * epsilon
            # deltas_dict['weights'].append(weights)
            # print(weights)

            global_weights = self.global_model.state_dict()

            # # 计算delta model weights
            # for key in self.weight_list[0].keys():
            #     # 计算每个设备模型权重与全局模型权重的差值，然后进行加权平均
            #     delta_weights[key] = torch.stack([w * (model_weights[key] - global_weights[key]) for w, model_weights in zip(weights, self.weight_list)], dim=0).sum(dim=0)
            
            delta_weights = {}
            for i in range(len(self.weight_list)):
                d_para = {}
                for key in self.weight_list[0].keys():
                    d_para[key] = global_weights[key] - self.weight_list[i][key]

                    delta_weights[key] = delta_weights.get(key, 0) + weights[i] * d_para[key]


            # 将平均后的差值加回到全局权重中，得到新的全局模型权重
            for key in global_weights.keys():
                global_weights[key] = global_weights[key] - delta_weights[key]
            
            self.global_model.load_state_dict(global_weights)


        self.weight_list = []

    async def run(self):
        # 如果还没有结束
        async with self.condition:
            while not all(agent.finish_train for agent in self.agents):
                # await asyncio.sleep(0.01)  # Polling delay
                # print('Server: Waiting for agents...')
                await self.condition.wait_for(lambda: all((agent.ready or agent.finish_train) for agent in self.agents))  # Wait for all agents to reach a sync point
                if all(agent.finish_train for agent in self.agents):
                    break
                # await self.condition.wait_for(lambda: all(agent.ready for agent in self.agents))  # Wait for all agents to reach a sync point

                if self.cfg['FL']:
                    # if all((agent.t % agent.cfg['FL_LOCAL_STEP'] == 0) for agent in self.agents):
                    # for agent in self.agents:
                    #     print(f'Agent {agent.id}: episode done in {agent.t}')
                    # 聚合所有Agent
                    # 上传
                    # tic = time.time()
                    for agent in self.agents:
                        self.update_model(agent, agent.i_episode)
                    # toc = time.time()
                    # print(f'Upload time: {toc-tic}')
                    # 聚合
                    # tic = time.time()
                    self.aggregate()
                    # toc = time.time()
                    # print(f'Aggregate time: {toc-tic}')
                    # 下载
                    # tic = time.time()
                    for agent in self.agents:
                        agent.policy_net.load_state_dict(self.global_model.state_dict())
                    # toc = time.time()
                    # print(f'Download time: {toc-tic}')

                # 继续运行
                for agent in self.agents:  # Reset all agents' ready status
                    agent.ready = False
                    # print(f'agent ready {agent.ready}')
                self.condition.notify_all()  # Wake up all agents
            # 如果所有agent都结束了，就结束server
                


# %%
class DQNAgent:
    def __init__(self, env, cfg, condition, id):
        self.cfg = cfg
        self.env = env
        self.id = id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = env.action_space.n
        state, _ = env.reset()
        self.n_observations = len(state)
        self.policy_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.cfg.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.episode_durations = []

        self.condition = condition
        self.ready = False
        self.finish_train = False
        self.optimize_t = 0

        self.current_step = 0

        self.i_episode = 0


    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.cfg.EPS_END + (self.cfg.EPS_START - self.cfg.EPS_END) * \
            math.exp(-1. * self.steps_done / self.cfg.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.cfg.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.cfg.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.cfg.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.cfg.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    async def train(self, num_episodes):
        for i_episode in trange(num_episodes):
            self.i_episode = i_episode
            # Initialize the environment and get it's state
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in count():

                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=device)
                self.done = terminated or truncated

                self.current_step += 1

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
                self.optimize_t += 1
                if self.optimize_t % self.cfg.local_step == 0:
                    async with self.condition:
                        self.ready = True
                        self.condition.notify_all()
                        # print(t)
                        # print('wait for server')
                        await self.condition.wait_for(lambda: not self.ready)  # Wait for the server
                        self.optimize_t = 0

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*cfg.TAU + target_net_state_dict[key]*(1-cfg.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                # if self.current_step % 100 == 0:
                #     self.target_net.load_state_dict(policy_net_state_dict)

                if self.done:
                    self.episode_durations.append(t + 1)
                    # plot_durations()
                    break
        # print('Complete')
        async with self.condition:
            self.finish_train = True
            self.condition.notify_all()

# 保证能跑
# 协程关键词
# 结束边界条件
# 保存训练过程代码
# %%
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
cfg = edict()

cfg.BATCH_SIZE = 128
cfg.GAMMA = 0.99
cfg.EPS_START = 0.9
cfg.EPS_END = 0.05
cfg.EPS_DECAY = 1000
cfg.TAU = 0.005
cfg.LR = 1e-4

cfg.num_agent = 3
cfg.local_step = 8


cfg.ShareQ = False
cfg.FL = True
METHODS = ['FedAvg', 'FedGradual']
cfg.aggregation_method = METHODS[1]



def plot_durations(episode_durations, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

if torch.cuda.is_available():
    num_episodes = 500
else:
    num_episodes = 50
# %%
# agents = [DQNAgent(gym.make("CartPole-v1"), cfg) for _ in range(cfg.num_agent)]
# agent = DQNAgent(gym.make("CartPole-v1"), cfg)
# agent.train(num_episodes)
condition = asyncio.Condition()
agents = [DQNAgent(gym.make("CartPole-v1"), cfg, condition, id) for id in range(cfg.num_agent)] 
server = FLServer(gym.make("CartPole-v1"), agents, cfg, condition)

for agent in agents:
    agent.policy_net.load_state_dict(server.global_model.state_dict())
    agent.target_net.load_state_dict(server.global_model.state_dict())

if cfg.ShareQ:
    for agent in agents:
        agent.policy_net = server.global_model
        agent.optimizer = optim.AdamW(agent.policy_net.parameters(), lr=agent.cfg.LR, amsgrad=True)
        # agent.target_net = server.global_model

async def train():
    await asyncio.gather(server.run(), *(agent.train(num_episodes) for agent in agents))

# %%

# for i in trange(cfg['training_episodes']):
await train()
    # asyncio.run(train()) # in Terminal
    # reward_temp = []
    # for agent in agents:
    #     reward_temp.append(agent.r_sum)
    #     agent.reset()
    # agent_rewards.append(reward_temp)

# %%

print('Complete')
plot_durations(agents[2].episode_durations, show_result=True)
plt.ioff()
plt.show()
# %%
import time
import os
import pickle

train_history = {'agent_rewards': [agent.episode_durations for agent in agents]}

floder_name = None
n_times = 0
if floder_name is None:
    prefix = 'CartPole_'

    floder_name = prefix  # M17_20_all_delta_

    # floder_name += 'dynamic_' if dynamic else ''
    if cfg['FL']:
        floder_name += 'FL_' if cfg['aggregation_method']=='FedAvg' else ''
        # floder_name += 'FLMax_' if FLMax else ''
        floder_name += 'FLAll_' if cfg['aggregation_method']=='QAll' else ''
        # floder_name += 'FLDelta_' if FLdelta else ''

        floder_name += 'FLDynamicAvg_' if cfg['aggregation_method']=='FedGradual' else ''
        # floder_name += 'FLRewardShape_' if FLRewardShape else ''

        # floder_name += 'FLdev_' if FLdev else ''

    # floder_name += '{}LStep_'.format(FL_LOCAL_EPOCH) if 'FL' in floder_name else ''

    floder_name += 'Paramsshare_' if cfg['ShareQ'] else ''

    # floder_name += 'onlyA_' if onlyA else ''
    # floder_name += 'role_' if role else ''
    # floder_name += 'Memoryshare_' if share_memory else ''

    # floder_name += 'ep{}_'.format(EPISODES)

    # floder_name += 'FLDynamicAvg_'  # 

    if floder_name == prefix:
        floder_name += 'Independent_'
    floder_name += time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

# create floder
if not os.path.exists('./logs/' + floder_name):
    print('create floder: ', floder_name)
    os.mkdir('./logs/' + floder_name)

# for i in range(1):
#     # save Q_tables
#     with open('./logs/{}/Q_table_{}_{}.pkl'.format(floder_name, i, n_times), 'wb') as f:
#         pickle.dump(agents[i].q_table, f)

with open('./logs/{}/train_history_{}.pkl'.format(floder_name, n_times), 'wb') as f:
    pickle.dump(train_history, f)
# %%
