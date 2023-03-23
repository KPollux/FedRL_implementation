import os
import random
import time

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import math
from collections import namedtuple, deque
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from games.GridWorld.GridWorldMeet import FedRLEnv
from games.GridWorld.setting import cfg_data


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
device = "cuda:0"


class DQN(nn.Module):
    def __init__(self, observation_size, maze_size, num_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(observation_size, maze_size * maze_size * 2)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(maze_size * maze_size * 2, maze_size * maze_size * 2)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(maze_size * maze_size * 2, num_actions)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.output_layer(x)
        return x


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


# def plot_rewards(episode_rewards, show_result=False, zero_point=None, ylabel='Rewards'):
#     plt.figure(1)
#     rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
#     if show_result:
#         plt.title('Result')
#     else:
#         plt.clf()
#         plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel(ylabel)
#     plt.plot(rewards_t.numpy())
#
#     if zero_point is None:
#         zero_point = (maze_size * maze_size * 0.5)
#
#     # Take 100 episode averages and plot them too
#     if len(rewards_t) >= 100:
#         means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99) - zero_point, means))
#         plt.plot(means.numpy())
#
#     plt.pause(0.001)  # pause a bit so that plots are updated
#     if is_ipython:
#         if not show_result:
#             display.display(plt.gcf())
#             display.clear_output(wait=True)
#         else:
#             display.display(plt.gcf())


class Agent:
    def __init__(self, agent_id, observation_size, cfg_):
        self.observation_size = observation_size
        self.cfg = cfg_

        self.policy_net = DQN(observation_size, self.cfg.maze_size, self.cfg.num_actions).to(device)
        self.target_net = DQN(observation_size, self.cfg.maze_size, self.cfg.num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.cfg.LR)
        self.agent_id = agent_id

        self.memory = ReplayMemory(int(maze.size * 0.5 / 0.04) * 2)

        self.episode_rewards = []
        self.episode_step = []

        self.steps_done = 0

    # 动作选取
    def select_action(self, state):
        sample = random.random()

        # 随着进行，eps_threshold逐渐降低
        # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        #     math.exp(-1. * steps_done / EPS_DECAY)

        self.steps_done += 1

        # 常规情况选择价值最高的动作
        if sample > self.cfg.eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                state = torch.tensor(state, dtype=torch.float32, device=device)
                return self.policy_net(state).max(1)[1].view(1, 1)

        # 当随机值超过阈值时，随机选取 - exploration
        else:
            # 探索时只探索可能的动作，增加探索效率？
            return torch.tensor([[random.choice(env.valid_actions(self.agent_id))]], device=device,
                                dtype=torch.long)

    def train(self):
        if len(self.memory) < self.cfg.BATCH_SIZE:
            return

        # 离线学习，从记忆池中抽取回忆
        transitions = self.memory.sample(self.cfg.BATCH_SIZE)
        # print(transitions)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.

        # 将([a, 1], [b, 2], [c, 3])转化为([a, b, c], [1, 2, 3])，一个zip的trick
        # 然后将他们分别放到tuples with names里（'state', 'action', 'next_state', and 'reward'）
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # 计算非最终状态的掩码，并将批处理元素连接起来
        # (最终状态是指模拟结束后的状态)
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
        # 模型计算Q价值，我们根据价值选择动作
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
        # 当前奖励+下一个状态的奖励，更新Q. 如果下一个状态为最终状态，则仅有当前奖励
        expected_state_action_values = (next_state_values * self.cfg.GAMMA) + reward_batch
        # print(expected_state_action_values)

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


from tqdm import trange


def evaluation():
    win = 0
    episode_rewards_eval = []

    env = FedRLEnv(maze, cfg_data)

    for j in range(1):

        # Initialize the environment and get it's state
        state, info = env.reset()
        # Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity
        state = torch.tensor(state, dtype=torch.float32, device=device)

        done = False
        while not done:
            action = policy_net(state).max(1)[1].view(1, 1)  # 选择一个动作
            # random.choice(env.valid_actions())
            observation, reward, done, info = env.step(action.item())  # 执行动作，返回{下一个观察值、奖励、是否结束、是否提前终止}
            # reward = torch.tensor([reward], device=device)
            # print(int(action[0][0]))
            # print(observation, reward, done, info)
            # print()
            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device)  # 如果没有终止则继续记录下一个状态

            # Store the transition in memory
            # memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

        episode_rewards_eval.append(env.total_reward)
        if info == 'win':
            win += 1

    win_rate = win / 1

    env.render()
    # print(env.visited)
    # print(env.state)
    # print(env.total_reward)

    return episode_rewards_eval, win_rate

# %%

# if __name__ == '__main__':
    # cfg = edict()
    # cfg.num_actions = 4
    # cfg.maze_size = 8
    # cfg.BATCH_SIZE = 32
    # cfg.GAMMA = 0.99
    # cfg.EPS_START = 0.9
    # cfg.EPS_END = 0.05
    # cfg.EPS_DECAY = 1000
    # cfg.eps_threshold = 0.1
    # cfg.TAU = 0.005
    # cfg.LR = 1e-3
    # cfg.num_episodes = 10000
    #
    # maze = np.loadtxt('maze8n_5.txt')
    #
env = FedRLEnv(maze, cfg_data)
print(env.reset())
    #
    # ENV_NAME = 'grid'
    # now = time.strftime("%m-%d_%H-%M-%S", time.localtime())
    # folder_name = f"runs/{ENV_NAME}/" + now
    # os.makedirs('runs/', exist_ok=True)
    # os.makedirs(f'runs/{ENV_NAME}/', exist_ok=True)
    # os.makedirs(folder_name, exist_ok=True)
    #
    # # tensorboard
    # writer = SummaryWriter(folder_name)

# [0, 1, 2, 3] [L, U, R, D]

env.render()

t_actions = {'alpha': 2, 'beta': 0}

t_observations, t_rewards, done, info = env.step(t_actions)

print(t_observations, t_rewards, done, info)

env.render()

# %%
t_actions = {'alpha': 2, 'beta': 0}
print(env.step(t_actions))
env.render()
# %%
t_actions = {'alpha': 3, 'beta': 1}
print(env.step(t_actions))
env.render()
# %%
t_actions = {'alpha': 2, 'beta': 1}
print(env.step(t_actions))
env.render()


# t_alpha_action = 0
# t_beta_action = 3
#
# t_actions = {'alpha': t_alpha_action, 'beta': t_beta_action}

# EXECUTE ACTION
# t_new_observations, t_rewards, done, infos = env.step(t_actions)
#
# print(t_new_observations)
# print(t_rewards)



# map = env.render()

# plt.imshow(map)
# plt.show()

# print(env.agents[0].x)


# print(map)

NUMBER_OF_GAMES = 10
