import os
import random
import time
from itertools import count

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
        observation_size = observation_size ** 2
        self.observation_size = observation_size
        self.cfg = cfg_

        self.policy_net = DQN(observation_size, self.cfg.maze_size, self.cfg.num_actions).to(device)
        self.target_net = DQN(observation_size, self.cfg.maze_size, self.cfg.num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.cfg.LR)
        self.agent_id = agent_id

        self.memory = ReplayMemory(10000)

        self.episode_rewards = []
        self.episode_step = []

        self.steps_done = 0

    # 动作选取
    def select_action(self, state):
        sample = random.random()

        # 随着进行，eps_threshold逐渐降低
        eps_threshold = self.cfg.EPS_END + (self.cfg.EPS_START - self.cfg.EPS_END) * \
            math.exp(-1. * self.steps_done / self.cfg.EPS_DECAY)

        # eps_threshold = self.cfg.eps_threshold

        self.steps_done += 1

        # 常规情况选择价值最高的动作
        if sample > eps_threshold:
            # print('利用')
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # print(state)
                # state = torch.tensor(state, dtype=torch.float32, device=device).reshape((1, -1))
                # print(state)
                return self.policy_net(state).max(1)[1].view(1, 1)

        # 当随机值超过阈值时，随机选取 - exploration
        else:
            # print('随机')
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

    # soft_sync_target_net
    def soft_sync_target_net(self):
        # self.target_net.load_state_dict(self.policy_net.state_dict())
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        # if (steps_done % sync_target_net_freq) == 0:
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.cfg.TAU \
                                         + target_net_state_dict[key] * (1 - self.cfg.TAU)

        self.target_net.load_state_dict(target_net_state_dict)

from tqdm import trange


def evaluation():
    win = 0
    # episode_rewards_eval = []
    episode_rewards_alpha_eval = []
    episode_rewards_beta_eval = []
    episode_step_eval = []

    for j in range(1):
        # Initialize the environment and get it's state
        state, info = env_eval.reset()

        done = False
        while not done:
            # 环境观察到的两个状态分别分配给两个agent
            state_alpha = torch.tensor(state['alpha'], dtype=torch.float32, device=device).reshape((1, -1))
            state_beta = torch.tensor(state['beta'], dtype=torch.float32, device=device).reshape((1, -1))

            with torch.no_grad():
                action_alpha = Agent_alpha.policy_net(state_alpha).max(1)[1].view(1, 1)  # 选择一个动作
                action_beta = Agent_beta.policy_net(state_beta).max(1)[1].view(1, 1)  # 选择一个动作

            t_actions = {'alpha': action_alpha.item(), 'beta': action_beta.item()}

            # random.choice(env.valid_actions())
            t_observations, t_rewards, done, info = env_eval.step(t_actions)  # 执行动作，返回{下一个观察值、奖励、是否结束、是否提前终止}
            # observation_alpha = torch.tensor(t_observations['alpha'], dtype=torch.float32, device=device).reshape((1, -1))
            # observation_beta = torch.tensor(t_observations['beta'], dtype=torch.float32, device=device).reshape((1, -1))
            #
            # if done:
            #     next_state_alpha = next_state_beta = None
            # else:  # 如果没有终止则继续记录下一个状态
            #     next_state_alpha = observation_alpha
            #     next_state_beta = observation_beta

            state = t_observations
            # # Store the transition in memory
            # # memory.push(state, action, next_state, reward)
            #
            # # Move to the next state
            # state = t_observations

        episode_rewards_alpha_eval.append(env_eval.agent_dict.alpha.total_reward)
        episode_rewards_beta_eval.append(env_eval.agent_dict.beta.total_reward)
        episode_step_eval.append(env_eval.total_Tstep)
        if info == 'win':
            win += 1

    win_rate = win / 1

    # env.render()
    # print(env.visited)
    # print(env.state)
    # print(env.total_reward)

    return [episode_rewards_alpha_eval, episode_rewards_beta_eval, episode_step_eval], win_rate


# %%

if __name__ == '__main__':
    # %%
    cfg = edict()
    cfg.num_actions = 4
    cfg.maze_size = 8
    cfg.BATCH_SIZE = 128
    cfg.GAMMA = 0.99
    cfg.EPS_START = 0.9
    cfg.EPS_END = 0.05
    cfg.EPS_DECAY = 1000
    cfg.eps_threshold = 0.1
    cfg.TAU = 0.005
    cfg.LR = 1e-4
    cfg.num_episodes = 500

    # maze = np.loadtxt('maze8n_2.txt')
    maze = np.loadtxt('games/GridWorld/maze8_0.1_5.txt')
    #
    env = FedRLEnv(maze, cfg_data)

    env_eval = FedRLEnv(maze, cfg_data)
    env_eval.max_Tstep = 38

    # [0, 1, 2, 3] [L, U, R, D]


# %%
    # print(env.reset())
    #
    ENV_NAME = 'grid-double-agent'
    # now = time.strftime("%m-%d_%H-%M-%S", time.localtime())
    # folder_name = f"runs/{ENV_NAME}/" + now
    # os.makedirs('runs/', exist_ok=True)
    # os.makedirs(f'runs/{ENV_NAME}/', exist_ok=True)
    # os.makedirs(folder_name, exist_ok=True)
    #
    # # tensorboard
    # writer = SummaryWriter(folder_name)

    Agent_alpha = Agent('alpha', 3, cfg)
    Agent_beta = Agent('beta', 5, cfg)

    episode_rewards_alpha = []
    episode_rewards_beta = []
    episode_step = []

    for i_episode in range(cfg.num_episodes):
        # Initialize the environment and get it's state
        state, info = env.reset()

        # 环境观察到的两个状态分别分配给两个agent
        # state = torch.tensor(state, dtype=torch.float32, device=device).reshape((1, -1))
        state_alpha = torch.tensor(state['alpha'], dtype=torch.float32, device=device).reshape((1, -1))
        state_beta = torch.tensor(state['beta'], dtype=torch.float32, device=device).reshape((1, -1))

        done = False
        for t in count():
            action_alpha = Agent_alpha.select_action(state_alpha)  # 选择一个动作
            action_beta = Agent_beta.select_action(state_beta)  # 选择一个动作

            # print(action_alpha, action_beta)
            t_actions = {'alpha': action_alpha.item(), 'beta': action_beta.item()}
            # print(t_actions)
            # random.choice(env.valid_actions())
            t_observations, t_rewards, done, info = env.step(t_actions)  # 执行动作，返回{下一个观察值、奖励、是否结束、是否提前终止}
            # print(t_observations, t_rewards, done, _)
            # break
            reward_alpha = torch.tensor([t_rewards['alpha']], device=device)
            reward_beta = torch.tensor([t_rewards['beta']], device=device)

            observation_alpha = torch.tensor(t_observations['alpha'], dtype=torch.float32, device=device).reshape((1, -1))
            observation_beta = torch.tensor(t_observations['beta'], dtype=torch.float32, device=device).reshape((1, -1))

            if done:
                next_state_alpha = next_state_beta = None
            else:  # 如果没有终止则继续记录下一个状态
                next_state_alpha = observation_alpha
                next_state_beta = observation_beta

            # print(reward_alpha, reward_beta)
            # print(observation_alpha, observation_beta)
            # print(next_state_alpha, next_state_beta)

            # break
            # Store the transition in memory
            Agent_alpha.memory.push(state_alpha, action_alpha, next_state_alpha, reward_alpha)
            Agent_beta.memory.push(state_beta, action_beta, next_state_beta, reward_beta)


            # Move to the next state
            state = t_observations  # state = next_state

            # Perform one step of the optimization (on the policy network)
            Agent_alpha.train()
            Agent_beta.train()

            # Soft update of the target network's weights
            Agent_alpha.soft_sync_target_net()
            Agent_beta.soft_sync_target_net()

            if done:
                break

        Agent_alpha.episode_rewards.append(env.agent_dict.alpha.total_reward)
        Agent_beta.episode_rewards.append(env.agent_dict.beta.total_reward)

        Agent_alpha.episode_step.append(env.total_Tstep)
        Agent_beta.episode_step.append(env.total_Tstep)

        _, win_eval = evaluation()

        print('Episode {}\tLast num step: {:.2f}\tLast reward Alpha: {:.2f}\t'
              'Last reward Beta: {:.2f}\tInfo: {}\tEval: {}'
              .format(i_episode, env.total_Tstep, env.agent_dict.alpha.total_reward,
                      env.agent_dict.beta.total_reward, info, win_eval))

        # print('')
        # if i_episode % 20 == 0:
    plt.title('alpha episode_rewards')
    plt.xlabel('episode')
    plt.ylabel('episode_rewards')
    plt.plot(Agent_alpha.episode_rewards)
    plt.show()

    plt.title('alpha episode_step')
    plt.xlabel('episode')
    plt.ylabel('episode_step')
    plt.plot(Agent_alpha.episode_step)
    plt.show()

    plt.title('beta episode_rewards')
    plt.xlabel('episode')
    plt.ylabel('episode_rewards')
    plt.plot(Agent_beta.episode_rewards)
    plt.show()

    plt.title('beta episode_step')
    plt.xlabel('episode')
    plt.ylabel('episode_step')
    plt.plot(Agent_beta.episode_step)
    plt.show()

    env.render()
    env_eval.render()




    # eps_threshold = []
    # samples = []
    # for steps_done in range(cfg.num_episodes):
    #     eps_threshold.append(cfg.EPS_END + (cfg.EPS_START - cfg.EPS_END) * \
    #                          math.exp(-1. * steps_done / cfg.EPS_DECAY))
    #     samples.append(random.random())
    # plt.plot([_ for _ in range(cfg.num_episodes)], samples)
    # plt.plot(eps_threshold)
    # plt.show()

    # [0, 1, 2, 3] [L, U, R, D]
    #
    # env.render()
    #
    # t_actions = {'alpha': 2, 'beta': 0}
    #
    # t_observations, t_rewards, done, info = env.step(t_actions)
    #
    # print(t_observations, t_rewards, done, info)
    #
    # env.render()
    #
    # # %%
    # t_actions = {'alpha': 2, 'beta': 0}
    # print(env.step(t_actions))
    # env.render()
    # # %%
    # t_actions = {'alpha': 3, 'beta': 1}
    # print(env.step(t_actions))
    # env.render()
    # # %%
    # t_actions = {'alpha': 2, 'beta': 1}
    # print(env.step(t_actions))
    # env.render()

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

    # NUMBER_OF_GAMES = 10


    # 手操
    # # %%
    # env.render()
    #
    # t_actions = {'alpha': 2, 'beta': 0}
    #
    # t_observations, t_rewards, done, info = env.step(t_actions)
    #
    # print(t_observations, t_rewards, done, info)
    #
    # env.render()
    # t_actions = {'alpha': 2, 'beta': 0}
    # print(env.step(t_actions))
    # env.render()
    # # %%
    # t_actions = {'alpha': 3, 'beta': 1}
    # print(env.step(t_actions))
    # env.render()
    # # %%
    # t_actions = {'alpha': 2, 'beta': 1}
    # print(env.step(t_actions))
    # env.render()
    #
    # # %%
    # env.reset()
    # env.render()