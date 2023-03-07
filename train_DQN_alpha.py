import json

# import gymnasium as gym
import math
import os
import random
import time

import matplotlib as mpl
import numpy as np
from tqdm import trange

# mpl.use('TkAgg')
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from alg_GLOBALS import *
from alg_plotter import ALGPlotter
from alg_env import FedRLEnv
from alg_nets import CriticNet, ActorNet

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------- # PARAMETERS # -------------------------- #
# M_EPISODE = 10
M_EPISODE = 6400
BATCH_SIZE = 128  # size of the batches
BUFFER_SIZE = 10000
LR = 1e-4  # learning rate
# LR_ACTOR = 1e-3  # learning rate
GAMMA = 0.99  # discount factor
EPSILON_MAX = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005  # for soft update of target parameters

# --------------------------- # ENVIRONMENT # -------------------------- #
MAX_STEPS = 38  # 38, 86, 178
SIDE_SIZE = 8  # 8, 16, 32

steps_done = 0
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
episode_durations = []


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


# 动作选取
def select_action(state, eval=False):
    global steps_done, EPS_START, EPS_END, EPS_DECAY
    sample = random.random()

    # 随着进行，eps_threshold逐渐降低
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    # t_alpha_action = torch.tensor(random.choice(env.action_spaces()['alpha'])).cuda()
    # print(t_alpha_action)
    # exit()
    # 常规情况选择价值最高的动作
    if sample > eps_threshold or eval:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.

            t_alpha_action = policy_net(state).max(1)[1].squeeze()
            # print(sample, eps_threshold)
            # print(policy_net(state))
            # print(t_alpha_action)
            # print(policy_net(state).max(1))
            # print(policy_net(state).max(1)[0])
            # exit()


    # 当随机值超过阈值时，随机选取 - exploration
    else:
        # return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        t_alpha_action = torch.tensor(random.choice(env.action_spaces()['alpha'])).cuda()
        # print(sample, eps_threshold)

    # print(t_alpha_action.shape)

    t_beta_action = torch.tensor(random.choice(env.action_spaces()['beta'])).cuda()

    # t_actions = torch.stack((t_alpha_action, t_beta_action))
    # t_actions = zip(*t_actions)
    # print(1 or 0)
    # print(1 and 0)
    # print(t_alpha_action, t_beta_action)

    return t_alpha_action, t_beta_action


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
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
    # plt.close()


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    # 离线学习，从记忆池中抽取回忆
    transitions = memory.sample(BATCH_SIZE)
    # print(transitions)
    # exit()

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

    # print(state_batch.shape)
    # print(action_batch.shape)
    # print(reward_batch.shape)
    # exit()

    # print(action_batch)
    # print(action_batch.shape)
    # 只保留alpha的动作
    # 只根据alpha的动作来计算价值，因为观测不到beta的动作
    action_batch = action_batch[..., 0]
    action_batch = action_batch.unsqueeze(1)
    # print(action_batch)
    # print(action_batch.shape)
    # exit()

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # 模型计算Q价值，我们根据价值选择动作
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # print(policy_net(state_batch))
    # print(state_action_values)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch  # 当前奖励+下一个状态的奖励，更新Q

    # Compute Huber loss
    # criterion = nn.SmoothL1Loss()
    criterion = nn.MSELoss()
    # print(state_action_values, expected_state_action_values.unsqueeze(1))
    # exit()
    # print(state_action_values.dtype)
    # print(expected_state_action_values.unsqueeze(1).dtype)
    loss = criterion(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if __name__ == '__main__':

    ENV_NAME = 'grid'
    env = FedRLEnv(max_steps=MAX_STEPS, side_size=SIDE_SIZE)

    folder_name = f"runs/{ENV_NAME}/" + time.asctime(time.gmtime()).replace(" ", "_").replace(":", "_")
    os.makedirs('runs/', exist_ok=True)
    os.makedirs(f'runs/{ENV_NAME}/', exist_ok=True)
    os.makedirs(folder_name, exist_ok=True)

    # print(env.action_space.n)  # ACTION: 0,1,2,3 = east > , south v , west < , north ^

    # state = env.reset()
    # state = state['alpha']
    # print(state)
    #
    # map = env.render()
    # plt.imshow(map)
    # plt.show()

    # --------------------------- # NETS # -------------------------- #
    # 初始化policy_net、target_net
    for i_agent in env.agents:
        if i_agent.type == 'alpha':
            policy_net = ActorNet(i_agent.state_size, 4).cuda()
            target_net = ActorNet(i_agent.state_size, 4).cuda()
            target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(BUFFER_SIZE)

    # --------------------------- # TRAIN # -------------------------- #
    logs = {
        "train": {
            "rewards": None,
        },
        "eval": {
            "rewards": [],
            "successful": [],
        }
    }

    num_episodes = M_EPISODE

    for i_episode in trange(num_episodes):
        # Initialize the environment and get it's state
        state = env.reset()
        state = state['alpha'].reshape((1, -1))

        # alpha 矩阵
        state = state.cuda()

        # count()，它返回一个迭代器，该迭代器从指定数字开始生成无限连续数字
        # 默认start=0, step=1
        for t in count():
            t_alpha_action, t_beta_action = select_action(state)  # 选择一个动作
            # t_actions = {'alpha': t_alpha_action, 'beta': t_beta_action}
            # print(t_actions)
            # merge two tensors
            t_actions = torch.stack((t_alpha_action, t_beta_action)).view(1, -1)
            # print(t_actions)
            # exit()
            observation, reward, done, _ = env.step(t_actions)  # 执行动作，返回{下一个观察值、奖励、是否结束、是否提前终止}
            # print(observation, reward, done, _)
            # exit()
            # 仅保留alpha
            observation = observation['alpha'].reshape((1, -1))
            reward = torch.tensor([reward['alpha']], device=device)

            if done:
                next_state = None
            else:
                next_state = observation.cuda()  # 如果没有终止则继续记录下一个状态

            # Store the transition in memory
            memory.push(state, t_actions, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(float(reward.cpu().numpy()[0]))
                # print(episode_durations)

                if i_episode % 100 == 0:
                    # plot_durations()
                    last_100 = episode_durations[-100:]
                    average = sum(last_100) / len(last_100)
                    print('i_episode', i_episode, 'running_avg_reward: ', average)
                break

        logs['train']['rewards'] = episode_durations

        with open(folder_name + "/train.txt", 'w') as convert_file:
            convert_file.write(json.dumps(logs))



    # eval
    torch.save(policy_net.state_dict(), f'{folder_name}/model.pt')

    NEPTUNE = False
    PLOT_LIVE = True
    plotter = ALGPlotter(plot_life=PLOT_LIVE, plot_neptune=NEPTUNE, name='my_run_FedRL', tags=[ENV_NAME], plot_per=1)
    # plotter.neptune_init()

    TotalCumulativeReward = []
    SuccessfulEpisode = 0
    eval_round = 800

    for i_eval in range(eval_round):
        state = env.reset()
        state = state['alpha'].reshape((1, -1))
        state = state.cuda()
        for t in count():
            t_alpha_action, t_beta_action = select_action(state, eval=True)
            t_actions = torch.stack((t_alpha_action, t_beta_action)).view(1, -1)
            observation, reward, done, info = env.step(t_actions)
            # observation = observation['alpha'].reshape((1, -1))
            reward = float(reward['alpha'])
            TotalCumulativeReward.append(reward)
            logs['eval']['rewards'].append(reward)
            if done:
                if info['success']:
                    SuccessfulEpisode += 1
                logs['eval']['successful'].append(info['success'])
                # plotter.plot(i_eval, env, TotalCumulativeReward)
                break

    # plotter.plot(i_eval, env, TotalCumulativeReward)
    # logs['eval']['rewards'] = TotalCumulativeReward

        with open(folder_name + "/train.txt", 'w') as convert_file:
            convert_file.write(json.dumps(logs))

    print('AvgCumulativeReward', np.sum(TotalCumulativeReward)/eval_round)
    print('SuccessfulRate', SuccessfulEpisode / eval_round)
    print()
    # PLOT
    # scores.append(sum(t_rewards.values()).item())
    # steps += 1
    # global_steps += 1
    # plotter.neptune_plot({
    #     'epsilon': epsilon,
    #     'alpha_loss': alpha_loss.item(), 'beta_loss': beta_loss.item(),
    #     'action alpha': t_alpha_action.item(), 'action beta': t_beta_action.item(),
    #     'state alpha': t_alpha_obs.mean().item(), 'state beta': t_beta_obs.mean().item(),
    #     'buffer size': len(replay_buffer_alpha),
    # })
    # if i_episode > M_EPISODE - PLOT_LAST and done:
    #     # print('plot', M_EPISODE, PLOT_LAST, done)
    #     plotter.plot(i_episode, env, scores)

    # print('Complete')
    # plot_durations(show_result=True)
    # plt.ioff()
    # plt.show()
