import random
import torch
import torch.nn as nn

from collections import namedtuple, deque

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_global_weights(agents_nets):
    number_of_agents = len(agents_nets)
    
    global_dqn_para = agents_nets[0].state_dict()
                  
    for idx in range(number_of_agents):
        net_dqn_para = agents_nets[idx].state_dict()
        if idx == 0:
            for key in net_dqn_para:
                global_dqn_para[key] = net_dqn_para[key] / number_of_agents
        else:
            for key in net_dqn_para:
                global_dqn_para[key] += net_dqn_para[key] / number_of_agents

    # for idx in range(number_of_agents):
    #     agents_nets[idx].load_state_dict(global_dqn_para)
    return global_dqn_para

def sync_Agents_weights(agents_nets):

    number_of_agents = len(agents_nets)

    global_dqn_para = get_global_weights(agents_nets)

    for idx in range(number_of_agents):
        agents_nets[idx].load_state_dict(global_dqn_para)

def plot_rewards(agent_episode_rewards, show_result=False, zero_point=None, ylabel='Rewards'):
    from IPython import display

    n_agents = len(agent_episode_rewards)
    fig, axs = plt.subplots(n_agents)

    for i, episode_rewards in enumerate(agent_episode_rewards):
        rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
        if show_result:
            axs[i].set_title('Result')
        else:
            axs[i].set_title('Training...')
        axs[i].set_xlabel('Episode')
        axs[i].set_ylabel(ylabel)
        axs[i].plot(rewards_t.cpu().numpy(), label=f'Agent {i+1}')
        
        # Take 100 episode averages and plot them too
        moving_averages = moving_average(episode_rewards, 100)
        axs[i].plot(moving_averages, label=f'Agent {i+1} Moving Average')
        
        axs[i].legend()

    plt.tight_layout()
    plt.pause(0.001)  # pause a bit so that plots are updated

    if not show_result:
        display.display(plt.gcf())
        display.clear_output(wait=True)
    else:
        display.display(plt.gcf())



def draw_history(agent_rewards, agent_paths_length, n_agents, EPISODES, window_size=100, title='', xlims=[None, None], ylims=[None, None]):
    # Convert agent_rewards and agent_paths_length to numpy arrays for easier manipulation
    agent_rewards_np = [np.array(rewards) for rewards in agent_rewards]
    agent_paths_length_np = [np.array(lengths) for lengths in agent_paths_length]

    # Mean and std for each agent's rewards and path length
    rewards_means = [rewards.mean() for rewards in agent_rewards_np]
    rewards_std = [rewards.std() for rewards in agent_rewards_np]
    paths_length_means = [lengths.mean() for lengths in agent_paths_length_np]
    paths_length_std = [lengths.std() for lengths in agent_paths_length_np]

    # Range of episodes
    episodes = np.arange(1, EPISODES+1)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))

    # Set larger font size
    plt.rcParams.update({'font.size': 15})

    # Rewards
    for i in range(n_agents):
        ax1.plot(episodes, moving_average(agent_rewards_np[i], window_size), label=f'Agent {i+1}')
        # ax1.fill_between(episodes, rewards_means[i]-rewards_std[i], rewards_means[i]+rewards_std[i], alpha=0.2)
    ax1.set_xlabel('Episodes', fontsize=15)
    ax1.set_ylabel('Cumulative Rewards', fontsize=15)
    ax1.set_title(title + 'Rewards per Episode for each Agent')
    ax1.grid(True)
    ax1.legend()
    if xlims[0] is not None:
        ax1.set_xlim(xlims[0])
    if ylims[0] is not None:
        ax1.set_ylim(ylims[0])
    ax1.tick_params(axis='both', which='major', labelsize=15)

    # Path lengths
    for i in range(n_agents):
        ax2.plot(episodes, moving_average(agent_paths_length_np[i], window_size), label=f'Agent {i+1}')
        # ax2.fill_between(episodes, paths_length_means[i]-paths_length_std[i], paths_length_means[i]+paths_length_std[i], alpha=0.2)
    ax2.set_xlabel('Episodes', fontsize=15)
    ax2.set_ylabel('Path Length', fontsize=15)
    ax2.set_title(title + 'Path Length per Episode for each Agent')
    ax2.grid(True)
    ax2.legend()
    if xlims[1] is not None:
        ax2.set_xlim(xlims[1])
    if ylims[1] is not None:
        ax2.set_ylim(ylims[1])
    ax2.tick_params(axis='both', which='major', labelsize=15)

    # Show the plots
    plt.tight_layout()
    plt.show()

def draw_path(episode_index, agent_paths, n_agents, env_size):
    # Choose an episode
    episode_index = episode_index  # Change this to select a different episode

    # Extract paths for each agent from the chosen episode
    agent_paths_episode = [agent_paths[i][episode_index] for i in range(n_agents)]

    # Create a new figure
    plt.figure(figsize=(10,10))

    # Plot grid
    for i in range(env_size):
        for j in range(env_size):
            plt.plot(i, j, 's', color='lightgray')

    # Plot paths for each agent
    colors = ['blue', 'red']
    for i, path in enumerate(agent_paths_episode):
        # Unzip the list of positions into two lists for x and y positions
        x_positions, y_positions = zip(*path)
        plt.plot(x_positions, y_positions, color=colors[i], label=f'Agent {i+1}')

    plt.gca().invert_yaxis()  # Optional: Invert y-axis to match gridworld's orientation
    plt.grid(True)
    plt.legend()
    plt.title(f'Paths for Agents in Episode {episode_index}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def moving_average(data_list, window_size=100):
    moving_averages = []
    for i in range(len(data_list)):
        if i < window_size:
            window_data = data_list[:i + 1]
        else:
            window_data = data_list[i - window_size + 1:i + 1]
        average = sum(window_data) / len(window_data)
        moving_averages.append(average)
    
    return moving_averages

# 经验回放
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

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

# LSTM神经网络
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        # 先将输入铺平
        x = x.view(batch_size * seq_len, self.input_dim)

        # 通过第一个全连接层
        x = torch.relu(self.fc1(x))

        # 重塑x以供LSTM使用
        x = x.view(batch_size, seq_len, self.hidden_dim)
        
        # 通过LSTM层
        x, _ = self.lstm(x)

        # print(x)
        # print(x.shape)

        # 只保留最后一个时间步的输出
        x = x[:, -1, :]

        # print(x.shape)

        # 通过最后一个全连接层
        x = self.fc2(x)

        return x

# LSTM神经网络
class DuelingDQNLast(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DuelingDQNLast, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # 决斗架构，包括值函数流和优势函数流
        self.value_stream = nn.Linear(hidden_dim, 1)
        self.advantage_stream = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        # 先将输入铺平
        x = x.view(batch_size * seq_len, self.input_dim)

        # 通过第一个全连接层
        x = torch.relu(self.fc1(x))

        # 重塑x以供LSTM使用
        x = x.view(batch_size, seq_len, self.hidden_dim)
        
        # 通过LSTM层
        x, _ = self.lstm(x)

        # 只保留最后一个时间步的输出
        x = x[:, -1, :]
        # x = torch.relu(x)

        # 计算值函数和优势函数
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # 计算Q值：Q(s, a) = V(s) + A(s, a) - mean(A(s, a))
        x = value + advantage - advantage.mean(dim=1, keepdim=True)

        # 重新shape成原始的batch_size * seq_len * output_dim
        # x = x.view(batch_size, seq_len, self.output_dim)

        return x
    


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_dim=32):
        super(DuelingDQN, self).__init__()

        self.input_shape = input_shape
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        self.conv = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(hidden_dim, hidden_dim)

        self.fc_adv = nn.Linear(hidden_dim, n_actions)
        self.fc_val = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden):
        x = x.view(x.size(0), -1)
        x = self.conv(x)
        x, hidden = self.lstm(x.unsqueeze(0), hidden)
        x = x.squeeze(0)

        adv = self.fc_adv(x)
        val = self.fc_val(x)

        return val + adv - adv.mean(), hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_dim),
                torch.zeros(1, batch_size, self.hidden_dim))

# define a function to create network
def create_dueling_dqn_network(input_shape, n_actions):
    return DuelingDQN(input_shape, n_actions)



# # model = DuelingDQN(input_shape=25, hidden_dim=128, n_actions=4)
# model = DuelingDQNLast(input_dim=25, hidden_dim=32, output_dim=4)
# sample = torch.randn(1, 5, 25)
# output = model(sample)
# print(output)
# # best = output.argmax(-1)
# # print(best)