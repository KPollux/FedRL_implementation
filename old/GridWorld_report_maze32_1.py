#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys, time, datetime, json, random
import numpy as np
import copy
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# ![image.png](attachment:153e8ada-51b8-4d9c-aa0c-614a6b3089f3.png)

# - 0 - 左
# - 1 - 向上
# - 2 - 右
# - 3 - 向下
# - 每次移动都会花费老鼠 -0.04 分
# - 奶酪，给予 1.0 分
# - 封锁的单元格-0.75 分，动作不会被执行
# - 迷宫边界之外的行为：-0.8 分，动作不会被执行
# - 已经访问过的单元格，-0.25 分
# - 总奖励低于负阈值：(-0.5 * maze.size)，lose

# In[2]:


visited_mark = 0.8  # Cells visited by the rat will be painted by gray 0.8
end_mark = 1.5
rat_mark = 0.5      # The current rat cell will be painteg by gray 0.5
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

num_actions = len(actions_dict)

# Exploration factor
epsilon = 0.1


# In[3]:


import random
import heapq

def generate_gridworld(n, prob):
    """
    生成大小为n x n的随机Gridworld，其中prob控制1出现的频率。
    返回值是一个包含n个列表的列表，每个列表包含n个随机数（0或1）。
    """
    gridworld = [[int(random.random() < prob) for j in range(n)] for i in range(n)]
    return gridworld

# def find_path(gridworld, start, end):
#     """
#     使用深度优先搜索算法查找从start到end的路径。
#     如果找到了一条路径，则返回True，否则返回False。
#     """
#     visited = set()
#     stack = [start]
#     while stack:
#         current = stack.pop()
#         if current == end:
#             return True
#         if current in visited:
#             continue
#         visited.add(current)
#         x, y = current
#         neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
#         for neighbor in neighbors:
#             i, j = neighbor
#             if i < 0 or i >= len(gridworld) or j < 0 or j >= len(gridworld[0]):
#                 continue
#             if gridworld[i][j] == 0:
#                 continue
#             stack.append((i, j))
#     return False

def find_path(maze, start, end):
    """
    使用 A* 算法搜索迷宫最优路径
    :param maze: 二维迷宫数组，0 表示障碍，1 表示可通行
    :param start: 起点坐标 (row, col)
    :param end: 终点坐标 (row, col)
    :return: 返回最优路径
    """
    ROW, COL = len(maze), len(maze[0])
    pq = []  # 使用优先队列存储搜索节点
    heapq.heappush(pq, (0, start, [start]))
    visited = set()  # 使用 set 存储已访问的节点
    while pq:
        f, (row, col), path = heapq.heappop(pq)
        if (row, col) in visited:
            continue
        visited.add((row, col))
        if (row, col) == end:
            return path
        for (r, c) in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]:
            if 0 <= r < ROW and 0 <= c < COL and maze[r][c] == 1 and (r, c) not in visited:
                g = len(path)  # 当前节点到起点的距离
                h = abs(r-end[0]) + abs(c-end[1])  # 当前节点到终点的曼哈顿距离
                f = g + h
                heapq.heappush(pq, (f, (r, c), path + [(r, c)]))
    return False


def generate_solvable_gridworld(n, prob=0.6):
    """
    生成一个可解的大小为n x n的Gridworld。
    返回值是一个元组，包含生成的Gridworld、起点和终点。
    """
    while True:
        gridworld = generate_gridworld(n, prob=prob)
        # start = (random.randint(0, n-1), random.randint(0, n-1))
        # end = (random.randint(0, n-1), random.randint(0, n-1))
        start = (0, 0)
        end = (n-1, n-1)
        
        optimal_path = find_path(gridworld, start, end)
        if gridworld[start[0]][start[1]] == 1 and gridworld[end[0]][end[1]] == 1 and start != end and optimal_path is not False:
            return gridworld, start, end, optimal_path

# 示例代码
# gridworld, start, end, optimal_path = generate_solvable_gridworld(8)
# gridworld = np.array(gridworld)*1.0
# print(gridworld)
# print("start:", start)
# print("end:", end)

# train_set = {'gridworld':[],
#             'start':[],
#             'end':[]}

# test_set = {'gridworld':[],
#             'start':[],
#             'end':[]}

# for i in range(6400):
#     gridworld, start, end = generate_solvable_gridworld(8)
#     train_set['gridworld'].append(gridworld)
#     train_set['start'].append(start)
#     train_set['end'].append(end)

# # 将字典保存到文件中
# with open("gridworld3x3_train_dict.pickle", "wb") as f:
#     pickle.dump(train_set, f)


# ## Q-maze

# In[4]:


# maze is a 2d Numpy array of floats between 0.0 to 1.0
# 1.0 corresponds to a free cell, and 0.0 an occupied cell
# rat = (row, col) initial rat position (defaults to (0,0))

class Qmaze(object):
    def __init__(self, maze, rat=(0,0), max_Tstep=800):
        # 允许的最大步数
        self.max_Tstep = max_Tstep
        self.action_space = [0, 1, 2, 3]
        # 初始化迷宫，老鼠可以从任意位置开始，默认为左上角
        self._maze = np.array(maze)
        nrows, ncols = self._maze.shape
        # 终点始终在右下角
        self.target = (nrows-1, ncols-1)   # target cell where the "cheese" is
        # 初始化空格list，maze为1表示空格，为0表示墙体
        self.free_cells = [(r,c) for r in range(nrows) for c in range(ncols) if self._maze[r,c] == 1.0]
        # 将目标格移出空格list
        self.free_cells.remove(self.target)
        # 检查左上和右下是否为空
        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if not rat in self.free_cells:
            raise Exception("Invalid Rat Location: must sit on a free cell")
        # 放置老鼠并初始化参数
        state, info = self.reset(rat)
    
        # return state, info

    def reset(self, rat=(0, 0)):
        self.rat = rat
        self.maze = np.copy(self._maze)
        nrows, ncols = self.maze.shape
        row, col = rat
        self.maze[row, col] = rat_mark
        self.maze[self.target[0], self.target[1]] = end_mark
        # 初始状态
        self.state = (row, col, 'start')
        # 设置最低奖励阈值
        self.min_reward = -0.5 * self.maze.size
        # 初始化总奖励
        self.total_reward = 0
        self.visited = list()
        self.total_Tstep = 0
        
        return self.observe(), self.game_status()

    def update_state(self, action):
        '''
            input: action [0, 1, 2, 3] [L, U, R, D]
        '''
        nrows, ncols = self.maze.shape
        nrow, ncol, nmode = rat_row, rat_col, mode = self.state
        
        # 如果老鼠访问的是空格，则记录
        if self.maze[rat_row, rat_col] > 0.0:
            self.visited.append((rat_row, rat_col))  # mark visited cell

        # 获取所有可能执行的动作
        valid_actions = self.valid_actions()
        # print('valid_actions', valid_actions)
        
        # 如果没有可以执行的动作（被围住了），则状态为 blocked，位置不变
        if not valid_actions:
            nmode = 'blocked'
            print('blocked')
        # 如果需要执行的动作在可执行动作列表中，那么状态为有效，并相应执行动作
        elif action in valid_actions:
            nmode = 'valid'
            if action == LEFT:
                ncol -= 1
            elif action == UP:
                nrow -= 1
            if action == RIGHT:
                ncol += 1
            elif action == DOWN:
                nrow += 1
        # 如果需要执行的动作不在可执行动作列表中（撞墙），位置不变
        else:                  # invalid action, no change in rat position
            nmode = 'invalid'
            
        self.total_Tstep += 1  # 每次执行动作+1
        # new state
        self.state = (nrow, ncol, nmode)

    def get_reward(self):
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        
        reward = 0
        rl = 0
        rg = 0
        
        if rat_row == nrows-1 and rat_col == ncols-1:
            rl = 1  # 奶酪，给予 1.0 分
        elif mode == 'blocked':
            rl = self.min_reward - 1
        elif mode == 'invalid':
            rl = -0.75  # 撞墙-0.75 分，动作不会被执行
        elif (rat_row, rat_col) in self.visited:
            rl = -0.25  # 访问已经访问过的单元格，-0.25 分
        elif mode == 'valid':
            rl = -0.04  # 每次移动都会花费老鼠 -0.04 分
        
        # rg = self._maze.shape[0]/(abs(self.state[0]-self.target[0]) + abs(self.state[1]-self.target[1]))
        # print(rl, rg)
        
        reward = rl + rg
        
        return reward

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status
    
    def step(self, action):
        envstate, reward, status = self.act(action)
        observation = envstate
        done = self.is_game_done()
        info = status
        return observation, reward, done, info

    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_env(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r,c] > 0.0:
                    canvas[r,c] = 1.0
        # draw the rat
        row, col, valid = self.state
        canvas[row, col] = rat_mark
        canvas[self.target[0], self.target[1]] = end_mark
        return canvas

    def game_status(self):
        if self.total_Tstep > self.max_Tstep or self.total_reward < self.min_reward:
        # if self.total_reward < self.min_reward:
            return 'lose'
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows-1 and rat_col == ncols-1:
            return 'win'

        return 'not_over'
    
    def is_game_done(self):
        game_status = self.game_status()
        
        if game_status == 'not_over':
            return False
        elif game_status == 'win' or game_status == 'lose':
            return True
        
        return -1

    def valid_actions(self, cell=None):
        # 默认验证当前位置
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = copy.deepcopy(self.action_space)
        nrows, ncols = self.maze.shape
        # 如果在第0行，则不能向上走；如果在最后一行，则不能向下走
        if row == 0:
            actions.remove(1)
        elif row == nrows-1:
            actions.remove(3)
        # 列-左右
        if col == 0:
            actions.remove(0)
        elif col == ncols-1:
            actions.remove(2)

        # 如果不在最左列，而左边是墙，则不能向左；右边同理
        if row>0 and self.maze[row-1,col] == 0.0:
            actions.remove(1)
        if row<nrows-1 and self.maze[row+1,col] == 0.0:
            actions.remove(3)

        # 上下同理
        if col>0 and self.maze[row,col-1] == 0.0:
            actions.remove(0)
        if col<ncols-1 and self.maze[row,col+1] == 0.0:
            actions.remove(2)

        # 返回所有可能执行的动作
        return actions


# In[5]:


def show(qmaze):
    plt.grid('on')
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    for row,col in qmaze.visited:
        canvas[row,col] = 0.6
    rat_row, rat_col, _ = qmaze.state
    canvas[rat_row, rat_col] = 0.3   # rat cell
    canvas[nrows-1, ncols-1] = 0.9 # cheese cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    return img


# In[6]:


maze_size = 32
obstical_prob = 0.2
gridworld, start, end, optimal_path = generate_solvable_gridworld(maze_size, 1-obstical_prob)
maze = np.array(gridworld)*1.0
print(maze)
print("start:", start)
print("end:", end)


# In[7]:


# maze = np.array([
#     [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
#     [ 1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.],
#     [ 1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.],
#     [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
#     [ 1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
#     [ 1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.],
#     [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
#     [ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.]
# ])
# maze =  np.array([
#     [ 1.,  0.,  1.,  1.,  1.,  1.,  1., 1.],
#     [ 1.,  1.,  1.,  0.,  0.,  1.,  0., 0.],
#     [ 1.,  0.,  1.,  1.,  1.,  1.,  0., 0.],
#     [ 1.,  1.,  1.,  1.,  0.,  1.,  1., 1.],
#     [ 1.,  0.,  0.,  1.,  1.,  1.,  1., 1.],
#     [ 1.,  0.,  1.,  1.,  1.,  1.,  1., 1.],
#     [ 1.,  1.,  1.,  0.,  1.,  1.,  1., 1.],
#     [ 1.,  1.,  1.,  0.,  1.,  1.,  1., 1.],
# ])


# In[8]:


# np.savetxt('maze32_1.txt', maze)
maze = np.loadtxt('maze32_1.txt')


# In[9]:


qmaze = Qmaze(maze)
maze_size = maze.shape[0]
optimal_path = find_path(maze, (0, 0), (maze_size-1, maze_size-1))
qmaze.visited = optimal_path
optimal_length = len(optimal_path)

# show(qmaze)
print('optimal path length is:', optimal_length)


# In[10]:


qmaze = Qmaze(maze)
canvas, reward, game_over = qmaze.act(DOWN)
print("reward=", reward)
# show(qmaze)


# In[11]:


canvas


# In[12]:


qmaze.act(RIGHT)  # move right
qmaze.act(RIGHT)  # move right
qmaze.act(UP)  # move up
# show(qmaze)


# ## DQN

# In[13]:


import math


# In[14]:


import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, observation_size, num_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(observation_size, observation_size*2)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(observation_size*2, observation_size*2)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(observation_size*2, num_actions)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.output_layer(x)
        return x


# In[15]:


from collections import namedtuple, deque
import matplotlib
import matplotlib.pyplot as plt
# set up matplotliba
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display
    
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
    
def plot_rewards(episode_rewards, show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.plot(rewards_t.numpy())
    
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99)-(maze_size*maze_size*0.5), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


# In[16]:


# 动作选取
def select_action(state):
    global steps_done
    sample = random.random()
    
    # 随着进行，eps_threshold逐渐降低
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #     math.exp(-1. * steps_done / EPS_DECAY)
    eps_threshold = 0.1
    steps_done += 1
    
    # 常规情况选择价值最高的动作
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    
    # 当随机值超过阈值时，随机选取 - exploration
    else:
        # 探索时只探索可能的动作，增加探索效率？
        return torch.tensor([[random.choice(env.valid_actions())]], device=device, dtype=torch.long)


# In[17]:


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    # 离线学习，从记忆池中抽取回忆
    transitions = memory.sample(BATCH_SIZE)
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
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    # 当前奖励+下一个状态的奖励，更新Q. 如果下一个状态为最终状态，则仅有当前奖励
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch  
    # print(expected_state_action_values)

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# optimize_model()


# In[18]:


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'


# In[19]:


# ss = []
# es = []
# EPS_DECAY = 10
# for i in range(1000):
#     sample = random.random()
#     eps_threshold = EPS_END + (EPS_START - EPS_END) * \
#             math.exp(-1. * i / EPS_DECAY)
#     # eps_threshold = 1.0 / (i + 1)
#     ss.append(sample)
#     es.append(eps_threshold)

# exploit = 0
# for i in range(1000):
#     if ss[i] > es[i]:
#         exploit += 1
# plt.plot(ss)
# plt.plot(es)
# print(exploit, exploit/1000)


# In[20]:


optimal_length


# In[21]:


from tqdm import trange

def evaluation():
    global optimal_length
    win = 0
    episode_rewards_eval = []
   
    env = Qmaze(maze, max_Tstep=optimal_length)

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
    
    # show(env)
    # print(env.visited)
    # print(env.state)
    # print(env.total_reward)
    
    return episode_rewards_eval, win_rate

# win_rate = evaluation(10)
# print(win_rate)


# In[38]:


ENV_NAME = 'grid_32_1'

folder_name = f"runs/{ENV_NAME}/" + time.asctime(time.gmtime()).replace(" ", "_").replace(":", "_")
os.makedirs('runs/', exist_ok=True)
os.makedirs(f'runs/{ENV_NAME}/', exist_ok=True)
os.makedirs(folder_name, exist_ok=True)


# In[39]:


history = {}


# In[45]:


# BATCH_SIZE是指从重放缓冲区采样的转换数
# GAMMA是上一节中提到的折扣系数
# EPS_START是EPSILON的起始值
# EPS_END是epsilon的最终值
# EPS_DECAY 控制epsilon的指数衰减率，越高意味着衰减越慢
# TAU是目标网络的更新率
# LR是AdamW优化器的学习率
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-3
num_episodes = 100000
steps_done = 0
# sync_target_net_freq = 1e4

# 初始化环境
env = Qmaze(maze, max_Tstep=int(maze.size*0.5/0.04))
# 重置环境获取信息
state, info = env.reset()

n_observations = state.size
state = torch.Tensor(state).to(device)

policy_net = DQN(n_observations, num_actions).to(device)
target_net = DQN(n_observations, num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(int(maze.size*0.5/0.04)*2)

episode_rewards = []

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset()
    # Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity
    state = torch.tensor(state, dtype=torch.float32, device=device)

    done = False
    while not done:
        action = select_action(state)  # 选择一个动作
        # random.choice(env.valid_actions())
        observation, reward, done, _ = env.step(action.item())  # 执行动作，返回{下一个观察值、奖励、是否结束、是否提前终止}
        reward = torch.tensor([reward], device=device)

        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device)  # 如果没有终止则继续记录下一个状态

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        
        # if (steps_done % sync_target_net_freq) == 0:
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
            # target_net.load_state_dict(policy_net_state_dict)

    episode_rewards.append(env.total_reward)
    
    history['episode_rewards'] = episode_rewards
    # 将字典保存成 txt 文件
    with open(folder_name+'/history.txt', 'w') as f:
        for key, value in history.items():
            f.write(f'{key}: {value}\n')
    
    # plot_rewards(episode_rewards)
    episode_rewards_eval, win_rate = evaluation()
    if win_rate == 1:
        break


# In[32]:


# episode_rewards


# In[ ]:


# plot_rewards(episode_rewards, True)


# In[71]:


# len(episode_rewards)


# In[33]:


# Initialize the environment and get it's state
state, info = env.reset()
# Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity
state = torch.tensor(state, dtype=torch.float32, device=device)

done = False
while not done:
    action = policy_net(state).max(1)[1].view(1, 1)  # 选择一个动作
    # random.choice(env.valid_actions())
    observation, reward, done, _ = env.step(action.item())  # 执行动作，返回{下一个观察值、奖励、是否结束、是否提前终止}
    # reward = torch.tensor([reward], device=device)

    if done:
        next_state = None
    else:
        next_state = torch.tensor(observation, dtype=torch.float32, device=device)  # 如果没有终止则继续记录下一个状态

    # Store the transition in memory
    # memory.push(state, action, next_state, reward)

    # Move to the next state
    state = next_state
    


# In[34]:


# show(env)
print(env.visited)
print(env.total_reward)


# In[44]:


history['test_route'] = env.visited
history['test_total_reward'] = env.total_reward

# 将字典保存成 txt 文件
with open(folder_name+'/history.txt', 'w') as f:
    for key, value in history.items():
        f.write(f'{key}: {value}\n')

torch.save(policy_net.state_dict(), folder_name+'/my_model_weights.pth')


# In[26]:


# env.total_Tstep


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[27]:


# def play_game(model, qmaze, rat_cell):
#     qmaze.reset(rat_cell)
#     envstate = qmaze.observe()
#     while True:
#         prev_envstate = envstate
#         # get next action
#         q = model.predict(prev_envstate)
#         action = np.argmax(q[0])

#         # apply action, get rewards and new state
#         envstate, reward, game_status = qmaze.act(action)
#         if game_status == 'win':
#             return True
#         elif game_status == 'lose':
#             return False
        
# def completion_check(model, qmaze):
#     for cell in qmaze.free_cells:
#         if not qmaze.valid_actions(cell):
#             return False
#         if not play_game(model, qmaze, cell):
#             return False
#     return True


# In[28]:


# class Experience(object):
#     def __init__(self, model, max_memory=100, discount=0.95):
#         self.model = model
#         self.max_memory = max_memory
#         self.discount = discount
#         self.memory = list()
#         self.num_actions = num_actions

#     def remember(self, episode):
#         # episode = [envstate, action, reward, envstate_next, game_over]
#         # memory[i] = episode
#         # envstate == flattened 1d maze cells info, including rat cell (see method: observe)
#         self.memory.append(episode)
#         if len(self.memory) > self.max_memory:
#             del self.memory[0]

#     def predict(self, envstate):
#         envstate = torch.Tensor(envstate)
#         return self.model(envstate)

#     def get_data(self, data_size=10):
#         env_size = self.memory[0][0].shape[1]   # envstate 1d size (1st element of episode)
#         mem_size = len(self.memory)
#         data_size = min(mem_size, data_size)
#         inputs = torch.zeros((data_size, env_size))
#         targets = torch.zeros((data_size, self.num_actions))
#         for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
#             envstate, action, reward, envstate_next, game_over = self.memory[j]
#             inputs[i] = envstate
#             # There should be no target values for actions not taken.
#             targets[i] = self.predict(envstate)
#             # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
#             Q_sa = torch.max(self.predict(envstate_next))
#             print(targets, reward)
#             if game_over:
#                 targets[i, action] = reward
#             else:
#                 # reward + gamma * max_a' Q(s', a')
#                 targets[i, action] = reward + self.discount * Q_sa
#             print(targets, reward)
#             return
#         return inputs, targets
    
#     def get_data_tensor(self, data_size=10):
#         env_size = self.memory[0][0].shape[1]  # envstate 1d size (1st element of episode)
#         mem_size = len(self.memory)
#         data_size = min(mem_size, data_size)
#         inputs = torch.zeros((data_size, env_size))
#         targets = torch.zeros((data_size, self.num_actions))

#         sample_indices = np.random.choice(range(mem_size), data_size, replace=False)
#         sampled_memory = [self.memory[j] for j in sample_indices]
       
#         envstates = torch.cat([m[0] for m in sampled_memory])
#         actions = torch.LongTensor([m[1] for m in sampled_memory]).view(-1, 1)
#         rewards = torch.FloatTensor([m[2] for m in sampled_memory]).view(-1, 1)
#         envstates_next = torch.cat([m[3] for m in sampled_memory])
#         game_over = torch.FloatTensor([m[4] for m in sampled_memory]).view(-1, 1)

#         inputs.copy_(envstates)
#         # There should be no target values for actions not taken.
#         targets.copy_(self.predict(envstates).data)
#         # print(torch.max(self.predict(envstates_next)))
#         # print(self.predict(envstates_next).shape)
#         # print(envstates_next.shape)
#         Q_sa = torch.max(self.predict(envstates_next), dim=1, keepdim=True)[0]
#         print('Q_sa', Q_sa)
#         # print(targets.gather(1, actions).view(-1, 1))
#         print(targets, rewards)
#         targets.gather(1, actions).view(-1, 1).copy_(rewards)
#         print(targets.gather(1, actions).view(-1, 1))
#         print(targets, rewards)
#         # return
#         mask = (game_over == 0)
#         masked_next_state_values = Q_sa * mask.float()
#         targets.gather(1, actions).view(-1, 1).add_(masked_next_state_values)

#         return inputs, targets


# In[29]:


# maze.shape


# In[30]:


# model = Model(maze.size, num_actions)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# loss = nn.MSELoss()


# In[31]:


# model


# In[32]:


# model = build_model(maze)
# qtrain(model, maze, epochs=1000, max_memory=8*maze.size, data_size=32)


# In[33]:


# # This is a small utility for printing readable time strings:
# def format_time(seconds):
#     if seconds < 400:
#         s = float(seconds)
#         return "%.1f seconds" % (s,)
#     elif seconds < 4000:
#         m = seconds / 60.0
#         return "%.2f minutes" % (m,)
#     else:
#         h = seconds / 3600.0
#         return "%.2f hours" % (h,)


# In[34]:


# global epsilon
# n_epoch = 1
# max_memory = 8*maze.size
# data_size = 32
# weights_file = ""
# name = 'model'
# start_time = datetime.datetime.now()

# # If you want to continue training from a previous model,
# # just supply the h5 file name to weights_file option
# if weights_file:
#     print("loading weights from file: %s" % (weights_file,))
#     model.load_weights(weights_file)

# # Construct environment/game from numpy array: maze (see above)
# qmaze = Qmaze(maze)

# # Initialize experience replay object
# experience = Experience(model, max_memory=max_memory)

# win_history = []   # history of win/lose game
# n_free_cells = len(qmaze.free_cells)
# hsize = qmaze.maze.size//2   # history window size
# win_rate = 0.0
# imctr = 1

# for epoch in range(n_epoch):
#     loss = 0.0
#     rat_cell = random.choice(qmaze.free_cells)
#     qmaze.reset(rat_cell)
#     game_over = False

#     # get initial envstate (1d flattened canvas)
#     envstate = qmaze.observe()  # (1, 64)
#     envstate = torch.Tensor(envstate)

#     n_episodes = 0
#     while not game_over:
#         valid_actions = qmaze.valid_actions()
#         if not valid_actions: break
#         prev_envstate = envstate  # 上一个时刻的状态
        
#         # Get next action
#         if np.random.rand() < epsilon:
#             action = random.choice(valid_actions)
#         else:
#             action = np.argmax(experience.predict(prev_envstate).cpu().detach().numpy())
            
#         # Apply action, get reward and new envstate
#         envstate, reward, game_status = qmaze.act(action)
#         envstate = torch.Tensor(envstate)
        
#         if game_status == 'win':
#             win_history.append(1)
#             game_over = True
#         elif game_status == 'lose':
#             win_history.append(0)
#             game_over = True
#         else:
#             game_over = False

#         # Store episode (experience)
#         episode = [prev_envstate, action, reward, envstate, game_over]
#         experience.remember(episode)
#         n_episodes += 1
        
#         # Train neural network model
#         inputs, targets = experience.get_data(data_size=data_size)

#         # break
        
# inputs, inputs.shape, targets, targets.shape






# #         h = model.fit(
# #             inputs,
# #             targets,
# #             epochs=8,
# #             batch_size=16,
# #             verbose=0,
# #         )
# #         loss = model.evaluate(inputs, targets, verbose=0)


# In[35]:


# for epoch in range(n_epoch):
#     loss = 0.0
#     rat_cell = random.choice(qmaze.free_cells)
#     qmaze.reset(rat_cell)
#     game_over = False

#     # get initial envstate (1d flattened canvas)
#     envstate = qmaze.observe()

#     n_episodes = 0
#     while not game_over:
#         valid_actions = qmaze.valid_actions()
#         if not valid_actions: break
#         prev_envstate = envstate
#         # Get next action
#         if np.random.rand() < epsilon:
#             action = random.choice(valid_actions)
#         else:
#             action = np.argmax(experience.predict(prev_envstate))

#         # Apply action, get reward and new envstate
#         envstate, reward, game_status = qmaze.act(action)
#         if game_status == 'win':
#             win_history.append(1)
#             game_over = True
#         elif game_status == 'lose':
#             win_history.append(0)
#             game_over = True
#         else:
#             game_over = False

#         # Store episode (experience)
#         episode = [prev_envstate, action, reward, envstate, game_over]
#         experience.remember(episode)
#         n_episodes += 1

#         # Train neural network model
#         inputs, targets = experience.get_data(data_size=data_size)
#         h = model.fit(
#             inputs,
#             targets,
#             epochs=8,
#             batch_size=16,
#             verbose=0,
#         )
#         loss = model.evaluate(inputs, targets, verbose=0)

#     if len(win_history) > hsize:
#         win_rate = sum(win_history[-hsize:]) / hsize

#     dt = datetime.datetime.now() - start_time
#     t = format_time(dt.total_seconds())
#     template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
#     print(template.format(epoch, n_epoch-1, loss, n_episodes, sum(win_history), win_rate, t))
#     # we simply check if training has exhausted all free cells and if in all
#     # cases the agent won
#     if win_rate > 0.9 : epsilon = 0.05
#     if sum(win_history[-hsize:]) == hsize and completion_check(model, qmaze):
#         print("Reached 100%% win rate at epoch: %d" % (epoch,))
#         break

# # Save trained model weights and architecture, this will be used by the visualization code
# h5file = name + ".h5"
# json_file = name + ".json"
# model.save_weights(h5file, overwrite=True)
# with open(json_file, "w") as outfile:
#     json.dump(model.to_json(), outfile)
# end_time = datetime.datetime.now()
# dt = datetime.datetime.now() - start_time
# seconds = dt.total_seconds()
# t = format_time(seconds)
# print('files: %s, %s' % (h5file, json_file))
# print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, data_size, t))

