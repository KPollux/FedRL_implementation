#!/usr/bin/env python
# coding: utf-8
import os, sys, time, datetime, json, random
import numpy as np
import copy
import random
import heapq
import random
import heapq
import math
import torch
import torch.nn as nn
import torch.optim as optim

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maze_names = ['maze16_1', 'maze16_2', 'maze16_3', 'maze16_4', 'maze16_5']

ENV_NAME = 'grid_fullob_largeDQN'

for maze_name in maze_names:
    
    maze = np.loadtxt(maze_name+'.txt')
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
            self.min_reward = -100 # -0.5 * self.maze.size
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
                rl = 50  # 奶酪，给予 1.0 分
            # elif mode == 'blocked':
            #     rl = self.min_reward - 1
            # elif (rat_row, rat_col) in self.visited:
            #     rl = -0.25  # 访问已经访问过的单元格，-0.25 分
            elif mode == 'invalid':
                rl = -10  # 撞墙-0.75 分，动作不会被执行
            elif mode == 'valid':
                rl = -1  # 每次移动都会花费老鼠 -0.04 分

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
            # canvas = self.get_observation()
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
            # if self.total_Tstep > self.max_Tstep:
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

#         def get_observation(self, size=3):
#             maze = self.draw_env()
#             row, col, _ = self.state
#             # 获取maze的行列数
#             ROWS = len(maze)
#             COLS = len(maze[0])

#             # 初始化结果二维数组
#             result = [[0 for _ in range(size)] for _ in range(size)]

#             # 将以指定点为中心指定尺寸范围的观测值存入结果二维数组
#             for i in range(row-size//2, row+size//2+1):
#                 for j in range(col-size//2, col+size//2+1):
#                     if i < 0 or i >= ROWS or j < 0 or j >= COLS:
#                         # 如果超出边界，则填充为1
#                         result[i-row+size//2][j-col+size//2] = 0.0
#                     else:
#                         result[i-row+size//2][j-col+size//2] = maze[i][j]

#             # 返回结果二维数组
#             result = np.array(result)
#             result[size//2][size//2] = 0.5
#             return result

    qmaze = Qmaze(maze)
    maze_size = maze.shape[0]
    optimal_path = find_path(maze, (0, 0), (maze_size-1, maze_size-1))
    qmaze.visited = optimal_path
    optimal_length = len(optimal_path)

    # show(qmaze)
    print('optimal path length is:', optimal_length)


    class DQN(nn.Module):
        def __init__(self, observation_size, num_actions):
            super(DQN, self).__init__()
            self.layer1 = nn.Linear(observation_size, maze_size*maze_size*2)
            self.relu1 = nn.ReLU()
            self.layer2 = nn.Linear(maze_size*maze_size*2, maze_size*maze_size*2)
            self.relu2 = nn.ReLU()
            self.layer3 = nn.Linear(maze_size*maze_size*2, maze_size*maze_size*2)
            self.relu3 = nn.ReLU()
            self.layer4 = nn.Linear(maze_size*maze_size*2, maze_size*maze_size*2)
            self.relu4 = nn.ReLU()
            self.output_layer = nn.Linear(maze_size*maze_size*2, num_actions)

        def forward(self, x):
            x = self.layer1(x)
            x = self.relu1(x)
            x = self.layer2(x)
            x = self.relu2(x)
            x = self.layer3(x)
            x = self.relu3(x)
            x = self.layer4(x)
            x = self.relu4(x)
            x = self.output_layer(x)
            return x


    # In[26]:


    from collections import namedtuple, deque
    # import matplotlib
    # import matplotlib.pyplot as plt
    # # set up matplotliba
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

#     def plot_rewards(episode_rewards, show_result=False, zero_point=None, ylabel='Rewards'):
#         plt.figure(1)
#         rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
#         if show_result:
#             plt.title('Result')
#         else:
#             plt.clf()
#             plt.title('Training...')
#         plt.xlabel('Episode')
#         plt.ylabel(ylabel)
#         plt.plot(rewards_t.numpy())

#         if zero_point is None:
#             zero_point = (maze_size*maze_size*0.5)

#         # Take 100 episode averages and plot them too
#         if len(rewards_t) >= 100:
#             means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
#             means = torch.cat((torch.zeros(99)-zero_point, means))
#             plt.plot(means.numpy())

#         plt.pause(0.001)  # pause a bit so that plots are updated
#         if is_ipython:
#             if not show_result:
#                 display.display(plt.gcf())
#                 display.clear_output(wait=True)
#             else:
#                 display.display(plt.gcf())


    # In[27]:


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


    # In[28]:


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


    # In[29]:


    # if gpu is to be used
    
    # device = 'cpu'


    # In[30]:


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


    # In[31]:


    optimal_length


    # In[33]:


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

            # episode_rewards_eval.append(env.total_reward)
            episode_rewards_eval = env.total_reward
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


    # In[246]:


    

    now = time.strftime("%m-%d_%H-%M-%S", time.localtime())
    folder_name = f"runs/{ENV_NAME}/" + now
    os.makedirs('runs/', exist_ok=True)
    os.makedirs(f'runs/{ENV_NAME}/', exist_ok=True)
    os.makedirs(folder_name, exist_ok=True)


    # In[ ]:


    # ENV_NAME = 'grid'
    # now = time.strftime("%m-%d_%H-%M-%S", time.localtime())
    # folder_name = f"runs/{ENV_NAME}/" + now
    # os.makedirs('runs/', exist_ok=True)
    # os.makedirs(f'runs/{ENV_NAME}/', exist_ok=True)
    # os.makedirs(folder_name, exist_ok=True)
    #
    # # tensorboard
    # writer = SummaryWriter(folder_name)


    # In[247]:


    history = {}
    history['win'] = []
    history['episode_rewards_eval'] = []


    # In[248]:


    from itertools import count

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
    num_episodes = 1000
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
    episode_step = []

    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        state, info = env.reset()
        # Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity
        state = torch.tensor(state, dtype=torch.float32, device=device)

        done = False
        for t in count():
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

            if done:
                break

        episode_rewards.append(env.total_reward)
        episode_step.append(t)

        history['episode_rewards'] = episode_rewards
        history['episode_step'] = episode_step
        # 将字典保存成 txt 文件
        with open(folder_name+'/history.txt', 'w') as f:
            for key, value in history.items():
                f.write(f'{key}: {value}\n')

        # plot_rewards(episode_step, False, -15, 'Steps')

        episode_rewards_eval, win_rate = evaluation()

        history['win'].append(win_rate)
        history['episode_rewards_eval'].append(episode_rewards_eval)


    history['test_route'] = env.visited
    history['test_total_reward'] = env.total_reward

    # 将字典保存成 txt 文件
    with open(folder_name+'/history.txt', 'w') as f:
        for key, value in history.items():
            f.write(f'{key}: {value}\n')

    torch.save(policy_net.state_dict(), folder_name+'/my_model_weights.pth')


    
