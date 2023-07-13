# https://github.com/Kirili4ik/HRL-taxi/blob/master/Taxi.py
# %%
import os
import pickle
import numpy as np
# import gymnasium as gym
import gym
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import trange

from taxi_grid_world import TaxiEnv
# %%
class Agent:
    def __init__(self, env, alpha, gamma):
        self.env = env  # Gym环境对象
        
        not_pr_acts = 2 + 1 + 1 + 1   # gotoS,D + put + get + root (非原子动作数目)
        nA = env.action_space_n + not_pr_acts  # 动作总数 6+5=11      
        nS = env.observation_space_n  # 状态空间大小 500
        self.V = np.zeros((nA, nS))  # 价值函数初始化 
        self.C = np.zeros((nA, nS, nA))  # 层次奖励初始化
        self.V_copy = self.V.copy()  # 复制价值函数，用于动作价值的评估
        
        # 为所有可能的动作定义编码
        s = self.south = 0
        n = self.north = 1
        e = self.east = 2
        w = self.west = 3
        pickup = self.pickup = 4
        dropoff = self.dropoff = 5
        gotoS = self.gotoS = 6
        gotoD = self.gotoD = 7
        get = self.get = 8
        put = self.put = 9
        root = self.root = 10
        
        # 为每个动作定义可能的子动作集合
        self.graph = [
            # is primitive 原子动作
            set(),  # south
            set(),  # north
            set(),  # east
            set(),  # west
            set(),  # pickup
            set(),  # dropoff

            # not primitive 可分解动作
            {s, n, e, w},  # gotoSource
            {s, n, e, w},  # gotoDestination
            {pickup, gotoS},  # get -> pickup, gotoSource
            {dropoff, gotoD},  # put -> dropoff, gotoDestination
            {put, get},  # root -> put, get
        ]
        
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.r_sum = 0  # 奖励累加器
        self.new_s = copy.copy(self.env.s)  # 新的状态
        self.done = False  # 是否完成一个episode
        self.num_of_ac = 0  # 执行的动作数目

        self.t = 0 # 时间步
        self.max_t = 512 # 最大时间步

    def is_primitive(self, act):
        # 判断给定的动作是否是原子动作（不可再分解的动作）
        if act <= 5:
            return True
        else:
            return False

    def is_terminal(self, a, done):
        # 判断当前的状态是否已经达到了给定的动作的终点
        # RGBY = self.env.dropoff_locations
        taxiloc, passloc, destloc = self.env.state
        # taxiloc = (taxirow, taxicol)
        # passidx = RGBY.index(passloc)
        # destidx = RGBY.index(destloc)
        if done:
            return True
        elif a == self.root:
            return done
        elif a == self.put:
            return passloc != "IN_TAXI"  # 如果要放下乘客，乘客在某个地方（不在出租车上），返回终止状态
        elif a == self.get:
            return passloc == "IN_TAXI"  # 如果要接起乘客，且4 乘客在出租车上，返回终止状态
        elif a == self.gotoD:
            return passloc == "IN_TAXI" and taxiloc == destloc  # 如果要去到目的地，乘客在出租车上，且出租车在目的地
                                                              # 那么就成功送到乘客，返回终止状态
        elif a == self.gotoS:
            return passloc != "IN_TAXI" and taxiloc == passloc  # 如果要接乘客，乘客在某个地方，且出租车在乘客所在的地方。
                                                             # 那么就成功接起乘客，返回终止状态
        elif self.is_primitive(a):
            # just else
            return True

    def evaluate(self, act, s):
        # 动作价值评估
        # 如果是原子动作（空集，没有子节点），直接返回价值函数
        if self.is_primitive(act):
            return self.V_copy[act, s]  # 当前动作-状态对的价值函数
        else:
            # 如果是可分解动作，递归计算（往下找子节点）
            for j in self.graph[act]:  # 对于所有子节点
                self.V_copy[j, s] = self.evaluate(j, s)  # 递归计算当前节点的价值
            Q = np.arange(0)  # 空的np数组
            for a2 in self.graph[act]:
                Q = np.concatenate((Q, [self.V_copy[a2, s]]))  # 将所有子节点的价值函数值放入Q中
            max_arg = np.argmax(Q)  # 找到最大的价值函数值
            return self.V_copy[max_arg, s]  # 返回最大的价值函数值

    def greed_act(self, act, s):
        # 采用 ε-greedy 策略选择动作
        e = 0.001
        Q = np.arange(0)
        possible_a = np.arange(0)
        for act2 in self.graph[act]:  # 对于所有子节点
            # 如果是原子动作，或者不是终止状态，就将其放入Q中
            if self.is_primitive(act2) or (not self.is_terminal(act2, self.done)):
                # 将所有子节点的价值函数值放入Q中
                Q = np.concatenate((Q, [self.V[act2, s] + self.C[act, s, act2]]))
                possible_a = np.concatenate((possible_a, [act2]))
        max_arg = np.argmax(Q)
        if np.random.rand(1) < e:
            return np.random.choice(possible_a)
        else:
            return possible_a[max_arg]

    def MAXQ_0(self, i, s):
        # MAXQ递归
        if self.done:
            i = 11                  # to end recursion
        self.done = False
        if self.is_primitive(i):
            self.new_s, r, done, _ = copy.copy(self.env.step(i))
            self.done = done
            self.t += 1
            if self.t > self.max_t:
                self.done = True
            self.r_sum += r
            self.num_of_ac += 1
            self.V[i, s] += self.alpha * (r - self.V[i, s])
            return 1
        elif i <= self.root:
            count = 0
            while not self.is_terminal(i, self.done):
                a = self.greed_act(i, s)
                N = self.MAXQ_0(a, s)
                self.V_copy = self.V.copy()
                evaluate_res = self.evaluate(i, self.new_s)
                self.C[i, s, a] += self.alpha * (self.gamma ** N * evaluate_res - self.C[i, s, a])
                count += N
                s = self.new_s
            return count


    def reset(self):
        # 重置环境
        self.env.reset()
        self.r_sum = 0
        self.num_of_ac = 0
        self.done = False
        self.new_s = copy.copy(self.env.s)
        self.t = 0

# %%
alpha = 0.2  # 设置学习率
gamma = 1  # 设置折扣因子
maze_cross = np.loadtxt('maze17_0.2.txt')
# maze_cross = np.loadtxt('mazeTaxi5.txt')
env = TaxiEnv(maze=maze_cross)
env.reset()
# list(env.decode(env.s))
# env.action_space.n
# env.observation_space.n
# env.step(1)

# %%
taxi = Agent(env, alpha, gamma)
episodes = 15000
sum_list = []
for j in range(episodes):
    taxi.reset()
    taxi.MAXQ_0(10, env.s)      # start in root
    sum_list.append(taxi.r_sum)
    if (j % 1000 == 0):
        print('already made', j, 'episodes')

# %%
def moving_average(data_list, window_size=100):
    moving_averages = []
    cumsum = [0]
    for i, x in enumerate(data_list, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=window_size:
            moving_aver = (cumsum[i] - cumsum[i-window_size])/window_size
            moving_averages.append(moving_aver)
    return moving_averages

sns.set(style='darkgrid')
sns.set(font_scale=1.5)
plt.figure(figsize=(15, 7.5))
sum_list_ = moving_average(sum_list)
plt.plot(sum_list_)
plt.xlabel('episode num')
plt.ylabel('points')
plt.show()
# %%
train_history = {'agent_rewards': sum_list}

floder_name = None
n_times = 0
if floder_name is None:
    prefix = 'Taxi_Q_learning_'

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

    floder_name += 'MAXQ_'

    # if floder_name == prefix:
    #     floder_name += 'Independent_'
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
