# https://github.com/Kirili4ik/HRL-taxi/blob/master/Taxi.py
# %%
from itertools import count
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
from easydict import EasyDict as edict

# %%
class FLServer:
    def __init__(self, env, aggregation_method='QAvg'):
        self.agent_V_tables = []
        self.agent_C_tables = []

        not_pr_acts = 2 + 1 + 1 + 1   # gotoS,D + put + get + root (非原子动作数目)
        nA = env.action_space_n + not_pr_acts # 动作总数 6+5=11      
        nS = env.observation_space_n  # 状态空间大小 500

        self.global_V = np.zeros((nA, nS))
        self.global_C = np.zeros((nA, nS, nA))  # 层次奖励初始化
        self.aggregation_method = aggregation_method
        self.i_episode = 0

    def update_table(self, agent_table, i_episode, table_type='V'):
        self.i_episode = i_episode
        # self.agent_tables.append(agent_table)
        if table_type == 'V':
            self.agent_V_tables.append(agent_table)
            if len(self.agent_V_tables) == NUMBER_OF_AGENTS: 
                self.aggregate_table('V')
        elif table_type == 'C':
            self.agent_C_tables.append(agent_table)
            if len(self.agent_C_tables) == NUMBER_OF_AGENTS: 
                self.aggregate_table('C')
        

    # def aggregate_all_tables(self):
    #     self.aggregate_table('V')
    #     self.aggregate_table('C')
        
    def aggregate_table(self, table_type='V'):
        agent_tables = self.agent_V_tables if table_type == 'V' else self.agent_C_tables
        global_table = self.global_V if table_type == 'V' else self.global_C

        if self.aggregation_method == 'QAvg':
            global_table = np.mean(agent_tables, axis=0)  # Aggregates the Q-tables by taking the mean.

        elif self.aggregation_method == 'QAll':
            # 分别计算每个Agent的Q表与全局Q表的差值，得到DeltaQ
            delta_Qs = [agent_table - global_table for agent_table in agent_tables]
            # 将所有DeltaQ相加，得到新的全局Q表
            for dQ in delta_Qs:
                global_table += dQ
            
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
            delta_Qs = [agent_table - global_table for agent_table in agent_tables]

            # Then use the adjusted weights to update the global Q table
            for i in range(NUMBER_OF_AGENTS):
                global_table += delta_Qs[i] * weights[i]
        else:
            raise ValueError('Invalid aggregation method.')
            
        self.agent_q_tables = []  # Clears the list of agent Q-tables.

    def distribute_V(self):
        return self.global_V
    
    def distribute_C(self):
        return self.global_C

class Agent:
    def __init__(self, env, server, alpha, gamma):
        self.env = env  # Gym环境对象
        self.env.reset()
        self.server = server
        
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
            set(),  # south 0
            set(),  # north 1
            set(),  # east 2
            set(),  # west 3 
            set(),  # pickup 4
            set(),  # dropoff 5

            # not primitive 可分解动作
            {s, n, e, w},  # gotoSource 6
            {s, n, e, w},  # gotoDestination 7
            {pickup, gotoS},  # get -> pickup, gotoSource 8
            {dropoff, gotoD},  # put -> dropoff, gotoDestination 9
            {put, get},  # root -> put, get 10
        ]
        
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.r_sum = 0  # 奖励累加器
        self.new_s = copy.copy(self.env.s)  # 新的状态
        self.done = False  # 是否完成一个episode
        self.num_of_ac = 0  # 执行的动作数目

        self.atomic_action_count = 0
        self.step_generator = self.MAXQ_generator()

        self.t = 0 # 时间步
        self.max_t = 512 # 最大时间步

        self.log = edict()
        self.log.action_history = []
        self.log.state_history = []

    def is_primitive(self, act):
        # 判断给定的动作是否是原子动作（不可再分解的动作）
        if act <= 5:
            return True
        else:
            return False

    def is_terminal(self, a, done):
        # 判断当前的状态是否已经达到了给定的动作的终点
        RGBY = self.env.dropoff_locations
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
                                                              # 那么就成功送乘客，返回终止状态
        elif a == self.gotoS:
            return passloc != "IN_TAXI" and taxiloc == passloc  # 如果要接乘客，乘客在某个地方，且出租车在乘客所在的地方。
                                                             # 那么就成功接乘客，返回终止状态
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
        e = 0.1
        Q = np.arange(0)
        possible_a = np.arange(0)
        for act2 in self.graph[act]:  # 对于所有子节点
            # 如果是原子动作，或者不是终止状态，就将其放入Q中
            if self.is_primitive(act2) or (not self.is_terminal(act2, self.done)):
                # 将所有子节点的价值函数值放入Q中
                # print(self.V.shape)
                # print(self.C.shape)
                # print(act, s, act2)
                # print(self.V[act2, s])
                # print(self.C[act, s, act2])
                Q = np.concatenate((Q, [self.V[act2, s] + self.C[act, s, act2]]))
                possible_a = np.concatenate((possible_a, [act2]))
        max_arg = np.argmax(Q)
        if np.random.rand(1) < e:
            return np.random.choice(possible_a)
        else:
            return possible_a[max_arg]

    def MAXQ_step(self, i, s, FL_LOCAL_STEP=8):
        self.FL_LOCAL_STEP = FL_LOCAL_STEP

        # MAXQ递归
        if self.done:
            i = 11                  # to end recursion
            return 0
        self.done = False
        if self.is_primitive(i):
            if self.t > self.max_t:
                self.done = True
                return 0
            self.new_s, r, done, _ = copy.copy(self.env.step(i))
            self.done = done
            self.t += 1
            self.r_sum += r
            self.num_of_ac += 1
            self.V[i, s] += self.alpha * (r - self.V[i, s])
            self.atomic_action_count += 1
            return 1
        elif i <= self.root:
            count = 0
            while not self.is_terminal(i, self.done):
                a = self.greed_act(i, s)
                self.log.action_history.append(a)
                self.log.state_history.append(self.env.state)
                N = self.MAXQ_step(a, s)
                self.V_copy = self.V.copy()
                evaluate_res = self.evaluate(i, self.new_s)
                self.C[i, s, a] += self.alpha * (self.gamma ** N * evaluate_res - self.C[i, s, a])
                count += N
                s = self.new_s
            return count

    def MAXQ_generator(self):
        while True:
            self.MAXQ_step(self.root, self.new_s)
            if self.atomic_action_count >= self.FL_LOCAL_STEP:
                yield self.C, self.V
                self.atomic_action_count = 0

    def step(self):
        return next(self.step_generator)

    def reset(self):
        # 重置环境
        self.env.reset()
        self.r_sum = 0
        self.num_of_ac = 0
        self.done = False
        self.new_s = copy.copy(self.env.s)
        self.t = 0
        self.step_generator = self.step_generator = self.MAXQ_generator()


# %%
alpha = 0.1  # 设置学习率
gamma = 1  # 设置折扣因子
# maze_cross = np.loadtxt('maze8n_1.txt')
# maze_cross = np.loadtxt('maze17_0.2.txt')
maze_cross = np.loadtxt('maze_cross_level4.txt') * 0.0 + 1.0
env = TaxiEnv
FL = False
ShareQ = False
FL_LOCAL_STEP = 8
MAX_LOCAL_STEP = 512
NUMBER_OF_AGENTS = 1
training_episodes = 100000

server = FLServer(env(maze_cross), aggregation_method='QAvg')
agents = [Agent(env(maze_cross), server, alpha, gamma) for _ in range(NUMBER_OF_AGENTS)] 

agent_rewards = []
for agent in agents:
    server.update_table(agent.V, 0, 'V')
    server.update_table(agent.C, 0, 'C')
for agent in agents:
    agent.V = copy.deepcopy(server.distribute_V())
    agent.C = copy.deepcopy(server.distribute_C())

if ShareQ:
    for agent in agents:
        agent.V = agents[0].V
        agent.C = agents[0].C
# list(env.decode(env.s))
# env.action_space.n
# env.observation_space.n
# env.step(1)

# %%

for i in trange(training_episodes):

    for t in count():
        # print(t)
        # 所有agent运行一步
        for agent in agents:
            # # 如果距离MAX_LOCAL_STEP还足够运行FL_LOCAL_STEP
            # if t + FL_LOCAL_STEP <= MAX_LOCAL_STEP:
            #     agent.step(FL_LOCAL_STEP)
            # else:
            #     agent.step(MAX_LOCAL_STEP - t)
            agent.step()  # 如果不加以限制，就是一直运行到终止状态

        # if all(agent.update_count % FL_LOCAL_STEP == 0 for agent in agents):
        if t % FL_LOCAL_STEP == 0 and FL:
            # 聚合所有Agent
            for agent in agents:
                server.update_table(agent.V, i, 'V')
                server.update_table(agent.C, i, 'C')
            # if len(server.agent_q_tables) == NUMBER_OF_AGENTS:
            #     server.aggregate_q_tables()
            for agent in agents:
                agent.V = copy.deepcopy(server.distribute_V())
                agent.C = copy.deepcopy(server.distribute_C())

        if all(agent.done for agent in agents):  #  or t > MAX_LOCAL_EPISODE:
            reward_temp = []
            for agent in agents:
                reward_temp.append(agent.r_sum)
                agent.reset()
            agent_rewards.append(reward_temp)
            break
        

print("Training finished.\n")
# %%
# taxi = Agent(env, alpha, gamma)
# episodes = 15000
# sum_list = []
# for j in trange(episodes):
#     taxi.reset()
#     while not taxi.done:
#         # taxi.MAXQ_step(10, env.s)      # start in root
#         taxi.step()
#     sum_list.append(taxi.r_sum)
    # if (j % 1000 == 0):
    #     print('already made', j, 'episodes')
for i, (s, a) in enumerate(zip(agents[0].log.state_history, agents[0].log.action_history)):
    
    if i >= 1460000:
        print(s, a)
    
    if i >= 1470000:
        break

# %%
agents[0].V.shape


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

agent_rewards = np.array(agent_rewards)
sum_list = agent_rewards[:, 0]

sns.set(style='darkgrid')
sns.set(font_scale=1.5)
plt.figure(figsize=(15, 7.5))
sum_list_ = moving_average(sum_list)
plt.plot(sum_list_)
plt.xlabel('episode num')
plt.ylabel('points')
plt.show()

# %%
show_env = env(maze_cross)
show_env.reset()
print(show_env.render())

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
