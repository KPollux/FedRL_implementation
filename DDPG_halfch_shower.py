# %%
import copy
from itertools import count
import pickle
from matplotlib import legend_handler, pyplot as plt
import numpy as np
# import torch
# from cross_gridworld import Gridworld
from utils import DuelingDQNLast, draw_history
from IPython.display import clear_output

import pickle
import os
# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# %% draw with multiple folds
def smooth(data, sm=1):
    if sm > 1:
        smooth_data = []
        for d in data:
            y = np.ones(sm)*1.0/sm
            d = np.convolve(y, d, "same")

            smooth_data.append(d)
    else:
        smooth_data = copy.deepcopy(data)

    return smooth_data
    
def moving_average(data_list, window_size=100):
    moving_averages = []
    cumsum = [0]
    for i, x in enumerate(data_list, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=window_size:
            moving_aver = (cumsum[i] - cumsum[i-window_size])/window_size
            moving_averages.append(moving_aver)
    return moving_averages

# %%

# Cartpole
log_paths = [
    'DDPG_HalfCheetah_1step_SGD_INDL_0.0001_2023-08-20-05-58-18',
    'DDPG_HalfCheetah_1step_SGD_ShareParameter_0.0001_2023-08-20-09-27-01',
    'DDPG_HalfCheetah_1step_SGD_QAvg_0.0001_2023-08-20-09-27-02',
    # 'DDPG_HalfCheetah_1step_ShareParameter_2023-08-15-19-21-21',
    # 'DDPG_HalfCheetah_1step_ShareParameter_0.0001_2023-08-18-22-34-31',
    'DDPG_HalfCheetah_1step_SGD_QGradual_37500_decay_gap_1_warmup_6250_s1.0_e1.0_0.0001_2023-08-21-14-13-20',
    # 'DDPG_HalfCheetah_1step_SGD_QGradual_37500_decay_gap_1_warmup_6250_s1.0_e0.5_0.0001_2023-08-21-12-26-39',
    # 'DDPG_HalfCheetah_1step_SGD_QGradual_37500_decay_gap_1_warmup_6250_s1.0_e0.5_0.0001_2023-08-21-11-25-41'
]
# for i, path in enumerate(log_paths):
#     log_paths[i] = './logs/' + path + '/'

# log_paths = [
#     'DDPG_HalfCheetah_1step_SGD_QGradual_1_decay_gap_1_0.0001_2023-08-21-00-07-02',
#     'DDPG_HalfCheetah_1step_SGD_QGradual_1_decay_gap_1_0.0001_2023-08-21-00-18-28',  # GC
#     'DDPG_HalfCheetah_1step_SGD_QGradual_inf_decay_gap_1_warmup_6250_s1.0_e0.3333333333333333_0.0001_2023-08-21-03-31-10',

# ]

# log_paths = [
#     'DDPG_HalfCheetah_1step_ShareParameter_0.0001_2023-08-18-22-34-31',
#     'DDPG_HalfCheetah_1step_SGD_ShareParameter_0.0001_2023-08-20-09-27-01',
#     'DDPG_HalfCheetah_1step_QAvg_0.0001_2023-08-18-09-23-53',
#     'DDPG_HalfCheetah_1step_SGD_QAvg_0.0001_2023-08-20-09-27-02'
# ]

n_agents = 3

log_agent_rewards_list = []
log_agent_paths_length_list = []
log_agent_wins_list = []
for floder_name in log_paths:
    temp_log_agent_rewards_list = []
    temp_log_agent_paths_length_list = []
    temp_log_agent_wins_list = []

    # for i in range(5):
    #     with open(path + f'train_history_{i}.pkl', 'rb') as f:
    #         train_history = pickle.load(f)
    #     temp_log_agent_rewards_list.append(train_history['agent_rewards'])
    #     temp_log_agent_paths_length_list.append(train_history['agent_paths_length'])
    #     # temp_log_agent_wins_list.append(train_history['agent_wins'])

    #     if '5flod' not in path:
    #         print(f'warning: not 5flod for {path}')
    #         # 复制到5折
    #         for _ in range(4):
    #             temp_log_agent_rewards_list.append(train_history['agent_rewards'])
    #             temp_log_agent_paths_length_list.append(train_history['agent_paths_length'])
    #             # temp_log_agent_wins_list.append(train_history['agent_wins'])
    #         break
    with open('./logs/{}/train_history.pkl'.format(floder_name), 'rb') as f:
        train_history = pickle.load(f)
    for idx in range(n_agents):
        idx = str(idx)
        temp_log_agent_rewards_list.append(train_history[idx]['agent_rewards'])
        # temp_log_agent_paths_length_list.append(train_history['agent_paths_length'])
        # temp_log_agent_wins_list.append(train_history['agent_wins'])

        # if '5flod' not in floder_name:
        #     print(f'warning: not 5flod for {floder_name}')
        #     # 复制到5折
        #     for _ in range(4):
        #         temp_log_agent_rewards_list.append(train_history[idx]['agent_rewards'])
        #         # temp_log_agent_paths_length_list.append(train_history['agent_paths_length'])
        #         # temp_log_agent_wins_list.append(train_history['agent_wins'])
        #     break

    log_agent_rewards_list.append(temp_log_agent_rewards_list)
    log_agent_paths_length_list.append(temp_log_agent_paths_length_list)
    # log_agent_wins_list.append(temp_log_agent_wins_list)



# EPISODES = 500# 50000
# env_size = 17
# legends = ['INDL', 'SQ', 'QAvg', 'QGradual', 'QAll']
# legends = ['IL', 'SQ', 'QAvg','QMax', 'QAll']
# legends = ['5', '10', '20', '50', '100']
# legends = ['INDL', 'SQ', 'QAvg', 'QAll']
# legends = ['FL', '5000', '10000', '500000']
# legends = ['INDL', 'SQ', 'QAvg', 'QGradual']  #, 'QMax', 'QAll', 'FLGreedEpsilon']
# legends = ['INDL', 'SQ', 'QAvg', 'QMax', 'QAll', 'FLGreedEpsilon']
legends = ['DDPG', 'DDPGShare', 'DDPGAvg', 'DDPGGradual']
# legends = ['DDPGGradual', 'DDPGGradual+GC', 'DDPGGradual+GC+Warmup']
# legends = ['DDPGShare-AdamInAgent', 'DDPGShare-SGDInAgent', 'DDPGAvg-AdamInAgent', 'DDPGAvg-SGDInAgent']



import numpy as np

def expand_arr(arr, size):
    # 假设arr是你的(2, 3, 300000)的数组
    # arr = np.array(log_agent_rewards_list)  # 这里只是为了示例，你可以使用你自己的数组

    # 增加一个新的轴并复制
    arr_expanded = arr[:, np.newaxis, :, :].repeat(size, axis=1)

    # 确认新的形状
    # print(arr_expanded.shape)  # 应该输出(2, 5, 3, 300000)

    return arr_expanded

log_agent_rewards_list_ori = np.array(log_agent_rewards_list)
print(log_agent_rewards_list_ori.shape)


# %%
x_axis = int(len(log_agent_rewards_list_ori[0, 0, :]) / 1000)

log_agent_rewards_avg_list = np.zeros((len(log_paths), n_agents, x_axis))
log_agent_rewards_sum_1000 = np.zeros((len(log_paths), n_agents, x_axis))
for i in range(len(log_paths)):
    for idx in range(3):
        # 在每一个step上求历史平均值
        log_array = log_agent_rewards_list_ori[i][idx]
        # avg_1000 = log_array.reshape(-1, 1000).sum(axis=1)
        # train_history[idx]['avg_rewards'] = np.array(train_history[idx]['agent_rewards']).cumsum() / np.arange(1, len(train_history[idx]['agent_rewards']) + 1)
        # log_agent_rewards_avg_list[i][idx] = avg_1000.cumsum() / np.arange(1, len(log_array)/1000 + 1)
        # print(avg_1000.shape)

        log_agent_rewards_sum_1000[i][idx] = log_array.reshape(-1, 1000).sum(axis=1)
    
# log_agent_paths_length_list = np.array(log_agent_paths_length_list)
# log_agent_wins_list = np.array(log_agent_wins_list)

# log_agent_rewards_avg_list.shape
log_agent_rewards_sum_1000.shape
print(log_agent_rewards_sum_1000)
# %%
log_agent_rewards_list = expand_arr(log_agent_rewards_sum_1000, 5)
# log_agent_rewards_list = expand_arr(log_agent_rewards_list_ori, 5)
log_agent_rewards_list.shape
# %%
# log_agent_rewards_list = expand_arr(log_agent_rewards_avg_list, 5)
# log_agent_paths_length_list = expand_arr(log_agent_paths_length_list, 5)

# print(log_agent_rewards_list.shape) # (3, 3, 1000) 3 tests, 3 agents, 1000 episodes

def tsplot(ax, data, **kw):
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd * 0.95, est + sd * 0.95)
    ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
    # ax.plot(x, est, **kw)
    est = moving_average(est, 64)
    line, = ax.plot(est, **kw)
    # ax.margins(x=0)
    return line

fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 创建1行3列的子图
agents = ['Agent 1', 'Agent 2', 'Agent 3']

# Set larger font size
plt.rcParams.update({'font.size': 15})
legend_handles = []

add_sd = False
window_size = 64
for i in range(3):
    for j in range(len(log_paths)):  # 循环遍历每个test
        data = copy.deepcopy(log_agent_rewards_list[j, :, i, :])

        if add_sd:
            line = tsplot(axs[i], data, label=legends[j])  # 绘制每个agent在每个test的数据
            if i == 0:  # 只收集一次图例句柄
                legend_handles.append(line)
        else:
            data = np.mean(data, axis=0)
            data = moving_average(data, window_size)
            # data = smooth(data, 2)
            axs[i].plot(range(window_size, len(data) + window_size), data, label=legends[j])  # 绘制每个agent在每个test的数据

    axs[i].set_title(agents[i])  # 设置子图标题
    axs[i].set_xlabel('Thousand Steps', fontsize=15)  # 设置x轴标签
    axs[i].set_ylabel('Running Average Rewards', fontsize=15)  # 设置y轴标签
    axs[i].grid(True)
    if not add_sd:
        axs[i].legend()  # 显示图例
    axs[i].tick_params(axis='both', which='major', labelsize=15)

    # axs[i].set_xlim([0, 30])
    # axs[i].set_ylim([-600, 500])

# 对每个子图添加图例
if add_sd:
    for ax in axs:
        ax.legend(handles=legend_handles, labels=[legends[j] for j in range(len(log_paths))])

plt.tight_layout()  # 调整子图间距
plt.show()  # 显示图像

# %% 正规化AUC
from scipy import integrate

# 初始化一个存储AUC的字典
auc_dict = {agent: [] for agent in agents}



for i in range(3):
    min_val = 1e5
    # 找到这个客户端的最小值
    for j in range(len(log_paths)):  # 循环遍历每个test
        data = log_agent_rewards_list[j, :, i, :]  # 取第j个算法的第i个agent的数据
        
        data = np.mean(data, axis=0)

        min_val = min(min_val, np.min(data))

    for j in range(len(log_paths)):  # 循环遍历每个test
        data = log_agent_rewards_list[j, :, i, :]  # 取第j个算法的第i个agent的数据
        
        data = np.mean(data, axis=0)

        # data = moving_average(data, 64)

        # print(min_val)
        # 计算需要平移的值，我们选择将最小值平移到0.001
        shift_val = 0.001 - min_val 

        # 对数据进行平移
        data_shifted = data + shift_val
        # if i == 1:
        #     plt.plot(data_shifted, label=legends[j])  # 绘制检马尔积后的数据

        # print(np.min(data_shifted))

        # print(min_val)

        # print(data_shifted)

        # 计算AUC，并将结果存储到字典中
        auc = integrate.simps(data_shifted[:30000])  # 使用Simpson积分法

        # # 计算每个区间的AUC
        # auc_early = integrate.simps(data_shifted[:10000])  # 计算前10000轮的AUC
        # auc_late = integrate.simps(data_shifted[10000:])  # 计算后40000轮的AUC

        # # # # 计算总的AUC，其中每个区间的AUC都乘以相应的权重
        # we = 0.2
        # wl = 1-we
        # auc = we * auc_early + wl * auc_late

        auc_dict[agents[i]].append(auc / (x_axis - 1))  # 正规化，除以x轴范围，x.shape[0] - 1 = len(x) - 1

# 打印每个agent的AUC结果
for agent in agents:
    print(f'{agent} AUCs: {auc_dict[agent]}')

# plt.legend()
# plt.show()

# print(auc_dict)

import numpy as np
import matplotlib.pyplot as plt

# 对AUC进行归一化
auc_norm_dict = {}
for agent in agents:
    max_val = max(auc_dict[agent])
    auc_norm_dict[agent] = [val / max_val for val in auc_dict[agent]]

# 设置柱状图的位置和宽度
N = len(agents)
ind = np.arange(N)  # x轴的位置
width = 0.1        # 柱子的宽度

fig, ax = plt.subplots(figsize=(10, 7))

# 绘制每个test的柱状图
for i in range(len(log_paths)):
    rects = ax.bar(ind + i * width - 0.1, [auc_norm_dict[agent][i] for agent in agents], width)

# 添加图例
ax.legend(legends, fontsize=15, loc='lower center')  # , loc='lower center'

# 添加标签和标题
ax.set_ylabel('Normalized AUC')
ax.set_xticks(ind + width)
ax.set_xticklabels(agents)
# ax.set_ylim([0.90, 1])
# ax.set_xlabel('Agents')
plt.grid(axis='y')
plt.show()


# %%
print(auc_norm_dict)



# %%
import pandas as pd
import numpy as np

num_algorithms = 6
num_folds = 5
num_agents = 3
num_episodes = 50000

# 假设你的log_agent_rewards_list的形状是(6, 5, 3, 50000)
# log_agent_rewards_list = np.random.rand(num_algorithms, num_folds, num_agents, num_episodes)  # 只是为了示例，你应该使用你的数据

algorithms = np.repeat(np.arange(num_algorithms), num_folds * num_agents * num_episodes)
folds = np.tile(np.repeat(np.arange(num_folds), num_agents * num_episodes), num_algorithms)
agents = np.tile(np.repeat(np.arange(num_agents), num_episodes), num_algorithms * num_folds)
episodes = np.tile(np.arange(num_episodes), num_algorithms * num_folds * num_agents)
rewards = log_agent_rewards_list.ravel()

df = pd.DataFrame({
    'algorithm': algorithms,
    'fold': folds,
    'agent': agents,
    'episode': episodes,
    'reward': rewards
})

# %%
# df

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# 创建子图
fig, axs = plt.subplots(1, num_agents, figsize=(18, 6)) 

# 创建一个字典，其中包含算法ID和算法名称
algorithm_names = {0: 'IL', 1: 'SQ', 2: 'QAvg', 3: 'QDynamicAvg', 4: 'QMax', 5: 'QAll'}

# 循环遍历每个Agent
for j in range(num_agents):
    # 创建一个空的DataFrame用于存储处理后的数据
    data_to_plot = pd.DataFrame()

    # 循环遍历每个算法
    for i in range(num_algorithms):
        # 筛选出当前算法和Agent的数据，并计算每个episode的平均奖励
        data = df[(df['algorithm'] == i) & (df['agent'] == j)].groupby('episode').mean().reset_index()
        # 计算滑动平均
        data['reward'] = data['reward'].rolling(window=1000).mean()
        # 使用算法名称替换算法ID
        data['algorithm'] = algorithm_names[i]
        # 将数据添加到存储处理后的数据的DataFrame中
        data_to_plot = pd.concat([data_to_plot, data])

    # 重置索引
    data_to_plot.reset_index(drop=True, inplace=True)

    # 使用Seaborn的lineplot函数绘制数据
    sns.lineplot(ax=axs[j], x='episode', y='reward', data=data_to_plot, errorbar='sd', hue='algorithm') 
    # 设置子图标题
    axs[j].set_title(f'Agent {j+1}')  
    # 设置x轴标签
    axs[j].set_xlabel('Episodes')  
    # 设置y轴标签
    axs[j].set_ylabel('Running Average Path Length')  

# 调整子图间距
plt.tight_layout()  
# 显示图像
plt.show() 


# %%
import seaborn as sns
import matplotlib.pyplot as plt

# 创建子图
fig, axs = plt.subplots(1, num_agents, figsize=(18, 6)) 

# 创建一个字典，其中包含算法ID和算法名称
algorithm_names = {0: 'IL', 1: 'SQ', 2: 'QAvg', 3: 'QDelta', 4: 'QMax', 5: 'QAll'}

# 循环遍历每个Agent
for j in range(num_agents):
    # 创建一个空的DataFrame用于存储处理后的数据
    data_to_plot = pd.DataFrame()

    # 循环遍历每个算法
    for i in range(num_algorithms):
        # 筛选出当前算法和Agent的数据
        data = df[(df['algorithm'] == i) & (df['agent'] == j)]
        # 使用算法名称替换算法ID
        data = df[(df['algorithm'] == i) & (df['agent'] == j)]
        data.loc[:, 'algorithm'] = algorithm_names[i]
        # 将数据添加到存储处理后的数据的DataFrame中
        data_to_plot = pd.concat([data_to_plot, data])

    # 重置索引
    data_to_plot.reset_index(drop=True, inplace=True)

    # 使用Seaborn的lineplot函数绘制数据，并计算并绘制置信区间
    sns.lineplot(ax=axs[j], x='episode', y='reward', data=data_to_plot, errorbar='sd', hue='algorithm') 
    # 设置子图标题
    axs[j].set_title(f'Agent {j+1}')  
    # 设置x轴标签
    axs[j].set_xlabel('Episodes')  
    # 设置y轴标签
    axs[j].set_ylabel('Running Average Path Length')  

# 调整子图间距
plt.tight_layout()  
# 显示图像
plt.show() 









# %%
log_agent_rewards_list.shape
# %%
def tsplot(ax, data, **kw):
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd * 0.95, est + sd * 0.95)
    ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
    # ax.plot(x, est, **kw)
    mdata = moving_average(est, 64)
    line, = ax.plot(mdata, **kw)
    # ax.margins(x=0)
    return line

fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 创建1行3列的子图
agents = ['Agent 1', 'Agent 2']

# Set larger font size
plt.rcParams.update({'font.size': 15})
legend_handles = []

add_sd = False

for i in range(2):
    for j in range(len(log_paths)):  # 循环遍历每个test
        data = log_agent_paths_length_list[j, :, i, :]

        if add_sd:
            line = tsplot(axs[i], data, label=legends[j])  # 绘制每个agent在每个test的数据
            if i == 0:  # 只收集一次图例句柄
                legend_handles.append(line)
        else:
            data = np.mean(data, axis=0)
            data = moving_average(data, 64)
            axs[i].plot(data, label=legends[j])  # 绘制每个agent在每个test的数据

    axs[i].set_title(agents[i])  # 设置子图标题
    axs[i].set_xlabel('Episodes', fontsize=15)  # 设置x轴标签
    axs[i].set_ylabel('Running Average Path Length', fontsize=15)  # 设置y轴标签
    axs[i].grid(True)
    if not add_sd:
        axs[i].legend()  # 显示图例
    axs[i].tick_params(axis='both', which='major', labelsize=15)

    axs[i].set_xlim([3000, 5000])
    axs[i].set_ylim([0, 500])

# 对每个子图添加图例
if add_sd:
    for ax in axs:
        ax.legend(handles=legend_handles, labels=[legends[j] for j in range(len(log_paths))])

plt.tight_layout()  # 调整子图间距
plt.show()  # 显示图像

# %%
with open('./logs/Q_learning_M17_20_0Reward_5flod_FLMax_2023-06-06-05-18-53/' +
           f'Q_table_0_4.pkl', 'rb') as f:
    Q_table = pickle.load(f)
# np.sum(Q_table)
Q_table.shape

# %%
import matplotlib.pyplot as plt

import numpy as np

# 假设你的Q-Table是 q_table
# q_table = np.random.rand(289, 4)

# 找出每个状态的最大动作
max_actions = np.argmax(Q_table, axis=1)

# 将它们重新排列成 17x17 的形式
max_actions = max_actions.reshape((17, 17))


plt.figure(figsize=(6, 6))
plt.imshow(max_actions, cmap='viridis')
plt.colorbar()
plt.title('Max actions for each state')
plt.show()

# %%
with open('./logs/Q_learning_M17_20_Independent_2023-05-30-00-15-20/train_history.pkl', 'rb') as f:
    train_history = pickle.load(f)

log_agent_paths_length = np.array(train_history['agent_paths_length'])
log_agent_paths_length_smooth = np.array(smooth(log_agent_paths_length, sm=10))

fig, ax = plt.subplots()

def tsplot(ax, data,**kw):
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd * 0.95, est + sd * 0.95)
    ax.fill_between(x,cis[0],cis[1],alpha=0.2, **kw)
    ax.plot(x,est,**kw)
    ax.margins(x=0)

tsplot(ax, log_agent_paths_length)

ax.set_title("tsplot")

plt.show()

# %%
log_agent_rewards_list = []
log_agent_paths_length_list = []

log_paths = [
    './logs/Q_learning_M17_20_Independent_2023-05-30-00-15-20/train_history.pkl',
    './logs/Q_learning_M17_20_Paramsshare_level4_2023-05-30-00-17-04/train_history.pkl',
    './logs/Q_learning_M17_20_FL_level4_2023-05-30-00-20-41/train_history.pkl',
    './logs/Q_learning_M17_20_FLDelta_2023-06-06-01-38-06/train_history_0.pkl',
    './logs/Q_learning_M17_20_FLMax_level4_2023-05-30-00-21-25/train_history.pkl',
    './logs/Q_learning_M17_20_all_delta_level4_2023-05-30-06-24-32/train_history_0.pkl',
]


for path in log_paths:
    with open(path, 'rb') as f:
        train_history = pickle.load(f)
    log_agent_rewards_list.append(train_history['agent_rewards'])
    log_agent_paths_length_list.append(train_history['agent_paths_length'])

n_agents = 3
EPISODES = 1000
env_size = 17
legends = ['IL', 'SQ', 'QAvg', 'QDelta', 'QMax', 'QAll']

log_agent_rewards_list = np.array(log_agent_rewards_list)
log_agent_paths_length_list = np.array(log_agent_paths_length_list)

print(log_agent_rewards_list.shape) # (3, 3, 1000) 3 tests, 3 agents, 1000 episodes

# draw_history(log_agent_rewards, log_agent_paths_length, n_agents, EPISODES, 64, title='Q_learning_Paramsshare ', ylims=[None, (0, 450)])
# %%
import matplotlib.pyplot as plt
import numpy as np

# 假设你的数据在 log_agent_rewards_list
# log_agent_rewards_list = np.random.rand(3, 3, 1000)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 创建1行3列的子图
agents = ['Agent 1', 'Agent 2', 'Agent 3']

# Set larger font size
plt.rcParams.update({'font.size': 15})

for i in range(3):
    for j in range(len(log_paths)):  # 循环遍历每个test
        data = log_agent_paths_length_list[j, i, :]
        data = moving_average(data, 64)
        axs[i].plot(data, label=legends[j])  # 绘制每个agent在每个test的数据

    axs[i].set_title(agents[i])  # 设置子图标题
    axs[i].set_xlabel('Episodes', fontsize=15)  # 设置x轴标签
    axs[i].set_ylabel('Running Average Path Length', fontsize=15)  # 设置y轴标签
    axs[i].grid(True)
    axs[i].legend()  # 显示图例
    axs[i].tick_params(axis='both', which='major', labelsize=15)

    axs[i].set_xlim([0, 400])
    axs[i].set_ylim([0, 200])

plt.tight_layout()  # 调整子图间距
plt.show()  # 显示图像

# %%
import matplotlib.pyplot as plt
import numpy as np

# 假设你的数据在 log_agent_rewards_list
# log_agent_rewards_list = np.random.rand(3, 3, 1000)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 创建1行3列的子图
agents = ['Agent 1', 'Agent 2', 'Agent 3']
num_stages = 10  # 将训练过程分成10个阶段

for i in range(3):
    # 计算每个阶段的奖励平均值
    stage_rewards = np.mean(log_agent_paths_length_list[:, i, :].reshape(3, num_stages, -1), axis=2)
    stage_rewards = stage_rewards[..., 1:]  # 去掉第一个阶段的数据
    box_plot = axs[i].boxplot(stage_rewards.T)  # 绘制箱型图
    axs[i].set_xticklabels([f'Test {j+1}' for j in range(len(box_plot['boxes']))])  # 设置x轴标签
    axs[i].set_title(agents[i])  # 设置子图标题
    axs[i].set_ylabel('Average rewards')  # 设置y轴标签

plt.tight_layout()  # 调整子图间距
plt.show()  # 显示图像

# %%
import matplotlib.pyplot as plt
import numpy as np

# 假设你的Q-Table是 q_table
# q_table = np.random.rand(289, 4)

# 找出每个状态的最大动作
max_actions = np.argmax(Q_table, axis=1)

# 将它们重新排列成 17x17 的形式
max_actions = max_actions.reshape((17, 17))

# 创建一个空的画布
fig, ax = plt.subplots(figsize=(8, 8))

# 创建一个代表每个动作的箭头的字典
arrows = {0: '↑', 1: '→', 2: '↓', 3: '←'}

# 循环遍历每个状态
for i in range(17):
    for j in range(17):
        # 在每个状态的位置画出代表该状态最优动作的箭头
        ax.text(j, i, arrows[max_actions[i, j]], ha='center', va='center')

# 设置x和y轴的范围
ax.set_xlim(-0.5, 16.5)
ax.set_ylim(16.5, -0.5)

# 隐藏坐标轴
ax.axis('off')

# 显示图像
plt.show()

# %%
