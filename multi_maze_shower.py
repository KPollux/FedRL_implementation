# %%
from itertools import count
import pickle
from matplotlib import legend_handler, pyplot as plt
import numpy as np
import torch
from cross_gridworld import Gridworld
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

    return smooth_data

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
log_agent_rewards_list = []
log_agent_paths_length_list = []

# log_paths = [
#     './logs/Q_learning_M17_20_5flod_Independent_2023-06-06-03-26-39/',
#     './logs/Q_learning_M17_20_5flod_Paramsshare_2023-06-06-03-28-42/',
#     './logs/Q_learning_M17_20_5flod_FL_2023-06-06-04-50-34/',
#     './logs/Q_learning_M17_20_5flod_FLDelta_2023-06-06-04-57-56/',
#     './logs/Q_learning_M17_20_5flod_FLMax_2023-06-06-04-55-58/',
#     './logs/Q_learning_M17_20_5flod_FLAll_2023-06-06-04-56-54/',
# ]

# log_paths = [
#     './logs/Q_learning_M17_20_0Reward_5flod_Independent_2023-06-06-05-08-56/',
#     './logs/Q_learning_M17_20_0Reward_5flod_Paramsshare_2023-06-06-05-11-33/',
#     './logs/Q_learning_M17_20_0Reward_5flod_FL_2023-06-06-05-13-39/',
#     './logs/Q_learning_M17_20_0Reward_5flod_FLDelta_2023-06-06-05-16-27/',
#     './logs/Q_learning_M17_20_0Reward_5flod_FLMax_2023-06-06-05-18-53/',
#     './logs/Q_learning_M17_20_0Reward_5flod_FLAll_2023-06-06-05-21-24/',
# ]

# log_paths = [
#     './logs/Q_learning_M17_20_5LE_5flod_FLAll_2023-06-11-23-49-43/',
#     './logs/Q_learning_M17_20_10LE_5flod_FLAll_2023-06-11-23-51-06/',
#     './logs/Q_learning_M17_20_20LE_5flod_FLAll_2023-06-11-23-52-10/',
#     './logs/Q_learning_M17_20_50LE_5flod_FLAll_2023-06-11-23-53-10/',
#     './logs/Q_learning_M17_20_100LE_5flod_FLAll_2023-06-11-23-54-10/'
# ]

# log_paths = [
#     './logs/Q_learning_M17_20_0Reward_static_10LE_ep5000_5flod_Independent_2023-06-12-17-59-11/',
#     './logs/Q_learning_M17_20_0Reward_static_10LE_ep5000_5flod_Paramsshare_2023-06-12-18-07-47/',
#     './logs/Q_learning_M17_20_0Reward_static_10LE_ep5000_5flod_FL_2023-06-12-18-10-35/',
#     './logs/Q_learning_M17_20_0Reward_static_10LE_ep5000_5flod_FLDelta_2023-06-12-18-13-37/',
#     './logs/Q_learning_M17_20_0Reward_static_10LE_ep5000_5flod_FLMax_2023-06-12-18-16-33/',
#     './logs/Q_learning_M17_20_0Reward_static_10LE_ep5000_5flod_FLAll_2023-06-12-18-19-30/',
# ]

# log_paths = [
#     './logs/Q_learning_M17_20_0Reward_dynamic_10LE_ep5000_5flod_Independent_2023-06-12-20-51-26/',
#     './logs/Q_learning_M17_20_0Reward_dynamic_10LE_ep5000_5flod_Paramsshare_2023-06-12-21-02-52/',
#     './logs/Q_learning_M17_20_0Reward_dynamic_10LE_ep5000_5flod_FL_2023-06-12-21-10-57/',
#     './logs/Q_learning_M17_20_0Reward_dynamic_10LE_ep5000_5flod_FLDelta_2023-06-12-21-18-07/',
#     './logs/Q_learning_M17_20_0Reward_dynamic_10LE_ep5000_5flod_FLMax_2023-06-12-21-25-47/',
#     './logs/Q_learning_M17_20_0Reward_dynamic_10LE_ep5000_5flod_FLAll_2023-06-12-21-35-43/',
# ]

# 静态难地图、三Agent、相同奖励（对比算法）
# log_paths = [
#     './logs/Q_learning_M1720_5flod_ep1000_2023-06-13-00-33-17/',
#     './logs/Q_learning_M1720_5flod_Paramsshare_ep1000_2023-06-13-00-33-59/',
#     './logs/Q_learning_M1720_5flod_FL_10LE_ep1000_2023-06-13-00-34-32/',
#     './logs/Q_learning_M1720_5flod_FLDelta_10LE_ep1000_2023-06-13-00-35-20/',
#     './logs/Q_learning_M1720_5flod_FLMax_10LE_ep1000_2023-06-13-00-36-06/',
#     './logs/Q_learning_M1720_5flod_FLAll_10LE_ep1000_2023-06-13-00-36-41/',
# ]

# 静态难地图、三Agent、相同奖励（调整FLMax的通信轮次）
# log_paths = [
#     './logs/Q_learning_M1720_5flod_FLMax_5LE_ep1000_2023-06-13-01-22-45/',
#     './logs/Q_learning_M1720_5flod_FLMax_10LE_ep1000_2023-06-13-00-36-06/',
#     './logs/Q_learning_M1720_5flod_FLMax_20LE_ep1000_2023-06-13-01-27-18/',
#     './logs/Q_learning_M1720_5flod_FLMax_50LE_ep1000_2023-06-13-01-30-22/',
#     './logs/Q_learning_M1720_5flod_FLMax_100LE_ep1000_2023-06-13-01-57-53/',
# ]

# # 动态难地图、三Agent、相同奖励（对比算法）
# log_paths = [
#     './logs/Q_learning_M1720_5flod_dynamic_ep1000_2023-06-13-02-01-51/',
#     './logs/Q_learning_M1720_5flod_dynamic_Paramsshare_ep1000_2023-06-13-02-03-29/',
#     './logs/Q_learning_M1720_5flod_dynamic_FL_10LE_ep1000_2023-06-13-02-04-32/',
#     './logs/Q_learning_M1720_5flod_dynamic_FLDelta_10LE_ep1000_2023-06-13-02-06-16/',
#     './logs/Q_learning_M1720_5flod_dynamic_FLMax_10LE_ep1000_2023-06-13-02-07-54/',
#     './logs/Q_learning_M1720_5flod_dynamic_FLAll_10LE_ep1000_2023-06-13-02-09-24/',
# ]

# 静态难地图、二Agent、不同奖励（对比算法）
# log_paths = [
#     './logs/Q_learning_M1720_5flod_0Reward_ep5000_2023-06-13-05-21-36/',
#     './logs/Q_learning_M1720_5flod_0Reward_Paramsshare_ep5000_2023-06-13-05-30-18/',
#     './logs/Q_learning_M1720_5flod_0Reward_FL_10LE_ep5000_2023-06-13-05-33-11/',
#     './logs/Q_learning_M1720_5flod_0Reward_FLDelta_10LE_ep5000_2023-06-13-05-39-28/',
#     './logs/Q_learning_M1720_5flod_0Reward_FLMax_10LE_ep5000_2023-06-13-05-44-44/',
#     './logs/Q_learning_M1720_5flod_0Reward_FLAll_10LE_ep5000_2023-06-13-05-49-38/',
# ]

# 动态难地图、二Agent、不同奖励（对比算法）
log_paths = [
    './logs/Q_learning_M1720_5flod_0Reward_dynamic_ep5000_2023-06-13-05-55-55/',
    './logs/Q_learning_M1720_5flod_0Reward_dynamic_Paramsshare_ep5000_2023-06-13-06-07-53/',
    './logs/Q_learning_M1720_5flod_0Reward_dynamic_FL_10LE_ep5000_2023-06-13-06-15-56/',
    './logs/Q_learning_M1720_5flod_0Reward_dynamic_FLDelta_10LE_ep5000_2023-06-13-06-27-54/',
    './logs/Q_learning_M1720_5flod_0Reward_dynamic_FLMax_10LE_ep5000_2023-06-13-06-39-28/',
    './logs/Q_learning_M1720_5flod_0Reward_dynamic_FLAll_10LE_ep5000_2023-06-13-06-49-09/',
]



for path in log_paths:
    temp_log_agent_rewards_list = []
    temp_log_agent_paths_length_list = []
    for i in range(5):
        with open(path + f'train_history_{i}.pkl', 'rb') as f:
            train_history = pickle.load(f)
        temp_log_agent_rewards_list.append(train_history['agent_rewards'])
        temp_log_agent_paths_length_list.append(train_history['agent_paths_length'])
    log_agent_rewards_list.append(temp_log_agent_rewards_list)
    log_agent_paths_length_list.append(temp_log_agent_paths_length_list)


n_agents = 2
EPISODES = 1000
env_size = 17
legends = ['IL', 'SQ', 'QAvg', 'QDelta', 'QMax', 'QAll']
# legends = ['5', '10', '20', '50', '100']

log_agent_rewards_list = np.array(log_agent_rewards_list)
log_agent_paths_length_list = np.array(log_agent_paths_length_list)

print(log_agent_rewards_list.shape) # (3, 3, 1000) 3 tests, 3 agents, 1000 episodes

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

fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 创建1行3列的子图
agents = ['Agent 1', 'Agent 2', 'Agent 3']

# Set larger font size
plt.rcParams.update({'font.size': 15})
legend_handles = []

add_sd = False

for i in range(3):
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

    axs[i].set_xlim([0, 400])
    axs[i].set_ylim([0, 200])

# 对每个子图添加图例
if add_sd:
    for ax in axs:
        ax.legend(handles=legend_handles, labels=[legends[j] for j in range(len(log_paths))])

plt.tight_layout()  # 调整子图间距
plt.show()  # 显示图像
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
