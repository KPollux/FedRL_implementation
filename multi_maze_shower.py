# %%
import copy
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
    else:
        smooth_data = copy.deepcopy(data)

    return smooth_data

# def moving_average(data_list, window_size=100):
#     moving_averages = []
#     for i in range(len(data_list)):
#         if i < window_size:
#             window_data = data_list[:i + 1]
#         else:
#             window_data = data_list[i - window_size + 1:i + 1]
#         average = sum(window_data) / len(window_data)
#         moving_averages.append(average)
    
#     return moving_averages
    
def moving_average(data_list, window_size=100):
    moving_averages = []
    cumsum = [0]
    for i, x in enumerate(data_list, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=window_size:
            moving_aver = (cumsum[i] - cumsum[i-window_size])/window_size
            moving_averages.append(moving_aver)
    return moving_averages


# def moving_average(data_list, window_size=100):
#     moving_averages = []
#     cumsum = [0]
#     for i, x in enumerate(data_list, 1):
#         cumsum.append(cumsum[i-1] + x)
#         if i >= window_size:
#             moving_aver = (cumsum[i] - cumsum[i-window_size]) / window_size
#             if i == window_size:
#                 for _ in range(window_size):
#                     moving_averages.append(moving_aver)
#             moving_averages.append(moving_aver)
#     return moving_averages





# %%

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
#     'Q_learning_M1720_5flod_ep1000_2023-06-13-00-33-17/',
#     # './logs/Q_learning_M1720_5flod_Paramsshare_ep1000_2023-06-13-00-33-59/',
#     'Q_learning_M1720_5flod_FL_10LE_ep1000_2023-06-13-00-34-32/',
#     # './logs/Q_learning_M1720_5flod_FLDelta_10LE_ep1000_2023-06-13-00-35-20/',
#     'Q_learning_M1720_5flod_FLAll_10LE_ep1000_2023-06-13-00-36-41/',
#     'Q_learning_M1720_5flod_FLMax_10LE_ep1000_2023-06-13-00-36-06/',
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
#     'Q_learning_M1720_5flod_dynamic_ep1000_2023-06-13-02-01-51/',
#     # './logs/Q_learning_M1720_5flod_dynamic_Paramsshare_ep1000_2023-06-13-02-03-29/',
#     'Q_learning_M1720_5flod_dynamic_FL_10LE_ep1000_2023-06-13-02-04-32/',
#     # './logs/Q_learning_M1720_5flod_dynamic_FLDelta_10LE_ep1000_2023-06-13-02-06-16/',
#     'Q_learning_M1720_5flod_dynamic_FLAll_10LE_ep1000_2023-06-13-02-09-24/',
#     'Q_learning_M1720_5flod_dynamic_FLMax_10LE_ep1000_2023-06-13-02-07-54/',
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
# log_paths = [
#     './logs/Q_learning_M1720_5flod_0Reward_dynamic_ep5000_2023-06-13-05-55-55/',
#     './logs/Q_learning_M1720_5flod_0Reward_dynamic_Paramsshare_ep5000_2023-06-13-06-07-53/',
#     './logs/Q_learning_M1720_5flod_0Reward_dynamic_FL_10LE_ep5000_2023-06-13-06-15-56/',
#     './logs/Q_learning_M1720_5flod_0Reward_dynamic_FLDelta_10LE_ep5000_2023-06-13-06-27-54/',
#     './logs/Q_learning_M1720_5flod_0Reward_dynamic_FLMax_10LE_ep5000_2023-06-13-06-39-28/',
#     './logs/Q_learning_M1720_5flod_0Reward_dynamic_FLAll_10LE_ep5000_2023-06-13-06-49-09/',
# ]

# # 静态难度地图、三Agent、相同奖励、8ls（文章1）
# log_paths = [
#     'Q_learning_M1720_5flod_ep1000_2023-06-18-00-04-04',
#     # 'Q_learning_M1720_5flod_Paramsshare_ep1000_2023-06-18-00-04-46',
#     'Q_learning_M1720_5flod_FL_8LStep_ep1000_2023-06-18-00-05-19',
#     # 'Q_learning_M1720_5flod_FLDelta_8LStep_ep1000_2023-06-18-00-06-08',
#     'Q_learning_M1720_5flod_FLMax_8LStep_ep1000_2023-06-18-00-06-53',
#     # 'Q_learning_M1720_5flod_FLAll_8LStep_ep1000_2023-06-18-00-07-24',
# ]

# 静态难度地图、三Agent、相同奖励(无启发式)、8ls（文章1）
# log_paths = [
#     'Q_learning_M1720_noHRwd_5flod_ep1000_2023-06-18-00-45-18',
#     'Q_learning_M1720_noHRwd_5flod_Paramsshare_ep1000_2023-06-18-00-46-01',
#     'Q_learning_M1720_noHRwd_5flod_FL_8LStep_ep1000_2023-06-18-00-46-33',
#     # 'Q_learning_M1720_noHRwd_5flod_FLDynamicAvg_8LStep_ep1000_2023-06-19-18-11-35',  # 衰减100轮
#     # 'Q_learning_M1720_noHRwd_5flod_FLDynamicAvg_8LStep_ep1000_2023-06-19-18-20-11',  # 衰减500轮
#     'Q_learning_M1720_noHRwd_5flod_FLDynamicAvg_8LStep_ep1000_2023-06-19-19-27-04',  # 衰减1000轮
#     # 'Q_learning_M1720_noHRwd_5flod_FLMax_8LStep_ep1000_2023-06-18-00-48-08',
#     # 'Q_learning_M1720_noHRwd_5flod_FLAll_8LStep_ep1000_2023-06-18-00-48-37',
#     # 'Q_learning_M1720_noHRwd_5flod_FLGreedEpsilon_8LStep_ep1000_2023-07-27-00-55-28',
#     'Q_learning_ER_M1720_noHRwd_5flod_ep1000_2023-08-07-02-37-54',
#     'Q_learning_M1720_noHRwd_5flod_Independent_DoubleQ_ep1000_2023-08-04-01-39-18',
# ]

# 静态地图参数
# log_paths = [
#     'Q_learning_M1720_noHRwd_5flod_FL_8LStep_ep1000_2023-06-18-00-46-33',
#     'Q_learning_M1720_noHRwd_5flod_FLDynamicAvg_8LStep_ep1000_2023-06-19-18-11-35',  # 衰减100轮
#     'Q_learning_M1720_noHRwd_5flod_FLDynamicAvg_8LStep_ep1000_2023-06-19-18-20-11',  # 衰减500轮
#     'Q_learning_M1720_noHRwd_5flod_FLDynamicAvg_8LStep_ep1000_2023-06-19-19-27-04',  # 衰减1000轮
#     'Q_learning_M1720_noHRwd_5flod_FLAll_8LStep_ep1000_2023-06-18-00-48-37',
# ]

# # 动态难度地图、三Agent、相同奖励(无启发式)、8ls（文章1）
# log_paths = [
#     'Q_learning_M1720_noHRwd_Dynamic_5flod_dynamic_ep1000_2023-06-18-00-50-47',
#     'Q_learning_M1720_noHRwd_Dynamic_5flod_dynamic_Paramsshare_ep1000_2023-06-18-00-52-57',
#     'Q_learning_M1720_noHRwd_Dynamic_5flod_dynamic_FL_8LStep_ep1000_2023-06-18-00-54-13',
#     # 'Q_learning_M1720_noHRwd_Dynamic_5flod_dynamic_FLDelta_8LStep_ep1000_2023-06-18-00-56-31',
#     'Q_learning_M1720_noHRwd_Dynamic_5flod_dynamic_FLMax_8LStep_ep1000_2023-06-18-00-58-33',
#     'Q_learning_M1720_noHRwd_Dynamic_5flod_dynamic_FLAll_8LStep_ep1000_2023-06-18-00-59-48',
#     'Q_learning_M1720_noHRwd_5flod_dynamic_FLGreedEpsilon_8LStep_ep1000_2023-07-27-02-48-24',
    
# ]



# # 冰冻难度地图、三Agent、相同奖励(无启发式)、8ls（文章1）
# log_paths = [
#     'Q_learning_M1720Forzen_noHRwd_5flod_ep50000_2023-06-18-03-07-40',
#     'Q_learning_M1720Forzen_noHRwd_5flod_Paramsshare_ep50000_2023-06-18-03-22-03',
#     'Q_learning_M1720Forzen_noHRwd_5flod_FL_8LStep_ep50000_2023-06-18-03-36-13',
#     'Q_learning_M1720Forzen_noHRwd_5flod_FLDynamicAvg_8LStep_ep50000_2023-06-20-02-51-11', # 'Q_learning_M1720Forzen_noHRwd_FLDynamicAvg_8LStep_ep50000_2023-06-19-01-43-33',  # 渐变1阶 5000轮
#     # 'Q_learning_M1720Forzen_noHRwd_5flod_FLMax_8LStep_ep50000_2023-06-18-04-04-30',
#     # 'Q_learning_M1720Forzen_noHRwd_5flod_FLAll_8LStep_ep50000_2023-06-18-04-19-31',
#     # 'Q_learning_M1720Forzen_noHRwd_FLDynamicAvg_8LStep_ep50000_2023-06-19-01-13-21',  # 跳变
#     # 'Q_learning_M1720Forzen_noHRwd_5flod_FLDynamicAvg_8LStep_ep50000_2023-06-27-02-38-04',  # 衰减2500轮
#     # 'Q_learning_M1720Forzen_noHRwd_FLDynamicAvg_8LStep_ep50000_2023-06-19-01-50-13', # 渐变2阶
#     # 'Q_learning_M1720Forzen_noHRwd_FLDynamicAvg_8LStep_ep50000_2023-06-19-18-25-35',  # 衰减10000轮
#     # 'Q_learning_M1720Forzen_noHRwd_5flod_FLDynamicAvg_8LStep_ep50000_2023-06-20-01-43-30', # 衰减50000轮
#     # 'Q_learning_M1720Forzen_noHRwd_5flod_FLGreedEpsilon_ep50000_2023-07-27-15-32-30',  # 衰减200轮 - 很慢
#     # 'Q_learning_M1720Forzen_noHRwd_5flod_FLGreedEpsilon_8LStep_ep50000_2023-07-27-15-53-00',  # 衰减5000轮
#     # 'Q_learning_M1720Forzen_noHRwd_5flod_FLGreedEpsilon_8LStep_ep50000_2023-07-27-16-18-14',  # 衰减15000轮
#     'Q_learning_ER_M1720Forzen_noHRwd_5flod_ep50000_2023-08-07-02-50-18',
#     'Q_learning_M1720Forzen_noHRwd_5flod_Independent_DoubleQ_ep50000_2023-08-04-02-10-26'
    
# ]

# log_paths = [
#     'Q_learning_M1720Forzen_noHRwd_5flod_FL_8LStep_ep50000_2023-06-18-03-36-13',
#     'Q_learning_M1720Forzen_noHRwd_5flod_FLGreedEpsilon_ep50000_2023-07-27-15-32-30',  # 衰减200轮 - 很慢
#     'Q_learning_M1720Forzen_noHRwd_5flod_FLDynamicAvg_8LStep_ep50000_2023-06-27-02-38-04',  # 衰减2500轮
#     'Q_learning_M1720Forzen_noHRwd_5flod_FLDynamicAvg_8LStep_ep50000_2023-06-20-02-51-11', # 'Q_learning_M1720Forzen_noHRwd_FLDynamicAvg_8LStep_ep50000_2023-06-19-01-43-33',  # 渐变1阶 5000轮
#     'Q_learning_M1720Forzen_noHRwd_FLDynamicAvg_8LStep_ep50000_2023-06-19-18-25-35',  # 衰减10000轮
#     'Q_learning_M1720Forzen_noHRwd_5flod_FLGreedEpsilon_8LStep_ep50000_2023-07-27-16-18-14',  # 衰减15000轮
#     'Q_learning_M1720Forzen_noHRwd_5flod_FLDynamicAvg_8LStep_ep50000_2023-06-20-01-43-30', # 衰减50000轮
#     'Q_learning_M1720Forzen_noHRwd_5flod_FLAll_8LStep_ep50000_2023-06-18-04-19-31',

# ]

# 冰冻动态地图、三Agent、相同奖励(无启发式)、8ls（文章1）
# log_paths = [
#     'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_ep50000_2023-06-23-22-27-20',
#     'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_Paramsshare_ep50000_2023-06-23-22-45-30',
#     'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_FL_8LStep_ep50000_2023-06-23-23-03-03',
#     # 'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_FLAll_8LStep_ep50000_2023-06-23-23-36-20',
#     'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_FLDynamicAvg_8LStep_ep50000_2023-06-23-23-19-39',
#     # 'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_FLRewardShape_8LStep_ep50000_2023-06-24-17-05-50',
#     # 'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_FLRewardShape_8LStep_ep50000_2023-06-24-16-38-18',
#     # 'Q_learning_M1720Forzen_noHRwd_5flod_FLRewardShape_8LStep_ep50000_2023-06-24-00-12-15',
#     'Q_learning_ER_M1720Forzen_noHRwd_5flod_dynamic_ep50000_2023-08-07-03-22-18',
#     'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_Independent_DoubleQ_ep50000_2023-08-04-02-25-40',
# ]

# 冰冻动态地图参数实验
log_paths = [
    'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_FL_8LStep_ep50000_2023-06-23-23-03-03',
    # 'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_FLGreedEpsilon_200_8LStep_ep50000_2023-08-30-17-34-58',
    # 'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_FLGreedEpsilon_2500_8LStep_ep50000_2023-08-30-17-34-59',
    # 'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_FLGreedEpsilon_5000_8LStep_ep50000_2023-08-30-17-35-30',
    # 'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_FLGreedEpsilon_10000_8LStep_ep50000_2023-08-30-17-36-09',
    # 'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_FLGreedEpsilon_15000_8LStep_ep50000_2023-08-30-17-36-17',
    # 'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_FLGreedEpsilon_50000_8LStep_ep50000_2023-08-30-17-37-49',
    # 'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_FLGreedEpsilon_inf_8LStep_ep50000_2023-08-30-17-39-31'
    'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_FLDynamicAvg_200_8LStep_ep50000_2023-08-30-20-03-30',
    'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_FLDynamicAvg_2500_8LStep_ep50000_2023-08-30-20-03-40',
    'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_FLDynamicAvg_5000_8LStep_ep50000_2023-08-30-20-03-02',
    'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_FLDynamicAvg_10000_8LStep_ep50000_2023-08-30-20-03-44',
    'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_FLDynamicAvg_15000_8LStep_ep50000_2023-08-30-20-04-25',
    'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_FLDynamicAvg_50000_8LStep_ep50000_2023-08-30-20-05-16',
    'Q_learning_M1720Forzen_noHRwd_5flod_dynamic_FLDynamicAvg_inf_8LStep_ep50000_2023-08-30-20-06-26',

]

for i, path in enumerate(log_paths):
    log_paths[i] = './logs/' + path + '/'


log_agent_rewards_list = []
log_agent_paths_length_list = []
log_agent_wins_list = []
for path in log_paths:
    temp_log_agent_rewards_list = []
    temp_log_agent_paths_length_list = []
    temp_log_agent_wins_list = []

    for i in range(5):
        with open(path + f'train_history_{i}.pkl', 'rb') as f:
            train_history = pickle.load(f)
        temp_log_agent_rewards_list.append(train_history['agent_rewards'])
        temp_log_agent_paths_length_list.append(train_history['agent_paths_length'])
        # temp_log_agent_wins_list.append(train_history['agent_wins'])

        if '5flod' not in path:
            print(f'warning: not 5flod for {path}')
            # 复制到5折
            for _ in range(4):
                temp_log_agent_rewards_list.append(train_history['agent_rewards'])
                temp_log_agent_paths_length_list.append(train_history['agent_paths_length'])
                # temp_log_agent_wins_list.append(train_history['agent_wins'])
            break

    log_agent_rewards_list.append(temp_log_agent_rewards_list)
    log_agent_paths_length_list.append(temp_log_agent_paths_length_list)
    # log_agent_wins_list.append(temp_log_agent_wins_list)


n_agents = 3
EPISODES = 50000# 50000
env_size = 17
# legends = ['INDL', 'SQ', 'QAvg', 'QGradual', 'QAll']
# legends = ['IL', 'SQ', 'QAvg','QMax', 'QAll']
# legends = ['5', '10', '20', '50', '100']
# legends = ['INDL', 'SQ', 'QAvg', 'QAll']
# legends = ['FL', '5000', '10000', '500000']
# legends = ['Q-Learning', 'SQ', 'QAvg', 'QGradual', 'Q-ER', 'DoubleQ']
# legends = ['Decay-0(FedAvg)', 'Decay-100', 'Decay-500', 'Decay-1000', 'NoDecay']
legends = ['Decay-0(FedAvg)', 'Decay-200', 'Decay-2500', 'Decay-5000', 'Decay-10000', 'Decay-15000', 'Decay-50000', 'NoDecay']
# legends = ['Q-Learning', 'SQ', 'QAvg', 'QGradual', 'DoubleQ']

# legends = ['INDL', 'SQ', 'QAvg', 'QMax', 'QAll', 'FLGreedEpsilon']


# %%
log_agent_rewards_list = np.array(log_agent_rewards_list)
log_agent_paths_length_list = np.array(log_agent_paths_length_list)
# log_agent_wins_list = np.array(log_agent_wins_list)

print(log_agent_rewards_list.shape) # (3, 3, 1000) 3 tests, 3 agents, 1000 episodes

# %%
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
window_size = 6400
for i in range(3):
    for j in range(len(log_paths)):  # 循环遍历每个test
        data = copy.deepcopy(log_agent_rewards_list[j, :, i, :])

        # if j == 2:
        #     data -= 0.3

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
    axs[i].set_xlabel('Episodes', fontsize=15)  # 设置x轴标签
    axs[i].set_ylabel('Running Average Rewards', fontsize=15)  # 设置y轴标签
    axs[i].grid(True)
    if not add_sd:
        axs[i].legend()  # 显示图例
    axs[i].tick_params(axis='both', which='major', labelsize=15)

    # axs[i].set_xlim([0, 400])
    # axs[i].set_ylim([-50, None])

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
        auc = integrate.simps(data_shifted[:])  # 使用Simpson积分法

        # # 计算每个区间的AUC
        # auc_early = integrate.simps(data_shifted[:10000])  # 计算前10000轮的AUC
        # auc_late = integrate.simps(data_shifted[10000:])  # 计算后40000轮的AUC

        # # # # 计算总的AUC，其中每个区间的AUC都乘以相应的权重
        # we = 0.2
        # wl = 1-we
        # auc = we * auc_early + wl * auc_late

        auc_dict[agents[i]].append(auc / (EPISODES - 1))  # 正规化，除以x轴范围，x.shape[0] - 1 = len(x) - 1

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
ax.set_ylim([0.95, 1])
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
