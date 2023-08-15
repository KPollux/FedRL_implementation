# %%
import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat, configure


import numpy as np
import os
import time

import torch
import pickle

import matplotlib.pyplot as plt
import multiprocessing as mp
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from easydict import EasyDict as edict
import math


class RewardLoggerCallback(BaseCallback):
    """
    A custom callback that logs the reward for each step.
    """

    def __init__(self, agent_idx=0, train_history=None):
        super(RewardLoggerCallback, self).__init__()
        self.train_history = train_history
        self.history = self.train_history[str(agent_idx)]

    def _on_step(self) -> bool:
        # Get the latest reward and print it
        reward = self.locals['rewards'][0]
        self.history['agent_rewards'].append(reward)
        # print(f"Reward at step {self.num_timesteps}: {reward}")
        # print(self.locals['rewards'])
        # print(self.locals)
        
        # Return True to continue training
        return True
# %%
from tqdm import trange
agent_num = 1
device = torch.device("cuda:1")

modes = ['INDL', 'ShareParameter', 'QAvg', 'QGradual']
mode = modes[0]
print(mode)

LOCAL_EPISODES = 8

# DDPG 中使用的噪声对象
env = gym.make('HalfCheetah-v3')
n_actions = env.action_space.shape[0]
# 创建 DDPG 模型
agent = DDPG(
    "MlpPolicy",
    env,
    # action_noise=NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)),
    verbose=0,
    device = device,
    train_freq = 1,
    gradient_steps = 1,
    # learning_starts = 100,
    # tensorboard_log="./custom_tensorboard/HalfCheetah/INDL/",
)


# global_weights = agents[0].policy.state_dict()
# global_weights.keys()

train_history = edict()
for idx in range(agent_num):
    idx = str(idx)
    train_history[idx] = {'agent_rewards':[], 'steps':[]}

for _ in trange(math.ceil(3_000_000/LOCAL_EPISODES)):
    agent.learn(total_timesteps=LOCAL_EPISODES, log_interval=1, reset_num_timesteps=False,
                            callback=RewardLoggerCallback(0, train_history))

# %%
plt.plot(train_history['0']['agent_rewards'])
# 按每1000求和
plt.plot(np.array(train_history['0']['agent_rewards']).reshape(-1, 1000).sum(axis=1))
# %%
# 绘制历史平均曲线
agent_num = 1

for idx in range(agent_num):
    idx = str(idx)
    # 在每一个step上求历史平均值
    train_history[idx]['avg_rewards'] = np.array(train_history[idx]['agent_rewards']).cumsum() / np.arange(1, len(train_history[idx]['agent_rewards']) + 1)
    plt.plot(train_history[idx]['avg_rewards'], label=f'agent_{idx}')

# %%
# 保存训练历史
mode_name = mode
if mode == 'QGradual':
    mode_name = mode + '_' + str(EPS_DECAY)
floder_name = 'DDPG_HalfCheetah_1step_' + mode_name + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

# create floder
if not os.path.exists('./logs/' + floder_name):
    os.mkdir('./logs/' + floder_name)

with open('./logs/{}/train_history.pkl'.format(floder_name), 'wb') as f:
    pickle.dump(train_history, f)

# %%
# 读取训练历史
# floder_name = 'DDPG_HalfCheetah_QGradual_1_2023-08-14-15-12-13'
# floder_name = 'DDPG_HalfCheetah_QAvg_2023-08-14-02-35-50'
floder_name = 'DDPG_HalfCheetah_ShareParameter_2023-08-13-20-41-57'
# floder_name = 'DDPG_HalfCheetah_QGradual_0_2023-08-14-18-59-49'
# floder_name = 'DDPG_HalfCheetah_QGradual_1_2023-08-14-15-12-13'
with open('./logs/{}/train_history.pkl'.format(floder_name), 'rb') as f:
    train_history = pickle.load(f)

agent_num = 3

for idx in range(agent_num):
    idx = str(idx)
    # 在每一个step上求历史平均值
    train_history[idx]['avg_rewards'] = np.array(train_history[idx]['agent_rewards']).cumsum() / np.arange(1, len(train_history[idx]['agent_rewards']) + 1)
    plt.plot(train_history[idx]['avg_rewards'], label=f'agent_{idx}')

# %%
# %%
# train_history['0']['agent_rewards']

# # %%
# agent.locals["rewards"]

# # %%
# # %%
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# def tensorboard_to_dict(log_dir):
#     event_acc = EventAccumulator(log_dir)
#     event_acc.Reload()  # 加载 event 文件

#     # 显示所有可以提取的张量名称
#     print(event_acc.Tags()['scalars'])

#     data = {}

#     for tag in event_acc.Tags()['scalars']:
#         values = []
#         for scalar_event in event_acc.Scalars(tag):
#             values.append(scalar_event.value)
#         data[tag] = values

#     return data

# log_dir = "./custom_tensorboard/HalfCheetah/INDL/DDPG_4"
# data = tensorboard_to_dict(log_dir)
# print(data)
# # %%
# print(data.keys())
# plt.plot(data['rollout/ep_rew_mean'])
# plt.xlabel('Thousands of Frames')
# plt.ylabel('Average Reward')

# # %%
# # 保存模型
# # model.save("ddpg_half_cheetah")

# # 加载模型并测试
# # loaded_model = DDPG.load("ddpg_half_cheetah")
# # loaded_model = model
# # obs = env.reset()
# # for _ in range(1000):
# #     action, _states = loaded_model.predict(obs, deterministic=True)
# #     obs, rewards, dones, info = env.step(action)
# #     env.render()

# # %%
# # weights = model.policy.state_dict()
# # %%
# import multiprocessing
# from multiprocessing import Value

# from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3 import DDPG
# import gym
# import numpy as np

# def agent_process(agent_id, weight_queue, updated_weight_event, done_counter, pause_interval=8):
#     env = gym.make('HalfCheetah-v3')
#     n_actions = env.action_space.shape[0]
#     action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
#     model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    
#     # Callback to handle communication with the central server
#     class FedCallback(BaseCallback):
#         def __init__(self, pause_interval=8, verbose=0):
#                 super(PauseCallback, self).__init__(verbose)
#                 self.pause_interval = pause_interval
#                 self.num_updates = 0  # 用于跟踪网络更新的次数

#         def _on_step(self) -> bool:              
#             return True

#         def _on_rollout_start(self):
#             self.num_timesteps += 1

#             if self.num_timesteps % pause_interval == 0:
#                 # Send weights to the central server
#                 weight_queue.put((agent_id, model.policy.state_dict()))
                
#                 # Wait for updated weights from the central server
#                 updated_weight_event.wait()
#                 updated_weights = weight_queue.get()
#                 model.policy.load_state_dict(updated_weights)
#                 updated_weight_event.clear()

#             self._on_rollout_start()
        
#     model.learn(total_timesteps=3_000, callback=FedCallback())
#     done_counter += 1

# # Central server logic to aggregate weights
# def aggregate_weights(agent_weights):
#     # Here, implement FedAvg aggregation logic
#     # This is a naive average for demonstration purposes
#     avg_weights = {k: sum(t[k] for t in agent_weights) / len(agent_weights) for k in agent_weights[0]}
#     return avg_weights

# if __name__ == "__main__":
#     agent_num = 3
#     manager = multiprocessing.Manager()
#     weight_queue = manager.Queue()
#     updated_weight_event = manager.Event()
#     done_counter = Value('i', 0)

#     processes = []
#     for i in range(agent_num):
#         p = multiprocessing.Process(target=agent_process, args=(i, weight_queue, updated_weight_event))
#         processes.append(p)
#         p.start()

#     while done_counter.value < agent_num:
#         collected_weights = []
#         for _ in range(agent_num):
#             agent_id, weights = weight_queue.get()
#             collected_weights.append(weights)
        
#         # Aggregate the weights
#         avg_weights = aggregate_weights(collected_weights)
        
#         for _ in range(agent_num):
#             weight_queue.put(avg_weights)
#             updated_weight_event.set()

#     for p in processes:
#         p.join()

            
