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

# %%

from stable_baselines3.common.callbacks import BaseCallback

class PauseCallback(BaseCallback):
    def __init__(self, pause_interval=8, verbose=0):
        super(PauseCallback, self).__init__(verbose)
        self.pause_interval = pause_interval
        self.num_updates = 0  # 用于跟踪网络更新的次数

    def _on_step(self) -> bool:              
        return True
    
    def on_rollout_start(self) -> None:
        self.num_updates += 1
        print(self.num_updates)
        if self.num_updates == self.pause_interval:
            print('aggreate')

        self._on_rollout_start()

# %%
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from easydict import EasyDict as edict
import math

class CustomLogger(BaseCallback):
    def __init__(self, agent_idx=0, train_history=None):
        super(CustomLogger, self).__init__()
        self.train_history = train_history
        self.history = self.train_history[str(agent_idx)]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos:
            latest_info = infos[-1]
            # print(latest_info.keys())
            # print(latest_info.get('episode', {}))
            ep_rew = latest_info.get('episode', {}).get('r', None)
            
            if ep_rew is not None:
                self.history['agent_rewards'].append(ep_rew)
                self.history['steps'].append(self.num_timesteps)
                # print(f"Step: {self.num_timesteps} - Episode Reward: {ep_rew}")

        return True


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
agent_num = 3
device = torch.device("cuda:0")

modes = ['INDL', 'ShareParameter', 'QAvg', 'QGradual']
mode = modes[2]
print(mode)

LOCAL_EPISODES = 8

# DDPG 中使用的噪声对象
envs = [gym.make('HalfCheetah-v3') for _ in range(agent_num)]
n_actions = envs[0].action_space.shape[0]
# 创建 DDPG 模型
agents = [DDPG(
    "MlpPolicy",
    envs[i],
    # action_noise=NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)),
    verbose=0,
    device = device,
    train_freq = 1,
    gradient_steps = 1,
    # learning_starts = 100,
    # tensorboard_log="./custom_tensorboard/HalfCheetah/INDL/",
) for i in range(agent_num)]


# global_weights = agents[0].policy.state_dict()
# global_weights.keys()

train_history = edict()
for idx in range(agent_num):
    idx = str(idx)
    train_history[idx] = {'agent_rewards':[], 'steps':[]}

# %%

global_weights = agents[0].policy.state_dict()
# FL设定中，每个agent应该从相同的起点开始训练
for agent in agents:
    agent.policy.load_state_dict(global_weights)

if mode == 'INDL':
    for _ in trange(math.ceil(300_000/LOCAL_EPISODES)):
        for i, agent in enumerate(agents):
            # agent.learn(total_timesteps=1000, log_interval=1, callback=CustomLogger(i, train_history))
            agent.learn(total_timesteps=LOCAL_EPISODES, log_interval=1, reset_num_timesteps=False,
                         callback=RewardLoggerCallback(i, train_history))
            # print(agent.replay_buffer.pos)

elif mode == 'ShareParameter':
    for _ in trange(math.ceil(300_000/LOCAL_EPISODES)):
        for i, agent in enumerate(agents):
            # 每次训练前，将agent的参数设置为全局参数
            agent.policy.load_state_dict(global_weights)
            # agent.learn(total_timesteps=1000, log_interval=1, callback=CustomLogger(i, train_history))
            agent.learn(total_timesteps=LOCAL_EPISODES, log_interval=1, reset_num_timesteps=False,
                         callback=RewardLoggerCallback(i, train_history))
            # 每次训练后，将agent的参数设置为全局参数，同步给其他agent
            global_weights = agent.policy.state_dict()

elif mode == 'QAvg':
    for _ in trange(math.ceil(300_000/LOCAL_EPISODES)):
        upload_weights = []
        for i, agent in enumerate(agents):
            # 每次训练前，下载权重
            agent.policy.load_state_dict(global_weights)
            # agent.learn(total_timesteps=LOCAL_EPISODES*1000, log_interval=1, callback=CustomLogger(i, train_history))
            agent.learn(total_timesteps=LOCAL_EPISODES, log_interval=1, reset_num_timesteps=False,
                         callback=RewardLoggerCallback(i, train_history))
            # 每次训练后，上传权重
            upload_weights.append(agent.policy.state_dict())
    
        num_selected_agent = len(upload_weights)
        avg_weights = np.ones(num_selected_agent) / num_selected_agent
        global_weights = {}
        for i in range(num_selected_agent):
            # if i == 0:
            for key in upload_weights[0].keys():
                global_weights[key] = global_weights.get(key, 0) + upload_weights[i][key] * avg_weights[i]
            # else:
            #     for key in global_weights.keys():
            #             global_weights[key] += upload_weights[i][key] * avg_weights[i]

elif mode == 'QGradual':
    for i_communication in trange(math.ceil(50/LOCAL_EPISODES)):
        # === Agent ===
        upload_weights = []
        for i, agent in enumerate(agents):
            # 每次训练前，下载权重
            agent.policy.load_state_dict(global_weights)
            agent.learn(total_timesteps=LOCAL_EPISODES*1000, log_interval=1, callback=CustomLogger(i, train_history))
            # 每次训练后，上传权重
            upload_weights.append(agent.policy.state_dict())
    
        # === Server ===
        num_selected_agent = agent_num
        # avg_weights = np.ones(num_selected_agent)

        # Define the starting and ending epsilon values
        EPS_START = 1.0
        EPS_END = 1.0   # / num_selected_agent

        # Define the decay rate
        EPS_DECAY = math.ceil(10/LOCAL_EPISODES)
        # EPS_DECAY = math.ceil(300/LOCAL_EPISODES/3)

        # Assuming that `current_round` is the current training round
        if i_communication < EPS_DECAY:
            epsilon = EPS_START - ((EPS_START - EPS_END) * (i_communication / EPS_DECAY))
        else:
            epsilon = EPS_END

        avg_weights = np.ones(num_selected_agent) * epsilon

        delta_weights = {}
        # global_weights = {}
        for i in range(num_selected_agent):
            d_para = {}
            for key in upload_weights[0].keys():
                d_para[key] = global_weights[key] - upload_weights[i][key]

                delta_weights[key] = delta_weights.get(key, 0) + avg_weights[i] * d_para[key]

            # for key in upload_weights[0].keys():
            #     global_weights[key] = global_weights.get(key, 0) + upload_weights[i][key] * avg_weights[i]


        # 将平均后的差值加回到全局权重中，得到新的全局模型权重
        for key in upload_weights[0].keys():
            global_weights[key] = global_weights[key] - delta_weights[key]
               

# %%
# plt.plot(train_history['0']['agent_rewards'])
# 按每1000求和
# plt.plot(np.array(train_history['0']['agent_rewards']).reshape(-1, 1000).sum(axis=1))
# %%
# 绘制历史平均曲线
# agent_num = 3

for idx in range(agent_num):
    idx = str(idx)
    # 在每一个step上求历史平均值
    avg_1000 = np.array(train_history[idx]['agent_rewards']).reshape(-1, 1000).sum(axis=1)
    # train_history[idx]['avg_rewards'] = np.array(train_history[idx]['agent_rewards']).cumsum() / np.arange(1, len(train_history[idx]['agent_rewards']) + 1)
    train_history[idx]['avg_rewards'] = avg_1000.cumsum() / np.arange(1, len(train_history[idx]['agent_rewards'])/1000 + 1)

    plt.plot(train_history[idx]['avg_rewards'], label=f'agent_{idx}')

# %%
# 保存训练历史
mode_name = mode
if mode == 'QGradual':
    mode_name = mode + '_' + str(EPS_DECAY)
floder_name = 'DDPG_HalfCheetah_1step_' + mode_name + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
print(floder_name)
# create floder
if not os.path.exists('./logs/' + floder_name):
    os.mkdir('./logs/' + floder_name)

with open('./logs/{}/train_history.pkl'.format(floder_name), 'wb') as f:
    pickle.dump(train_history, f)

# %%
# 读取训练历史
# floder_name = 'DDPG_HalfCheetah_1step_INDL_2023-08-15-01-15-14'
# floder_name = 'DDPG_HalfCheetah_1step_QAvg_2023-08-15-02-25-34'
floder_name = 'DDPG_HalfCheetah_1step_ShareParameter_2023-08-15-02-16-50'
with open('./logs/{}/train_history.pkl'.format(floder_name), 'rb') as f:
    train_history = pickle.load(f)

agent_num = 3

for idx in range(agent_num):
    idx = str(idx)
    # 在每一个step上求历史平均值
    avg_1000 = np.array(train_history[idx]['agent_rewards']).reshape(-1, 1000).sum(axis=1)
    # train_history[idx]['avg_rewards'] = np.array(train_history[idx]['agent_rewards']).cumsum() / np.arange(1, len(train_history[idx]['agent_rewards']) + 1)
    train_history[idx]['avg_rewards'] = avg_1000.cumsum() / np.arange(1, len(train_history[idx]['agent_rewards'])/1000 + 1)

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

            
