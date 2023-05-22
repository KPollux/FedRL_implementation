# %%
from itertools import count
import pickle
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import trange
from cross_gridworld import Gridworld
from utils import DuelingDQNLast, ReplayMemory, DQN, Transition, draw_history, draw_path, moving_average, plot_rewards, sync_Agents_weights
import torch.nn.functional as F
import math
from IPython.display import clear_output

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def select_action(state, policy_net, steps_done):
    EPS_START = 0.9
    EPS_END = 0.1
    EPS_DECAY = 200

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was found,
            # so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1).item()
    else:
        return random.randrange(4)

    
def flatten(observation):
    return torch.tensor(observation, device=device).view(-1)


def optimize_model(policy_net, target_net, memory, batch_size, gamma, optimizer):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
    #                                       batch.next_state)), device=device, dtype=torch.bool)
    # non_final_next_states = torch.cat([s for s in batch.next_state
    #                                             if s is not None])
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not True,
                                          batch.done)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s, mask in zip(batch.next_state, non_final_mask) if mask])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    next_state_values = next_state_values.view(-1, 1)
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    # print(expected_state_action_values)

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# %%
def main(share_params=False, FL=False):
    # Hyperparameters
    EPISODES = 500
    MAX_STEPS_PER_EPISODE = 256  # Maximum steps per episode
    BATCH_SIZE = 64
    GAMMA = 0.9
    MEMORY_SIZE = 10000
    HISTORY_LENGTH = 20
    LR = 5e-5  # 5e-5ForOpenMap
    TARGET_UPDATE = 20
    TAU = 0.005
    MAZE = np.loadtxt('maze_cross.txt') * 0.0 + 1.0
    SIZE = MAZE.shape[0]
    FL_GLOBAL_UPDATE = 5

    # Create environment
    env = Gridworld(size=SIZE, n_agents=2, heuristic_reward=True, maze=MAZE)

    # Create DQN networks and optimizers for each agent
    policy_nets = [DuelingDQNLast(25, 32, 4).to(device) for _ in range(env.n_agents)]
    target_nets = [DuelingDQNLast(25, 32, 4).to(device) for _ in range(env.n_agents)]
    optimizers = [torch.optim.AdamW(net.parameters(), lr = LR) for net in policy_nets]

    # Initialize networks
    for i in range(env.n_agents):
        target_nets[i].load_state_dict(policy_nets[i].state_dict())
        target_nets[i].eval()

    # 共享参数
    if share_params:
        policy_net = policy_nets[0]
        target_net = target_nets[0]
        for i in range(env.n_agents):
            policy_nets[i] = policy_net
            target_nets[i] = target_net
    
    # Federated Learning Init
    if FL:
        global_paras = policy_nets[0].state_dict()
        for i in range(env.n_agents):
            policy_nets[i].load_state_dict(global_paras)
            target_nets[i].load_state_dict(global_paras)
        FL_sync_count = 0

    # Create memory
    memory = ReplayMemory(MEMORY_SIZE)

    # Initialize epsilon
    episode_durations = [[] for _ in range(env.n_agents)]
    agent_paths = [[] for _ in range(env.n_agents)]
    agent_rewards = [[] for _ in range(env.n_agents)]
    agent_paths_length = [[] for _ in range(env.n_agents)]

    train_history = {'episode_durations': episode_durations, \
                        'agent_paths': agent_paths, \
                        'agent_rewards': agent_rewards, \
                        'agent_paths_length': agent_paths_length}

    for i_episode in trange(EPISODES):
        # Initialize the environment and state
        env.reset()
        steps_done = [0 for _ in range(env.n_agents)]

        # Record the history path for each agent
        # history = [[] for _ in range(env.n_agents)]
        history = [[torch.zeros(5*5, device=device)] * (HISTORY_LENGTH-1) for _ in range(env.n_agents)]

        # Log intermediate variables
        rewards = [0.0 for _ in range(env.n_agents)]
        # Record the full history for each agent
        full_history_position = [[] for _ in range(env.n_agents)]

        # Store the initial state in history
        for i in range(env.n_agents):
            initial_observation = flatten(env.observe(i))
            # history[i].extend([initial_observation for _ in range(HISTORY_LENGTH-1)])
            history[i].append(initial_observation)
            full_history_position[i].append(env.agents[i]['pos'])

        # raise
        for t in count():
            # Break the loop if the maximum steps per episode is reached or all agents are done
            if t >= MAX_STEPS_PER_EPISODE or all([env.agents[i]['done'] for i in range(env.n_agents)]):
                # 对于没有完成的agent，记录其历史
                for i in range(env.n_agents):
                    if not env.agents[i]['done']:  # Only record for agents that are not done
                        episode_durations[i].append(t + 1)
                        agent_paths[i].append(full_history_position[i])
                        agent_paths_length[i].append(len(full_history_position[i]))
                        agent_rewards[i].append(rewards[i])     # Log cumulative reward
                break

            # Select and perform an action, for each agent
            # clear_output(wait=True)  # clears the output of the cell
            # env.render()
            for i in range(env.n_agents):
                if FL:
                    if len(memory) < BATCH_SIZE:
                        pass
                    else:
                        FL_sync_count += 1  # 每个agent都会增加计数
                        # Sync global paras
                        # 只有当所有agent都更新完，且达到了FL_GLOBAL_UPDATE次数，才会更新参数
                        if FL_sync_count % env.n_agents == 0 and FL_sync_count/env.n_agents % FL_GLOBAL_UPDATE == 0:
                            sync_Agents_weights(policy_nets)

                # The agent might have been done
                if env.agents[i]['done']:
                    continue

                state = torch.stack(history[i][-HISTORY_LENGTH:]).unsqueeze(0).float().to(device)
                action = select_action(state, policy_nets[i], steps_done[i])

                # Perform action
                next_state, reward, done = env.step(i, action)
                steps_done[i] += 1  # Increment steps_done for this agent

                # Flatten next_state and add time dimension
                next_state_flat = torch.tensor(next_state.flatten(), device=device).float()

                # Record agent's path
                full_history_position[i].append(env.agents[i]['pos'])

                # Update history and reward
                history[i].append(next_state_flat)
                if len(history[i]) > HISTORY_LENGTH:
                    history[i].pop(0)

                # Flatten next_state and add time dimension
                next_state = torch.stack(history[i][-HISTORY_LENGTH:]).unsqueeze(0).to(device).float()

                # Record agent's cumulative reward
                rewards[i] += float(reward)

                # Store the transition in memory
                action = torch.tensor(action).view(1, -1).to(device)
                reward = torch.tensor(reward).view(1, -1).to(device)
                memory.push(state, action, reward, next_state, done)

                # Move to the next state
                state = next_state_flat

                # Perform one step of the optimization (on the policy network)
                optimize_model(policy_nets[i], target_nets[i], memory, BATCH_SIZE, GAMMA, optimizers[i])

                if done:
                    # full_history_position[i].append(env.end_point)
                    episode_durations[i].append(t + 1)
                    agent_paths[i].append(full_history_position[i])
                    agent_paths_length[i].append(len(full_history_position[i]))
                    agent_rewards[i].append(rewards[i])     # Log cumulative reward
                    continue
            
            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                for i in range(env.n_agents):
                    # target_nets[i].load_state_dict(policy_nets[i].state_dict())
                    # soft update
                    for target_param, param in zip(target_nets[i].parameters(), policy_nets[i].parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        # print(f'i_episode{i_episode}, steps_done{[agent_paths_length[k][-1] for k in range(2)]}, rewards{rewards}')
        # plot_rewards(agent_paths_length)
    
    return train_history


train_history = main(FL=True)


# %% 绘制参数
n_agents = 2
EPISODES = 500
env_size = 17

agent_rewards = train_history['agent_rewards']
agent_paths_length = train_history['agent_paths_length']
agent_paths = train_history['agent_paths']

# %%
draw_history(agent_rewards, agent_paths_length, n_agents, EPISODES)

# %%
episode_index = 400
draw_path(episode_index, agent_paths, n_agents, env_size)
print(agent_paths)
# %%
plt.plot(agent_paths_length[0])
# %%
# save train_history
with open('logs/train_history_open_FL.pkl', 'wb') as f:
    pickle.dump(train_history, f)
# %%
# load train_history
with open('logs/train_history_open_normal.pkl', 'rb') as f:
    train_history = pickle.load(f)
