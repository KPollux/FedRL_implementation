# %%
from itertools import count
import pickle
from matplotlib import pyplot as plt
import numpy as np
import torch
from cross_gridworld import Gridworld
from utils import DuelingDQNLast, draw_history
from IPython.display import clear_output

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flatten(observation):
    return torch.tensor(observation, device=device).float().view(-1)

def test():
    MAZE = np.loadtxt('maze_cross_level4.txt')
    SIZE = MAZE.shape[0]
    OBSERVATION_SPACE = 25
    HISTORY_LENGTH = 20
    MAX_STEPS_PER_EPISODE = 256  # Maximum steps per episode
    role = False

    log_path = './logs/level4_2023-05-22-23-41-34'

    # Create environment
    env = Gridworld(size=SIZE, n_agents=1, heuristic_reward=True, maze=MAZE)
    
    # Create role matrix
    roles = torch.eye(env.n_agents).cuda()
    if role:
        OBSERVATION_SPACE += env.n_agents

    # Create DQN networks and optimizers for each agent
    policy_nets = [DuelingDQNLast(OBSERVATION_SPACE, 32, 4).to(device)]

    # Load trained models
    # for i, policy_net in enumerate(policy_nets):
    policy_nets[0].load_state_dict(torch.load(log_path + f'/LR5e-06_agent{1}.pth'))
    
    # Initialize the environment and state
    size = 17
    # start = (np.random.randint(0, self.size//2), np.random.randint(self.size//2, self.size))
    # start = (np.random.randint(self.size//2, self.size), np.random.randint(0, self.size//2))
    range_quadrant_1 = [range(0, size//2), range(size//2+1, size)]
    range_quadrant_3 = [range(size//2+1, size), range(0, size//2)]

    range_quadrant = range_quadrant_3  # 0-1 1-3 
    
    # Initialize epsilon
    episode_durations = [[] for _ in range(env.n_agents)]
    agent_paths = [[] for _ in range(env.n_agents)]
    agent_rewards = [[] for _ in range(env.n_agents)]
    agent_paths_length = [[] for _ in range(env.n_agents)]
    
    for x in range_quadrant[0]:
        for y in range_quadrant[1]:
            # 求每个(x,y)的路径长度和累计奖励
            env.reset()
            env.agents[0]['pos'] = (x, y)
    
            steps_done = [0 for _ in range(env.n_agents)]

            # Record the history path for each agent
            # history = [[] for _ in range(env.n_agents)]
            history = [[torch.zeros(OBSERVATION_SPACE, device=device)] * (HISTORY_LENGTH-1) for _ in range(env.n_agents)]

            # Log intermediate variables
            rewards = [0.0 for _ in range(env.n_agents)]
            # Record the full history for each agent
            full_history_position = [[] for _ in range(env.n_agents)]

            # Store the initial state in history
            for i in range(env.n_agents):
                initial_observation = flatten(env.observe(i)) if not role else torch.cat((flatten(env.observe(i)), roles[i]))
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
                clear_output(wait=True)  # clears the output of the cell
                env.render()
                for i in range(env.n_agents):

                    state = torch.stack(history[i][-HISTORY_LENGTH:]).unsqueeze(0).float().to(device)
                    with torch.no_grad():
                        action = policy_nets[i](state).max(1)[1].view(1, 1).item()

                    # Perform action
                    next_state, reward, done = env.step(i, action)
                    steps_done[i] += 1  # Increment steps_done for this agent

                    # Flatten next_state and add time dimension
                    if role:
                        next_state_flat = torch.cat((flatten(next_state), roles[i]))
                    else:
                        next_state_flat = flatten(next_state)
                    
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
                    # memorys[i].push(state, action, reward, next_state, done)

                    # Move to the next state
                    state = next_state_flat

                    # Perform one step of the optimization (on the policy network)
                    # optimize_model(policy_nets[i], target_nets[i], memorys[i], BATCH_SIZE, GAMMA, optimizers[i])

                    if done:
                        # full_history_position[i].append(env.end_point)
                        episode_durations[i].append(t + 1)
                        agent_paths[i].append(full_history_position[i])
                        agent_paths_length[i].append(len(full_history_position[i]))
                        agent_rewards[i].append(rewards[i])     # Log cumulative reward
                        continue

    return agent_paths, agent_paths_length, agent_rewards, episode_durations
                    
# %%
agent_paths, agent_paths_length, agent_rewards, episode_durations = test()
# %%

# %%
with open('./logs/FL_role_level4_2023-05-23-09-09-12/train_history.pkl', 'rb') as f:
    train_history = pickle.load(f)

log_agent_rewards = train_history['agent_rewards']
log_agent_paths_length = train_history['agent_paths_length']
log_agent_paths = train_history['agent_paths']

n_agents = 2
EPISODES = 1000
env_size = 17

draw_history(log_agent_rewards, log_agent_paths_length, n_agents, EPISODES, 64, title='VFL ID Embedding ', ylims=[None, (100, 260)])
# %%
plt.plot(log_agent_paths_length[0])
plt.xlabel('Episode')
plt.ylabel('Path length')
plt.title('Path length of agent 1')
# %%
plt.plot(log_agent_paths_length[1])
plt.xlabel('Episode')
plt.ylabel('Path length')
plt.title('Path length of agent 2')
# %%
