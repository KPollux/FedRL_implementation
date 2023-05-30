# %%
import pickle

from matplotlib import pyplot as plt
import numpy as np

from utils import moving_average

# from utils import draw_history

# %%
def draw_history(agent_rewards, agent_paths_length, n_agents, EPISODES, window_size=100, title='', xlims=[None, None], ylims=[None, None]):
    # Convert agent_rewards and agent_paths_length to numpy arrays for easier manipulation
    agent_rewards_np = [np.array(rewards) for rewards in agent_rewards]
    agent_paths_length_np = [np.array(lengths) for lengths in agent_paths_length]

    # Mean and std for each agent's rewards and path length
    rewards_means = [rewards.mean() for rewards in agent_rewards_np]
    rewards_std = [rewards.std() for rewards in agent_rewards_np]
    paths_length_means = [lengths.mean() for lengths in agent_paths_length_np]
    paths_length_std = [lengths.std() for lengths in agent_paths_length_np]

    # Range of episodes
    episodes = np.arange(1, EPISODES+1)

    # Plotting
    fig, ax2 = plt.subplots(1, 1, figsize=(10, 7.5))

    # Set larger font size
    plt.rcParams.update({'font.size': 15})

    # # Rewards
    # for i in range(n_agents):
    #     ax1.plot(episodes, moving_average(agent_rewards_np[i], window_size), label=f'Agent {i+1}')
    #     # ax1.fill_between(episodes, rewards_means[i]-rewards_std[i], rewards_means[i]+rewards_std[i], alpha=0.2)
    # ax1.set_xlabel('Episodes', fontsize=15)
    # ax1.set_ylabel('Cumulative Rewards', fontsize=15)
    # ax1.set_title(title + 'Rewards per Episode for each Agent')
    # ax1.grid(True)
    # ax1.legend()
    # if xlims[0] is not None:
    #     ax1.set_xlim(xlims[0])
    # if ylims[0] is not None:
    #     ax1.set_ylim(ylims[0])
    # ax1.tick_params(axis='both', which='major', labelsize=15)

    # Path lengths
    for i in range(n_agents):
        ax2.plot(episodes, moving_average(agent_paths_length_np[i], window_size), label=f'Agent {i+1}')
        # ax2.fill_between(episodes, paths_length_means[i]-paths_length_std[i], paths_length_means[i]+paths_length_std[i], alpha=0.2)
    ax2.set_xlabel('Episodes', fontsize=15)
    ax2.set_ylabel('Path Length', fontsize=15)
    ax2.set_title(title + 'Path Length per Episode for each Agent')
    ax2.grid(True)
    ax2.legend()
    if xlims[0] is not None:
        ax2.set_xlim(xlims[0])
    if xlims[1] is not None:
        ax2.set_xlim(xlims[1])
    if xlims[0] is not None and xlims[1] is not None:
        ax2.set_xlim(xlims)

    if ylims[0] is not None:
        ax2.set_ylim(ylims[0])
    if ylims[1] is not None:
        ax2.set_ylim(ylims[1])
    if ylims[0] is not None and ylims[1] is not None:
        ax2.set_ylim(ylims)
    ax2.tick_params(axis='both', which='major', labelsize=15)

    # Show the plots
    plt.tight_layout()
    plt.show()

# %%

with open('./logs/Q_learning_M17_20_level4_2023-05-30-00-21-25/train_history.pkl', 'rb') as f:
    train_history = pickle.load(f)

log_agent_rewards = train_history['agent_rewards']
log_agent_paths_length = train_history['agent_paths_length']
log_agent_paths = train_history['agent_paths']

n_agents = 3
EPISODES = 1000
env_size = 17

TITLE = 'M17_20_Q_learning_FLMax '
draw_history(log_agent_rewards, log_agent_paths_length, n_agents, EPISODES, 64, title=TITLE) #, xlims=[0, 400], ylims=[0, 200])
draw_history(log_agent_rewards, log_agent_paths_length, n_agents, EPISODES, 64, title=TITLE, xlims=[0, 400], ylims=[0, 200])
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
