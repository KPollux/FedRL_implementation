# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import namedtuple, deque
import gym

# %%
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.max_action * torch.tanh(self.layer3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.layer1 = nn.Linear(state_dim + action_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)
        # Q2 architecture
        self.layer4 = nn.Linear(state_dim + action_dim, 400)
        self.layer5 = nn.Linear(400, 300)
        self.layer6 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        # Q1 output
        x1 = F.relu(self.layer1(xu))
        x1 = F.relu(self.layer2(x1))
        x1 = self.layer3(x1)
        # Q2 output
        x2 = F.relu(self.layer4(xu))
        x2 = F.relu(self.layer5(x2))
        x2 = self.layer6(x2)
        return x1, x2

# %%
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

    def push(self, *args):
        self.buffer.append(self.Transition(*args))

    def sample(self, batch_size):
        transitions = np.random.choice(self.buffer, batch_size, replace=False)
        batch = self.Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.buffer)

# %%
class DDPG:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005):
        self.actor = Actor(state_dim, action_dim, max_action).cuda()
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).cuda()
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(1000000)

    def select_action(self, state):
        state = torch.Tensor(state).cuda()
        return self.actor(state).cpu().data.numpy()

    def train(self, batch_size=64):
        batch = self.replay_buffer.sample(batch_size)
        state = torch.Tensor(batch.state).cuda()
        action = torch.Tensor(batch.action).cuda()
        next_state = torch.Tensor(batch.next_state).cuda()
        reward = torch.Tensor(batch.reward).cuda().unsqueeze(1)
        done = torch.Tensor(batch.done).cuda().unsqueeze(1)

        with torch.no_grad():
            target_Q1, target_Q2 = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (1 - done) * self.gamma * torch.min(target_Q1, target_Q2)

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add_to_replay_buffer(self, state, next_state, action, reward, done):
        self.replay_buffer.push(state, action, next_state, reward, done)

# %%
if __name__ == '__main__':
    env = gym.make("HalfCheetah-v2")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPG(state_dim, action_dim, max_action)

    episode_rewards = []

    for episode in range(500):
        state = env.reset()
        episode_reward = 0

        for step in range(1000):  # assuming max steps per episode is 1000
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.add_to_replay_buffer(state, next_state, action, reward, done)

            state = next_state
            episode_reward += reward

            if len(agent.replay_buffer) > 100:
                agent.train()

            if done:
                break

        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}")

    env.close()

# %%
