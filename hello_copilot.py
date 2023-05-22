import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class GridWorld:
    def __init__(self):
        self.grid = np.zeros((4, 4))
        self.grid[0, 0] = 1
        self.grid[3, 3] = 2
        self.agent_pos = [0, 0]

    def reset(self):
        self.agent_pos = [0, 0]
        return self.agent_pos

    def step(self, action):
        if action == 0:  # move right
            self.agent_pos[1] = min(self.agent_pos[1] + 1, 3)
        elif action == 1:  # move left
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 2:  # move up
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 3:  # move down
            self.agent_pos[0] = min(self.agent_pos[0] + 1, 3)

        reward = self.grid[self.agent_pos[0], self.agent_pos[1]]
        done = (reward == 2)
        return self.agent_pos, reward, done, {}

    def render(self):
        for i in range(4):
            for j in range(4):
                if self.agent_pos == [i, j]:
                    print("A", end=" ")
                elif self.grid[i, j] == 1:
                    print("S", end=" ")
                elif self.grid[i, j] == 2:
                    print("G", end=" ")
                else:
                    print("0", end=" ")
            print()
        print()


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Agent:
    def __init__(self, input_size, output_size, lr, gamma, epsilon):
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy_net = DQN(input_size, output_size)
        self.target_net = DQN(input_size, output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = []

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.output_size)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.policy_net(state)
                return q_values.argmax().item()

    def store_transition(self, transition):
        self.memory.append(transition)

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return

        transitions = random.sample(self.memory, batch_size)
        batch = list(zip(*transitions))

        states = torch.tensor(batch[0], dtype=torch.float32)
        actions = torch.tensor(batch[1], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(batch[3], dtype=torch.float32)
        dones = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())