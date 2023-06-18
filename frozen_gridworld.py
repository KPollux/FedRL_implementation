# %% 
import numpy as np
import copy

class FrozenLake:
    def __init__(self, size=17, n_agents=2, heuristic_reward=True, lake=None, same_room=False,
                 reward_dict=None, is_slippery=True, dynamic=False, hole_positions=None, hole_odds=None):
        self.size = size
        self.n_agents = n_agents
        self.heuristic_reward = heuristic_reward
        self.agents = []
        self.is_slippery = is_slippery

        if reward_dict is not None:
            self.reward_dict = reward_dict
        else:
             self.reward_dict = [{'step': -1, 'fall': -10, 'goal': 50, 'heuristic': True},
                                {'step': -1, 'fall': -10, 'goal': 50, 'heuristic': True},
                                {'step': -1, 'fall': -10, 'goal': 50, 'heuristic': True}]
        
        self.dynamic = dynamic
        if self.dynamic:
            self.hole_positions = hole_positions
            self.hole_odds = hole_odds

        self.lake = lake if lake is not None else np.zeros((self.size, self.size))  
        self.map = copy.deepcopy(self.lake)
        self.same_room = same_room

        self.action_space = [0, 1, 2, 3]  
        self.end_point = (size-1, size-1)

        self.init_map_and_agents()

    def init_map_and_agents(self):
        if self.lake is None: 
            self.lake = np.ones((self.size, self.size))
            self.lake[self.end_point] = 2
            # Add holes
            for _ in range(self.size):
                hole = (np.random.randint(0, self.size), np.random.randint(0, self.size))
                if hole != self.end_point:
                    self.lake[hole] = 0
        else:  
            self.lake = self.lake

        for i in range(self.n_agents):
            while True:
                if not self.same_room:
                    if i == 0:
                        start = (np.random.randint(0, self.size//2), np.random.randint(self.size//2, self.size))
                    elif i == 1:
                        start = (np.random.randint(self.size//2, self.size), np.random.randint(0, self.size//2))
                    else:
                        start = (np.random.randint(0, self.size//2), np.random.randint(0, self.size//2))
                else:
                    start = (np.random.randint(0, self.size//2), np.random.randint(self.size//2, self.size))

                if self.lake[start] == 1: 
                    self.agents.append({'pos': start, 'done': False, 'reward': 0, 'alive': True})
                    break

    def step(self, agent_idx, action):
        if self.dynamic:
            self.generate_dynamic_hole()

        reward = 0
        if self.agents[agent_idx]['done']:
            return self.observe(agent_idx), reward, True, self.agents[agent_idx]['alive']

        old_pos = list(self.agents[agent_idx]['pos'])
        new_pos = list(old_pos)
        
        if self.is_slippery:
            action = np.random.choice([action, (action + 1) % 4, (action + 3) % 4], p=[0.8, 0.1, 0.1])

        # Perform action
        if action == 0:   # up
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 1: # right
            new_pos[1] = min(self.size - 1, new_pos[1] + 1)
        elif action == 2: # down
            new_pos[0] = min(self.size - 1, new_pos[0] + 1)
        elif action == 3: # left
            new_pos[1] = max(0, new_pos[1] - 1)

        # Determine new state
        if old_pos == new_pos:  # hit border
            reward = self.reward_dict[agent_idx]['fall'] 
        elif self.lake[tuple(new_pos)] == 0:  # fell into hole
            self.agents[agent_idx]['alive'] = False
            self.agents[agent_idx]['done'] = True
            reward = self.reward_dict[agent_idx]['fall'] 
        elif tuple(new_pos) == self.end_point:  # reach goal
            self.agents[agent_idx]['pos'] = tuple(new_pos)
            self.agents[agent_idx]['done'] = True
            reward = self.reward_dict[agent_idx]['goal']
        else:  # free cell
            self.agents[agent_idx]['pos'] = tuple(new_pos)
            reward = self.reward_dict[agent_idx]['step']
            if self.heuristic_reward and self.reward_dict[agent_idx]['heuristic']:
                reward += - (np.abs(np.array(new_pos) - np.array(self.end_point))).sum() / self.size

        return self.observe(agent_idx), reward, self.agents[agent_idx]['done'], self.agents[agent_idx]['alive']

    # Remaining methods will be the same as original class with minor changes


    def reset(self):
        self.agents = []
        self.init_map_and_agents()
        observations = []
        for i in range(self.n_agents):
            observations.append(self.observe(i))
        return observations

    def render(self):
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.end_point:
                    print('G', end=' ')
                elif self.map[i, j] == 0:
                    print('W', end=' ')
                else:
                    agent_here = [idx for idx, agent in enumerate(self.agents) if agent['pos'] == (i, j)]
                    if agent_here:
                        print(agent_here[0], end=' ')
                    else:
                        print('.', end=' ')
            print()

    def observe(self, agent_idx):
        observation = np.pad(self.map, ((2,2), (2,2)), 'constant', constant_values=(0))
        # print(observation)
        pos = [self.agents[agent_idx]['pos'][0] + 2, self.agents[agent_idx]['pos'][1] + 2]
        # print(pos)
        observation[pos[0], pos[1]] = 3  # Mark the agent itself with a unique value
        # print(observation)
        observation = observation[pos[0]-2:pos[0]+3, pos[1]-2:pos[1]+3]
        # print(observation)
        return observation
    
    def get_state(self, agent_idx):
        return self.agents[agent_idx]['pos']
    
    def generate_dynamic_hole(self):
        hole_positions = self.hole_positions
        hole_odds = self.hole_odds
        for i, hole_position in enumerate(hole_positions):
            if np.random.uniform() < hole_odds[i]:
                agent_here = False
                for agent in self.agents:
                    if agent['pos'] == hole_position:
                        agent_here = True
                        break
                if not agent_here:
                    self.lake[hole_position] = 0                  
            else:
                self.lake[hole_position] = 1



# %%
# maze_cross = np.array([
#     [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
# ])

# np.savetxt('maze_cross.txt', maze_cross, fmt='%f')
# if __main__():

maze_cross = np.loadtxt('maze_cross_level4.txt')

# %%
env = FrozenLake(size=17, n_agents=3, heuristic_reward=False, lake=maze_cross, is_slippery=False)

# %%
env.render()
# %%
env.observe(0)
# %%
observations, rewards, dones, status = env.step(1, 0)
env.render()
# print(env.observe(0))
# print(env.observe(1))
print(rewards, dones)
# %%
env.agents[1]['pos']
# %%
env.observe(1)
# %%
env.reset()
env.render()
# %%
size = 17
# start = (np.random.randint(0, self.size//2), np.random.randint(self.size//2, self.size))
# start = (np.random.randint(self.size//2, self.size), np.random.randint(0, self.size//2))
mdis = []
for x in range(0, 17//2):
    for y in range (0, 17//2):
        # 求每个(x,y)到(15, 15)的Taxicab geometry
        mdis.append(abs(x-16) + abs(y-16))
        
        
# %%
np.mean(mdis)
# # %%

# %%
