# %% 
import numpy as np

class MeetingGridworld:
    def __init__(self, size=17, n_agents=2, heuristic_reward=True, maze=None):
        self.size = size
        self.n_agents = n_agents
        self.heuristic_reward = heuristic_reward
        self.agents = []
        self.map = maze if maze is not None else np.zeros((self.size, self.size))  # Use provided maze, if any
        self.maze = maze

        self.reward_dict = [
            {'step': 0, 'collision': -10, 'goal': 50, 'heuristic': True},
            # {'step': 0, 'collision': 0, 'goal': 0, 'heuristic': False},
            {'step': 0, 'collision': -10, 'goal': 50, 'heuristic': True},
            # {'step': 0, 'collision': -10, 'goal': 50, 'heuristic': True}
                            ]

        # self.end_point = (0, 0)
        self.action_space = [0, 1, 2, 3]  # Up, right, down, left

        self.init_map_and_agents()

    def init_map_and_agents(self):
        if self.maze is None:  # If no maze was provided
            self.map = np.ones((self.size, self.size))
            # self.map[self.end_point] = 2
            # Add obstacles
            # for _ in range(self.size):
            #     obstacle = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            #     if obstacle != self.end_point:
            #         self.map[obstacle] = 0
        else:  # If a maze was provided
            self.map = self.maze

        for i in range(self.n_agents):
            while True:
                if i == 0:
                    start = (np.random.randint(0, self.size//2-1), np.random.randint(self.size//2+1, self.size))
                elif i == 1:
                    start = (np.random.randint(self.size//2+1, self.size), np.random.randint(0, self.size//2-1))
                else:
                    start = (np.random.randint(0, self.size//2), np.random.randint(0, self.size//2))

                if self.map[start] == 1:  # Ensure agents do not start on obstacles
                    self.agents.append({'pos': start, 'done': False, 'reward': 0})
                    break

    def step(self, agent_idx, action):
        reward = 0  # reset the reward at each step
        if self.agents[agent_idx]['done']:
            return self.get_state(agent_idx), reward, True

        old_pos = list(self.agents[agent_idx]['pos'])
        new_pos = list(old_pos)
        if action == 0:   # up
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 1: # right
            new_pos[1] = min(self.size - 1, new_pos[1] + 1)
        elif action == 2: # down
            new_pos[0] = min(self.size - 1, new_pos[0] + 1)
        elif action == 3: # left
            new_pos[1] = max(0, new_pos[1] - 1)

        # 修改结束条件
        for i, agent in enumerate(self.agents):
            # 遍历所有agent，如果当前agent不是正在行走的Agent(自己)，
            # 且当前agent的位置和新位置相同，那么当前agent和自己都结束
            if agent != self.agents[agent_idx] and agent['pos'] == tuple(new_pos):
                self.agents[agent_idx]['pos'] = tuple(new_pos)
                self.agents[agent_idx]['done'] = True
                agent['done'] = True
                reward = self.reward_dict[i]['goal']
                
                return self.get_state(agent_idx), reward, True

        if old_pos == new_pos or self.map[tuple(new_pos)] == 0:  # hit wall or border
            reward = self.reward_dict[agent_idx]['collision']
        else:  # free cell
            self.agents[agent_idx]['pos'] = tuple(new_pos)
            reward = self.reward_dict[agent_idx]['step']
            # if self.heuristic_reward and self.reward_dict[agent_idx]['heuristic']:
            #     reward += - (np.abs(np.array(new_pos) - np.array(self.end_point))).sum() / self.size
            if self.heuristic_reward and self.reward_dict[agent_idx]['heuristic']:
                for agent in self.agents:
                    if agent != self.agents[agent_idx]:
                        # 两个agent之间的距离越近，reward越接近0，越远reward越是负数
                        reward += - (np.abs(np.array(new_pos) - np.array(agent['pos']))).sum() / self.size
                        # reward += self.size/(np.abs(np.array(new_pos) - np.array(agent['pos']))).sum()


        return self.get_state(agent_idx), reward, self.agents[agent_idx]['done']

    def reset(self):
        self.agents = []
        self.init_map_and_agents()
        states = []
        for i in range(self.n_agents):
            states.append(self.get_state(i))
        return states

    def render(self):
        for i in range(self.size):
            for j in range(self.size):
                # if (i, j) == self.end_point:
                #     print('G', end=' ')
                if self.map[i, j] == 0:
                    print('W', end=' ')
                else:
                    agent_here = [idx for idx, agent in enumerate(self.agents) if agent['pos'] == (i, j)]
                    if agent_here:
                        print(agent_here[0], end=' ')
                    else:
                        print('.', end=' ')
            print()

    # def observe(self, agent_idx):
    #     observation = np.pad(self.map, ((2,2), (2,2)), 'constant', constant_values=(0))
    #     # print(observation)
    #     pos = [self.agents[agent_idx]['pos'][0] + 2, self.agents[agent_idx]['pos'][1] + 2]
    #     # print(pos)
    #     observation[pos[0], pos[1]] = 3  # Mark the agent itself with a unique value
    #     # print(observation)
    #     observation = observation[pos[0]-2:pos[0]+3, pos[1]-2:pos[1]+3]
    #     # print(observation)
    #     return observation
    
    def get_state(self, agent_idx):
        return self.agents[agent_idx]['pos']



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
env = MeetingGridworld(size=17, n_agents=2, heuristic_reward=True, maze=maze_cross)

# %%
env.render()
# %%
env.get_state(1)
# %%
observations, rewards, dones = env.step(0, 1)
env.render()
print(env.get_state(0))
print(env.get_state(1))
print(rewards, dones)
# %%
env.agents[1]['pos']
# %%
env.get_state(1)
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
