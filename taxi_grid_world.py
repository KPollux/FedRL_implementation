# %%
import numpy as np
import random

import numpy as np

class TaxiEnv:

    def __init__(self, maze):
        self.ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "PICKUP", "DROPOFF"]

        self.maze = maze
        self.pickup_locations = [(0, 0), (0, maze.shape[1]-1), (maze.shape[0]-1, 0), (maze.shape[0]-1, maze.shape[1]-1)]
        self.dropoff_locations = self.pickup_locations.copy()

        self.num_rows = 17
        self.num_cols = 17
        self.num_pass_locs = 5  # 4 pickup locations + 1 in taxi
        self.num_dests = 4  # 4 dropoff locations

    def reset(self):
        while True:
            self.current_location = (np.random.randint(self.maze.shape[0]), np.random.randint(self.maze.shape[1]))
            if self.maze[self.current_location] == 1:  # ensure start location is not a wall
                break

        self.pickup_location = self.pickup_locations[np.random.randint(len(self.pickup_locations))]
        self.dropoff_location = self.dropoff_locations[np.random.randint(len(self.dropoff_locations))]
        while self.dropoff_location == self.pickup_location:
            self.dropoff_location = self.dropoff_locations[np.random.randint(len(self.dropoff_locations))]

        self.passenger_location = self.pickup_location
        self.state = (self.current_location, self.passenger_location, self.dropoff_location)
        return self.state

    def step(self, action_index):
        if self.get_done():
            return self.state, 0, True, {}
        old_state = self.state
        # 可能不移动
        new_location = self.current_location

        # 如果移动，更新位置
        if action_index == 0:  # "UP"
            new_location = (max(self.current_location[0] - 1, 0), self.current_location[1])
        elif action_index == 1:  #"DOWN":
            new_location = (min(self.current_location[0] + 1, self.maze.shape[0] - 1), self.current_location[1])
        elif action_index == 2:  #"LEFT":
            new_location = (self.current_location[0], max(self.current_location[1] - 1, 0))
        elif action_index == 3:  #"RIGHT":
            new_location = (self.current_location[0], min(self.current_location[1] + 1, self.maze.shape[1] - 1))
        
        # 如果不移动，更新乘客状态
        elif action_index == 4:  #"PICKUP":
            if self.current_location == self.passenger_location:
                self.passenger_location = "IN_TAXI"
        elif action_index == 5:  #"DROPOFF":
            if self.current_location == self.dropoff_location and self.passenger_location == "IN_TAXI":
                self.passenger_location = self.dropoff_location

        if self.maze[new_location] == 1:  # only move to new location if it is valid (i.e., not a wall)
            self.current_location = new_location

        self.state = (self.current_location, self.passenger_location, self.dropoff_location)

        reward = self.get_reward(self.ACTIONS[action_index], old_state)
        done = self.get_done()
        return self.state, reward, done, {}

    def get_reward(self, action, old_state):
        # 行动暂时没有任何奖励（实际上是-1，但在分支最后）
        # if action in ACTIONS[:4]:
        reward = -1
        last_passenger_location = old_state[1]  # 乘客的位置已经被改变了，所以需要记录上一次的位置
        if action == "PICKUP":
            if self.current_location == last_passenger_location and last_passenger_location != 'IN_TAXI':  # self.passenger_location:
                reward = 0  # 正确位置接到乘客，奖励为0
            else:
                reward =  -10  # 错误位置接乘客，奖励为-10
        elif action == "DROPOFF":
            if self.current_location == self.dropoff_location and last_passenger_location == "IN_TAXI":
                reward = 20  # 正确位置放下乘客，奖励为20
            else:
                reward =  -10  # 错误位置放下乘客，奖励为-10
        
        return reward


    def get_done(self):
        if self.passenger_location == self.dropoff_location and self.current_location == self.dropoff_location:
            return True
        else:
            return False
        

    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
        num_rows = self.num_rows
        num_cols = self.num_cols
        num_pass_locs = self.num_pass_locs
        num_dests = self.num_dests

        i = taxi_row
        i *= num_cols
        i += taxi_col
        i *= num_pass_locs
        i += pass_loc
        i *= num_dests
        i += dest_idx
        return i

    def decode(self, i):
        num_rows = self.num_rows
        num_cols = self.num_cols
        num_pass_locs = self.num_pass_locs
        num_dests = self.num_dests

        out = []
        out.append(i % num_dests)
        i = i // num_dests
        out.append(i % num_pass_locs)
        i = i // num_pass_locs
        out.append(i % num_cols)
        i = i // num_cols
        out.append(i)
        assert 0 <= i < num_rows
        return reversed(out)
    
    
    def render(self):
        output = ""

        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i][j] == 0:  # wall
                    output += '#'
                elif self.current_location == (i, j):  # taxi
                    if self.passenger_location == "IN_TAXI":
                        output += 'L'
                    else:
                        output += 'T'
                elif self.passenger_location == (i, j):  # passenger
                    output += 'P'
                elif (i, j) in self.dropoff_locations:  # possible destination
                    if (i, j) == self.dropoff_location:
                        output += 'G'
                    else:
                        output += 'D'
                else:  # empty space
                    output += '.'
            output += '\n'

        return output


# %%
maze_cross = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
])

env = TaxiEnv(maze_cross)
# %%
env.reset()


# %%
print(env.step(5))
print(env.render())
# %%
