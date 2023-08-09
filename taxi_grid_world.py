# %%
import numpy as np
import random

import numpy as np

class TaxiEnv:

    def __init__(self, maze, fickle=False, allow_collision=True):
        self.ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "PICKUP", "DROPOFF"]

        self.maze = maze
        self.pickup_locations = [(0, 0), (0, maze.shape[1]-1), (maze.shape[0]-1, 0), (maze.shape[0]-1, maze.shape[1]-1)]
        self.dropoff_locations = self.pickup_locations.copy()

        self.num_rows = maze.shape[0]
        self.num_cols = maze.shape[1]
        self.num_pass_locs = 5  # 4 pickup locations + 1 in taxi
        self.num_dests = 4  # 4 dropoff locations

        self.observation_space = (self.num_rows, self.num_cols, self.num_pass_locs, self.num_dests)
        self.observation_space_n = np.prod(self.observation_space)

        self.action_space = [_ for _ in range(len(self.ACTIONS))]
        self.action_space_n = len(self.action_space)

        self.fickle = fickle
        self.reward = -1
        self.change_flag = False

        # 碰撞
        self.collision = False
        self.allow_collision = allow_collision

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

        self.collision = False
        return self.encode(*self.state)  # 默认输出编码后的状态

    def step(self, action_index):
        if self.get_done():
            return self.state, 0, True, {}
        
        if self.fickle:
            # 如果乘客被成功接起，那么在下一轮有概率改变目的地
            if self.reward == 0:
                # 如果乘客被成功接起，且没有改变目的地，那么有概率改变目的地
                self.change_flag = True
            elif self.change_flag and self.reward == -1 and self.passenger_location == "IN_TAXI":
                if random.random() < 0.3:
                    self.dropoff_location = self.dropoff_locations[np.random.randint(len(self.dropoff_locations))]
                    # while self.dropoff_location == self.pickup_location:
                        # self.dropoff_location = self.dropoff_locations[np.random.randint(len(self.dropoff_locations))]
                    self.state = (self.current_location, self.passenger_location, self.dropoff_location)
                    self.change_flag = False
            
            # 如果行动是上下左右，那么有概率改变行动
            if action_index < 4:
                # 80%的概率执行原来的行动
                if random.random() < 0.8:
                    pass
                    # action_index = action_index
                # 10%的概率执行当前行动的左右
                else:
                    if action_index in [0, 1]:
                        action_index = random.choice([2, 3])
                    else:
                        action_index = random.choice([0, 1])
                        
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

        # 检查是否撞墙
        if self.current_location == old_state[0] and action_index in self.ACTIONS[:4]:
            self.collision = True

        done = self.get_done()
        return self.encode(*self.state), reward, done, {}

    # def get_reward(self, action, old_state):
    #     # 行动暂时没有任何奖励（实际上是-1，但在分支最后）
    #     # if action in ACTIONS[:4]:
    #     reward = -0.04
    #     last_passenger_location = old_state[1]  # 乘客的位置已经被改变了，所以需要记录上一次的位置
    #     # 如果撞墙了，奖励为-10
    #     if self.current_location == old_state[0] and action in self.ACTIONS[:4]:
    #         reward = -1
    #     elif action == "PICKUP":
    #         if self.current_location == last_passenger_location and last_passenger_location != 'IN_TAXI':  # self.passenger_location:
    #             reward = 25  # 正确位置接到乘客，奖励为0
    #         else:
    #             reward =  -1  # 错误位置接乘客，奖励为-10
    #     elif action == "DROPOFF":
    #         if self.current_location == self.dropoff_location and last_passenger_location == "IN_TAXI":
    #             reward = 50  # 正确位置放下乘客，奖励为20
    #         else:
    #             reward =  -1  # 错误位置放下乘客，奖励为-10
        
    #     return reward
    
    def get_reward(self, action, old_state):
        # 行动暂时没有任何奖励（实际上是-1，但在分支最后）
        # if action in ACTIONS[:4]:
        reward = -1
        last_passenger_location = old_state[1]  # 乘客的位置已经被改变了，所以需要记录上一次的位置
        # 如果撞墙了，奖励为-10
        # if self.current_location == old_state[0] and action in self.ACTIONS[:4]:
        #     reward = -1
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
        self.reward = reward
        return reward


    def get_done(self):
        if not self.allow_collision:
            if self.collision:
                return True
 
        if self.passenger_location == self.dropoff_location and self.current_location == self.dropoff_location:
            return True
        else:
            return False
        

    def encode(self, taxi_loc, pass_loc, dest_loc):
        # 与多维数组类似，加上该维度的偏移量再乘以该维度的长度
        # ((taxi_row * num_cols + taxi_col) * num_pass_locs + pass_loc) * num_dests + dest_idx
        num_rows = self.num_cols
        num_cols = self.num_rows
        num_pass_locs = len(self.pickup_locations) + 1  # four pickup locations and one in-taxi
        num_dests = len(self.dropoff_locations)

        i = taxi_loc[0]  # taxi_row
        i *= num_cols
        i += taxi_loc[1]  # taxi_col
        i *= num_pass_locs

        # Encode passenger location
        if pass_loc == "IN_TAXI":
            i += len(self.pickup_locations)  # consider in-taxi as an additional location
        else:
            i += self.pickup_locations.index(pass_loc)

        i *= num_dests
        dest_idx = self.dropoff_locations.index(dest_loc)
        i += dest_idx
        return i


    def decode(self, i):
        num_rows = self.num_rows
        num_cols = self.num_cols
        num_pass_locs = len(self.pickup_locations) + 1  # four pickup locations and one in-taxi
        num_dests = len(self.dropoff_locations)

        out = []

        # decode
        dest_idx = i % num_dests
        i = i // num_dests
        pass_loc = i % num_pass_locs
        i = i // num_pass_locs
        
        taxi_col = i % num_cols
        i = i // num_cols
        taxi_row = i
        assert 0 <= i < num_rows

        # Decode taxi location
        out.append((taxi_row, taxi_col))

        # Decode passenger location
        if pass_loc == len(self.pickup_locations):  # consider in-taxi as an additional location
            out.append("IN_TAXI")
        else:
            out.append(self.pickup_locations[pass_loc])

        # Decode destination location
        out.append(self.dropoff_locations[dest_idx])

        # [taxi_loc, pass_loc, dest_loc]
        return list(out)

    
    
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
    
    @property
    def s(self):
        return self.encode(*self.state)


# # %%
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
maze_cross = np.loadtxt('maze17_0.2.txt')
env = TaxiEnv(maze_cross)
# # %%
state = env.reset()
# print(state)
# state_code = env.encode(*state)
# print(state_code)
# print(env.decode(state_code))



# # %%
# print(env.step(5))
print(env.render())
# %%
