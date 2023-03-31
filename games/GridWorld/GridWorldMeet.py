import copy
import math
import random

import cv2
import matplotlib

# from alg_GLOBALS import *
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from easydict import EasyDict as edict


# class Position:
#     def __init__(self, x, y, block=False):
#         self.x = x
#         self.y = y
#         self.block = block
#         self.occupied = block
#
#         self.color = 'k' if self.block else 'lightgray'
#         self.marker = 's' if self.block else '.'


class Agent:
    def __init__(self, x, y, ob_size=3):  # , agent_id, agent_type, metric_radius=1):
        self.init_x = x
        self.init_y = y
        self.x = x
        self.y = y
        self.path = []
        self.observe_size = ob_size
        self.mark = 0.5
        self.path_mark = 0.6
        self.status = 'Start'
        self.total_reward = 0

    def reset(self):
        self.x = self.init_x
        self.y = self.init_y
        self.path = []
        self.status = 'Start'
        self.total_reward = 0

        # self.id = agent_id

        # self.id = agent_id
        # self.type = agent_type
        # self.metric_radius = metric_radius
        # self.state_side_size = self.metric_radius * 2 + 1
        # self.state_size = self.state_side_size ** 2
        # # self.name = f'agent_{self.type}_{self.id}'
        # self.name = f'{self.type}'
        # self.domain = []
        # self.state = []
        # self.distance_type = 'chebyshev'
        # # self.distance_type = 'cityblock'
        # self.marker = 'p'
        #
        # if self.type == 'alpha':
        #     self.color = 'r'
        #
        # elif self.type == 'beta':
        #     self.color = 'b'
        #
        # else:
        #     raise RuntimeError('Unknown type!')


# def set_domain_and_state_of_agents(agents, positions):
#     # print(agents, positions)
#     for agent in agents:
#         agent.domain = []
#         # positions 中是所有的点，点是Position对象
#         # 将对象根据坐标转换为字典
#         pos_dict = {(pos.x, pos.y): pos for pos in positions}
#         # self position
#         # 位置对象，根据坐标取出
#         agent.domain.append(pos_dict[(agent.x, agent.y)])
#
#         # 对于所有坐标，如果与agent的距离小于等于metric_radius，则加入domain
#         for pos in positions:
#             dist = cdist([[agent.x, agent.y]], [[pos.x, pos.y]], agent.distance_type)[0, 0]
#             # print(dist, agent.metric_radius)
#             if dist <= agent.metric_radius:
#                 # if not pos.occupied:
#                 agent.domain.append(pos)
#
#         # print(agent.x, agent.y)
#         # for d in agent.domain:
#         #     print(d.x, d.y)
#
#         side = agent.metric_radius * 2 + 1
#         # 超出地图应该全为1
#         agent.state = np.ones((side, side))
#         for pos in agent.domain:
#             status = 1 if pos.block else 0
#             status = 0.5 if (pos.occupied and not pos.block) else status
#             agent.state[agent.metric_radius - (agent.y - pos.y), agent.metric_radius - (agent.x - pos.x)] = float(
#                 status)

# break


# def distance(pos1, pos2):
#     return math.sqrt(((pos1.x - pos2.x) ** 2) + ((pos1.y - pos2.y) ** 2))


class FedRLEnv:
    """
    The main duties of this wrapper:

    1. Receive inputs and transform outputs as tensors
    2. Normalize states of observations

    """

    def __init__(self, maze, cfg_data):
        self.visited_mark = 0.5  # Cells visited by the rat will be painted by gray 0.8
        self.end_mark = 0.7
        self.rat_mark = 0.3  # The current rat cell will be painteg by gray 0.5
        self.LEFT = 0
        self.UP = 1
        self.RIGHT = 2
        self.DOWN = 3
        self.cfg_data = cfg_data

        # 允许的最大步数
        self.max_Tstep = cfg_data.max_Tstep
        self.action_space = [0, 1, 2, 3]
        # 初始化迷宫，老鼠可以从任意位置开始，默认为左上角
        self._maze = np.array(maze)
        nrows, ncols = self._maze.shape

        self.name = 'FedRL_env'
        # self.agents = []
        self.agent_dict = edict()
        self.agent_dict.alpha = Agent(cfg_data.rat[0], cfg_data.rat[1], 3)  # rat
        if cfg_data.target is None:
            self.agent_dict.beta = Agent(nrows - 1, ncols - 1, 5)  # target
        else:
            self.agent_dict.beta = Agent(cfg_data.target[0], cfg_data.target[1], 5)

        self.agent_dict.alpha.mark = self.rat_mark
        self.agent_dict.beta.mark = self.end_mark
        self.agent_dict.alpha.path_mark = 0.4
        self.agent_dict.beta.path_mark = 0.6

        # 终点可以是任意位置
        # if cfg_data.target is None:
        #     self.target = (nrows - 1, ncols - 1)  # target cell where the "cheese" is
        # else:
        #     self.target = cfg_data.target
        # # 初始化空格list，maze为1表示空格，为0表示墙体
        # self.free_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self._maze[r, c] == 1.0]
        # # 将目标格移出空格list
        # self.free_cells.remove(self.target)
        # 检查左上和右下是否为空
        if self._maze[self.agent_dict.beta.x, self.agent_dict.beta.y] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if self._maze[self.agent_dict.alpha.x, self.agent_dict.alpha.y] == 0.0:
            raise Exception("Invalid Rat Location: must sit on a free cell")
        # 放置两个agent并初始化参数
        self.reset()

    def reset(self):
        self.maze = np.copy(self._maze)  # self.maze is the maze that will be modified during the simulation
        for _, agent in self.agent_dict.items():
            agent.reset()

        nrows, ncols = self.maze.shape
        alpha_x = self.agent_dict.alpha.x
        alpha_y = self.agent_dict.alpha.y
        beta_x = self.agent_dict.beta.x
        beta_y = self.agent_dict.beta.y
        self.maze[alpha_x, alpha_y] = self.rat_mark
        self.maze[beta_x, beta_y] = self.end_mark

        # 初始状态
        # self.state = (row, col, 'start')
        # 设置最低奖励阈值
        self.min_reward = 38 * self.cfg_data.REWARD.WALL  # -0.5 * self.maze.size
        # 初始化总奖励
        # self.total_reward = 0
        self.total_Tstep = 0

        t_observations = {name: self.get_observation(agent) for name, agent in self.agent_dict.items()}

        # self.visited = list()
        # for _, agent in self.agent_dict.items():
        #     agent.path = []
        #     agent.status = 'Start'
        #     agent.total_reward = 0

        return t_observations, self.game_status()

    def get_observation(self, agent):
        maze = np.copy(self._maze)

        # draw the rat
        for name, _agent in self.agent_dict.items():
            maze[_agent.x, _agent.y] = _agent.mark

        size = agent.observe_size

        row, col = agent.x, agent.y
        # 获取maze的行列数
        ROWS = len(maze)
        COLS = len(maze[0])

        # 初始化结果二维数组
        result = [[0 for _ in range(size)] for _ in range(size)]

        # 将以指定点为中心指定尺寸范围的观测值存入结果二维数组
        for i in range(row - size // 2, row + size // 2 + 1):
            for j in range(col - size // 2, col + size // 2 + 1):
                if i < 0 or i >= ROWS or j < 0 or j >= COLS:
                    # 如果超出边界，则填充为1
                    result[i - row + size // 2][j - col + size // 2] = 0.0
                else:
                    result[i - row + size // 2][j - col + size // 2] = maze[i][j]

        # 返回结果二维数组
        result = np.array(result)
        result[size // 2][size // 2] = agent.mark
        return result

    def render(self):
        plt.grid('on')
        ax = plt.gca()
        nrows, ncols = self.maze.shape
        ax.set_xticks(np.arange(0.5, nrows, 1))
        ax.set_yticks(np.arange(0.5, ncols, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        canvas = np.copy(self.maze)

        for k, agent in self.agent_dict.items():

            for row, col in agent.path:
                # if canvas[row, col] == 1.0:
                canvas[row, col] = agent.path_mark
                # else:
                #     canvas[row, col] = self.visited_mark

        alpha_x = self.agent_dict.alpha.x
        alpha_y = self.agent_dict.alpha.y
        beta_x = self.agent_dict.beta.x
        beta_y = self.agent_dict.beta.y
        canvas[alpha_x, alpha_y] = self.rat_mark
        canvas[beta_x, beta_y] = self.end_mark
        plt.imshow(canvas, interpolation='none', cmap='gray')
        plt.show()
        # return img

    def step(self, action_dict):
        self.update_state(action_dict)

        t_observations = {name: self.get_observation(agent) for name, agent in self.agent_dict.items()}
        t_rewards = self.get_reward()
        info = self.game_status()
        done = info != 'not_over'

        return t_observations, t_rewards, done, info

    def game_status(self):
        flag = 0

        # 如果所有老鼠的奖励都小于最低奖励阈值，则游戏结束
        for name, agent in self.agent_dict.items():
            if agent.total_reward > self.min_reward:
                flag = 1

        if self.total_Tstep > self.max_Tstep or flag == 0:
            # if self.total_reward < self.min_reward:
            # if self.total_Tstep > self.max_Tstep:
            return 'lose'

        if self.get_distance() <= 1:
            return 'win'

        return 'not_over'

    def update_state(self, action_dict):
        '''
            input: {'alpha': t_alpha_action, 'beta': t_beta_action} [0, 1, 2, 3] [L, U, R, D]
        '''
        for name, agent in self.agent_dict.items():
            action = action_dict[name]

            # 如果老鼠访问的是空格，则记录
            if self.maze[agent.x, agent.y] > 0.0:
                agent.path.append((agent.x, agent.y))  # mark visited cell

            # 获取所有可能执行的动作
            valid_actions = self.valid_actions(name)
            # print('valid_actions', valid_actions)

            # 如果没有可以执行的动作（被围住了），则状态为 blocked，位置不变
            if not valid_actions:
                agent.status = 'blocked'
                print('blocked')
            # 如果需要执行的动作在可执行动作列表中，那么状态为有效，并相应执行动作
            elif action in valid_actions:
                agent.status = 'valid'
                if action == self.LEFT:
                    agent.y -= 1
                elif action == self.UP:
                    agent.x -= 1
                if action == self.RIGHT:
                    agent.y += 1
                elif action == self.DOWN:
                    agent.x += 1
            # 如果需要执行的动作不在可执行动作列表中（撞墙），位置不变
            else:  # invalid action, no change in rat position
                agent.status = 'invalid'

        # 两次动作都执行完毕后，更新步数
        self.total_Tstep += 1  # 每次执行动作+1

    def valid_actions(self, agent_name):
        # print('agent_name', agent_name)
        # print(self.agent_dict)
        agent = self.agent_dict[agent_name]
        # 默认验证当前位置
        row, col = agent.x, agent.y
        actions = copy.deepcopy(self.action_space)
        nrows, ncols = self.maze.shape
        # 如果在第0行，则不能向上走；如果在最后一行，则不能向下走
        if row == 0:
            actions.remove(1)
        elif row == nrows - 1:
            actions.remove(3)
        # 列-左右
        if col == 0:
            actions.remove(0)
        elif col == ncols - 1:
            actions.remove(2)

        # 如果不在最左列，而左边是墙，则不能向左；右边同理
        if row > 0 and self.maze[row - 1, col] == 0.0:
            actions.remove(1)
        if row < nrows - 1 and self.maze[row + 1, col] == 0.0:
            actions.remove(3)

        # 上下同理
        if col > 0 and self.maze[row, col - 1] == 0.0:
            actions.remove(0)
        if col < ncols - 1 and self.maze[row, col + 1] == 0.0:
            actions.remove(2)

        # 返回所有可能执行的动作
        return actions

    def get_reward(self):

        # nrows, ncols = self.maze.shape
        # reward = 0
        L1_distance = self.get_distance()
        if L1_distance <= 1:
            t_rewards = dict()
            for name, agent in self.agent_dict.items():
                t_rewards[name] = self.cfg_data.REWARD.WIN
                agent.total_reward += self.cfg_data.REWARD.WIN
            return t_rewards

        if self.cfg_data.REWARD.HEURISTIC:
            # 估计距离
            rg = self._maze.shape[0] / L1_distance
            # rg = 1 / L1_distance
        else:
            rg = 0

        t_rewards = dict()
        for name, agent in self.agent_dict.items():
            rl = 0
            if L1_distance <= 1:
                rl = self.cfg_data.REWARD.WIN  # 奶酪，给予 1.0 分
            # elif mode == 'blocked':
            #     rl = self.min_reward - 1
            elif agent.status == 'invalid':
                rl = self.cfg_data.REWARD.WALL  # 撞墙-0.75 分，动作不会被执行
            # elif (rat_row, rat_col) in self.visited:
            #     rl = self.cfg_data.REWARD.VISITED  # 访问已经访问过的单元格，-0.25 分
            elif agent.status == 'valid':
                rl = self.cfg_data.REWARD.MOVE  # 每次移动都会花费老鼠 -0.04 分

            reward = rl + rg

            t_rewards[name] = reward
            agent.total_reward += reward

        return t_rewards

    def get_distance(self):
        rat_row, rat_col = self.agent_dict.alpha.x, self.agent_dict.alpha.y
        target_row, target_col = self.agent_dict.beta.x, self.agent_dict.beta.y
        L1_distance = abs(rat_row - target_row) + abs(rat_col - target_col)

        return L1_distance
    # # get to batch
    # def step(self, t_actions):
    #     # actions = {agent_name: action.detach().item() for agent_name, action in t_actions.items()}
    #     # print(t_actions)
    #     # t_actions = t_actions.unsqueeze(0)
    #     # for t_action in t_actions:
    #
    #     # t_action = t_actions.squeeze(0)
    #     # actions = {'alpha': t_action[0].detach().item(), 'beta': t_action[1].detach().item()}
    #     # t_action = t_actions
    #     # print(t_action)
    #     actions = t_actions  # {'alpha': 0, 'beta': 3}
    #
    #     # ACTION: 0,1,2,3,4 = stay ! ,  east > , south v , west < , north ^
    #     # ori no stay
    #     # ACTION: 0,1,2,3 = east > , south v , west < , north ^
    #     # 初始化下一个状态的变量
    #     observations, done, infos = {}, False, {}
    #     rewards = {agent.name: 0 for agent in self.agents}
    #
    #     # EXECUTE ACTIONS
    #     for agent_name, action in actions.items():
    #         r_l_1 = self._execute_action(agent_name, action)
    #         rewards[agent_name] += r_l_1
    #         agent = self.agent_dict[agent_name]
    #         agent.path.append((agent.x, agent.y))
    #     set_domain_and_state_of_agents(self.agents, self.positions)
    #
    #     # OBSERVATIONS
    #     observations = {agent.name: agent.state for agent in self.agents}
    #
    #     # REWARDS
    #     dist = cdist([[self.agents[0].x, self.agents[0].y]], [[self.agents[1].x, self.agents[1].y]], 'cityblock')[0, 0]
    #     r_g = 50 if dist <= 2 else self.side_size / dist
    #     # r_g += self.side_size / dist
    #     for agent in self.agents:
    #         rewards[agent.name] += r_g
    #
    #     # DONE
    #     self.steps_counter += 1
    #     if self.steps_counter == self.max_steps or dist <= 2:
    #         done = True
    #
    #     # INFO
    #     # if self.steps_counter == self.max_steps or dist <= 2:
    #     infos = {'success': True if dist <= 2 else False}
    #
    #     # TO TENSOR
    #     t_observations = {agent_name: obs for agent_name, obs in observations.items()}
    #     t_rewards = {agent_name: reward for agent_name, reward in rewards.items()}
    #
    #     return t_observations, t_rewards, done, infos
    #
    # def close(self):
    #     pass
    #
    # def seed(self, seed=None):
    #     pass
    #
    # def _execute_action(self, agent_name, action):
    #     # ACTION: 0,1,2,3,4 = stay ! ,  east > , south v , west < , north ^
    #     # ori no stay
    #     # ACTION: 0,1,2,3 = east > , south v , west < , north ^
    #     agent = self.agent_dict[agent_name]
    #     print(agent)
    #     print(agent.x, agent.y)
    #     return_value = -1
    #     # stay
    #     # if action != 0:
    #     new_x, new_y = agent.x, agent.y
    #     curr_pos = self.pos_dict[(new_x, new_y)]
    #
    #     new_x = new_x + 1 if action == 0 else new_x  # east >
    #     new_y = new_y - 1 if action == 1 else new_y  # south v
    #     new_x = new_x - 1 if action == 2 else new_x  # west <
    #     new_y = new_y + 1 if action == 3 else new_y  # north ^
    #
    #     if (new_x, new_y) in self.pos_dict:
    #         pos = self.pos_dict[(new_x, new_y)]
    #         if not pos.block:
    #             # print(f'occ: {pos.occupied}, block: {pos.block}')
    #             curr_pos.occupied = False
    #             agent.x = new_x
    #             agent.y = new_y
    #             pos.occupied = True
    #         else:
    #             return_value = -10
    #
    #     print(agent)
    #     print(agent.x, agent.y)
    #
    #     # t_return_value = torch.tensor(return_value)
    #     return return_value
    #
    # def render(self, mode='human'):
    #     map = np.zeros((self.side_size, self.side_size))
    #     for P in self.positions:
    #         map[P.y][P.x] = 1 if P.block else 0
    #
    #     # for A in self.agents:
    #     #     map[A.y][A.x] = 2
    #
    #     A1 = self.agents[0]
    #     map[A1.y][A1.x] = 2
    #     rect1 = matplotlib.patches.Rectangle((A1.x - (A1.metric_radius + .5),
    #                                           A1.y - (A1.metric_radius + .5)),
    #                                          A1.state_side_size, A1.state_side_size, fill=False)
    #
    #     A2 = self.agents[1]
    #     map[A2.y][A2.x] = 1.5
    #     rect2 = matplotlib.patches.Rectangle((A2.x - (A2.metric_radius + .5),
    #                                           A2.y - (A2.metric_radius + .5)),
    #                                          A2.state_side_size, A2.state_side_size, fill=False)
    #
    #     fig, ax = plt.subplots()
    #     ax.add_patch(rect1)
    #     ax.add_patch(rect2)
    #
    #     return map
    #
    # def observation_spaces(self):
    #     return {agent.name: agent.state for agent in self.agents}
    #
    # def observation_sizes(self):
    #     return {agent.name: agent.state_size for agent in self.agents}
    #
    # def action_spaces(self):
    #     return {agent.name: list(range(4)) for agent in self.agents}
    #
    # def action_sizes(self):
    #     return {agent.name: 4 for agent in self.agents}
    #
    # @staticmethod
    # def _get_two_distant_positions(free_positions):
    #     # pos1 = random.sample(free_positions, 1)[0]
    #     # max_dist = 0
    #     # max_pos = pos1
    #     # for pos in free_positions:
    #     #     dist = distance(pos1, pos)
    #     #     if dist > max_dist:
    #     #         max_dist = dist
    #     #         max_pos = pos
    #     # pos2 = max_pos
    #
    #     pos1, pos2 = random.sample(free_positions, 2)
    #
    #     return pos1, pos2


if __name__ == '__main__':
    # %%
    # --------------------------- # CREATE ENV # -------------------------- #
    MAX_STEPS = 40
    SIDE_SIZE = 8
    # SIDE_SIZE = 16
    # SIDE_SIZE = 32
    ENV_NAME = 'grid'

    env = FedRLEnv(max_steps=MAX_STEPS, side_size=SIDE_SIZE)
    print(env.reset())

    map = env.render()

    plt.imshow(map)
    plt.show()

    # print(env.agents[0].x)

    # print(map)

    NUMBER_OF_GAMES = 10
