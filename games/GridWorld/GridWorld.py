import os, sys, time, datetime, json, random
import numpy as np
import copy
import matplotlib.pyplot as plt


# maze is a 2d Numpy array of floats between 0.0 to 1.0
# 1.0 corresponds to a free cell, and 0.0 an occupied cell
# rat = (row, col) initial rat position (defaults to (0,0))
from games.GridWorld.gen_gridworld import generate_solvable_gridworld, find_path


class GridWorld(object):
    def __init__(self, maze, cfg_data, ob_size=0):
        self.visited_mark = 0.6  # Cells visited by the rat will be painted by gray 0.8
        self.end_mark = 0.9
        self.rat_mark = 0.3  # The current rat cell will be painteg by gray 0.5
        self.LEFT = 0
        self.UP = 1
        self.RIGHT = 2
        self.DOWN = 3
        self.cfg_data = cfg_data

        self.observe_size = ob_size

        # 允许的最大步数
        self.max_Tstep = cfg_data.max_Tstep
        self.action_space = [0, 1, 2, 3]
        # 初始化迷宫，老鼠可以从任意位置开始，默认为左上角
        self._maze = np.array(maze)
        nrows, ncols = self._maze.shape
        # 终点可以是任意位置
        if cfg_data.target is None:
            self.target = (nrows - 1, ncols - 1)  # target cell where the "cheese" is
        else:
            self.target = cfg_data.target
        # # 初始化空格list，maze为1表示空格，为0表示墙体
        # self.free_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self._maze[r, c] == 1.0]
        # # 将目标格移出空格list
        # self.free_cells.remove(self.target)
        # 检查左上和右下是否为空
        if self._maze[self.target[0], self.target[1]] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if self._maze[cfg_data.rat[0], cfg_data.rat[1]] == 0.0:
            raise Exception("Invalid Rat Location: must sit on a free cell")
        # 放置老鼠并初始化参数
        state, info = self.reset(cfg_data.rat)

        # return state, info

    def reset(self, rat=(0, 0)):
        self.rat = rat
        self.maze = np.copy(self._maze)
        nrows, ncols = self.maze.shape
        row, col = rat
        self.maze[row, col] = self.rat_mark
        self.maze[self.target[0], self.target[1]] = self.end_mark
        # 初始状态
        self.state = (row, col, 'start')
        # 设置最低奖励阈值
        self.min_reward = 10*cfg_data.REWARD.WALL  # -0.5 * self.maze.size
        # 初始化总奖励
        self.total_reward = 0
        self.visited = list()
        self.total_Tstep = 0

        return self.observe(), self.game_status()

    def update_state(self, action):
        '''
            input: action [0, 1, 2, 3] [L, U, R, D]
        '''
        nrows, ncols = self.maze.shape
        nrow, ncol, nmode = rat_row, rat_col, mode = self.state

        # 如果老鼠访问的是空格，则记录
        if self.maze[rat_row, rat_col] > 0.0:
            self.visited.append((rat_row, rat_col))  # mark visited cell

        # 获取所有可能执行的动作
        valid_actions = self.valid_actions()
        # print('valid_actions', valid_actions)

        # 如果没有可以执行的动作（被围住了），则状态为 blocked，位置不变
        if not valid_actions:
            nmode = 'blocked'
            print('blocked')
        # 如果需要执行的动作在可执行动作列表中，那么状态为有效，并相应执行动作
        elif action in valid_actions:
            nmode = 'valid'
            if action == self.LEFT:
                ncol -= 1
            elif action == self.UP:
                nrow -= 1
            if action == self.RIGHT:
                ncol += 1
            elif action == self.DOWN:
                nrow += 1
        # 如果需要执行的动作不在可执行动作列表中（撞墙），位置不变
        else:  # invalid action, no change in rat position
            nmode = 'invalid'

        self.total_Tstep += 1  # 每次执行动作+1
        # new state
        self.state = (nrow, ncol, nmode)

    def get_reward(self):
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        target_row, target_col = self.target

        reward = 0
        rl = 0

        if rat_row == target_row and rat_col == target_col:
            rl = self.cfg_data.REWARD.WIN  # 奶酪，给予 1.0 分
        # elif mode == 'blocked':
        #     rl = self.min_reward - 1
        elif mode == 'invalid':
            rl = self.cfg_data.REWARD.WALL  # 撞墙-0.75 分，动作不会被执行
        elif (rat_row, rat_col) in self.visited:
            rl = self.cfg_data.REWARD.VISITED  # 访问已经访问过的单元格，-0.25 分
        elif mode == 'valid':
            rl = self.cfg_data.REWARD.MOVE  # 每次移动都会花费老鼠 -0.04 分

        if self.cfg_data.REWARD.HEURISTIC:
            # 估计距离
            rg = self._maze.shape[0] / (abs(rat_row - target_row) + abs(rat_col - target_col))
        else:
            rg = 0

        reward = rl + rg

        return reward

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status

    def step(self, action):
        envstate, reward, status = self.act(action)
        observation = envstate
        done = self.is_game_done()
        info = status
        return observation, reward, done, info

    def observe(self):
        canvas = self.get_observation()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_env(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r, c] > 0.0:
                    canvas[r, c] = 1.0
        # draw the rat
        row, col, valid = self.state
        canvas[row, col] = self.rat_mark
        canvas[self.target[0], self.target[1]] = self.end_mark
        return canvas

    def game_status(self):
        if self.total_Tstep > self.max_Tstep or self.total_reward < self.min_reward:
            # if self.total_reward < self.min_reward:
            # if self.total_Tstep > self.max_Tstep:
            return 'lose'
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows - 1 and rat_col == ncols - 1:
            return 'win'

        return 'not_over'

    def is_game_done(self):
        game_status = self.game_status()

        if game_status == 'not_over':
            return False
        elif game_status == 'win' or game_status == 'lose':
            return True

        return -1

    def valid_actions(self, cell=None):
        # 默认验证当前位置
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
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

    def get_observation(self):
        maze = self.draw_env()
        if self.observe_size <= 0:
            return maze
        else:
            size = self.observe_size

        row, col, _ = self.state
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
        result[size // 2][size // 2] = self.rat_mark
        return result

    def render(self):
        plt.grid('on')
        nrows, ncols = self.maze.shape
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, nrows, 1))
        ax.set_yticks(np.arange(0.5, ncols, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        canvas = np.copy(self.maze)
        for row, col in self.visited:
            canvas[row, col] = self.visited_mark
        rat_row, rat_col, _ = self.state
        target_row, target_col = self.target
        canvas[rat_row, rat_col] = self.rat_mark  # rat cell
        canvas[target_row, target_col] = self.end_mark  # cheese cell

        return canvas


if __name__ == '__main__':
    from setting import cfg_data

    print(cfg_data.rat, cfg_data.target)
    gridworld, _, _, _ = generate_solvable_gridworld(8, 0.8, cfg_data.rat, cfg_data.target)
    qmaze = GridWorld(gridworld, cfg_data, 3)

    maze_size = gridworld.shape[0]
    optimal_path = find_path(gridworld, cfg_data.rat, cfg_data.target)
    qmaze.visited = optimal_path
    optimal_length = len(optimal_path)

    print(qmaze.observe())

    canvas = qmaze.render()
    plt.imshow(canvas, interpolation='none', cmap='gray')
    plt.show()
