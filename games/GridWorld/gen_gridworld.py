import pickle
import random
import heapq

import numpy as np
from matplotlib import pyplot as plt

import random
import heapq


def generate_gridworld(n, prob):
    """
    生成大小为n x n的随机Gridworld，其中prob控制1出现的频率。
    返回值是一个包含n个列表的列表，每个列表包含n个随机数（0或1）。
    """
    gridworld = [[int(random.random() < prob) for j in range(n)] for i in range(n)]
    return gridworld


def find_path(maze, start, end):
    """
    使用 A* 算法搜索迷宫最优路径
    :param maze: 二维迷宫数组，0 表示障碍，1 表示可通行
    :param start: 起点坐标 (row, col)
    :param end: 终点坐标 (row, col)
    :return: 返回最优路径
    """
    ROW, COL = len(maze), len(maze[0])
    pq = []  # 使用优先队列存储搜索节点
    heapq.heappush(pq, (0, start, [start]))
    visited = set()  # 使用 set 存储已访问的节点
    while pq:
        f, (row, col), path = heapq.heappop(pq)
        if (row, col) in visited:
            continue
        visited.add((row, col))
        if (row, col) == end:
            return path
        for (r, c) in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]:
            if 0 <= r < ROW and 0 <= c < COL and maze[r][c] == 1 and (r, c) not in visited:
                g = len(path)  # 当前节点到起点的距离
                h = abs(r-end[0]) + abs(c-end[1])  # 当前节点到终点的曼哈顿距离
                f = g + h
                heapq.heappush(pq, (f, (r, c), path + [(r, c)]))
    return False


def generate_solvable_gridworld(n, prob=0.6):
    """
    生成一个可解的大小为n x n的Gridworld。
    返回值是一个元组，包含生成的Gridworld、起点和终点。
    """
    while True:
        gridworld = generate_gridworld(n, prob=prob)
        # start = (random.randint(0, n-1), random.randint(0, n-1))
        # end = (random.randint(0, n-1), random.randint(0, n-1))
        start = (0, 0)
        end = (n - 1, n - 1)

        optimal_path = find_path(gridworld, start, end)
        if gridworld[start[0]][start[1]] == 1 and gridworld[end[0]][
            end[1]] == 1 and start != end and optimal_path is not False:
            return gridworld, start, end, optimal_path


if __name__ == '__main__':
    # 示例代码
    gridworld, start, end, optimal_path = generate_solvable_gridworld(16, prob=0.98)
    gridworld = np.array(gridworld) * 1.0
    print(gridworld)
    print("start:", start)
    print("end:", end)

    plt.imshow(gridworld)
    plt.show()

    # train_set = {'gridworld': [],
    #              'start': [],
    #              'end': []}
    #
    # test_set = {'gridworld': [],
    #             'start': [],
    #             'end': []}
    #
    # for i in range(6400):
    #     gridworld, start, end = generate_solvable_gridworld(8)
    #     train_set['gridworld'].append(gridworld)
    #     train_set['start'].append(start)
    #     train_set['end'].append(end)

    # 将字典保存到文件中
    # with open("gridworld3x3_train_dict.pickle", "wb") as f:
    #     pickle.dump(train_set, f)
    np.savetxt('maze16_0.02_5.txt', gridworld)
    # maze = np.loadtxt('maze32_1.txt')
