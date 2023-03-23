from easydict import EasyDict as edict

# init
__C_GridWorld = edict()

cfg_data = __C_GridWorld

__C_GridWorld.rat = (0, 0)
__C_GridWorld.target = (7, 7)
__C_GridWorld.max_Tstep = 100  # 实际最大步数考察(MAX_TSTEP, reward<10*__C_GridWorld.REWARD.WALL)

__C_GridWorld.REWARD = edict()

__C_GridWorld.REWARD.WIN = 50
__C_GridWorld.REWARD.WALL = -10
__C_GridWorld.REWARD.MOVE = -1
__C_GridWorld.REWARD.VISITED = __C_GridWorld.REWARD.MOVE

__C_GridWorld.REWARD.HEURISTIC = True
__C_GridWorld.REWARD.HEURISTIC_FACTOR = None



