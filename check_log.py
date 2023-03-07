import json
import os

import matplotlib.pyplot as plt
import numpy as np

rootpath = r'runs/grid/Mon_Mar__6_23_48_31_2023'

# open txt file with json
with open(os.path.join(rootpath, 'train.txt'), 'r') as f:
    log = json.load(f)

# print(log['eval'])
rewards = log['eval']['rewards']

plt.plot(rewards)
plt.show()
rewards = np.array(rewards)

print(len(rewards[rewards > 10]))
print(np.sum(rewards)/800)
# print(1727/6400)

