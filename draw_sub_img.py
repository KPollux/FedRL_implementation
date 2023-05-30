# %%
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# 假设你的数据
x = np.linspace(0, 600, 600)
y = np.sin(x) * np.exp(-0.01*x)

fig, ax = plt.subplots() # 创建一个新的图表和坐标轴
ax.plot(x, y) # 在主坐标轴上画出原始数据

# 创建放大的区域
axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47]) # 这个列表分别代表了放大区域左下角的x，y坐标以及区域的宽度和高度
axins.plot(x, y) # 在新的坐标轴上画出原始数据

# 设定放大的区域
x1, x2, y1, y2 = 0, 100, -0.25, 0.25
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

# 在主图和放大区域之间添加连接线
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

# 使用plt.show()来显示图像
plt.show()


# %%
