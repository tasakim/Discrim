import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
# 设置图形参数
# params = {
#     'font.family': 'serif',
#     'font.serif': ['Times New Roman'],
#     'font.size': 12,
#     'axes.labelsize': 20,
#     'axes.linewidth': 2,
#     'xtick.labelsize': 18,
#     'ytick.labelsize': 18,
#     'legend.fontsize': 20,
#     'legend.frameon': True,
#     'legend.framealpha': 0.8,
#     'legend.facecolor': 'white',
#     'legend.edgecolor': 'gray',
#     'grid.color': 'gray',
#     'grid.linestyle': ':',
#     'grid.linewidth': 1,
#     'figure.figsize': [8, 6],
#     'lines.linewidth': 2,
#     'lines.markersize': 8,
#     'lines.markeredgecolor': 'black',
#     'lines.markerfacecolor': 'white'
# }
#
# plt.rcParams.update(params)

palette = sns.color_palette('colorblind', n_colors=10)
# 设置图形参数
params = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.linewidth': 1.5,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.facecolor': 'white',
    'legend.edgecolor': 'gray',
    'grid.color': 'gray',
    'grid.linestyle': ':',
    'grid.linewidth': 1,
}

plt.rcParams.update(params)


# 准备数据
epochs = np.arange(1, 800)
with open('./loss/N_m=0.json', 'r') as f:
    data = f.read()
test_loss1 = np.log(np.array(json.loads(data))[:, 2]).tolist()
with open('./loss/N_m=1.json', 'r') as f:
    data = f.read()
test_loss2 = np.log(np.array(json.loads(data))[:, 2]).tolist()
with open('./loss/N_m=5.json', 'r') as f:
    data = f.read()
test_loss3 = np.log(np.array(json.loads(data))[:, 2]).tolist()
with open('./loss/N_m=10.json', 'r') as f:
    data = f.read()
test_loss4 = np.log(np.array(json.loads(data))[:, 2]).tolist()


# 绘制折线图
fig, ax = plt.subplots()

ax.plot(epochs, test_loss1, linestyle='-', linewidth=2, color=palette[0], label='$\gamma=0$')
ax.plot(epochs, test_loss2, linestyle='-', linewidth=2, color=palette[1], label='$\gamma=1$')
ax.plot(epochs, test_loss3, linestyle='-', linewidth=2, color=palette[2], label='$\gamma=5$')
ax.plot(epochs, test_loss4, linestyle='-', linewidth=2, color=palette[3], label='$\gamma=10$')

# 添加标题和标签
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('log(Test Loss)', fontsize=12, fontweight='bold')

ax.grid(True)

# 添加图例
ax.legend(loc='upper right')

# 调整边框
for spine in ax.spines.values():
    spine.set_linewidth(0.5)

# plt.show()
plt.savefig('./N_m.png', dpi=150)