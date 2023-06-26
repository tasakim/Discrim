import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    'legend.fontsize': 12,
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
layer_id = [x for x in range(1, 28)]
strategy1_100 = [15.7, 15.7, 15.7, 15.7, 15.7, 15.7, 15.7, 15.7, 15.7, 31.3, 31.3, 31.3, 31.3, 31.3, 31.3, 31.3, 31.3, 31.3, 62.7, 62.7, 62.7, 62.7, 62.7, 62.7, 62.7, 62.7, 62.7]
strategy1_5000 = [783.5, 783.5, 783.5, 783.5, 783.5, 783.5, 783.5, 783.5, 783.5, 1567.0, 1567.0, 1567.0, 1567.0, 1567.0, 1567.0, 1567.0, 1567.0, 1567.0, 3133.9, 3133.9, 3133.9, 3133.9, 3133.9, 3133.9, 3133.9, 3133.9, 3133.9]
strategy2 = [4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 19.4, 19.4, 19.4, 19.4, 19.4, 19.4, 19.4, 19.4, 19.4, 79.0, 79.0, 79.0, 79.0, 79.0, 79.0, 79.0, 79.0, 79.0]
strategy3 = [23.3, 23.3, 23.3, 23.3, 23.3, 23.3, 23.3, 23.3, 23.3, 187.1, 187.1, 187.1, 187.1, 187.1, 187.1, 187.1, 187.1, 187.1, 1497.4, 1497.4, 1497.4, 1497.4, 1497.4, 1497.4, 1497.4, 1497.4, 1497.4]

# 绘制折线图
fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(pruning_rate, strategy1, color='blue', linestyle='-', linewidth=2, marker='o', markersize=8, label='Strategy 1')
# ax.plot(pruning_rate, strategy2, color='red', linestyle='--', linewidth=2, marker='s', markersize=8, label='Strategy 2')
# ax.plot(pruning_rate, strategy3, color='green', linestyle='-.', linewidth=2, marker='^', markersize=8, label='Strategy 3')
ax.plot(layer_id, np.log(strategy1_100), color=palette[0], linestyle='-', linewidth=2, marker='o', markersize=2, label='Strategy 1 ($N_{m}=100$)')
ax.plot(layer_id, np.log(strategy1_5000), color=palette[0], linestyle=':', linewidth=2, marker='o', markersize=2, label='Strategy 1 ($N_{m}=5000$)')
ax.plot(layer_id, np.log(strategy2), color=palette[1], linestyle='--', linewidth=2, marker='s', markersize=2, label='Strategy 2')
ax.plot(layer_id, np.log(strategy3), color=palette[2], linestyle='-.', linewidth=2, marker='^', markersize=2, label='Strategy 3')


# 添加标题和标签
# ax.set_title('折线图示例', fontsize=18, fontweight='bold')
ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
ax.set_ylabel('$log(FLOPs)$ (G)', fontsize=12, fontweight='bold')

# 调整坐标轴范围
ax.set_xlim(0, 30)
ax.set_ylim(0, 10)

# 添加网格线
ax.grid(True)

# 添加图例
ax.legend(loc='lower right')

# 调整边框
for spine in ax.spines.values():
    spine.set_linewidth(0.5)

# 显示图像
plt.savefig('flops_compare.png', dpi=180)
# plt.show()