import matplotlib.pyplot as plt
import seaborn as sns

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
pruning_rate = [10, 30, 50, 70, 90]
strategy1_100 = [93.86, 93.45, 93.05, 92.06, 87.96]
strategy1_1000 = [94.19, 94.08, 93.77, 92.95, 89.45]
strategy2 = [94.07, 93.96, 93.50, 92.49, 88.82]
strategy3 = [94.09, 94.08, 93.72, 92.90, 89.23]

# 绘制折线图
fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(pruning_rate, strategy1, color='blue', linestyle='-', linewidth=2, marker='o', markersize=8, label='Strategy 1')
# ax.plot(pruning_rate, strategy2, color='red', linestyle='--', linewidth=2, marker='s', markersize=8, label='Strategy 2')
# ax.plot(pruning_rate, strategy3, color='green', linestyle='-.', linewidth=2, marker='^', markersize=8, label='Strategy 3')
ax.plot(pruning_rate, strategy1_100, color=palette[0], linestyle='-', linewidth=2, marker='o', markersize=8, label='Strategy 1 ($N_{m}=100$)')
ax.plot(pruning_rate, strategy1_1000, color=palette[0], linestyle=':', linewidth=2, marker='o', markersize=8, label='Strategy 1 ($N_{m}=5000$)')
ax.plot(pruning_rate, strategy2, color=palette[1], linestyle='--', linewidth=2, marker='s', markersize=8, label='Strategy 2')
ax.plot(pruning_rate, strategy3, color=palette[2], linestyle='-.', linewidth=2, marker='^', markersize=8, label='Strategy 3')


# 添加标题和标签
# ax.set_title('折线图示例', fontsize=18, fontweight='bold')
ax.set_xlabel('Pruning Rate (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')

# 调整坐标轴范围
ax.set_xlim(0, 100)
ax.set_ylim(85, 95)

# 添加网格线
ax.grid(True)

# 添加图例
ax.legend(loc='lower left')

# 调整边框
for spine in ax.spines.values():
    spine.set_linewidth(0.5)

# 显示图像
plt.savefig('strategy_compare.png', dpi=140)
