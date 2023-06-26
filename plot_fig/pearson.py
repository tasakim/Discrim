
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# 设置图像参数
plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(figsize=(8, 6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_tick_params(width=0.5, length=2)
ax.yaxis.set_tick_params(width=0.5, length=2)
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.set_xlabel('Test Loss', fontsize=12)
ax.set_ylabel('Accuracy(%)', fontsize=12)

# 生成示例数据
x = np.array([41.79, 38.84, 35.39, 32.13, 29.32, 27.02, 25.19, 23.68, 22.43, 21.34, 20.38, 19.52, 18.74, 18.02, 17.37, 16.76, 16.17, 15.63, 15.11, 14.62, 14.15, 13.69, 13.25, 12.85, 12.46, 12.08, 11.71, 11.37, 11.05, 10.74, 10.42, 10.13, 9.85, 9.57, 9.3, 9.04, 8.81, 8.58, 8.35, 8.12, 7.91, 7.71, 7.5, 7.31, 7.12, 6.94, 6.77, 6.6, 6.44, 6.28])
y = np.array([27.11, 27.32, 22.15, 21.73, 23.96, 19.98, 23.44, 24.33, 27.37, 22.65, 22.67, 29.44, 33.51, 40.22, 44.16, 43.26, 42.0, 43.05, 49.16, 44.53, 43.53, 45.59, 44.91, 48.95, 49.63, 47.65, 47.65, 48.51, 49.01, 50.17, 50.17, 50.59, 54.28, 53.36, 53.46, 53.35, 53.11, 53.13, 52.82, 52.23, 56.97, 57.32, 57.18, 56.48, 56.17, 57.68, 58.46, 58.42, 58.44, 52.76])



# 计算皮尔森系数
corr, _ = pearsonr(x, y)

# 绘制散点图
ax.scatter(x, y, s=40, c='#1f77b4', alpha=0.8, edgecolor='none')

# 添加皮尔森系数文本框
textstr = r'$\rho =$' + f'{corr:.2f}'
props = dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3, edgecolor='black')
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', horizontalalignment='left', bbox=props)

fit = np.polyfit(x, y, deg=1)
ax.plot(x, fit[0] * x + fit[1], color='black', linewidth=1)

# 设置图像标题和字体
# plt.title('Scatter Plot', fontsize=16, fontweight='bold', y=1.05)

# 添加标尺
# ax.axhline(y=np.mean(y), color='black', linestyle='--', linewidth=1)
# ax.axvline(x=np.mean(x), color='black', linestyle='--', linewidth=1)
ax.text(np.mean(x)-1.5, 0.92, 'Mean of Test Loss', fontsize=12, transform=ax.transAxes)
ax.text(0.92, np.mean(y)+0.05, 'Mean of Accuracy', fontsize=12, transform=ax.transAxes)

# 添加图例
ax.annotate('Pearson correlation coefficient', xy=(0.12, 0.88), xycoords='axes fraction', fontsize=12,
            xytext=(0.05, 0.8), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3,rad=0.2'))

# 反转x轴
ax.invert_xaxis()

# 调整图像位置
fig.tight_layout(pad=2)

# 保存图像
fig.savefig('./pearson.png', dpi=210)
# plt.show()