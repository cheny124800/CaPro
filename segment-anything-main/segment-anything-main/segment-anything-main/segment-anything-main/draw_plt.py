import matplotlib.pyplot as plt

# 数据
K_values = [5, 15, 25, 35]
MIoU = [0.41099, 0.5362, 0.53349, 0.51603]
F1 = [0.39161, 0.57725, 0.57747, 0.55607]

# 绘图
plt.figure(figsize=(8, 6), dpi=200)
plt.plot(K_values, MIoU, marker='o', markersize=5, linestyle='-', color='b', label='MIoU')
plt.plot(K_values, F1, marker='s', markersize=5, linestyle='--', color='r', label='F1')

# 标注数据点，避免标签遮挡
for x, y in zip(K_values, MIoU):
    if x == 5:
        plt.text(x, y + 0.02, f'{y:.4f}', ha='center', va='bottom', fontsize=10, color='b')
    elif x == 15:
        plt.text(x+0.55, y + 0.007, f'{y:.4f}', ha='center', va='bottom', fontsize=10, color='b')
    elif x == 25:
        plt.text(x, y + 0.007, f'{y:.4f}', ha='center', va='bottom', fontsize=10, color='b')
    else:
        plt.text(x, y + 0.007, f'{y:.4f}', ha='center', va='bottom', fontsize=10, color='b')
for x, y in zip(K_values, F1):
    if x == 5:
        plt.text(x, y - 0.01, f'{y:.4f}', ha='center', va='top', fontsize=10, color='r')
    elif x == 15:
        plt.text(x, y + 0.016, f'{y:.4f}', ha='center', va='top', fontsize=10, color='r')
    elif x == 25:
        plt.text(x, y + 0.016, f'{y:.4f}', ha='center', va='top', fontsize=10, color='r')
    else:
        plt.text(x, y + 0.017, f'{y:.4f}', ha='center', va='top', fontsize=10, color='r')


# 设置轴标签
plt.xlabel('K', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.xticks(K_values)
plt.ylim(0.35, 0.62)  # 进一步拉长y轴，避免数据重叠
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

# 显示图表
plt.show()
#plt.savefig(r'C:\Users\86181\Desktop\sim2teal\test\plot3.png', dpi=300, bbox_inches='tight')