import matplotlib.pyplot as plt
import numpy as np

# 定义矩形框的坐标 (每组是 x1, y1, x2, y2, x3, y3, x4, y4)
rectangles = [
    (132.5589099187031, 333.18407349429697, 132.5589099187031, 283.3758950423188,
     131.32526072070306, 284.60954424031877, 181.1334391726813, 234.80136578834055),

    (170.21699390571317, 292.2842809248667, 170.21699390571317, 284.1449979385735,
     231.17251249970326, 223.1894793445834, 239.31179548599647, 215.05019635829018),

    (116.69741183968134, 341.28159185388984, 116.69741183968134, 348.8938666283826,
     193.46453319564887, 272.12674527241506, 185.8522584211561, 279.73902004690785)
]

# 创建绘图
fig, ax = plt.subplots(figsize=(8, 8))

# 遍历每个矩形框，绘制四个顶点连线
for rect in rectangles:
    x_vals = [rect[0], rect[2], rect[4], rect[6], rect[0]]  # 将四个顶点闭合
    y_vals = [rect[1], rect[3], rect[5], rect[7], rect[1]]  # 将四个顶点闭合
    ax.plot(x_vals, y_vals, marker='o', label="crack")

# 设置图像的显示范围
ax.set_xlim(0, 512)
ax.set_ylim(0, 512)

# 反转 y 轴，使得 y 轴正方向向下
ax.invert_yaxis()

# 设置坐标轴比例
ax.set_aspect('equal')

# 添加标签和标题
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Rectangles with Coordinates")

# 显示图形
plt.show()
