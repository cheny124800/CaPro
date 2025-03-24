import os
from PIL import Image, ImageDraw

# 定义文件夹路径
img_dir = r'C:\Users\86181\Desktop\sim2teal\222Our_Result222\1\DEEPCRACK'
txt_dir = r'C:\Users\86181\Desktop\sim2teal\ours\DEEPCRACK\txt_top6(vitB)'
output_dir = r'C:\Users\86181\Desktop\sim2teal\222Our_Result222\8\DEEPCRACK'


# 绘制四个点连接成四边形
def draw_lines(img, points):
    draw = ImageDraw.Draw(img)
    # 获取四个点的坐标（前四个点）
    x1, y1, x2, y2, x3, y3, x4, y4, m1, m2 = map(int, map(round, points))  # 四舍五入并转换为整数

    # 用线条连接四个点
    draw.line([(x1, y1), (x2, y2)], fill='red', width=3)  # 连接第一条线
    draw.line([(x2, y2), (x3, y3)], fill='blue', width=3)  # 连接第二条线
    draw.line([(x3, y3), (x4, y4)], fill='red', width=3)  # 连接第三条线
    draw.line([(x4, y4), (x1, y1)], fill='blue', width=3)  # 连接第四条线


# 绘制四个点作为小圆点
def draw_points(img, points):
    draw = ImageDraw.Draw(img)
    # 获取前四个点的坐标
    greem_points = [(int(round(points[i])), int(round(points[i + 1]))) for i in range(0, 8, 2)]  # 前四个点
    # 获取第五个点
    red_point = (int(round(points[8])), int(round(points[9])))  # 第五个点
    r = 1.5
    # 绘制绿色的小圆点（前四个点）
    for (x, y) in greem_points:
        draw.ellipse([x - r, y - r, x + r, y + r], fill='#00FF00', outline='#00FF00')

    # 绘制红色的小圆点（第五个点）
    draw.ellipse([red_point[0] - r, red_point[1] - r, red_point[0] + r, red_point[1] + r], fill='#FF0000', outline='#FF0000')


# 获取jpg文件列表
img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

# 遍历每个jpg文件
for img_file in img_files:
    img_path = os.path.join(img_dir, img_file)
    txt_path = os.path.join(txt_dir, img_file.replace('.jpg', '_O.txt'))

    if os.path.exists(txt_path):
        # 打开图像文件
        img = Image.open(img_path)

        # 读取txt文件内容
        with open(txt_path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                points = list(map(float, line.split()))  # 将每一行的点转换为浮动数
                if len(points) >= 10:  # 每行包含五个点
                    # 调用函数绘制四条线
                    #draw_lines(img, points)
                    # 调用函数绘制四个点和红色小圆点
                    draw_points(img, points)

        # 保存绘制后的图像
        img.save(os.path.join(output_dir, img_file))

        print(f"Processed {img_file}")

print("All images processed.")
