import os
from PIL import Image
import numpy as np

# 输入和输出路径
input_folder = r'C:\Users\86181\Desktop\sim2teal\cracktree200\Annotations\val'
output_folder = r'C:\Users\86181\Desktop\sim2teal\cracktree200\Annotations\val_gt'

# 获取输入文件夹中所有的PNG文件
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        # 构建完整的文件路径
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 打开图片
        img = Image.open(input_path)
        # 将图像转换为NumPy数组
        img_array = np.array(img)

        # 将像素值为1或2的点置为1
        img_array = np.where(np.isin(img_array, [1, 2]), 1, img_array)

        # 将所有像素值乘以255
        img_array = img_array * 255

        # 确保像素值在0到255之间
        img_array = np.clip(img_array, 0, 255)

        # 将处理后的数组转换回图像
        processed_img = Image.fromarray(img_array.astype(np.uint8))

        # 保存图像
        processed_img.save(output_path)

        print(f'处理并保存了：{filename}')
