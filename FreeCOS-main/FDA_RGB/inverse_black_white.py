import os
from PIL import Image, ImageOps

def invert_gif_colors(input_dir):
    # 遍历指定目录下的所有文件
    for filename in os.listdir(input_dir):
        # 判断文件是否为 gif 格式
        if filename.endswith('.png'):
            # 拼接出文件的完整路径
            file_path = os.path.join(input_dir, filename)

            # 打开 gif 图像
            img = Image.open(file_path)

            # 转为灰度图像
            img_gray = img.convert('L')

            # 对灰度图像进行黑白反转
            img_inverted = ImageOps.invert(img_gray)

            # 直接保存反转后的图像覆盖原图
            img_inverted.save(file_path)

            print(f"Processed and saved: {file_path}")

# 设置你的文件夹路径
input_directory = r'C:\Users\86181\Desktop\cvpr25\FreeCOS-main\FreeCOS-main\Data\CrackTree\fake_gtvessel_thin'

# 调用函数进行处理
invert_gif_colors(input_directory)
