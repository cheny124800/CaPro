from PIL import Image
import os

# 设置源目录和目标目录
src_dir = r"C:\Users\86181\Downloads\CrackTree260\gt"
dst_dir = r"C:\Users\86181\Downloads\CrackTree260\gt_png"

# 如果目标目录不存在，则创建
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# 遍历源目录中的所有bmp文件
for filename in os.listdir(src_dir):
    if filename.endswith(".bmp"):
        # 构造完整的文件路径
        bmp_path = os.path.join(src_dir, filename)

        # 打开BMP文件并转换为PNG
        with Image.open(bmp_path) as img:
            # 构造目标文件路径，保持文件名不变，只修改扩展名为png
            png_filename = os.path.splitext(filename)[0] + ".png"
            png_path = os.path.join(dst_dir, png_filename)

            # 保存为PNG文件
            img.save(png_path)

print("转换完成！")
