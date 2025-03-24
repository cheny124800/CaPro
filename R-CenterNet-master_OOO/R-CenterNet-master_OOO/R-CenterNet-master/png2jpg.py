import os
from PIL import Image

def convert_png_to_jpg(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".png"):  # 只处理png文件
            # 构建完整的文件路径
            png_path = os.path.join(input_folder, filename)
            jpg_filename = os.path.splitext(filename)[0] + ".jpg"
            jpg_path = os.path.join(output_folder, jpg_filename)

            # 打开 PNG 图片
            with Image.open(png_path) as img:
                # 转换为 RGB 模式（PNG 图片有可能是 RGBA，JPG 不支持透明通道）
                img = img.convert("RGB")
                # 保存为 JPG 格式
                img.save(jpg_path, "JPEG")

            print(f"Converted {filename} to {jpg_filename}")

if __name__ == "__main__":
    input_folder = r"C:\Users\86181\Desktop\cvpr25\R-CenterNet-master_OOO\R-CenterNet-master\imgs"  # 输入文件夹路径
    output_folder = r"C:\Users\86181\Desktop\cvpr25\R-CenterNet-master_OOO\R-CenterNet-master\imgs"  # 输出文件夹路径
    convert_png_to_jpg(input_folder, output_folder)
