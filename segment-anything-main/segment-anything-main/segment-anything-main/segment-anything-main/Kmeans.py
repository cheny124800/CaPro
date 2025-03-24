import numpy as np
import cv2
import os
from sklearn.cluster import KMeans

# 输入输出目录
input_dir = './input'
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

# 设定类别数
k = 2
# 选择要分割的类别（假设我们要提取类别 2，可自行调整）
target_class = 0

# 处理 input 目录下的所有 jpg 图像
for filename in os.listdir(input_dir):
    if filename.lower().endswith('.jpg'):
        img_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace('.jpg', '_mask.png'))

        # 读取图像
        img = cv2.imread(img_path)  # 读取 BGR 图像
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB

        width, height, _ = img.shape

        # 重新调整形状进行 K-Means 聚类
        img_flattened = img.reshape(-1, 3)
        classifier = KMeans(n_clusters=k, max_iter=100, init='k-means++', random_state=42)
        kmeans = classifier.fit(img_flattened)
        labels = kmeans.labels_.reshape(width, height)  # 重新塑形为图像尺寸

        # 生成二值掩码（前景=1，背景=0）
        mask = (labels == target_class).astype(np.uint8)

        # 保存掩码
        cv2.imwrite(output_path, mask * 255)  # 乘以 255 以便可视化
        print(f"掩码已保存至 {output_path}")
