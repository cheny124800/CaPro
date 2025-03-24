import time
import cv2
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import warnings

warnings.filterwarnings("ignore")


def save_mask_image(mask, output_dir, base_filename_O):
    # 将掩码值为1的地方设置为白色，0的地方设置为黑色
    mask_image = np.zeros_like(mask, dtype=np.uint8)
    mask_image[mask == 1] = 255  # 前景为白色
    mask_image[mask == 0] = 0    # 背景为黑色
    print("生成掩码")
    # 保存掩码图像到 output_dir
    mask_output_path = os.path.join(output_dir, f"{base_filename_O}_mask_vitb.png")  # 修改文件扩展名为.png
    print(mask_output_path)
    cv2.imwrite(mask_output_path, mask_image)
    # print(f"Saved mask image to {mask_output_path}")


# 输入： 原图 + 蒙版 -》 输出： 图像最终结果（数组形式）
def show_mask(image, mask):
    color_ = [255, 0, 0]  # 绿色蒙版
    color = np.array(color_)
    h, w = mask.shape[-2:]
    imageResult = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    for i in range(0, h):
        for j in range(0, w):
            if color_[0] == imageResult[i][j][0] and color_[1] == imageResult[i][j][1] and color_[2] == imageResult[i][j][2]:
                image[i][j] = color_
    return image

# 计算IoU的函数
# 计算IoU（前景 + 背景）
def calculate_iou_big(pred_mask, gt_mask):
    # 前景 IoU
    foreground_intersection = np.sum(np.logical_and(pred_mask == 1, gt_mask == 1))
    foreground_union = np.sum(np.logical_or(pred_mask == 1, gt_mask == 1))
    foreground_iou = foreground_intersection / foreground_union if foreground_union != 0 else 0

    # 背景 IoU
    background_intersection = np.sum(np.logical_and(pred_mask == 0, gt_mask == 0))
    background_union = np.sum(np.logical_or(pred_mask == 0, gt_mask == 0))
    background_union = np.sum(np.logical_or(pred_mask == 0, gt_mask == 0))
    background_iou = background_intersection / background_union if background_union != 0 else 0

    # 总体 IoU：前景和背景 IoU 的平均值
    total_iou = (foreground_iou + background_iou) / 2
    return total_iou

def calculate_iou_small(pred_mask, gt_mask):
    # 交集：预测为前景且真实标签也为前景的区域
    intersection = np.sum(np.logical_and(pred_mask == 1, gt_mask == 1))
    # 并集：预测为前景或真实标签为前景的区域
    union = np.sum(np.logical_or(pred_mask == 1, gt_mask == 1))
    # 计算IoU，防止除以0
    iou = intersection / union if union != 0 else 0
    return iou

# 计算F1的函数
def calculate_f1(pred_mask, gt_mask):
    # True Positives
    tp = np.sum(np.logical_and(pred_mask == 1, gt_mask == 1))
    # False Positives
    fp = np.sum(np.logical_and(pred_mask == 1, gt_mask == 0))
    # False Negatives
    fn = np.sum(np.logical_and(pred_mask == 0, gt_mask == 1))

    # Precision 和 Recall
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    return f1




# SAM模型初始化
sam_checkpoint = "sam_vit_b_01ec64.pth"  # 定义模型路径
model_type = "vit_b"  # 定义模型类型
device = "cuda"  # "cpu" or "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)  # 定义模型参数
predictor = SamPredictor(sam)  # 调用预测模型

# 定义文件夹路径
input_dir = r"C:\Users\86181\Desktop\sim2teal\OCTA500\all\test"  # 目标文件夹路径
txt_O_dir = r"C:\Users\86181\Desktop\sim2teal\test\top5"  # txt文件所在目录
backup_txt_O_dir = r"C:\Users\86181\Desktop\sim2teal\test\top5"
output_dir = r"C:\Users\86181\Desktop\sim2teal\test\top5_output"  # 输出文件夹路径（确保该路径已存在）
gt_dir = r"C:\Users\86181\Desktop\sim2teal\OCTA500\all\test_label"  # GT掩码所在目录

files = os.listdir(input_dir)

# 统计 .png 文件的个数
png_count = len([file for file in files if file.endswith('.jpg')])

# 创建输出目录（如果不存在）
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 创建计数器，避免文件名覆盖
counter = 0
cnt = 0
# 存储所有图像的IoU值
iou_values_big = []
iou_values_small = []
# 存储所有图像的F1值
f1_values = []

# 遍历目录中的所有jpg文件
for filename in os.listdir(input_dir):
    if filename.lower().endswith('.jpg'):

        # O_prompt_path
        base_filename_O = os.path.splitext(filename)[0]
        txt_filename_O = f"{base_filename_O}_O.txt"  # 对应的txt文件名
        txt_file_path_O = os.path.join(txt_O_dir, txt_filename_O)

        # Check if the txt file exists in the primary directory
        if not os.path.exists(txt_file_path_O):
            # If not found, try the backup directory
            txt_file_path_O = os.path.join(backup_txt_O_dir, txt_filename_O)

            # If the file is still not found, print a message and continue to the next file
            if not os.path.exists(txt_file_path_O):
                print(f"Warning: {txt_filename_O} not found in both the primary and backup directories. Skipping.")
                continue

        input_points_O = []
        input_labels_O = []  # 用来存储每个点的标签（背景点为0，对象点为1）

        with open(txt_file_path_O, 'r') as f:
            for line in f:
                coords = list(map(float, line.split()))
                if len(coords) == 10:
                    # 提取四个背景点坐标
                    x0n, y0n, x1n, y1n, x2n, y2n, x3n, y3n, target_x, target_y = coords

                    # 将背景点作为prompt输入到SAM
                    input_points_O.append([x0n, y0n])
                    input_points_O.append([x1n, y1n])
                    input_points_O.append([x2n, y2n])
                    input_points_O.append([x3n, y3n])
                    input_points_O.append([target_x, target_y])
                    input_labels_O.extend([0, 0, 0, 0, 1])  # 背景点标签

        # 获取图片路径并读取
        photoPath = os.path.join(input_dir, filename)
        image = cv2.imread(photoPath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

        # 获取真实标签掩码
        gt_mask_path = os.path.join(gt_dir, f"{base_filename_O}.png")  # 假设GT掩码是png格式
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
        gt_mask = np.where(gt_mask > 127, 1, 0)  # 假设GT掩码为二值图像（1为前景，0为背景）

        # 确保将图像传递给预测器
        predictor.set_image(image)  # 必须先调用此方法设置图像

        # SAM模型预测
        masks_with_O_prompt, scores, logits = predictor.predict(
            point_coords=np.array(input_points_O),
            point_labels=np.array(input_labels_O),
            multimask_output=False,
        )

        # 生成预测掩码后，处理图像
        imageResult_with_prompt = image.copy()

        # 将掩码应用到图像
        mask = masks_with_O_prompt[0]  # 获取第一个掩码
        # 保存二值化掩码图像
        save_mask_image(mask, output_dir, base_filename_O)

        imageResult_with_prompt = show_mask(imageResult_with_prompt, mask)

        # 保存结果图像
        counter += 1  # 递增计数器
        output_filename_with_prompt = os.path.join(output_dir, f"{base_filename_O}_vitb.png")
        pil_image = Image.fromarray(imageResult_with_prompt)
        # pil_image.save(output_filename_with_prompt)

        # 计算IoU并存储
        iou_big = calculate_iou_big(mask, gt_mask)
        iou_values_big.append(iou_big)
        iou_small = calculate_iou_small(mask, gt_mask)
        iou_values_small.append(iou_small)

        # 计算F1并存储
        f1 = calculate_f1(mask, gt_mask)
        f1_values.append(f1)
        cnt += 1
        print(f"当前进度：{(cnt * 100/png_count):.2f}%, 背景：{iou_big:.4f}, 前景: {iou_small:.4f}")

# 计算平均IoU
average_iou_big = np.mean(iou_values_big) if iou_values_big else 0
average_iou_small = np.mean(iou_values_small) if iou_values_small else 0
print(f"Average IoU: {(average_iou_big + average_iou_small)/2:.5f}")
# 计算平均F1
average_f1 = np.mean(f1_values) if f1_values else 0
print(f"Average F1: {average_f1:.5f}")