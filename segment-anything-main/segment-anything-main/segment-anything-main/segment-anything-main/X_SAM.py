import time
import cv2
import os
from PIL import ImageDraw
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
# segment_anything 是SAM算法的依赖库
from segment_anything import sam_model_registry, SamPredictor
# 屏蔽后台警告
import warnings

warnings.filterwarnings("ignore")


# 输入： 原图 + 蒙版 -》 输出： 图像最终结果（数组形式）
def show_mask(image, mask):
    # mask 的形状是（1 X 图像高 X 图像宽）
    color_ = [0, 255, 0]  # 把mask做成 绿色蒙版
    color = np.array(color_)  # 把mask做成 绿色蒙版
    h, w = mask.shape[-2:]  # 从mask中 取出蒙版的 高 与 宽
    imageResult = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)  # 把mask蒙版形状改为 （图像高 X 图像宽 X 3）
    # 便利原图像素点，将蒙版点更新到原图的
    for i in range(0, h):
        for j in range(0, w):
            # 判断mask蒙版中的rbg是否为绿色
            if color_[0] == imageResult[i][j][0] and color_[1] == imageResult[i][j][1] and color_[2] == \
                    imageResult[i][j][2]:
                image[i][j] = color_  # 将image原图中对应mask蒙版的点，改为绿色
    return image


# 计算IoU（只计算前景的IoU）

def calculate_iou(pred_mask, gt_mask):
    # 交集：预测为前景且真实标签也为前景的区域
    intersection = np.sum(np.logical_and(pred_mask == 1, gt_mask == 1))
    # 并集：预测为前景或真实标签为前景的区域
    union = np.sum(np.logical_or(pred_mask == 1, gt_mask == 1))
    # 计算IoU，防止除以0
    iou = intersection / union if union != 0 else 0
    return iou


'''''''''
# 计算IoU（前景 + 背景）
def calculate_iou(pred_mask, gt_mask):
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
'''''''''
sam_checkpoint = "sam_vit_b_01ec64.pth"  # 定义模型路径
model_type = "vit_b"  # 定义模型类型
device = "cuda"  # "cpu"  or  "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)  # 定义模型参数
predictor = SamPredictor(sam)  # 调用预测模型

# 定义文件夹路径
input_dir = r"C:\Users\86181\Desktop\CRACK_SIM\images"  # 修改为目标文件夹路径
txt_X_dir = r"C:\Users\86181\Desktop\CRACK_SIM\txt_X"  # txt文件所在目录
txt_O_dir = r"C:\Users\86181\Desktop\CRACK_SIM\txt_O"
output_dir = r"C:\Users\86181\Desktop\CRACK_SIM\output"  # 输出文件夹路径（请确保该路径已存在）
gt_dir = r"C:\Users\86181\Desktop\CRACK_SIM\gt"  # 真实掩码文件夹路径

# 创建输出目录（如果不存在）
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 初始化计数器和IoU列表
iou_with_X_prompt = []  # 存储输入点提示情况下的IoU
iou_with_O_prompt = []  # 存储输入点提示情况下的IoU
iou_without_prompt = []  # 存储没有输入点提示情况下的IoU

# 遍历目录中的所有jpg文件
for filename in os.listdir(input_dir):
    if filename.lower().endswith('.png'):

        # X_prompt_path
        base_filename_X = os.path.splitext(filename)[0]
        txt_filename_X = f"{base_filename_X}_mid.txt"  # 生成对应的txt文件名
        txt_file_path_X = os.path.join(txt_X_dir, txt_filename_X)
        input_points_X = []
        input_labels_X = []  # 用来存储每个点的标签（背景点为0，对象点为1）
        with open(txt_file_path_X, 'r') as f:
            for line in f:
                coords = list(map(float, line.split()))
                if len(coords) == 10:
                    # 提取每行的两个坐标点 (x1, y1) 和 (x2, y2)
                    input_points_X.append([coords[0], coords[1]])
                    input_points_X.append([coords[2], coords[3]])
                    input_points_X.append([coords[4], coords[5]])
                    input_points_X.append([coords[6], coords[7]])
                    input_labels_X.append(0)  # 背景点标签
                    input_labels_X.append(0)  # 背景点标签
                    input_labels_X.append(0)  # 背景点标签
                    input_labels_X.append(0)  # 背景点标签
                    # 后两个坐标是目标对象点（x3, y3），标记为 1
                    input_points_X.append([int(coords[8]), int(coords[9])])
                    input_labels_X.append(1)  # 对象点标签
        # 如果没有提取到足够的点，跳过当前文件
        if len(input_points_X) == 0:
            print(f"Warning: No valid points found in {txt_file_path_X}. Skipping {filename}.")
            continue

        # O_prompt_path
        base_filename_O = os.path.splitext(filename)[0]
        txt_filename_O = f"{base_filename_O}_O.txt"  # 生成对应的txt文件名
        txt_file_path_O = os.path.join(txt_O_dir, txt_filename_O)
        input_points_O = []
        input_labels_O = []  # 用来存储每个点的标签（背景点为0，对象点为1）
        with open(txt_file_path_O, 'r') as f:
            for line in f:
                coords = list(map(float, line.split()))
                if len(coords) == 10:
                    # 提取每行的4个坐标点
                    input_points_O.append([coords[0], coords[1]])
                    input_points_O.append([coords[2], coords[3]])
                    input_points_O.append([coords[4], coords[5]])
                    input_points_O.append([coords[6], coords[7]])
                    input_labels_O.append(0)  # 背景点标签
                    input_labels_O.append(0)  # 背景点标签
                    input_labels_O.append(0)  # 背景点标签
                    input_labels_O.append(0)  # 背景点标签
                    # 后两个坐标是目标对象点，标记为 1
                    input_points_O.append([int(coords[8]), int(coords[9])])
                    input_labels_O.append(1)  # 对象点标签
        # 如果没有提取到足够的点，跳过当前文件
        if len(input_points_O) == 0:
            print(f"Warning: No valid points found in {txt_file_path_O}. Skipping {filename}.")
            continue



        # 记录处理时间
        start_time = time.time()

        # 构建文件路径
        photoPath = os.path.join(input_dir, filename)
        image = cv2.imread(photoPath)  # opencv 方式读取图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将图像从BGR颜色空间转换为RGB颜色空间
        predictor.set_image(image)  # 预测图像



        # SAM_O:
        # 将输入点转换为 numpy 数组
        input_points_O = np.array(input_points_O)
        input_labels_O = np.array(input_labels_O)
        masks_with_O_prompt, scores, logits = predictor.predict(
            point_coords=input_points_O,
            point_labels=input_labels_O,
            multimask_output=False,
        )
        gt_mask_path = os.path.join(gt_dir, f"{base_filename_O}_mask.png")  # 真实掩码路径
        if not os.path.exists(gt_mask_path):
            print(f"Warning: Ground truth mask for {filename} not found. Skipping IoU calculation.")
            continue
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
        gt_mask = (gt_mask > 128).astype(np.uint8)  # 将真实掩码二值化（前景为白色，背景为黑色）
        pred_mask_bin_with_prompt = (masks_with_O_prompt > 0.5).astype(np.uint8)  # 将预测掩码二值化
        iou_with_O_prompt_score = calculate_iou(pred_mask_bin_with_prompt, gt_mask)
        print(f"iou_with_O_prompt_score : {iou_with_O_prompt_score}")
        iou_with_O_prompt.append(iou_with_O_prompt_score)
        imageResult_with_prompt = show_mask(image.copy(), masks_with_O_prompt)  # 最终图像结果
        output_filename_with_prompt = os.path.join(output_dir, f"{base_filename_O}_result_with_O_prompt.png")
        plt.imshow(imageResult_with_prompt)
        plt.imsave(output_filename_with_prompt, imageResult_with_prompt)

        # SAM_X:
        # 将输入点转换为 numpy 数组
        input_points = np.array(input_points_X)
        input_labels = np.array(input_labels_X)
        masks_with_X_prompt, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False,
        )
        gt_mask_path = os.path.join(gt_dir, f"{base_filename_X}.png")  # 真实掩码路径
        if not os.path.exists(gt_mask_path):
            print(f"Warning: Ground truth mask for {filename} not found. Skipping IoU calculation.")
            continue
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
        gt_mask = (gt_mask > 128).astype(np.uint8)  # 将真实掩码二值化（前景为白色，背景为黑色）
        pred_mask_bin_with_prompt = (masks_with_X_prompt > 0.5).astype(np.uint8)  # 将预测掩码二值化
        iou_with_X_prompt_score = calculate_iou(pred_mask_bin_with_prompt, gt_mask)
        print(f"iou_with_X_prompt_score : {iou_with_X_prompt_score}")
        iou_with_X_prompt.append(iou_with_X_prompt_score)
        imageResult_with_prompt = show_mask(image.copy(), masks_with_X_prompt)  # 最终图像结果
        output_filename_with_prompt = os.path.join(output_dir, f"{base_filename_X}_result_with_X_prompt.png")
        plt.imshow(imageResult_with_prompt)
        plt.imsave(output_filename_with_prompt, imageResult_with_prompt)



        # ONLY_SAM：：：计算IoU并保存结果图（SAM）
        masks_without_prompt, scores, logits = predictor.predict(
            point_coords=None,  # 不提供输入点
            point_labels=None,
            multimask_output=False,
        )
        pred_mask_bin_without_prompt = (masks_without_prompt > 0.5).astype(np.uint8)  # 将预测掩码二值化
        iou_without_prompt_score = calculate_iou(pred_mask_bin_without_prompt, gt_mask)
        iou_without_prompt.append(iou_without_prompt_score)
        print(f"iou_without_prompt_score : {iou_without_prompt_score}")
        imageResult_without_prompt = show_mask(image.copy(), masks_without_prompt)  # 最终图像结果
        output_filename_without_prompt = os.path.join(output_dir, f"{base_filename_O}_result_without_prompt.png")
        plt.imshow(imageResult_without_prompt)
        plt.imsave(output_filename_without_prompt, imageResult_without_prompt)


        # 记录运行时间
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Processed {filename}, Score: {scores}, Execution Time: {execution_time:.2f}s")

# 计算平均IoU
avg_iou_with_X_prompt = np.mean(iou_with_X_prompt)
avg_iou_with_O_prompt = np.mean(iou_with_O_prompt)
avg_iou_without_prompt = np.mean(iou_without_prompt)

# 输出总平均IoU
print(f"\nAverage IoU with X prompt: {avg_iou_with_X_prompt:.4f}")
print(f"Average IoU with O prompt: {avg_iou_with_O_prompt:.4f}")
print(f"Average IoU without prompt: {avg_iou_without_prompt:.4f}")


