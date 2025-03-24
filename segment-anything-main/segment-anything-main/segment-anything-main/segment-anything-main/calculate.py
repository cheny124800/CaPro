import torch
import numpy as np
from sklearn.metrics import auc, roc_curve
# common
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from numpy import zeros
from numpy.random import randint
import torch
import os
import cv2
from statistics import mean
from torch.nn.functional import threshold, normalize
# Data Viz
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2




image_path = r"C:\Users\86181\Desktop\cvpr25\DataSet\GAPS384(done\sam_val_image"
Val_total_images = len(os.listdir(image_path))
all_image_paths = sorted(glob(image_path + "/*.jpg"))
print(f"Total Number of Images : {Val_total_images}")
lable_path = r"C:\Users\86181\Desktop\cvpr25\DataSet\GAPS384(done\sam_val_label"
Val_total_lables = len(os.listdir(lable_path))
all_lable_paths = sorted(glob(lable_path + "/*.png"))
print(f"Total Number of Images : {Val_total_lables}")
Test_image_paths = all_image_paths[0:Val_total_images]
Test_lable_paths = all_lable_paths[0:Val_total_lables]

#Import SAM model
model_type = 'vit_b'
model_path = r".\SAM5122weights_ViTB_GAPS.pth"
checkpoint = model_path
device = 'cuda:0'
from segment_anything import SamPredictor, sam_model_registry
sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictor_tuned = SamPredictor(sam_model)

desired_size=(400, 400)
ground_truth_test_masks = {}
for k in range(0, len(all_image_paths)):
  gt_grayscale = cv2.imread(Test_lable_paths[k])
  ground_truth_test = (gt_grayscale[:,:,2]>0).astype(np.float32)
  if desired_size is not None:
      ground_truth_test = cv2.resize(ground_truth_test, desired_size, interpolation=cv2.INTER_NEAREST)

  ground_truth_test_masks[k] = (ground_truth_test).astype(np.float32)



masks_tuned_list = {}
images_tuned_list= {}

for k in range(0, len(all_image_paths)):
    # Load the image and convert color space
    image = cv2.cvtColor(cv2.imread(Test_image_paths[k]), cv2.COLOR_BGR2RGB)
    if desired_size is not None:
       image = cv2.resize(image, desired_size, interpolation=cv2.INTER_LINEAR)
    predictor_tuned.set_image(image)

    # Perform prediction using predictor_tuned object
    masks_tuned, _, _ = predictor_tuned.predict(
        point_coords=None,
        box=None,
        multimask_output=False,
    )
    # Get the first mask from the predictions
    kk = masks_tuned[0, :, :]
    binary_mask = (kk > 0).astype(np.float32)

    # Resize the mask to the desired dimensions using nearest neighbor interpolation
    images_tuned_list[k]  = image
    masks_tuned_list[k]  = binary_mask


# 创建输出文件夹
output_folder_path = "./predicted_masks"
os.makedirs(output_folder_path, exist_ok=True)

# Assuming images_tuned_list and masks_binary_list are lists of RGB images and binary masks, respectively

# Create a 12x20 subplot grid
fig, axs = plt.subplots(12, 20, figsize=(20, 12))

# Iterate over rows and columns
for i in range(12):
    for j in range(20):
        # Calculate the index for images_tuned_list and masks_binary_list
        index = i * 20 + j

        # Check if index is within the range of your data
        if index < len(images_tuned_list):
            # Display the RGB image
            axs[i, j].imshow(images_tuned_list[index], interpolation='none')

            # Create a blue mask with alpha channel
            blue_mask = np.zeros_like(masks_tuned_list[index])
            blue_mask[masks_tuned_list[index] == 1] = 1
            blue_mask_rgb = np.stack([np.zeros_like(blue_mask), np.zeros_like(blue_mask), blue_mask], axis=-1)
            # Overlay the blue mask on the RGB image
            axs[i, j].imshow(blue_mask_rgb, alpha=0.50)
            axs[i, j].axis('off')


# Reduce spacing between images
plt.subplots_adjust(wspace=0.03, hspace=0.03)

# Adjust layout to prevent clipping of subplot labels
#plt.tight_layout()
output_path = 'test1.png'
plt.savefig(output_path, dpi=1200, bbox_inches='tight')
plt.show()
def binary_segmentation_metrics(predictions, targets):
    # Convert torch tensors to numpy arrays
    # Convert torch tensors to numpy arrays
    predictions = predictions.squeeze()
    targets = targets
    # Convert predictions to binary values (0 or 1)
    predictions_binary = (predictions > 0.5).astype(int)
    targets_binary = targets.astype(int)

    # Convert predictions to binary values (0 or 1)
    predictions_binary = (predictions > 0.5).astype(int)

    # True Positives (TP): prediction and target both are positive
    TP = np.sum((predictions_binary == 1) & (targets == 1))

    # False Positives (FP): prediction is positive but target is negative
    FP = np.sum((predictions_binary == 1) & (targets == 0))

    # False Negatives (FN): prediction is negative but target is positive
    FN = np.sum((predictions_binary == 0) & (targets == 1))

    # True Negatives (TN): prediction and target both are negative
    TN = np.sum((predictions_binary == 0) & (targets == 0))

    eps = 1e-5
    accuracy = (TP + TN + eps) / (TP + FP + FN + TN + eps)
    precision = (TP + eps) / (TP + FP + eps)
    recall = (TP + eps) / (TP + FN + eps)
    f_score = 2 * (precision * recall) / (precision + recall + eps)
    dice = (2 * TP + eps) / (2 * TP + FP + FN + eps)
    iou = (TP + eps) / (TP + FP + FN + eps)

    total = TP + FP + FN + TN + eps
    p_o = (TP + TN) / total
    p_e = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (total ** 2 + eps)
    kappa = (p_o - p_e) / (1 - p_e + eps)

    return accuracy, precision, recall, f_score, iou, kappa, FP, FN, TP, TN, dice

def calculate_average_metrics(predictions_list, targets_list):
    num_masks = len(predictions_list)

    total_metrics = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f_score': 0.0,
        'iou': 0.0,
        'kappa': 0.0,
        'FP': 0,
        'FN': 0,
        'MAR': 0.0,
        'FAR': 0.0,
        'dice': 0.0
    }

    for i in range(num_masks):
        metrics = binary_segmentation_metrics(predictions_list[i], targets_list[i])
        # 打印每张图片的性能
        print(f"Image {i + 1}:")
        print(f"  Accuracy: {metrics[0]:.4f}")
        print(f"  Precision: {metrics[1]:.4f}")
        print(f"  Recall: {metrics[2]:.4f}")
        print(f"  F1 Score: {metrics[3]:.4f}")
        print(f"  IoU: {metrics[4]:.4f}")
        print(f"  Kappa: {metrics[5]:.4f}")
        print(f"  Dice: {metrics[10]:.4f}")
        print(f"  TP: {metrics[6]} | FP: {metrics[7]} | FN: {metrics[8]} | TN: {metrics[9]}\n")
        for metric_name, value in zip(total_metrics.keys(), metrics):
            total_metrics[metric_name] += value

        total_metrics['MAR'] += metrics[8] / (metrics[8] + metrics[9])
        total_metrics['FAR'] += metrics[6] / (metrics[9] + metrics[6])

    avg_metrics = {k: v / num_masks for k, v in total_metrics.items()}

    return avg_metrics


ground_truth_test_masks = {}
# Example usage
for k in range(0, len(all_image_paths)):
    gt_grayscale = cv2.imread(Test_lable_paths[k])
    ground_truth_test = (gt_grayscale[:, :, 2] > 0).astype(np.float32)
    if desired_size is not None:
        ground_truth_test = cv2.resize(ground_truth_test, desired_size, interpolation=cv2.INTER_NEAREST)
    ground_truth_test_masks[k] = ground_truth_test

masks_tuned_list = {}
images_tuned_list = {}

for k in range(0, len(all_image_paths)):
    # 加载图像并转换颜色空间
    image = cv2.cvtColor(cv2.imread(Test_image_paths[k]), cv2.COLOR_BGR2RGB)
    if desired_size is not None:
        image = cv2.resize(image, desired_size, interpolation=cv2.INTER_LINEAR)

    predictor_tuned.set_image(image)

    # 执行预测
    masks_tuned, _, _ = predictor_tuned.predict(
        point_coords=None,
        box=None,
        multimask_output=False,
    )

    # 获取预测的第一个掩码
    kk = masks_tuned[0, :, :]
    binary_mask = (kk > 0).astype(np.float32)
    images_tuned_list[k] = image
    masks_tuned_list[k] = binary_mask


avg_metrics = calculate_average_metrics(masks_tuned_list, ground_truth_test_masks)

for metric_name, value in avg_metrics.items():
    print(f"Average {metric_name}: {value}")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.color import rgb2gray
from skimage.measure import find_contours

# Assuming images_tuned_list and masks_binary_list are lists of RGB images and binary masks, respectively
# Border parameters
border_color = 'black'
border_width = 0.35

# Create a 13x1 subplot grid for 13 images
fig, axs = plt.subplots(1, 13, figsize=(20, 10))

# Flatten the subplot grid for easy iteration
axs = axs.flatten()

# Select 13 images to display
selected_indices = [0, 1, 5, 12, 18, 19, 20, 21, 23, 24, 26, 27, 28]
# Iterate over selected indices
for i, index in enumerate(selected_indices):
    # Check if index is within the range of your data
    if index < len(images_tuned_list):
        # Display the RGB image
        gray_image = rgb2gray(images_tuned_list[index])

        axs[i].imshow(gray_image, cmap='gray', interpolation='none', alpha=0.5)  # Adjust alpha value as needed

        # Create a mask with alpha channel for segmentation result
        ground_truth_mask = (ground_truth_test_masks[index])
        mask = (masks_tuned_list[index])

        # Create a blue mask with alpha channel for true positives
        true_positive_mask = (mask == 1) & (ground_truth_mask == 1)
        blue_mask = np.stack([np.zeros_like(true_positive_mask), np.zeros_like(true_positive_mask), true_positive_mask], axis=-1)
        blue_mask_rgba = np.concatenate([blue_mask, true_positive_mask[:, :, None].astype(float)], axis=-1)
        axs[i].imshow(blue_mask_rgba, alpha=0.50)

        # Create a yellow mask with alpha channel for false alarms
        false_alarm_mask = (mask == 1) & (ground_truth_mask == 0)
        tight_green_mask = np.stack([np.zeros_like(false_alarm_mask), false_alarm_mask, np.zeros_like(false_alarm_mask)], axis=-1)
        tight_green_mask_rgba = np.concatenate([tight_green_mask, false_alarm_mask[:, :, None].astype(float)], axis=-1)
        axs[i].imshow(tight_green_mask_rgba, alpha=0.60)

        # Create a red mask with alpha channel for missed alarms
        missed_alarm_mask = (mask == 0) & (ground_truth_mask == 1)
        magenta_mask = np.stack([missed_alarm_mask, np.zeros_like(missed_alarm_mask), missed_alarm_mask], axis=-1)
        magenta_mask_rgba = np.concatenate([magenta_mask, missed_alarm_mask[:, :, None].astype(float)], axis=-1)
        axs[i].imshow(magenta_mask_rgba, alpha=0.60)

        # Add a black border around the image
        rect = Rectangle((0, 0), gray_image.shape[1], gray_image.shape[0], linewidth=border_width, edgecolor=border_color, facecolor='none')
        axs[i].add_patch(rect)

        # Turn off axis labels for better visualization
        axs[i].axis('off')
        # 保存每个图像的预测掩码
        mask_image = (mask * 255).astype(np.uint8)  # 将掩码归一化到 [0, 255]
        output_image_path = os.path.join(output_folder_path, f"prediction_{index}.png")
        cv2.imwrite(output_image_path, mask_image)  # 保存掩码

        print(f"Saved mask for image {index} to {output_image_path}")

# Adjust layout to prevent clipping of subplot labels
plt.subplots_adjust(wspace=0.02, hspace=0.02)
# Adjust layout to prevent clipping of subplot labels
# Save the figure with 600 DPI
plt.subplots_adjust(wspace=0.02, hspace=0.02)


output_path2 = 'test2.png'
plt.savefig(output_path2, dpi=600, bbox_inches='tight')
plt.show()