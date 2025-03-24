# common
import numpy as np
from glob import glob
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



image_path = r"C:\Users\86181\Desktop\cvpr25\DataSet\CRACK500(done\sam_val_image"
Val_total_images = len(os.listdir(image_path))
all_image_paths = sorted(glob(image_path + "/*.jpg"))
print(f"Total Number of Images : {Val_total_images}")
lable_path = r"C:\Users\86181\Desktop\cvpr25\DataSet\CRACK500(done\sam_val_label"
Val_total_lables = len(os.listdir(lable_path))
all_lable_paths = sorted(glob(lable_path + "/*.png"))
print(f"Total Number of Images : {Val_total_lables}")
Test_image_paths = all_image_paths[0:Val_total_images]
Test_lable_paths = all_lable_paths[0:Val_total_lables]

#Import SAM model
model_type = 'vit_b'
model_path = r".\SAM5122weights_ViTB.pth"
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

import matplotlib.pyplot as plt
import numpy as np

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
output_path = './test.png'
plt.savefig(output_path, dpi=1200, bbox_inches='tight')
plt.show()