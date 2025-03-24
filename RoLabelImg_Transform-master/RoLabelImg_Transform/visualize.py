#encoding utf-8

import numpy as np
import os
import cv2


def draw(image_data, line):
    line = line.strip()
    points_list = line.split(',')[:-1]
    if points_list:
        points = [int(float(point)) for point in points_list]
        points = np.reshape(points, (-1, 2))
        points = points[[0, 2, 1, 3]]  # 这里可以根据需要调整点的顺序
        contours = np.asarray([[list(p) for p in points]])
        cv2.drawContours(image_data, contours, -1, (0, 255, 0), 2)
    
       
def visualize(image_root, gt_root, det_root, output_root):
    def read_gt_file(image_name):
        gt_file = os.path.join(gt_root, '%s.txt'%(image_name.split('.png')[0]))
        print(f"Reading GT file: {gt_file}")  # 打印路径
        with open(gt_file, 'r') as gt_f:
            return gt_f.readlines()

    def read_det_file(image_name):
        det_file = os.path.join(det_root, '%s.txt'%(image_name.split('.png')[0]))
        with open(det_file, 'r') as det_f:
            return det_f.readlines()
    
    def read_image_file(image_name):
        img_file = os.path.join(image_root, image_name)
        img_array = cv2.imread(img_file)
        return img_array
    # 创建输出文件夹
    det_output_dir = os.path.join(output_root, 'det')
    gt_output_dir = os.path.join(output_root, 'gt')
    os.makedirs(det_output_dir, exist_ok=True)
    os.makedirs(gt_output_dir, exist_ok=True)

    for image_name in os.listdir(image_root):
        image_data = read_image_file(image_name)
        gt_image_data = image_data.copy()
        det_image_data = image_data.copy()

        if det_root:
            det_list = read_det_file(image_name)
            for det in det_list:
                draw(det_image_data, det)
                cv2.imwrite(os.path.join(output_root, '/det/' + image_name), det_image_data)
        if gt_root:
            gt_list = read_gt_file(image_name)
            for gt in gt_list:
                draw(gt_image_data, gt)
                print(output_root + '/gt/' + image_name)
                cv2.imwrite(output_root + '/gt/' + image_name, gt_image_data)


if __name__ == '__main__':    
    image_root = './RoLabelImg_Transform/img'
    gt_root = './RoLabelImg_Transform/txt'
    det_root = None
    output_root = './RoLabelImg_Transform/visualized_img'
    visualize(image_root, gt_root, det_root, output_root)
