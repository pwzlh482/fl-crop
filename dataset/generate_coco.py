import os
import sys
import numpy as np
import random
from utils.dataset_utils import check, save_file, separate_data, split_data

random.seed(1)
np.random.seed(1)

num_clients = 20
dir_path = "COCO_YOLO/"

def generate_dataset(dir_path, num_clients, niid, balance, partition):
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    image_dir = os.path.join(dir_path, "rawdata", "images")
    label_dir = os.path.join(dir_path, "rawdata", "labels")
    
    # 1. 筛选出既有图片又有标签的样本
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    valid_images = []
    temp_labels = []

    print("正在扫描 5000 张 COCO 图片...")
    for img_name in image_files:
        label_name = img_name.replace('.jpg', '.txt')
        label_path = os.path.join(label_dir, label_name)
        
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path, 'r') as f:
                line = f.readline()
                if line:
                    cls = int(line.split()[0])
                    temp_labels.append(cls)
                    valid_images.append(img_name)

    # 2. 标签重映射 (COCO类别ID不连续，必须映射到 0-79)
    unique_labels = sorted(np.unique(temp_labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    dataset_label = np.array([label_map[l] for l in temp_labels])
    dataset_image = np.array(valid_images)
    
    num_classes = len(unique_labels)
    print(f'有效样本数: {len(dataset_image)}, 映射后类别总数: {num_classes}')

    # 3. 执行联邦划分 (5000张够用，可以用 dir 模式)
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=5)
    
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
              statistic, niid, balance, partition)
    print("COCO-Val 联邦数据集生成完毕！")

if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    generate_dataset(dir_path, num_clients, niid, balance, partition)
