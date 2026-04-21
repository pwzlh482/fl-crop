import os
import sys
import numpy as np
from PIL import Image
import random
import torch
import torchvision.transforms as transforms
import ujson
from sklearn.model_selection import train_test_split

random.seed(1)
np.random.seed(1)

# 参数设置
num_clients = 20
niid = True
balance = True
partition = 'dir'  # Dirichlet分布
alpha = 1.0  # 与Cifar10一致
class_per_client = 2
train_ratio = 0.75

# 类别映射
class_names = ['rice', 'wheat', 'corn', 'weed']
class_to_label = {'rice': 0, 'wheat': 1, 'corn': 2, 'weed': 3}

# 图像尺寸
img_size = 64  # 与MobileNet V2中的crop数据集一致

def load_images():
    """加载所有图像，调整大小，转换为numpy数组"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    images = []
    labels = []
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    for class_name, label in class_to_label.items():
        folder_path = os.path.join(base_dir, class_name)
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹不存在 {folder_path}")
            continue
        
        extensions = ('.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG')
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]
        files.sort()
        
        print(f"加载 {class_name} 图像: {len(files)} 张")
        
        for filename in files:
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                # (C, H, W) -> (C, H, W), 转为uint8 [0-255]
                img_np = (img_tensor.numpy() * 255).astype(np.uint8)
                images.append(img_np)
                labels.append(label)
            except Exception as e:
                print(f"加载失败 {img_path}: {e}")
    
    images_np = np.array(images)
    labels_np = np.array(labels, dtype=np.int64)
    
    print(f"总计加载 {len(images_np)} 张图像，标签分布: {np.bincount(labels_np)}")
    return images_np, labels_np


def separate_data_dir(data, num_clients, num_classes, alpha=1.0):
    """Dirichlet非IID划分（简化版，适合小数据集）"""
    dataset_content, dataset_label = data
    K = num_classes
    N = len(dataset_label)
    
    # 初始化
    client_data = [[] for _ in range(num_clients)]
    idx = [np.where(dataset_label == k)[0] for k in range(K)]
    
    # Dirichlet分布
    min_size = 0
    while min_size < 2:  # 每个客户端至少2个样本
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(K):
            np.random.shuffle(idx[k])
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx[k])).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx[k], proportions))]
        min_size = min([len(idx_j) for idx_j in idx_batch])
    
    for j in range(num_clients):
        np.random.shuffle(idx_batch[j])
        client_data[j] = idx_batch[j]
    
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]
    
    for client in range(num_clients):
        idxs = client_data[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]
        
        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))
        print(f"Client {client}\t Size: {len(X[client])}\t Labels: {np.unique(y[client])}")
    
    return X, y, statistic


def split_data(X, y):
    """分割训练/测试集"""
    train_data, test_data = [], []
    
    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_ratio, shuffle=True)
        
        train_data.append({'x': X_train, 'y': y_train})
        test_data.append({'x': X_test, 'y': y_test})
    
    print(f"Total train samples: {sum(len(d['y']) for d in train_data)}")
    print(f"Total test samples: {sum(len(d['y']) for d in test_data)}")
    
    return train_data, test_data


def save_file(config_path, train_path, test_path, train_data, test_data, num_clients, 
              num_classes, statistic, niid, balance, partition):
    """保存文件"""
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    
    config = {
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'non_iid': niid, 
        'balance': balance, 
        'partition': partition, 
        'Size of samples for labels in clients': statistic, 
        'alpha': alpha, 
        'batch_size': 64,
    }
    
    print("Saving to disk...")
    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)
    
    print("Finish generating dataset.")


def generate_dataset():
    """生成数据集"""
    dir_path = "crop/"
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    
    print("加载图像...")
    images, labels = load_images()
    
    print("\n分配数据到客户端 (Dirichlet)...")
    X, y, statistic = separate_data_dir((images, labels), num_clients, len(class_names), alpha=alpha)
    
    print("\n分割训练/测试集...")
    train_data, test_data = split_data(X, y)
    
    print("\n保存文件...")
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, 
              len(class_names), statistic, niid, balance, partition)
    
    print("\n=== 数据集生成完成！===")


if __name__ == "__main__":
    generate_dataset()
