import numpy as np
import os
import torch
from torch.utils.data import TensorDataset
from collections import defaultdict

def read_data(dataset, idx, is_train=True):
    """
    dataset: 数据集名（crop）
    idx: 客户端编号（0-9）
    is_train: 是否训练集
    """
    # 指向你生成的客户端数据目录（config.py的输出路径）
    data_root = "./crop_data/clients"  # 关键：和你的目录对应
    split = "train" if is_train else "test"
    # 拼接npy文件路径（client_0.npy ~ client_9.npy）
    file_path = os.path.join(data_root, split, f"client_{idx}.npy")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"客户端数据文件不存在：{file_path}\n请先执行 dataset/config.py 生成数据！")
    
    # 读取npy文件（config.py生成的）
    data = np.load(file_path, allow_pickle=True).item()
    return data

def read_data1(dataset, idx, is_train=True):
    if is_train:
        data_dir = os.path.join('../dataset', dataset, 'train/')
    else:
        data_dir = os.path.join('../dataset', dataset, 'test/')

    file = data_dir + str(idx) + '.npz'
    with open(file, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].tolist()
    return data

def read_client_data(dataset, idx, is_train=True, few_shot=False):
    # 1. 读取npy字典数据
    data = read_data(dataset, idx, is_train)
    
    # 2. 转换为框架原生的 list 格式 [(img, label), ...]
    imgs = torch.Tensor(data['imgs']).type(torch.float32)  # 形状：(N, 3, H, W)
    labels = torch.Tensor(data['labels']).type(torch.int64)
    if len(labels.shape) == 2:
        labels = labels.squeeze(1)
    
    # 3. 转成list（框架默认格式，len()能正确计算）
    data_list = [(imgs[i], labels[i]) for i in range(len(imgs))]
    
    # 4. 兼容few-shot逻辑（和原有read_client_data1对齐）
    if is_train and few_shot > 0:
        from collections import defaultdict
        shot_cnt_dict = defaultdict(int)
        data_list_new = []
        for data_item in data_list:
            label = data_item[1].item()
            if shot_cnt_dict[label] < few_shot:
                data_list_new.append(data_item)
                shot_cnt_dict[label] += 1
        data_list = data_list_new
    
    return data_list


def read_client_data1(dataset, idx, is_train=True, few_shot=0):
    data = read_data(dataset, idx, is_train)
    if "News" in dataset:
        data_list = process_text(data)
    elif "Shakespeare" in dataset:
        data_list = process_Shakespeare(data)
    else:
        data_list = process_image(data)

    if is_train and few_shot > 0:
        shot_cnt_dict = defaultdict(int)
        data_list_new = []
        for data_item in data_list:
            label = data_item[1].item()
            if shot_cnt_dict[label] < few_shot:
                data_list_new.append(data_item)
                shot_cnt_dict[label] += 1
        data_list = data_list_new
    return data_list

def process_image(data):
    X = torch.Tensor(data['imgs']).type(torch.float32)
    y = torch.Tensor(data['labels']).type(torch.int64)
    if len(y.shape) == 2:
        y = y.squeeze(1)
    return [(x, y) for x, y in zip(X, y)]


def process_text(data):
    X, X_lens = list(zip(*data['x']))
    y = data['y']
    X = torch.Tensor(X).type(torch.int64)
    X_lens = torch.Tensor(X_lens).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [((x, lens), y) for x, lens, y in zip(X, X_lens, y)]


def process_Shakespeare(data):
    X = torch.Tensor(data['x']).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]

