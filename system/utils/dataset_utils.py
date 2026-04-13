import numpy as np
import json
import os

def separate_data(dataset, num_clients, num_classes, partition="noniid", alpha=0.5):
    """
    适配小数据量的非IID划分（30张图+20客户端），避免死循环
    """
    imgs, labels = dataset
    K = num_classes
    N = len(labels)

    # 初始化客户端数据容器
    client_data = [[] for _ in range(num_clients)]
    
    if partition == "noniid":
        # 简单非IID：按类别分组，每个客户端优先分配某一类数据，剩余数据随机分配
        # 1. 按类别拆分索引
        cls_idx = [np.where(labels == k)[0] for k in range(K)]
        # 2. 每个客户端先分配1张某类数据（保证min_size >=1）
        client_cls = [i % K for i in range(num_clients)]  # 客户端对应类别：0,1,2,0,1,2...
        for client_id in range(num_clients):
            target_cls = client_cls[client_id]
            if len(cls_idx[target_cls]) > 0:
                # 取该类的1个样本
                idx = cls_idx[target_cls][0]
                client_data[client_id].append(idx)
                # 从原列表删除，避免重复分配
                cls_idx[target_cls] = np.delete(cls_idx[target_cls], 0)
        # 3. 剩余数据随机分配给客户端
        remaining_idx = np.concatenate(cls_idx)
        np.random.shuffle(remaining_idx)
        for idx in remaining_idx:
            # 随机选一个客户端分配
            client_id = np.random.randint(0, num_clients)
            client_data[client_id].append(idx)
    else:
        # IID划分：随机均分
        all_idx = np.arange(N)
        np.random.shuffle(all_idx)
        client_data = np.array_split(all_idx, num_clients)

    # 整理最终数据
    final_client_data = []
    statistic = []
    for i in range(num_clients):
        client_idx = np.array(client_data[i])
        client_labels = labels[client_idx]
        cls_count = {k: int(np.sum(client_labels == k)) for k in range(K)}
        statistic.append(cls_count)
        final_client_data.append((imgs[client_idx], labels[client_idx]))

    return final_client_data, statistic
    
def separate_data1(dataset, num_clients, num_classes, partition="noniid", alpha=0.5):
    """
    非IID数据划分
    """
    imgs, labels = dataset
    min_size = 0
    K = num_classes
    N = len(labels)

    # 初始化客户端数据容器
    client_data = [[] for _ in range(num_clients)]
    idx = [np.where(labels == k)[0] for k in range(K)]
    num_per_class = [len(idx[k]) for k in range(K)]

    # 非IID划分（Dirichlet分布）
    if partition == "noniid":
        while min_size < 1:
            dist = np.random.dirichlet([alpha] * num_clients, K)
            dist = dist / dist.sum(axis=1, keepdims=True)
            client_class_num = np.array([np.floor(dist[k] * num_per_class[k]) for k in range(K)]).astype(int)
            client_num = client_class_num.sum(axis=0)
            min_size = min(client_num)

        # 分配数据索引
        for k in range(K):
            np.random.shuffle(idx[k])
            pos = 0
            for i in range(num_clients):
                end = pos + client_class_num[k][i]
                client_data[i].extend(idx[k][pos:end])
                pos = end
    # IID划分（备用）
    else:
        all_idx = np.arange(N)
        np.random.shuffle(all_idx)
        client_data = np.array_split(all_idx, num_clients)

    # 整理最终数据
    final_client_data = []
    statistic = []
    for i in range(num_clients):
        client_idx = np.array(client_data[i])
        client_labels = labels[client_idx]
        cls_count = {k: int(np.sum(client_labels == k)) for k in range(K)}
        statistic.append(cls_count)
        final_client_data.append((imgs[client_idx], labels[client_idx]))

    return final_client_data, statistic

def save_file(config_path, train_path, val_path, test_path, train_data, val_data, test_data, statistic):
    """
    保存客户端数据集（适配你的config.py逻辑）
    """
    # 保存配置
    config = {
        "num_clients": len(train_data),
        "num_classes": len(statistic[0]),
        "statistic": statistic,
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    # 保存每个客户端数据
    for client_id in range(len(train_data)):
        # 训练数据
        train_imgs, train_labels, train_annots = train_data[client_id]
        np.save(os.path.join(train_path, f"client_{client_id}.npy"), 
                {"imgs": train_imgs, "labels": train_labels, "annot": train_annots})
        # 验证数据
        val_imgs, val_labels, val_annots = val_data[client_id]
        np.save(os.path.join(val_path, f"client_{client_id}.npy"), 
                {"imgs": val_imgs, "labels": val_labels, "annot": val_annots})
        # 测试数据
        test_imgs, test_labels, test_annots = test_data[client_id]
        np.save(os.path.join(test_path, f"client_{client_id}.npy"), 
                {"imgs": test_imgs, "labels": test_labels, "annot": test_annots})

    print(f"✅ 数据集已保存至: {train_path}/{val_path}/{test_path}")