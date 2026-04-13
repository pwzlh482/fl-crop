import torch
import cv2
import os
import numpy as np

def yolov8_collate_fn(batch):
    """关键函数：将变长的检测标签组合成Batch"""
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, 0)
    new_targets = []
    for i, det in enumerate(targets):
        if det.shape[0] > 0:
            # 在标签前加上batch索引 [batch_idx, cls, x, y, w, h]
            ti = torch.cat([torch.full((det.shape[0], 1), i), torch.tensor(det)], dim=1)
            new_targets.append(ti)
    return imgs, torch.cat(new_targets, 0) if len(new_targets) > 0 else torch.zeros((0, 6))

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, img_names, labels, img_dir, size=320):
        self.img_names = img_names
        self.labels = labels 
        self.img_dir = img_dir
        self.size = size

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.size, self.size))
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        return img, self.labels[idx]
