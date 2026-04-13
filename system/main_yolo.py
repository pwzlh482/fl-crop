#!/usr/bin/env python
import copy
import torch
import argparse
import os
import torch.nn as nn
from ultralytics import YOLO
from flcore.servers.serverprox import FedProx
from flcore.clients.clientprox import clientProx
from ultralytics.utils.loss import v8DetectionLoss

# ===================== 1. YOLO 模型包装逻辑 =====================
class YoloDetModel(nn.Module):
    def __init__(self, weights="./yolov8n.pt"):
        super().__init__()
        os.environ["YOLO_AUTO_DOWNLOAD"] = "False"
        # 加载 YOLOv8n 的模型结构
        self.model = YOLO(weights).model 
    def forward(self, x):
        return self.model(x)

# ===================== 2. 适配检测任务的 Client =====================
class clientProxYolo(clientProx):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        # 核心修改：分类 CrossEntropyLoss 换成 YOLO 检测 Loss
        self.loss_func = v8DetectionLoss(self.model)

    def train(self):
        self.model.train()
        for epoch in range(self.local_epochs):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(x)
                # YOLO 要求的数据字典：[batch_idx, cls, x, y, w, h]
                batch_dict = {"img": x, "bboxes": y[:, 2:], "cls": y[:, 1], "batch_idx": y[:, 0]}
                loss, _ = self.loss_func(preds, batch_dict)
                
                # 保持 FedProx 特有的正则项逻辑
                if self.mu > 0:
                    proximal_term = 0.0
                    for w, w_t in zip(self.model.parameters(), self.global_params):
                        proximal_term += (w - w_t).norm(2)
                    loss += 0.5 * self.mu * proximal_term

                loss.backward()
                self.optimizer.step()

def main():
    parser = argparse.ArgumentParser()
    # ===================== 保持你原始 main.py 的所有参数定义 =====================
    parser.add_argument('--dataset', type=str, default="COCO_YOLO")
    parser.add_argument('--num_classes', type=int, default=80)
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--algorithm', type=str, default="FedProx")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--local_learning_rate', type=float, default=0.01)
    parser.add_argument('--global_rounds', type=int, default=100)
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--mu', type=float, default=0.01)
    parser.add_argument('--join_ratio', type=float, default=1.0)
    parser.add_argument('--random_join_ratio', type=bool, default=False)
    # 补齐之前报错的那些“基建参数”
    parser.add_argument('--client_drop_rate', type=float, default=0.0)
    parser.add_argument('--num_new_clients', type=int, default=0)
    parser.add_argument('--batch_num_per_client', type=int, default=0)
    # 其他琐碎参数（直接沿用你原始文件的默认值）
    parser.add_argument('--times', type=int, default=1)
    parser.add_argument('--eval_gap', type=int, default=1)
    parser.add_argument('--save_folder_name', type=str, default="yolo_results")
    parser.add_argument('--learning_rate_decay', type=bool, default=False)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--auto_break', type=bool, default=False)
    parser.add_argument('--top_cnt', type=int, default=100)
    parser.add_argument('--few_shot', type=bool, default=False)
    parser.add_argument('--goal', type=str, default="test")
    parser.add_argument('--time_select', type=bool, default=False)
    parser.add_argument('--time_threthold', type=float, default=10000)
    
    args = parser.parse_args()

    # 1. GPU 分配逻辑 (沿用你原始 main.py)
    args.client_device_map = {i: torch.device(args.device) for i in range(args.num_clients)}

    # 2. 核心修改：换成 YOLO 模型
    args.model = YoloDetModel(weights="./yolov8n.pt")
    
    # 3. 实例化 Server 并注入我们适配的 YOLO Client 类
    server = FedProx(args, times=1)
    server.set_clients(clientProxYolo) 
    
    print(">>> 基础参数已拉齐至原始 main.py 水准，YOLO 联邦训练启动...")
    server.train()

if __name__ == "__main__":
    main()

