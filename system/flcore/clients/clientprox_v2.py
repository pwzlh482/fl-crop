"""
clientProx V2 - FedProx 优化客户端
基于实际服务器 clientprox.py，仅新增，不修改原文件

优化点：
1. Label Smoothing (0.1) 防止过拟合
2. PerturbedGradientDescent1 替代原版 SGD，支持 momentum + weight_decay
3. MultiStepLR 学习率调度（与原版 FedProx 一致，前期高 lr 学得快）
4. 梯度裁剪 max_norm=5.0 防止梯度爆炸
5. 动态 mu 衰减 — 高 mu 抑制早期漂移，低 mu 释放后期个性化
"""

import torch
import numpy as np
import time
import copy
import torch.nn as nn
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent, PerturbedGradientDescent1
from flcore.clients.clientbase import Client


class clientProxV2(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.mu = args.mu
        self.momentum = getattr(args, 'momentum', 0.9)
        self.weight_decay = getattr(args, 'weight_decay', 1e-4)
        self.global_params = copy.deepcopy(list(self.model.parameters()))

        # 优化1: Label Smoothing 防止过拟合
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.learning_rate_decay = args.learning_rate_decay

        # 优化2: 使用 PerturbedGradientDescent1 (带 momentum + weight_decay)
        self.optimizer = PerturbedGradientDescent1(
            self.model.parameters(),
            lr=self.learning_rate,
            mu=self.mu,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        # 优化3: MultiStepLR（与原版 FedProx 一致，前期高 lr 学得快）
        # 默认在第 40、60 轮衰减，可通过 -ldm 参数自定义
        milestones = getattr(args, 'lr_decay_milestones', [40, 60])
        self.learning_rate_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self.optimizer,
            milestones=milestones,
            gamma=args.learning_rate_decay_gamma
        )

        # 优化4: 梯度裁剪阈值
        self.max_grad_norm = 5.0

        # 优化5: 动态 mu 参数
        self.base_mu = args.mu
        self.current_round = 0

        # 优化6: Mixup 数据增强
        self.mixup_alpha = getattr(args, 'mixup_alpha', 0.2)

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(x)

                # Mixup 数据增强
                if hasattr(self, 'mixup_alpha') and self.mixup_alpha > 0 and self.model.training:
                    lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                    if lam > 1 - lam:
                        lam = 1 - lam  # 确保 lam 较小
                    batch_size = x.size(0)
                    index = torch.randperm(batch_size, device=x.device)
                    mixed_x = lam * x + (1 - lam) * x[index]
                    mixed_output = self.model(mixed_x)
                    loss = lam * self.loss(mixed_output, y) + (1 - lam) * self.loss(mixed_output, y[index])
                else:
                    loss = self.loss(output, y)

                # FedProx 核心：手动添加近端项
                if self.mu > 0:
                    proximal_term = 0.0
                    for w, w_t in zip(self.model.parameters(), self.global_params):
                        proximal_term += (w - w_t).norm(2)**2
                    loss += (self.mu / 2) * proximal_term

                loss.backward()

                # 优化4: 梯度裁剪防止爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step(self.global_params, self.device)

        # 学习率调度（由客户端自主调度，与服务器端lr控制独立）
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model):
        for new_param, global_param, param in zip(model.parameters(), self.global_params, self.model.parameters()):
            global_param.data = new_param.data.clone()
            param.data = new_param.data.clone()

    def update_mu(self, current_round, total_rounds):
        """优化5: 动态 mu 衰减 — 前期高 mu 抑制漂移，后期低 mu 释放个性化

        衰减公式: mu = base_mu * (0.5 + 0.5 * cos(pi * min(progress, 1)))
        progress 从 0 到 1，mu 从 base_mu 平滑衰减到 0
        """
        self.current_round = current_round
        progress = min(current_round / max(total_rounds, 1), 1.0)
        decay_factor = 0.5 + 0.5 * np.cos(np.pi * progress)
        self.mu = self.base_mu * decay_factor

        # 同步更新优化器中的 mu
        for param_group in self.optimizer.param_groups:
            param_group['mu'] = self.mu

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)

                gm = torch.cat([p.data.view(-1) for p in self.global_params], dim=0)
                pm = torch.cat([p.data.view(-1) for p in self.model.parameters()], dim=0)
                loss += 0.5 * self.mu * torch.norm(gm - pm, p=2)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
