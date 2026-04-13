import torch
import numpy as np
import time
import copy
import torch.nn as nn
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
from flcore.clients.clientbase import Client


class clientProxV2(Client):
    """
    FedProx客户端优化版（不修改原clientprox.py）
    
    优化内容：
    1. 标签平滑（label_smoothing=0.1）— 防止模型过度自信
    2. 余弦退火学习率 — 替代ExponentialLR，训练更稳定
    3. mu动态衰减 — 前期抑制漂移，后期释放个性化能力
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.mu = args.mu
        self.initial_mu = args.mu  # 保存初始mu，用于动态衰减

        self.global_params = copy.deepcopy(list(self.model.parameters()))

        # 优化1：标签平滑
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.optimizer = PerturbedGradientDescent(
            self.model.parameters(), lr=self.learning_rate, mu=self.mu)

        # 优化2：余弦退火学习率
        self.learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=args.local_epochs * 3,
            eta_min=self.learning_rate * 0.01
        )

        # 优化3：mu动态衰减
        self.current_round = 0
        self.total_rounds = args.global_rounds

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
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step(self.global_params, self.device)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model):
        for new_param, global_param, param in zip(model.parameters(), self.global_params, self.model.parameters()):
            global_param.data = new_param.data.clone()
            param.data = new_param.data.clone()

        # 优化3：mu动态衰减
        self.current_round += 1
        if self.total_rounds > 0:
            decay_ratio = 1.0 - (self.current_round / self.total_rounds)
            self.mu = max(self.initial_mu * decay_ratio, 0.001)
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
