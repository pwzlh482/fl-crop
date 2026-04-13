import torch
import numpy as np
import time
import copy
import torch.nn as nn
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
from flcore.clients.clientbase import Client


class clientProx(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.mu = args.mu
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.global_params = copy.deepcopy(list(self.model.parameters()))

        self.loss = nn.CrossEntropyLoss()
        self.learning_rate_decay = args.learning_rate_decay

        
        self.optimizer = PerturbedGradientDescent(
            self.model.parameters(), lr=self.learning_rate, mu=self.mu
        )
        """
        #原生 SGD
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.learning_rate, 
            momentum=0.9, 
            weight_decay=1e-4
        )
        """
        
        
        """
        self.learning_rate_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self.optimizer, 
            milestones=args.lr_decay_milestones, # 如传入[60, 120] 
            gamma=args.learning_rate_decay_gamma
        )
        """

    def train1(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
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
                #self.optimizer.step(self.global_params, self.device)
                self.optimizer.step()

        # self.model.cpu()
        
        """
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        """
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()
        self.model.train()
    
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
    
        for epoch in range(max_local_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
    
                # ---这里是 FedProx 的核心：手动添加近端项 ---
                if self.mu > 0:
                    proximal_term = 0.0
                    # 遍历当前模型参数和全局模型参数
                    for w, w_t in zip(self.model.parameters(), self.global_params):
                        # 计算 (w - w_t) 的 L2 范数的平方
                        proximal_term += (w - w_t).norm(2)**2
                    
                    # 将惩罚项加到 Loss 上
                    loss += (self.mu / 2) * proximal_term
                # ----------------------------------------------
    
                loss.backward()
                self.optimizer.step(self.global_params, self.device)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model):
        for new_param, global_param, param in zip(model.parameters(), self.global_params, self.model.parameters()):
            global_param.data = new_param.data.clone()
            param.data = new_param.data.clone()

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
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
                loss += 0.5 * self.mu * torch.norm(gm-pm, p=2)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

