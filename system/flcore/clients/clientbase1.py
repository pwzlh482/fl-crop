import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay
    
    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
            
        # 1. 读取训练数据 + 过滤空值（和test逻辑一致）
        train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        train_data = [item for item in train_data if item is not None]
        
        # 2. 训练数据为空时，生成320×320虚拟数据（和test逻辑一致）
        if len(train_data) == 0:
            import torch
            dummy_img = torch.randn(3, 320, 320).type(torch.float32)
            dummy_label = torch.tensor(0).type(torch.int64)
            train_data = [(dummy_img, dummy_label)]
            batch_size = 1  # 强制批量为1
            print(f"Warning: Client {self.id} train data empty, use 1 dummy sample (320x320)")
        
        # 3. 避免样本被丢弃（和test逻辑一致）
        return DataLoader(train_data, batch_size=batch_size, drop_last=False, shuffle=True)
    
    def load_train_data1(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
            
        train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        
        # 修复1：变量名错误（test_data → train_data）+ 过滤空数据
        train_data = [item for item in train_data if item is not None]
        # 修复2：训练数据为空时的兜底逻辑
        if len(train_data) == 0:
            import torch
            # 关键：改为320×320匹配你的数据集
            dummy_img = torch.randn(3, 320, 320).type(torch.float32)  
            dummy_label = torch.tensor(0).type(torch.int64)            
            train_data = [(dummy_img, dummy_label)]
            batch_size = 1  
            print(f"Warning: Client {self.id} train data is empty, use dummy sample (320x320)")
        
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)
           
    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        
        # 1. 读取数据并过滤空值
        test_data = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
        # 强制过滤空数据项
        test_data = [item for item in test_data if item is not None]
        
        # 2. 若数据为空，强制设batch_size=1并生成1条虚拟数据
        if len(test_data) == 0:
            import torch
            # 修复3：虚拟数据尺寸改为320×320
            dummy_img = torch.randn(3, 320, 320).type(torch.float32)  # 匹配你的数据集
            dummy_label = torch.tensor(0).type(torch.int64)            # 单标签
            test_data = [(dummy_img, dummy_label)]
            batch_size = 1  # 强制批量为1
            print(f"Client {self.id} test data empty, use 1 dummy sample (320x320)")
        
        # 3. 初始化DataLoader（强制shuffle=False避免空数据报错）
        return DataLoader(test_data, batch_size=batch_size, drop_last=False, shuffle=False)


    def load_test_data1(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

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
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))