import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from flcore.servers.server_crop import ServerCrop
from config import cfg
import torch.nn as nn
from torchvision.models import resnet18
from ultralytics import YOLO

# ===================== 核心修改1：禁用YOLO自动下载 + 固定本地权重路径 =====================
# 强制使用本地权重，禁止ultralytics自动下载
os.environ["YOLO_AUTO_DOWNLOAD"] = "False"
# 指定权重加载目录为当前代码目录（需确保yolov8n.pt已放在此目录）
os.environ["ULTRALYTICS_ASSETS_DIR"] = "./"

# 1. 定义双模型融合类
class DualModel(nn.Module):
    def __init__(self, num_classes=3, device="cuda"):
        super().__init__()
        self.device = device
        # 1. ResNet18 特征提取（固定输出512通道）
        self.resnet = resnet18(weights=None, num_classes=num_classes).to(device)
        self.resnet_features = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3
        ).to(device)

        # ===================== 核心修改2：加载本地YOLO权重，避免下载 =====================
        # 直接指定本地yolov8n.pt路径（需提前上传到当前目录）
        yolo = YOLO("./yolov8n.pt").to(device)  # 绝对路径：/home/user-lbhzj/pw/PFLlib-master/system/yolov8n.pt
        self.yolo_backbone = yolo.model.model[:10].to(device)
        # 冻结YOLO底层
        for param in self.yolo_backbone.parameters():
            param.requires_grad = False

        # 3. 动态计算融合层输入通道（初始化时先占位，forward时再赋值）
        self.fusion = None
        self.cls_head = None
  
    def forward(self, x):
        if x.dim() == 4 and x.shape[1] != 3 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)  # 仅调整维度顺序，尺寸还是320×320
        
        # 2. 原有归一化逻辑（保留）
        x = x / 255.0 if x.max() > 1.0 else x
    
        # 3. 原有特征提取逻辑（全部保留，尺寸还是320×320）
        res_feat = self.resnet_features(x)  # (N, 512, H, W)
        res_channels = res_feat.shape[1]
    
        yolo_feat = self.yolo_backbone(x)
        if yolo_feat.shape[2:] != res_feat.shape[2:]:
            yolo_feat = nn.functional.interpolate(yolo_feat, size=res_feat.shape[2:], mode='bilinear')
        yolo_channels = yolo_feat.shape[1]
    
        if self.fusion is None:
            total_channels = res_channels + yolo_channels
            self.fusion = nn.Sequential(
                nn.Conv2d(total_channels, 256, kernel_size=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            ).to(self.device)
            self.cls_head = nn.Linear(256, 3).to(self.device)
    
        concat_feat = torch.cat([res_feat, yolo_feat], dim=1)
        fusion_feat = self.fusion(concat_feat)
        fusion_feat = fusion_feat.flatten(1)
        cls_out = self.cls_head(fusion_feat)
        return cls_out
    
    # 兼容框架的children()方法
    def children(self):
        return self.resnet.children()
    
    def forward1(self, x):
        # 1. 输入归一化
        x = x / 255.0 if x.max() > 1.0 else x

        # 2. ResNet特征提取（512通道）
        res_feat = self.resnet_features(x)  # (N, 512, H, W)
        res_channels = res_feat.shape[1]

        # 3. YOLO特征提取 + 尺寸对齐
        yolo_feat = self.yolo_backbone(x)
        # 尺寸对齐（H/W和ResNet一致）
        if yolo_feat.shape[2:] != res_feat.shape[2:]:
            yolo_feat = nn.functional.interpolate(yolo_feat, size=res_feat.shape[2:], mode='bilinear')
        yolo_channels = yolo_feat.shape[1]

        # 4. 动态初始化融合层（仅第一次forward时执行）
        if self.fusion is None:
            total_channels = res_channels + yolo_channels
            # 动态创建融合层（输入通道=实际拼接通道数）
            self.fusion = nn.Sequential(
                nn.Conv2d(total_channels, 256, kernel_size=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            ).to(self.device)
            # 动态创建分类头
            self.cls_head = nn.Linear(256, 3).to(self.device)  # 3类：小麦/玉米/水稻

        # 5. 特征拼接 + 融合 + 输出
        concat_feat = torch.cat([res_feat, yolo_feat], dim=1)
        fusion_feat = self.fusion(concat_feat)
        fusion_feat = fusion_feat.flatten(1)
        cls_out = self.cls_head(fusion_feat)
        return cls_out


# 2. 定义模型自动识别函数（根据名称切换单/双模型）
def get_model(model_name, num_classes=3, device="cuda"):
    """
    根据模型名称自动返回单/双模型实例
    model_name: "resnet18" → 单模型；"resnet18+yolov8" → 双模型
    """
    if model_name == "resnet18":
        # 单模型：返回纯ResNet18
        model = resnet18(pretrained=False, num_classes=num_classes).to(device)
    elif model_name == "resnet18+yolov8n":
        # 双模型：返回自定义融合模型
        model = DualModel(num_classes=num_classes, device=device).to(device)  # 补充device参数，避免漏传
    else:
        raise ValueError(f"不支持的模型名称：{model_name}，可选：resnet18 / resnet18+yolov8n")
    model.train()
    return model

class CropDataset(Dataset):
    def __init__(self, data):
        self.imgs = data[0]
        self.cls_labels = data[1]
        self.det_annots = data[2]
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img = self.imgs[idx]
        if img.ndim == 3 and img.shape[2] == 3:  # 原始是HWC格式
            img = np.transpose(img, (2, 0, 1))   # 转为CHW，尺寸还是320×320
        img = torch.FloatTensor(img)
        cls_label = torch.LongTensor([self.cls_labels[idx]])[0]
        det_annot = self.det_annots[idx]
        return img, (cls_label, det_annot)
        
# 数据集类（保留原有定义，避免兼容性问题）
class CropDataset1(Dataset):
    def __init__(self, data):
        self.imgs = data[0]
        self.cls_labels = data[1]
        self.det_annots = data[2]
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img = torch.FloatTensor(self.imgs[idx])
        cls_label = torch.LongTensor([self.cls_labels[idx]])[0]
        det_annot = self.det_annots[idx]
        return img, (cls_label, det_annot)

# 加载数据集
def load_crop_data(client_path):
    train_data = []
    val_data = []
    test_data = []
    for client_id in range(cfg.num_clients):
        # 加载训练集
        train_file = os.path.join(client_path, 'train', f'client_{client_id}.npy')
        train_d = np.load(train_file, allow_pickle=True).item()
        train_data.append(DataLoader(CropDataset((train_d['imgs'], train_d['labels'], train_d['annot'])), batch_size=cfg.batch_size, shuffle=True))
        # 加载验证集
        val_file = os.path.join(client_path, 'val', f'client_{client_id}.npy')
        val_d = np.load(val_file, allow_pickle=True).item()
        val_data.append(DataLoader(CropDataset((val_d['imgs'], val_d['labels'], val_d['annot'])), batch_size=cfg.batch_size, shuffle=False))
        # 加载测试集
        test_file = os.path.join(client_path, 'test', f'client_{client_id}.npy')
        test_d = np.load(test_file, allow_pickle=True).item()
        test_data.append(DataLoader(CropDataset((test_d['imgs'], test_d['labels'], test_d['annot'])), batch_size=cfg.batch_size, shuffle=False))
    # 全局测试集
    global_test = []
    for d in test_data:
        global_test += [x for x in d]
    return train_data, val_data, global_test

# 主函数
def main():
    # 打印配置信息
    print("="*50)
    print("Federated Learning System Configuration")
    print(f"Device: {cfg.device}")
    print(f"Global Rounds: {cfg.global_rounds} | Local Epochs: {cfg.local_epochs} | LR: {cfg.local_learning_rate}")
    print(f"Clients: {cfg.num_clients} | Batch Size: {cfg.batch_size}")
    print(f"Dataset Path: {cfg.data_path} | Save Dir: {cfg.save_dir}")
    print("="*50)
    
    # 创建保存目录
    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)
    
    # 加载数据集
    train_data, val_data, test_data = load_crop_data(cfg.data_path)
    
    # 构造参数实例
    class Args:
        def __init__(self):
            self.dataset = cfg.data_name
            self.num_classes = 3
            self.join_ratio = 1.0
            self.time_threthold = 0
            self.top_cnt = 10
            self.random_join_ratio = True
            self.few_shot = False
            self.time_select = False
            self.auto_break = False
            self.goal = "accuracy"
            self.data = cfg.data_name
            self.model = cfg.model_name
            self.algo = "ServerCrop"
            self.global_rounds = cfg.global_rounds
            self.local_epochs = cfg.local_epochs
            self.local_learning_rate = cfg.local_learning_rate
            self.batch_size = cfg.batch_size
            self.num_clients = cfg.num_clients
            self.num_new_clients = 0
            self.device_id = cfg.device_id
            self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
            self.eval_gap = cfg.eval_gap
            self.algorithm = "ServerCrop"
            self.save_folder_name = "crop_fl_results"
            self.optimizer = "adam"
            self.loss = "cross_entropy"
            self.seed = 42
            self.log_interval = 10
            self.early_stop = False
            self.patience = 50
            self.client_drop_rate = 0.0
            self.train_slow_rate = 0.0
            self.send_slow_rate = 0.0
            self.dlg_gap = 10
            self.dlg_eval = False
            self.weight_decay = 1e-4
            self.save_dir = cfg.save_dir
            self.batch_num_per_client = 0
            self.num_groups = 1
            self.group_par = 1.0
            self.learning_rate = self.local_learning_rate
            self.num_groups = 1
            self.group_par = 1.0
            self.fine_tuning_epoch_new = 0
            self.visible_gpu = "0"
            self.num_workers = 0
            self.load_model = False
            self.save_model = True

    # ===================== 核心补充：实例化ServerCrop并启动训练 =====================
    # 原有代码缺失启动逻辑，补充后避免运行时报“无实际执行逻辑”
    args = Args()
    # 获取模型实例（根据配置的model_name）
    model = get_model(args.model, args.num_classes, args.device)
    # 实例化联邦学习服务器并启动
    server = ServerCrop(args, model, train_data, val_data, test_data)
    server.train()

# 程序入口
if __name__ == "__main__":
    main()