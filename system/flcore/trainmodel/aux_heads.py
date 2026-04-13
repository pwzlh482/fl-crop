"""
aux_heads.py - 深度监督辅助分类头
用于 ResNet18 / MobileNetV2 中间层，帮助浅层学到判别性特征

原理：
  在网络中间层（layer1/2/3）各挂一个轻量分类头，
  训练时辅助分类头也计算分类损失，让梯度直达浅层，
  推理/联邦聚合时辅助分类头不参与，零开销。

用法：
  from flcore.trainmodel.aux_heads import AuxHeads
  aux = AuxHeads(model, num_classes)
  aux_loss = aux.compute_aux_loss(features_list, labels, criterion)
"""

import torch
import torch.nn as nn


class AuxHead(nn.Module):
    """单个辅助分类头：GAP + FC，极轻量"""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.gap(x)        # [B, C, 1, 1]
        x = x.view(x.size(0), -1)  # [B, C]
        return self.fc(x)       # [B, num_classes]


class AuxHeads(nn.Module):
    """深度监督辅助分类头集合

    根据 ResNet18 / MobileNetV2 的中间层通道数自动创建辅助分类头。

    Args:
        model: 主干模型（ResNet18 或 MobileNetV2）
        num_classes: 分类类别数
    """

    # ResNet18 各层输出通道数
    RESNET18_CHANNELS = [64, 128, 256, 512]
    # MobileNetV2 各 stage 输出通道数
    MOBILENET_CHANNELS = [16, 24, 32, 96]

    def __init__(self, model, num_classes):
        super().__init__()

        # 检测模型类型
        model_name = self._detect_model(model)

        if model_name == 'resnet18':
            channels = self.RESNET18_CHANNELS
        elif model_name == 'mobilenet':
            channels = self.MOBILENET_CHANNELS
        else:
            # 未知模型，不创建辅助头
            self.heads = nn.ModuleList()
            self.available = False
            return

        # 创建 3 个辅助分类头（对应 layer1/2/3，跳过 layer4 因为就是主分类头）
        self.heads = nn.ModuleList([
            AuxHead(ch, num_classes) for ch in channels[:3]
        ])
        self.available = True

    def _detect_model(self, model):
        """检测模型类型"""
        if hasattr(model, 'layer1') and hasattr(model, 'layer2'):
            return 'resnet18'
        elif hasattr(model, 'features') and hasattr(model, 'classifier'):
            return 'mobilenet'
        return 'unknown'

    def compute_aux_loss(self, features_list, labels, criterion):
        """计算深度监督辅助损失

        Args:
            features_list: 中间层特征列表 [feat1, feat2, feat3]
            labels: 标签
            criterion: 损失函数（含 label_smoothing）

        Returns:
            aux_loss: 辅助损失之和
        """
        if not self.available or len(self.heads) == 0:
            return torch.tensor(0.0, device=labels.device)

        aux_loss = torch.tensor(0.0, device=labels.device)
        for i, (feat, head) in enumerate(zip(features_list, self.heads)):
            if feat is not None:
                aux_out = head(feat)
                aux_loss = aux_loss + criterion(aux_out, labels)

        return aux_loss / len(self.heads)  # 取平均
