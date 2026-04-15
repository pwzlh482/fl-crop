#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main_v2.py - 优化版入口
基于实际服务器 main.py，仅新增，不修改原文件

优化点：
1. 使用 FedProxV2（label_smoothing + momentum + CosineAnnealing + 梯度裁剪 + 动态mu衰减 + Warmup）
2. 支持离线加载预训练 ResNet18 / MobileNetV2 权重
3. 增强数据增强：ColorJitter + RandomErasing
4. 优化默认参数（更适合CIFAR10训练）
5. 新增 warmup_rounds 参数
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import copy
import torch
import argparse
import os
import sys
import time
import warnings
import numpy as np
import torchvision
import torch
import torch.nn as nn
import copy
import logging
import torchvision.models as models

# 原版算法导入（保持兼容）
from flcore.servers.serveravg import FedAvg
from flcore.servers.serverpFedMe import pFedMe
from flcore.servers.serverperavg import PerAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverfomo import FedFomo
from flcore.servers.serveramp import FedAMP
from flcore.servers.servermtl import FedMTL
from flcore.servers.serverlocal import Local
from flcore.servers.serverper import FedPer
from flcore.servers.serverapfl import APFL
from flcore.servers.serverditto import Ditto
from flcore.servers.serverrep import FedRep
from flcore.servers.serverphp import FedPHP
from flcore.servers.serverbn import FedBN
from flcore.servers.serverrod import FedROD
from flcore.servers.serverproto import FedProto
from flcore.servers.serverdyn import FedDyn
from flcore.servers.servermoon import MOON
from flcore.servers.serverbabu import FedBABU
from flcore.servers.serverapple import APPLE
from flcore.servers.servergen import FedGen
from flcore.servers.serverscaffold import SCAFFOLD
from flcore.servers.serverfd import FD
from flcore.servers.serverala import FedALA
from flcore.servers.serverpac import FedPAC
from flcore.servers.serverlg import LG_FedAvg
from flcore.servers.servergc import FedGC
from flcore.servers.serverfml import FML
from flcore.servers.serverkd import FedKD
from flcore.servers.serverpcl import FedPCL
from flcore.servers.servercp import FedCP
from flcore.servers.servergpfl import GPFL
from flcore.servers.serverntd import FedNTD
from flcore.servers.servergh import FedGH
from flcore.servers.serverdbe import FedDBE
from flcore.servers.servercac import FedCAC
from flcore.servers.serverda import PFL_DA
from flcore.servers.serverlc import FedLC
from flcore.servers.serveras import FedAS
from flcore.servers.servercross import FedCross

# V2 优化版 FedProx
from flcore.servers.serverprox_v2 import FedProxV2
from flcore.trainmodel.attention import SEBlock, ECABlock, CBAM, SimAM

from flcore.trainmodel.models import *
from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter


class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def setup_device(args):
    if args.device == "cuda":
        if args.device_id:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
        if not torch.cuda.is_available():
            warnings.warn("CUDA不可用，自动切换到CPU")
            args.device = torch.device("cpu")
        else:
            args.device = torch.device("cuda")
            args.multi_gpu = torch.cuda.device_count() > 1
    else:
        args.device = torch.device("cpu")
        args.multi_gpu = False


def wrap_model_for_parallel(model, args):
    if args.device.type == "cuda":
        model = nn.DataParallel(model)
    return model


def load_pretrained_resnet18(model, pretrained_path, num_classes):
    """离线加载预训练ResNet18权重（不含fc层）- 保留旧接口兼容"""
    return load_pretrained_weights(model, pretrained_path, skip_keys=('fc',))


def load_pretrained_weights(model, pretrained_path, skip_keys=('fc', 'classifier')):
    """离线加载预训练权重（通用版，支持 ResNet18 / MobileNetV2 等）

    自动处理：
    - DataParallel 的 module. 前缀
    - fc / classifier 层类别数不匹配时跳过
    - 形状不匹配的层跳过（如 BN→GN 后参数形状变了）

    Args:
        model: 目标模型
        pretrained_path: 预训练权重文件路径
        skip_keys: 包含这些关键字的层如果形状不匹配则跳过（分类头）

    Returns:
        加载了预训练权重的模型（文件不存在则原样返回）
    """
    if not os.path.exists(pretrained_path):
        print(f"[预训练] 权重文件不存在: {pretrained_path}，使用随机初始化")
        return model

    print(f"[预训练] 正在加载: {pretrained_path}")
    try:
        state_dict = torch.load(pretrained_path, map_location='cpu')

        # 处理 DataParallel 的 module. 前缀
        if any(k.startswith('module.') for k in state_dict.keys()):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.', '')] = v
            state_dict = new_state_dict

        model_dict = model.state_dict()
        loaded_count = 0
        skipped_mismatch = []
        skipped_head = []

        for k, v in state_dict.items():
            if k not in model_dict:
                continue
            if v.shape != model_dict[k].shape:
                if any(s in k for s in skip_keys):
                    skipped_head.append(k)
                else:
                    skipped_mismatch.append(k)
                continue
            model_dict[k] = v
            loaded_count += 1

        model.load_state_dict(model_dict)
        print(f"[预训练] 成功加载 {loaded_count}/{len(model_dict)} 层权重")
        if skipped_head:
            print(f"[预训练] 跳过分类头(类别数不匹配): {skipped_head}")
        if skipped_mismatch:
            print(f"[预训练] 跳过形状不匹配层: {skipped_mismatch}")
    except Exception as e:
        print(f"[预训练] 加载失败: {e}，使用随机初始化")

    return model


def auto_load_pretrained(model, model_name, args, skip_keys=('fc', 'classifier')):
    """自动检测并加载预训练权重

    优先级：
    1. 命令行 -pp 指定的路径
    2. system/ 目录下的默认文件:
       - ResNet18  → resnet18_imagenet.pth
       - MobileNet → mobilenet_v2_imagenet.pth

    文件不存在则静默跳过，不影响原功能

    Returns:
        bool: 是否成功加载了预训练权重
    """
    default_files = {
        'ResNet18': 'resnet18_imagenet.pth',
        'MobileNet': 'mobilenet_v2_imagenet.pth',
        'ResNet34': 'resnet34_imagenet.pth',
    }

    pretrained_path = ""

    if hasattr(args, 'pretrained_path') and args.pretrained_path:
        pretrained_path = args.pretrained_path
    elif model_name in default_files:
        auto_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), default_files[model_name])
        if os.path.exists(auto_path):
            pretrained_path = auto_path
            print(f"[预训练] 自动检测到: {auto_path}")

    if pretrained_path:
        model = load_pretrained_weights(model, pretrained_path, skip_keys=skip_keys)
        return True
    else:
        if model_name in default_files:
            print(f"[预训练] 未找到 {default_files[model_name]}，使用随机初始化")
        else:
            print(f"[预训练] {model_name} 暂无自动预训练支持，使用随机初始化")
        return False


# ════════════════════════════════════════════════════════════════
# SE / 注意力机制注入辅助函数
# ════════════════════════════════════════════════════════════════

def _inject_se_resnet18(model):
    """在 ResNet18 每个 BasicBlock 的 conv2 后注入 SE Block

    BasicBlock forward: x → conv1 → bn1 → relu → conv2 → bn2 → SE → +x → relu
    SE 加在 shortcut 之前，让每个 block 学会强调/抑制通道。

    通过 setattr 将 SE 注册为 BasicBlock 的子模块，
    这样 model.to(device) 时 SE 会自动跟着移动。
    """
    block_idx = 0
    for name, child in model.named_children():
        if name in ('layer1', 'layer2', 'layer3', 'layer4'):
            _inject_se_resnet18(child)
        elif hasattr(child, 'conv2') and hasattr(child, 'bn2'):
            # 这是一个 BasicBlock，注入 SE
            ch = child.conv2.in_channels
            se = SEBlock(ch, reduction=16)

            # ★ 关键：用 setattr 注册为子模块，.to(device) 时会自动移动
            setattr(child, f'se_block', se)

            block_idx += 1

            # 保存原始 forward 引用（备用）
            orig_downsample = child.downsample

            def make_se_forward(block=child, ds=orig_downsample):
                def forward(x):
                    identity = x
                    out = block.conv1(x)
                    out = block.bn1(out)
                    out = block.relu(out)
                    out = block.conv2(out)
                    out = block.bn2(out)
                    out = block.se_block(out)   # ← SE 注入位置
                    if ds is not None:
                        identity = ds(x)
                    out += identity
                    out = block.relu(out)
                    return out
                return forward

            child.forward = make_se_forward()


def _inject_se_mobilenetv2(model):
    """在 MobileNetV2 每个 InvertedResidual 的 depthwise conv 后注入 SE Block

    InvertedResidual forward:
        x → conv[0] (expand) → BN → ReLU → conv[1] (dw) → BN → conv[2] (project) → +x

    SE 插在 conv[1] (depthwise) 之后，在 BN 之后（通道数最大，效果最好）。
    """
    for name, child in model.named_children():
        if name == 'features':
            for layer in child:
                if not hasattr(layer, 'conv') or not hasattr(layer, 'use_res_connect'):
                    continue
                if getattr(layer, '_se_injected', False):
                    continue

                conv = layer.conv  # nn.Sequential: [expand_conv, dw_conv, project_conv]
                conv_list = list(conv.children())
                if len(conv_list) < 2:
                    continue

                # 找 depthwise conv: groups > 1 的 Conv2d
                dw_idx = -1
                for i, m in enumerate(conv_list):
                    if isinstance(m, nn.Conv2d) and m.groups > 1:
                        dw_idx = i
                        break

                if dw_idx < 0:
                    continue

                # depthwise conv 输出通道 = SE 输入通道
                dw_out_ch = conv_list[dw_idx].out_channels
                se = SEBlock(dw_out_ch, reduction=8)

                # 在 dw conv 之后、project conv 之前插入 SE
                new_conv_list = conv_list[:dw_idx + 1]  # up to dw conv (included)
                new_conv_list.append(se)               # SE
                new_conv_list.extend(conv_list[dw_idx + 1:])  # rest

                layer.conv = nn.Sequential(*new_conv_list)
                layer._se_injected = True
        else:
            _inject_se_mobilenetv2(child)


logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)


def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "MLR":
            if "MNIST" in args.dataset:
                args.model = Mclr_Logistic(1*28*28, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(3*32*32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        elif model_str == "CNN":
            if "Cifar10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "Omniglot" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "DNN":
            if "MNIST" in args.dataset:
                args.model = DNN(1*28*28, 100, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = DNN(3*32*32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)

        elif model_str == "ResNet18":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)

            args.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            args.model.maxpool = nn.Identity()

            # ★ 预训练权重在 BN→GN 替换之前加载，卷积权重能被保留
            has_pretrained = auto_load_pretrained(args.model, 'ResNet18', args)

            # 替换 BN 为 GN (32 groups)
            def replace_bn(module):
                for name, child in module.named_children():
                    if isinstance(child, nn.BatchNorm2d):
                        setattr(module, name, nn.GroupNorm(32, child.num_features))
                    else:
                        replace_bn(child)

            replace_bn(args.model)

            # ══ SE 注意力注入（在 BN→GN 替换之后）═════════════════════
            att_type = getattr(args, 'attention', 'none')
            if att_type not in ('none', ''):
                _inject_se_resnet18(args.model)
                print(f">>> [注意力] ResNet18 已注入 SE Block (type={att_type})")
            # ════════════════════════════════════════════════════════

            # 初始化（只对 GN 和 fc 层做初始化，卷积层保留预训练值）
            for m in args.model.modules():
                if isinstance(m, nn.Conv2d):
                    if not has_pretrained:
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.GroupNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

            # CIFAR10 / Crop 数据增强（含 ColorJitter + RandomErasing）
            if "Cifar10" in args.dataset or "crop" in args.dataset.lower():
                from torchvision import transforms
                import utils.data_utils

                train_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
                ])
                test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                ])

                original_read_client_data = utils.data_utils.read_client_data
                def augmented_read_client_data(dataset, client_id, is_train=True, few_shot=None):
                    data = original_read_client_data(dataset, client_id, is_train=is_train, few_shot=few_shot)
                    if hasattr(data, 'dataset'):
                        data.dataset.transform = train_transform if is_train else test_transform
                    else:
                        data.transform = train_transform if is_train else test_transform
                    return data

                utils.data_utils.read_client_data = augmented_read_client_data
                print(">>> [V2增强] 已注入 ColorJitter + RandomErasing 数据增强")

            elif "MNIST" in args.dataset:
                args.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
                args.model.maxpool = nn.Identity()
            args.model = args.model.to(args.device)

        elif model_str == "MobileNet":
            args.model = models.mobilenet_v2(pretrained=False, num_classes=args.num_classes)

            # ★ 预训练权重在 BN→GN 替换之前加载
            has_pretrained = auto_load_pretrained(args.model, 'MobileNet', args,
                                                  skip_keys=('fc', 'classifier'))

            # 替换 BN 为 GN
            def replace_bn(module):
                for name, child in module.named_children():
                    if isinstance(child, nn.BatchNorm2d):
                        num_features = child.num_features
                        num_groups = 8 if num_features % 8 == 0 else 4
                        if num_features < num_groups:
                            num_groups = 1
                        setattr(module, name, nn.GroupNorm(num_groups, num_features))
                    else:
                        replace_bn(child)

            replace_bn(args.model)

            # ══ SE 注意力注入（在 BN→GN 替换之后）═════════════════════
            att_type = getattr(args, 'attention', 'none')
            if att_type not in ('none', ''):
                _inject_se_mobilenetv2(args.model)
                print(f">>> [注意力] MobileNetV2 已注入 SE Block (type={att_type})")
            # ════════════════════════════════════════════════════════

            # 初始化（只对 GN 和 fc 层做初始化，卷积层保留预训练值）
            for m in args.model.modules():
                if isinstance(m, nn.Conv2d):
                    if not has_pretrained:
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.GroupNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

            # 小图数据集适配
            if "Cifar10" in args.dataset or "crop" in args.dataset.lower():
                args.model.features[0][0].stride = (1, 1)
                print(">>> [优化] 已修改第一层 Stride 为 1，防止小图特征丢失")

                from torchvision import transforms
                import utils.data_utils

                img_size = 64
                train_transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
                ])
                test_transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                ])

                original_read_data = utils.data_utils.read_client_data
                def augmented_read_client_data(dataset, client_id, is_train=True, few_shot=None):
                    data = original_read_data(dataset, client_id, is_train=is_train, few_shot=few_shot)
                    data.transform = train_transform if is_train else test_transform
                    return data

                utils.data_utils.read_client_data = augmented_read_client_data
                print(f">>> [V2增强] 已注入数据增强并统一 Resize 至 {img_size}x{img_size}")

            elif "MNIST" in args.dataset:
                args.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
                print(">>> [优化] 已适配 MNIST 单通道，Stride设为1以保持特征图尺寸")

            args.model = args.model.to(args.device)

        elif model_str == "ResNet10":
            args.model = resnet10(num_classes=args.num_classes).to(args.device)

        elif model_str == "ResNet34":
            args.model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "AlexNet":
            args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "GoogleNet":
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False,
                                                      num_classes=args.num_classes).to(args.device)

        elif model_str == "MobileNet1":
            args.model = models.mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "LSTM":
            args.model = LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "BiLSTM":
            args.model = BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim,
                                                   output_size=args.num_classes, num_layers=1,
                                                   embedding_dropout=0, lstm_dropout=0, attention_dropout=0,
                                                   embedding_length=args.feature_dim).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=args.feature_dim, max_len=args.max_len, vocab_size=args.vocab_size,
                                 num_classes=args.num_classes).to(args.device)

        elif model_str == "Transformer":
            args.model = TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2,
                                          num_classes=args.num_classes, max_len=args.max_len).to(args.device)

        elif model_str == "AmazonMLP":
            args.model = AmazonMLP().to(args.device)

        elif model_str == "HARCNN":
            if args.dataset == 'HAR':
                args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                                    pool_kernel_size=(1, 2)).to(args.device)
            elif args.dataset == 'PAMAP2':
                args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                                    pool_kernel_size=(1, 2)).to(args.device)

        else:
            raise NotImplementedError

        args.model = args.model
        print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.module.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)

        elif args.algorithm == "Local":
            server = Local(args, i)

        elif args.algorithm == "FedMTL":
            server = FedMTL(args, i)

        elif args.algorithm == "PerAvg":
            server = PerAvg(args, i)

        elif args.algorithm == "pFedMe":
            server = pFedMe(args, i)

        elif args.algorithm == "FedProx":
            server = FedProx(args, i)

        # ===== V2 优化版 FedProx（含动态mu + 增强数据增强） =====
        elif args.algorithm == "FedProxV2":
            server = FedProxV2(args, i)

        elif args.algorithm == "FedFomo":
            server = FedFomo(args, i)

        elif args.algorithm == "FedAMP":
            server = FedAMP(args, i)

        elif args.algorithm == "APFL":
            server = APFL(args, i)

        elif args.algorithm == "FedPer":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPer(args, i)

        elif args.algorithm == "Ditto":
            server = Ditto(args, i)

        elif args.algorithm == "FedRep":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedRep(args, i)

        elif args.algorithm == "FedPHP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPHP(args, i)

        elif args.algorithm == "FedBN":
            server = FedBN(args, i)

        elif args.algorithm == "FedROD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedROD(args, i)

        elif args.algorithm == "FedProto":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedProto(args, i)

        elif args.algorithm == "FedDyn":
            server = FedDyn(args, i)

        elif args.algorithm == "MOON":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = MOON(args, i)

        elif args.algorithm == "FedBABU":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedBABU(args, i)

        elif args.algorithm == "APPLE":
            server = APPLE(args, i)

        elif args.algorithm == "FedGen":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGen(args, i)

        elif args.algorithm == "SCAFFOLD":
            server = SCAFFOLD(args, i)

        elif args.algorithm == "FD":
            server = FD(args, i)

        elif args.algorithm == "FedALA":
            server = FedALA(args, i)

        elif args.algorithm == "FedPAC":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPAC(args, i)

        elif args.algorithm == "LG-FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = LG_FedAvg(args, i)

        elif args.algorithm == "FedGC":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGC(args, i)

        elif args.algorithm == "FML":
            server = FML(args, i)

        elif args.algorithm == "FedKD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedKD(args, i)

        elif args.algorithm == "FedPCL":
            args.model.fc = nn.Identity()
            server = FedPCL(args, i)

        elif args.algorithm == "FedCP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedCP(args, i)

        elif args.algorithm == "GPFL":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = GPFL(args, i)

        elif args.algorithm == "FedNTD":
            server = FedNTD(args, i)

        elif args.algorithm == "FedGH":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGH(args, i)

        elif args.algorithm == "FedDBE":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedDBE(args, i)

        elif args.algorithm == 'FedCAC':
            server = FedCAC(args, i)

        elif args.algorithm == 'PFL-DA':
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = PFL_DA(args, i)

        elif args.algorithm == 'FedLC':
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedLC(args, i)

        elif args.algorithm == 'FedAS':
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAS(args, i)

        elif args.algorithm == "FedCross":
            server = FedCross(args, i)

        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()


def main():
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="Cifar10")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="MobileNet")
    parser.add_argument('-lbs', "--batch_size", type=int, default=64)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.05,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=True)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.2)
    parser.add_argument('-ldm', "--lr_decay_milestones", type=int, nargs='+', default=[40, 60],
                        help="轮次节点，比如输入 40 60 表示在40、60轮衰减学习率")
    parser.add_argument('-gr', "--global_rounds", type=int, default=70)
    parser.add_argument('-tc', "--top_cnt", type=int, default=10,
                        help="For auto_break")
    parser.add_argument('-ls', "--local_epochs", type=int, default=3,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedProxV2")

    parser.add_argument('--weight-decay', '--wd', dest='weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('-jr', "--join_ratio", type=float, default=1,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=True)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=10)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=80,
                        help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL / FedCAC
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.1)
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0,
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL / FedCross
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=1)
    # MOON / FedCAC / FedLC
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument('-fte', "--fine_tuning_epochs", type=int, default=10)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
    # SCAFFOLD / FedGH
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-graind than its original paper.")
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    # FedDBE
    parser.add_argument('-mo', "--momentum", type=float, default=0.9)
    parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)

    # FedCross
    parser.add_argument('-fsb', "--first_stage_bound", type=int, default=0)
    parser.add_argument('-ca', "--fedcross_alpha", type=float, default=0.99)
    parser.add_argument('-cmss', "--collaberative_model_select_strategy", type=int, default=1)

    # ===== V2 新增参数 =====
    parser.add_argument('-wr', "--warmup_rounds", type=int, default=0,
                        help="Warmup rounds for V2 (0=关闭)")
    parser.add_argument('-pp', "--pretrained_path", type=str, default="",
                        help="离线预训练权重路径")
    parser.add_argument('-mn', "--model_name", type=str, default="",
                        help="模型名称，用于保存文件命名")
    parser.add_argument('-ma', "--mixup_alpha", type=float, default=0,
                        help="Mixup 数据增强 alpha 值，0=关闭，推荐0.2")
    parser.add_argument('-aw', "--aux_weight", type=float, default=0,
                        help="深度监督辅助损失权重，0=关闭，推荐0.3")
    parser.add_argument('-at', "--attention", type=str, default='none',
                        choices=['none', 'se', 'eca', 'cbam', 'simam'],
                        help="注意力机制类型：se/eca/cbam/simam/none，se 推荐")

    args = parser.parse_args()

    if args.device == "cuda" and args.device_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    setup_device(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = torch.device("cpu")
        args.multi_gpu = False

    total_clients = args.num_clients

    if args.device == "cuda" and "," in args.device_id:
        num_gpus = len(args.device_id.split(','))
        available_gpus = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
    else:
        available_gpus = [torch.device(args.device)]

    args.client_device_map = {}
    for client_id in range(total_clients):
        gpu_idx = client_id % len(available_gpus)
        args.client_device_map[client_id] = available_gpus[gpu_idx]

    # 自动设置 model_name
    if not args.model_name:
        args.model_name = args.model

    print("=" * 50)
    for arg in vars(args):
        print(arg, '=', getattr(args, arg))
    print("=" * 50)

    run(args)


if __name__ == "__main__":
    main()
