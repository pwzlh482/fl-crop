#!/usr/bin/env python
"""
模型预测脚本 - 加载训练好的模型并在测试集上评估准确率
支持 PyTorch (.pt) 和 Keras (.h5) 模型
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from utils.data_utils import read_client_data

# 添加模型导入路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from flcore.trainmodel.models import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *

def create_model(model_str, dataset, num_classes, device):
    """
    根据模型字符串和数据集创建模型架构（与main.py保持一致）
    """
    model = None
    
    if model_str == "MLR":  # convex
        if "MNIST" in dataset:
            model = Mclr_Logistic(1*28*28, num_classes=num_classes).to(device)
        elif "Cifar10" in dataset:
            model = Mclr_Logistic(3*32*32, num_classes=num_classes).to(device)
        else:
            model = Mclr_Logistic(60, num_classes=num_classes).to(device)

    elif model_str == "CNN":  # non-convex
        if "Cifar10" in dataset:
            model = FedAvgCNN(in_features=3, num_classes=num_classes, dim=1600).to(device)
        elif "Omniglot" in dataset:
            model = FedAvgCNN(in_features=1, num_classes=num_classes, dim=33856).to(device)
        elif "Digit5" in dataset:
            model = Digit5CNN().to(device)
        else:
            model = FedAvgCNN(in_features=3, num_classes=num_classes, dim=10816).to(device)

    elif model_str == "DNN":  # non-convex
        if "MNIST" in dataset:
            model = DNN(1*28*28, 100, num_classes=num_classes).to(device)
        elif "Cifar10" in dataset:
            model = DNN(3*32*32, 100, num_classes=num_classes).to(device)
        else:
            model = DNN(60, 20, num_classes=num_classes).to(device)

    elif model_str == "ResNet18":
        model = models.resnet18(pretrained=False, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        
        # 替换 BN 为 GN
        def replace_bn(module):
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    setattr(module, name, nn.GroupNorm(32, child.num_features))
                else: 
                    replace_bn(child)
        replace_bn(model)
        
        # 初始化
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
        if "MNIST" in dataset:
            model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        model = model.to(device)

    elif model_str == "MobileNet":
        model = models.mobilenet_v2(pretrained=False, num_classes=num_classes)
        
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
        replace_bn(model)
        
        # 初始化
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
        if "Cifar10" in dataset:
            model.features[0][0].stride = (1, 1)
        elif "MNIST" in dataset:
            model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        model = model.to(device)

    elif model_str == "ResNet10":
        model = resnet10(num_classes=num_classes).to(device)
    
    elif model_str == "ResNet34":
        model = models.resnet34(pretrained=False, num_classes=num_classes).to(device)

    elif model_str == "AlexNet":
        model = alexnet(pretrained=False, num_classes=num_classes).to(device)

    elif model_str == "GoogleNet":
        model = models.googlenet(pretrained=False, aux_logits=False, num_classes=num_classes).to(device)

    elif model_str == "MobileNet1":
        model = models.mobilenet_v2(pretrained=False, num_classes=num_classes).to(device)

    else:
        raise NotImplementedError(f"模型 {model_str} 尚未实现")
    
    return model

def load_pytorch_model(model_path, model_str, dataset, num_classes, device):
    """加载PyTorch模型"""
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    
    # 如果checkpoint已经是模型实例
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
        model = model.to(device)
        model.eval()
        return model
    
    # 否则，创建新模型并加载状态字典
    model = create_model(model_str, dataset, num_classes, device)
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 适配DataParallel包装
    if hasattr(model, 'module'):
        model = model.module
    
    # 加载状态字典
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def load_keras_model(model_path):
    """加载Keras模型（.h5格式）"""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        return model
    except ImportError:
        print("错误: 需要TensorFlow来加载.h5模型")
        sys.exit(1)
    except Exception as e:
        print(f"加载Keras模型失败: {e}")
        sys.exit(1)

def evaluate_pytorch_model(model, dataset, device, num_clients=10):
    """评估PyTorch模型在测试集上的准确率"""
    model.eval()
    correct = 0
    total = 0
    
    print(f"开始评估，数据集={dataset}，客户端数量={num_clients}")
    
    # 直接加载npz文件，避免路径问题
    test_dir = f"dataset/{dataset}/test"
    if not os.path.exists(test_dir):
        print(f"错误: 测试目录不存在 {test_dir}")
        return 0, 0, 0
    
    print(f"测试目录: {os.path.abspath(test_dir)}")
    
    # 遍历所有客户端文件
    for client_idx in range(num_clients):
        npz_file = os.path.join(test_dir, f"{client_idx}.npz")
        if not os.path.exists(npz_file):
            print(f"客户端 {client_idx}: 文件不存在 {npz_file}")
            continue
        
        try:
            # 加载npz文件
            loaded = np.load(npz_file, allow_pickle=True)
            data_dict = loaded['data'].tolist()
            x_data = torch.tensor(data_dict['x'], dtype=torch.float32)
            y_data = torch.tensor(data_dict['y'], dtype=torch.int64)
            
            samples = len(x_data)
            print(f"客户端 {client_idx}: 加载 {samples} 个样本")
            
            client_correct = 0
            client_total = 0
            
            # 批量处理，提高效率
            batch_size = 64
            for i in range(0, len(x_data), batch_size):
                x_batch = x_data[i:i+batch_size].to(device)
                y_batch = y_data[i:i+batch_size].to(device)
                
                # 调整维度: [batch, channel, height, width]
                if len(x_batch.shape) == 3:
                    x_batch = x_batch.unsqueeze(1)  # 对于单通道数据
                elif len(x_batch.shape) == 4:
                    # 已经是正确的形状
                    pass
                else:
                    # 转置到标准格式
                    x_batch = x_batch.permute(0, 3, 1, 2)
                
                with torch.no_grad():
                    outputs = model(x_batch)
                    _, predicted = torch.max(outputs.data, 1)
                    batch_total = y_batch.size(0)
                    client_total += batch_total
                    client_correct += (predicted == y_batch).sum().item()
            
            total += client_total
            correct += client_correct
            if client_total > 0:
                client_acc = 100 * client_correct / client_total
                print(f"客户端 {client_idx}: 准确率 {client_acc:.2f}% ({client_correct}/{client_total})")
            
        except Exception as e:
            print(f"客户端 {client_idx}: 加载失败 - {e}")
            continue
    
    if total > 0:
        accuracy = 100 * correct / total
        print(f"整体准确率: {accuracy:.2f}% ({correct}/{total})")
    else:
        accuracy = 0
        print("警告: 没有加载到任何测试数据")
    
    return accuracy, correct, total

def evaluate_keras_model(model, dataset, num_clients=10):
    """评估Keras模型在测试集上的准确率"""
    # 注意：Keras模型需要不同的数据处理方式
    # 这里简化为使用PyTorch数据加载，需要转换
    print("警告: Keras模型评估需要适配数据预处理")
    # 暂时返回0
    return 0, 0, 0

def find_model_file(model_path, project_root):
    """查找模型文件，支持相对路径和文件名自动搜索"""
    # 如果是绝对路径且存在，直接返回
    if os.path.isabs(model_path) and os.path.exists(model_path):
        return model_path
    
    # 尝试在当前目录下查找
    if os.path.exists(model_path):
        return os.path.abspath(model_path)
    
    # 尝试在项目根目录下查找
    full_path = os.path.join(project_root, model_path)
    if os.path.exists(full_path):
        return full_path
    
    # 如果只是文件名，在 system/models/ 目录下递归搜索
    basename = os.path.basename(model_path)
    models_dir = os.path.join(project_root, "system", "models")
    if os.path.exists(models_dir):
        for root, dirs, files in os.walk(models_dir):
            if basename in files:
                return os.path.join(root, basename)
    
    # 未找到文件
    return None

def parse_model_info_from_path(model_path):
    """从文件路径解析数据集、模型和准确率信息"""
    # 从目录名推断数据集
    dataset = None
    model_str = None
    accuracy = None
    
    # 常见数据集
    datasets = ['Cifar10', 'MNIST', 'Omniglot', 'Digit5', 'HAR', 'PAMAP2', 'Shakespeare']
    models = ['ResNet', 'MobileNet', 'DNN', 'CNN', 'MLR', 'AlexNet', 'GoogleNet', 'LSTM', 'BiLSTM']
    
    # 从路径中提取数据集：检查每个目录层级
    path_parts = model_path.replace('\\', '/').split('/')
    for part in path_parts:
        for d in datasets:
            if d in part:
                dataset = d
                break
        if dataset:
            break
    
    # 从文件名解析
    basename = os.path.basename(model_path)
    name_parts = basename.replace('.pt', '').replace('.h5', '').replace('.pth', '').split('_')
    
    # 从文件名中查找数据集（如果路径中没有找到）
    if dataset is None:
        for part in name_parts:
            for d in datasets:
                if d in part:
                    dataset = d
                    break
            if dataset:
                break
    
    # 从文件名中查找模型类型
    for part in name_parts:
        for m in models:
            if m in part:
                model_str = m
                break
        if model_str:
            break
    
    # 特殊处理 MobileNetV2
    if 'MobileNetV2' in basename:
        model_str = 'MobileNet'
    
    # 从文件名中提取准确率
    for part in name_parts:
        if 'acc' in part:
            try:
                accuracy = float(part.replace('acc', ''))
            except:
                pass
    
    return dataset, model_str, accuracy

def main():
    parser = argparse.ArgumentParser(description='加载训练好的模型并评估')
    parser.add_argument('--model-path', '-path', type=str, required=True, 
                       help='模型文件路径 (.pt 或 .h5)，支持相对路径或文件名自动搜索')
    parser.add_argument('--dataset', type=str, default=None,
                       help='数据集名称 (如 Cifar10, MNIST)，如未指定则从文件名或路径推断')
    parser.add_argument('--model-str', type=str, default=None,
                       help='模型类型 (如 MobileNet, ResNet18)，如未指定则从文件名推断')
    parser.add_argument('--num-classes', type=int, default=10,
                       help='类别数量')
    parser.add_argument('--num-clients', type=int, default=20,
                       help='测试客户端数量')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cpu', 'cuda'], help='设备')
    
    args = parser.parse_args()
    
    # 获取项目根目录（假设脚本在 system/ 目录下）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # 上一级目录
    print(f"项目根目录: {project_root}")
    
    # 切换到项目根目录，确保数据加载路径正确
    original_cwd = os.getcwd()
    os.chdir(project_root)
    print(f"工作目录已切换到: {os.getcwd()}")
    
    # 查找模型文件（支持相对路径和文件名自动搜索）
    found_path = find_model_file(args.model_path, project_root)
    if found_path is None:
        print(f"错误: 找不到模型文件 '{args.model_path}'")
        os.chdir(original_cwd)
        sys.exit(1)
    
    args.model_path = found_path
    print(f"模型绝对路径: {args.model_path}")
    
    # 从文件路径推断数据集和模型类型
    inferred_dataset, inferred_model, inferred_acc = parse_model_info_from_path(args.model_path)
    
    if args.dataset is None:
        args.dataset = inferred_dataset
    if args.model_str is None:
        args.model_str = inferred_model
    
    if args.dataset is None:
        print("错误: 无法从文件名或路径推断数据集，请使用 --dataset 指定")
        os.chdir(original_cwd)
        sys.exit(1)
    if args.model_str is None:
        print("错误: 无法从文件名推断模型类型，请使用 --model-str 指定")
        os.chdir(original_cwd)
        sys.exit(1)
    
    print(f"模型文件: {args.model_path}")
    print(f"数据集: {args.dataset}")
    print(f"模型类型: {args.model_str}")
    print(f"类别数: {args.num_classes}")
    print(f"设备: {args.device}")
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"使用设备: {device}")
    
    # 根据文件扩展名选择加载方式
    if args.model_path.endswith('.pt'):
        print("加载PyTorch模型...")
        model = load_pytorch_model(args.model_path, args.model_str, args.dataset, args.num_classes, device)
        print("评估模型...")
        accuracy, correct, total = evaluate_pytorch_model(model, args.dataset, device, args.num_clients)
        print(f"测试结果: {correct}/{total} = {accuracy:.2f}%")
        
    elif args.model_path.endswith('.h5'):
        print("加载Keras模型...")
        model = load_keras_model(args.model_path)
        print("评估Keras模型...")
        accuracy, correct, total = evaluate_keras_model(model, args.dataset, args.num_clients)
        print(f"测试结果: {correct}/{total} = {accuracy:.2f}%")
    else:
        print(f"错误: 不支持的文件格式 {args.model_path}")
        os.chdir(original_cwd)
        sys.exit(1)
    
    # 恢复原始工作目录
    os.chdir(original_cwd)
    print("评估完成!")

if __name__ == '__main__':
    main()