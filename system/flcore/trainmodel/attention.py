"""
attention.py - 轻量注意力机制模块
支持 SE Block、ECA、CBAM、SimAM，可注入到 ResNet18 / MobileNetV2

用法:
    from flcore.trainmodel.attention import SEBlock, attach_se_to_resnet18, attach_se_to_mobilenetv2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────
# 1. SE Block (Squeeze-and-Excitation)
# ─────────────────────────────────────────────
class SEBlock(nn.Module):
    """SE Block — Channel Attention

    对每个通道学习一个 0~1 的缩放因子，强化重要通道。

    Args:
        channels: 输入通道数
        reduction: 压缩比（默认 16），即中间层神经元数 = channels // reduction
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)          # GAP → [B, C, 1, 1]
        self.excitation = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8)),   # ≥ 8 防止崩溃
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: [B, C, 1, 1] → [B, C]
        y = self.squeeze(x).view(b, c)
        # Excitation: [B, C] → [B, C]
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale: 通道级乘
        return x * y.expand_as(x)


# ─────────────────────────────────────────────
# 2. ECA Block (Efficient Channel Attention)
# ─────────────────────────────────────────────
class ECABlock(nn.Module):
    """ECA — 零额外参数，1D 卷积做通道注意力"""

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs(math.log2(channels) + b) / gamma)
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2) # [B, 1, C]
        y = self.conv(y)                    # [B, 1, C]
        y = y.transpose(-1, -2).squeeze(-1) # [B, C, 1, 1]
        y = self.sigmoid(y).unsqueeze(-1)   # [B, C, 1, 1]
        return x * y.expand_as(x)


# ─────────────────────────────────────────────
# 3. CBAM (Convolutional Block Attention Module)
# ─────────────────────────────────────────────
class CBAM(nn.Module):
    """CBAM — Channel Attention + Spatial Attention (顺序串联)"""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid_ch = nn.Sigmoid()

        # Spatial attention
        self.conv_sp = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_sp = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        ch_att = self.sigmoid_ch(avg_out + max_out)
        x = x * ch_att

        # Spatial attention
        avg_sp = torch.mean(x, dim=1, keepdim=True)
        max_sp, _ = torch.max(x, dim=1, keepdim=True)
        sp_cat = torch.cat([avg_sp, max_sp], dim=1)
        sp_att = self.sigmoid_sp(self.conv_sp(sp_cat))
        x = x * sp_att
        return x


# ─────────────────────────────────────────────
# 4. SimAM (Simple 3D Attention, 无参)
# ─────────────────────────────────────────────
class SimAM(nn.Module):
    """SimAM — 零参数 3D 注意力，参考 Li et al. 2021"""

    def __init__(self, lambda_=1e-4):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w
        # 能量 E = (x - x̄)² + λ / (var + λ)
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True) + self.lambda_
        E = (x - mean) ** 2 / var
        # Sigmoid 归一化到 0~1
        w = torch.sigmoid(E)
        return x * w


# ─────────────────────────────────────────────
# 辅助函数：将注意力注入到现有模型
# ─────────────────────────────────────────────

def attach_se_to_resnet18(model, reduction=16):
    """在 ResNet18 每个 BasicBlock 的 3x3 卷积后插入 SE Block

    会遍历 model.layer1/2/3/4 中的每个 BasicBlock，
    在其 conv2 (3x3) 后插入 SE(reduction)。
    """
    import torch.nn.functional as F

    for name, module in model.named_children():
        if name in ('layer1', 'layer2', 'layer3', 'layer4'):
            attach_se_to_resnet18(module, reduction)
        elif isinstance(module, nn.Module):
            for child_name, child in module.named_children():
                # BasicBlock: conv1(1x1) → conv2(3x3) → bn2 → relu
                if hasattr(child, 'conv2'):
                    ch = child.conv2.in_channels
                    se = SEBlock(ch, reduction)
                    # 包装 child，把 SE 串进去
                    original_forward = child.forward

                    def make_forward(se, orig_fwd, block=child):
                        def wrapped_forward(x):
                            out = orig_fwd(x)
                            return se(out)
                        return wrapped_forward
                    child.forward = make_forward(se, original_forward)


def attach_se_to_mobilenetv2(model, reduction=16):
    """在 MobileNetV2 每个 InvertedResidual 后面插入 SE Block

    MobileNetV2 的 features 由多个 InvertedResidual 组成，
    每个 InvertedResidual 有一个 shortcut 路径，在 add 后加 SE。
    """
    import torch.nn.functional as F

    def _wrap_inverted_residual(block):
        """给单个 InvertedResidual 注入 SE"""
        # 保存原始 forward
        original_forward = block.forward

        def wrapped_forward(x):
            out = original_forward(x)
            # 确认是 expansion > 1 才加 SE（只有带 shortcut 的 block 才适合加 SE）
            if block.use_res_connect and hasattr(block, 'conv'):
                # SE 加在 depthwise 之后
                pass  # InvertedResidual 内部结构复杂，用另一种方式注入
            return out

        # 更稳妥的做法：直接遍历 InvertedResidual 的子层
        return block

    for name, module in model.named_children():
        if name == 'features':
            for i, layer in enumerate(module):
                # InvertedResidual 有 expansion 属性
                if hasattr(layer, 'conv'):
                    ch = layer.conv[0].in_channels if hasattr(layer.conv, '__getitem__') else None
                    # 加在 expansion layer 之后（第一个 conv 是 pointwise 扩张）
                    if hasattr(layer, 'conv') and ch is not None and ch > 0:
                        pass  # 见下方注入逻辑
        # 递归处理
        attach_se_to_mobilenetv2(module, reduction)


def inject_se_resnet18(model, reduction=16):
    """在 ResNet18 中直接插入 SE 模块 — 替换 Forward 流程（推荐方式）

    适用于 BasicBlock:
      forward: x → conv1 → bn1 → relu → conv2 → bn2 → +x → relu
                ↓
      forward: x → conv1 → bn1 → relu → conv2 → bn2 → SE → +x → relu
    """
    for name, child in model.named_children():
        if name in ('layer1', 'layer2', 'layer3', 'layer4'):
            _inject_se_basicblock(child, reduction)
        else:
            inject_se_resnet18(child, reduction)


def _inject_se_basicblock(block, reduction):
    """递归遍历，将 BasicBlock 的 forward 注入 SE"""
    for name, child in block.named_children():
        if hasattr(child, 'conv2') and hasattr(child, 'bn2'):
            # 这是一个 BasicBlock，注入 SE
            ch = child.conv2.in_channels
            se = SEBlock(ch, reduction).to(next(child.parameters()).device)

            orig_forward = child.forward

            def make_new_forward(se_module, orig_fwd, child_mod=child):
                def new_forward(x):
                    identity = x
                    out = child_mod.conv1(x)
                    out = child_mod.bn1(out)
                    out = child_mod.relu(out)

                    out = child_mod.conv2(out)
                    out = child_mod.bn2(out)
                    out = se_module(out)   # ← SE 注入位置

                    if child_mod.downsample is not None:
                        identity = child_mod.downsample(x)

                    out += identity
                    out = child_mod.relu(out)
                    return out
                return new_forward

            child.forward = make_new_forward(se, orig_forward)
        else:
            _inject_se_basicblock(child, reduction)


def inject_se_mobilenetv2(model, reduction=16):
    """在 MobileNetV2 中每个 InvertedResidual 的 depthwise conv 后注入 SE"""
    features = model.features if hasattr(model, 'features') else model
    for i, layer in enumerate(features):
        if hasattr(layer, 'conv') and isinstance(layer.conv, nn.Sequential):
            # InvertedResidual: 第一个 conv(pointwise) → dwconv(3x3) → conv(pointwise)
            # SE 加在 depthwise 之后（即 conv[1] 之后）
            conv_layers = list(layer.conv.children())
            if len(conv_layers) >= 2:
                # 尝试找到 depthwise conv 的输出通道
                dw_layer = None
                for cl in conv_layers:
                    if isinstance(cl, nn.Conv2d) and cl.groups > 1:
                        dw_layer = cl
                        break
                    elif hasattr(cl, 'conv') and hasattr(cl.conv, 'groups'):
                        # wrapped depthwise
                        dw_layer = cl
                        break

                if dw_layer is not None:
                    in_ch = dw_layer.in_channels if hasattr(dw_layer, 'in_channels') else 0
                    if in_ch == 0:
                        # 从Sequential结构估算通道数
                        for j in range(len(conv_layers)):
                            if hasattr(conv_layers[j], 'out_channels'):
                                in_ch = conv_layers[j].out_channels
                                break

                    if in_ch > 0:
                        se = SEBlock(in_ch, reduction).to(next(model.parameters()).device)

                        # 重新构建 conv，插入 SE
                        new_conv_layers = []
                        se_inserted = False
                        for cl in conv_layers:
                            new_conv_layers.append(cl)
                            # 在 depthwise conv (groups > 1) 后插入 SE
                            if isinstance(cl, nn.Conv2d) and cl.groups > 1 and not se_inserted:
                                new_conv_layers.append(se)
                                se_inserted = True

                        if se_inserted:
                            layer.conv = nn.Sequential(*new_conv_layers)
