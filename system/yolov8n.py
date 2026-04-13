import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    """YOLOv8基础卷积模块（Conv + BN + SiLU）"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class C2f(nn.Module):
    """YOLOv8 C2f模块（替代C3，更高效）"""
    def __init__(self, in_channels, out_channels, n=1, shortcut=True):
        super().__init__()
        self.cv1 = Conv(in_channels, out_channels, 1, 1)
        self.cv2 = Conv(in_channels, out_channels, 1, 1)
        self.cv3 = Conv(2 * out_channels, out_channels, 1)
        self.m = nn.ModuleList([Conv(out_channels, out_channels, 3) for _ in range(n)])
        self.shortcut = shortcut

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x).chunk(2, 1)[0] if self.shortcut else self.cv2(x)
        
        for m in self.m:
            y2 = m(y2)
            y1 = torch.cat((y1, y2), 1)
        
        return self.cv3(y1)

class SPPF(nn.Module):
    """YOLOv8 SPPF模块（空间金字塔池化）"""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.cv1 = Conv(in_channels, out_channels, 1, 1)
        self.cv2 = Conv(out_channels * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))

class YOLOv8n(nn.Module):
    """手写YOLOv8n核心框架（检测+分类二合一，适配农作物任务）"""
    def __init__(self, num_classes=3):  # 3类农作物
        super().__init__()
        # ========== Backbone ==========
        self.backbone = nn.Sequential(
            Conv(3, 16, 3, 2),  # 0
            Conv(16, 32, 3, 2), # 1
            C2f(32, 32, 1),     # 2
            Conv(32, 64, 3, 2), # 3
            C2f(64, 64, 2),     # 4
            Conv(64, 128, 3, 2),# 5
            C2f(128, 128, 2),   # 6
            Conv(128, 256, 3, 2),#7
            C2f(256, 256, 1),   # 8
            SPPF(256, 256),     # 9
        )

        # ========== Neck ==========
        self.neck = nn.Sequential(
            Conv(256, 128, 1, 1),#10
            nn.Upsample(scale_factor=2, mode='nearest'),#11
            C2f(128+128, 128, 1),#12 (concat 6+11)
            Conv(128, 64, 1, 1), #13
            nn.Upsample(scale_factor=2, mode='nearest'),#14
            C2f(64+64, 64, 1),   #15 (concat 4+14)
            Conv(64, 64, 3, 2),  #16
            C2f(64+64, 128, 1),  #17 (concat 12+16)
            Conv(128, 128, 3, 2),#18
            C2f(128+128, 256, 1),#19 (concat 8+18)
        )

        # ========== Head（检测+分类） ==========
        # 检测头（简化版，输出xyxy+conf+cls）
        self.detect_head = nn.Conv2d(256, 4 + 1 + num_classes, 1)
        # 分类头（适配农作物分类）
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Backbone特征提取
        x = self.backbone(x)
        
        # Neck特征融合（手动concat，简化版）
        x10 = self.neck[0](x)  # Conv 256→128
        x11 = self.neck[1](x10)# Upsample
        x12 = self.neck[2](torch.cat([x11, self.backbone[6](self.backbone[:7](x))], 1)) # C2f
        
        x13 = self.neck[3](x12)# Conv 128→64
        x14 = self.neck[4](x13)# Upsample
        x15 = self.neck[5](torch.cat([x14, self.backbone[4](self.backbone[:5](x))], 1)) # C2f
        
        x16 = self.neck[6](x15)# Conv 64→64, stride=2
        x17 = self.neck[7](torch.cat([x16, x12], 1)) # C2f
        
        x18 = self.neck[8](x17)# Conv 128→128, stride=2
        x19 = self.neck[9](torch.cat([x18, x], 1)) # C2f

        # 检测输出
        detect_out = self.detect_head(x19)
        # 分类输出
        cls_out = self.cls_head(x19)

        # 返回检测+分类结果（适配联邦学习训练）
        return detect_out, cls_out

# 测试代码（验证网络结构）
if __name__ == "__main__":
    model = YOLOv8n(num_classes=3)
    dummy_input = torch.randn(1, 3, 640, 640)  # YOLOv8默认输入640x640
    detect_out, cls_out = model(dummy_input)
    print(f"检测输出维度: {detect_out.shape}")  # [1, 8, 80, 80] (4+1+3=8)
    print(f"分类输出维度: {cls_out.shape}")    # [1, 3]