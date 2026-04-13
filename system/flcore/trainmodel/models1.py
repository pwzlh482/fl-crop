import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

def get_model(model_name, num_classes=3):
    """Get model by name"""
    if model_name == "ResNet18":
        return ResNet18(num_classes=num_classes)
    elif model_name == "YOLOv8n":
        return YOLOv8n(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}, only ResNet18/YOLOv8n are available")

class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head
        
    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out


# ResNet18 Model
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet18, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    def get_loss(self, x, y):
        """Calculate classification loss"""
        pred = self.forward(x)
        return F.cross_entropy(pred, y), pred

# YOLOv8n Model
class YOLOv8n(nn.Module):
    def __init__(self, num_classes=3):
        super(YOLOv8n, self).__init__()
        self.model = YOLO("yolov8n.pt")
        self.model.model.nc = num_classes
        self.model.model.names = ["wheat", "corn", "rice"]
        # Freeze backbone layers for lightweight training
        for i, param in enumerate(self.model.model.parameters()):
            if i < 10:
                param.requires_grad = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    def forward(self, x):
        """Forward propagation"""
        x = x.to(self.device)
        results = self.model(x, verbose=False, device=self.device)
        return results[0].boxes.data
    def get_loss(self, x, annots):
        """Calculate detection loss"""
        x = x.to(self.device)
        annots = [a for a in annots]
        results = self.model.train_step(imgs=x, batch=annots, verbose=False, device=self.device)
        return results["loss"], results["pred"]