import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# 1. 基础设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = '../dataset/crop_weeds' # 指向你整理好的 5 分类文件夹
num_classes = 5
batch_size = 32
epochs = 30

print(f"========== 开始集中式训练 (设备: {device}) ==========")

# 2. 数据增强与加载 (和你联邦学习中的 train_transform 保持一致)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 直接用 ImageFolder 读取整个文件夹 (这就是集中式的体现：不切分客户端)
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# 简单划分 80% 训练集，20% 测试集
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print(f"总数据量: {len(full_dataset)} 张. 训练集: {train_size}, 测试集: {test_size}")

# 3. 初始化模型 (加载 ImageNet 预训练权重并修改分类头)
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
# 替换最后的全连接分类层，从 1000 类改为 5 类
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

# 4. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 标准的单机训练循环
best_acc = 0.0
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels).sum().item()
        
    train_acc = 100. * correct_train / total_train
    
    # 验证环节
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total_test += labels.size(0)
            correct_test += predicted.eq(labels).sum().item()
            
    test_acc = 100. * correct_test / total_test
    
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/train_size:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    
    # 保存最佳模型权重
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'centralized_crop_best.pth')

print(f"========== 训练结束！最高准确率: {best_acc:.2f}% ==========")
print("最优权重已保存为 centralized_crop_best.pth")
