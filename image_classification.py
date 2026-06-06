import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm


# ----------------- 1. 定义多模态数据集类 (Dataset) -----------------
class WindPowerDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 读取图片名
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx]['image_id'])
        # 必须转换为 RGB，因为保存的透明无边框图可能有 Alpha 通道
        image = Image.open(img_name).convert('RGB')

        # 获取数值标签 (真实功率)
        power = self.data_frame.iloc[idx]['target_power']

        # 【核心操作：满足老师的分类要求】
        # 将连续的功率数值，动态离散化为 3 个类别 (0: 低效, 1: 正常波动, 2: 满发)
        # --- 方案：基于统计分位数的阈值划分 ---
        if power < 0.3:
            label = 0  # 低效期
        elif power < 0.7:
            label = 1  # 正常波动期
        else:
            label = 2  # 满发期

        if self.transform:
            image = self.transform(image)

        return image, label, power, idx


# ----------------- 2. 图像预处理 (ResNet 专属格式) -----------------
# 采用 ImageNet 的标准归一化参数，这是使用预训练模型的标准动作
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ----------------- 3. 构建数据加载器 (DataLoader) -----------------
# 实例化数据集
dataset = WindPowerDataset(csv_file='multimodal_labels.csv',
                           img_dir='dataset_images',
                           transform=data_transforms)

# 划分训练集和测试集 (比如取前 80% 训练，后 20% 测试)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ----------------- 4. 搭建 ResNet-18 分类模型 -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用计算设备: {device}")

# 加载带有 ImageNet 预训练权重的 ResNet-18 (迁移学习，降维打击)
model = models.resnet18(pretrained=True)

# 修改最后一层全连接层 (原来是输出 1000 类，现在我们改为输出 3 类)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

model = model.to(device)

# 定义损失函数 (交叉熵，专门用于分类) 和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------- 5. 训练循环 (Training Loop) -----------------
num_epochs = 5  # 因为用了预训练，5 个 epoch 通常就能看到极好的效果

print("开始训练模型...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 加入 tqdm 进度条，看着比较专业
    for images, labels, _, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播与优化
        loss.backward()
        optimizer.step()

        # 统计准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_acc = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(train_loader):.4f} | Accuracy: {epoch_acc:.2f}%")

# 保存训练好的模型权重
torch.save(model.state_dict(), 'resnet18_wind_classifier.pth')
print("模型训练完成并保存为 resnet18_wind_classifier.pth")