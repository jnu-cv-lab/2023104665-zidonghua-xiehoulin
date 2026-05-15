import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

# ======================
# 自动保存到 lab08 文件夹
# ======================
SAVE_DIR = os.path.abspath("./lab08")
os.makedirs(SAVE_DIR, exist_ok=True)
print("📂 图片保存路径:", SAVE_DIR)

# ======================
# 任务1：环境检查
# ======================
print("===== 任务1：环境检查 =====")
print("PyTorch 版本:", torch.__version__)
print("GPU 是否可用:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# ======================
# 任务2：加载 CIFAR-10 数据集
# ======================
print("\n===== 任务2：加载 CIFAR-10 数据集 =====")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 类别名称
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 划分训练/验证集
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 画图函数
def imshow(img, title):
    img = img / 2 + 0.5
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')

# 保存 8 张训练图
print("生成 8 张训练样本...")
plt.figure(figsize=(12, 4))
for i in range(8):
    data, target = train_dataset[i]
    plt.subplot(1, 8, i+1)
    imshow(data, classes[target])
plt.savefig(os.path.join(SAVE_DIR, "cifar10_train_samples.png"), bbox_inches='tight', dpi=150)
plt.close('all')

# ======================
# 任务3：CIFAR-10 CNN 模型
# ======================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN().to(device)
print("\n===== 任务3：CIFAR-10 模型结构 =====")
print(model)

# ======================
# 任务4、5：训练 + 验证
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

print("\n===== 开始训练 CIFAR-10 =====")
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    train_loss /= len(train_loader)
    train_acc = 100 * correct / total

    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct_val += pred.eq(target).sum().item()
            total_val += target.size(0)

    val_loss /= len(val_loader)
    val_acc = 100 * correct_val / total_val

    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch+1:2d} | Train {train_acc:5.1f}% | Val {val_acc:5.1f}%")

# ======================
# 任务6：测试集
# ======================
print("\n===== 任务6：测试集评估 =====")
model.eval()
correct_test = 0
total_test = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct_test += pred.eq(target).sum().item()
        total_test += target.size(0)

test_acc = 100 * correct_test / total_test
print(f"✅ CIFAR-10 测试集准确率: {test_acc:.2f}%")

# 保存测试预测图
print("生成测试集预测图...")
plt.figure(figsize=(12, 4))
for i in range(8):
    data, target = test_dataset[i]
    input_tensor = data.unsqueeze(0).to(device)
    pred = model(input_tensor).argmax(dim=1).item()
    plt.subplot(1, 8, i+1)
    imshow(data, f"T:{classes[target]}\nP:{classes[pred]}")
plt.savefig(os.path.join(SAVE_DIR, "cifar10_test_pred.png"), bbox_inches='tight', dpi=150)
plt.close('all')

# ======================
# 任务7：保存曲线
# ======================
print("\n===== 任务7：保存训练曲线 =====")

plt.figure()
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.title('CIFAR-10 Loss Curve')
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, "cifar10_loss.png"), dpi=150)
plt.close('all')

plt.figure()
plt.plot(train_acc_list, label='Train Acc')
plt.plot(val_acc_list, label='Val Acc')
plt.title('CIFAR-10 Accuracy Curve')
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, "cifar10_acc.png"), dpi=150)
plt.close('all')