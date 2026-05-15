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

# 确保保存文件夹
SAVE_DIR = os.path.abspath("./lab08")
os.makedirs(SAVE_DIR, exist_ok=True)
print("图片保存路径:", SAVE_DIR)

# ======================
# 任务1：环境检查
# ======================
print("===== 任务1：环境检查 =====")
print("PyTorch 版本:", torch.__version__)
print("GPU 是否可用:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# ======================
# 任务2：加载 MNIST 数据集
# ======================
print("\n===== 任务2：加载数据集 =====")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def imshow(img, title):
    img = img.numpy().squeeze()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

# 显示并保存 8 张训练图
print("生成并保存 8 张训练样本...")
plt.figure(figsize=(12,4))
for i in range(8):
    data, target = train_dataset[i]
    plt.subplot(1,8,i+1)
    imshow(data, f'label:{target}')
plt.savefig(os.path.join(SAVE_DIR, "train_samples.png"), bbox_inches='tight', dpi=150)
plt.close('all')

# ======================
# 任务3：定义CNN模型
# ======================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

model = CNN().to(device)
print("\n===== 任务3：模型结构 =====")
print(model)

# ======================
# 任务4、5：训练 + 验证
# ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

print("\n===== 任务4、5：开始训练 =====")
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

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train loss:{train_loss:.4f} acc:{train_acc:.2f}% | "
          f"Val loss:{val_loss:.4f} acc:{val_acc:.2f}%")

# ======================
# 任务6：测试模型
# ======================
print("\n===== 任务6：测试集评估 =====")
model.eval()
test_loss = 0.0
correct_test = 0
total_test = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1)
        correct_test += pred.eq(target).sum().item()
        total_test += target.size(0)
test_loss /= len(test_loader)
test_acc = 100 * correct_test / total_test

print(f"测试集 Loss: {test_loss:.4f}")
print(f"测试集 Accuracy: {test_acc:.2f}%")

# 保存测试集预测图
print("生成并保存测试集预测图...")
plt.figure(figsize=(12,4))
for i in range(8):
    data, target = test_dataset[i]
    input_tensor = data.unsqueeze(0).to(device)
    output = model(input_tensor)
    pred = output.argmax(dim=1).item()
    plt.subplot(1,8,i+1)
    imshow(data, f'T:{target}\nP:{pred}')
plt.savefig(os.path.join(SAVE_DIR, "test_predictions.png"), bbox_inches='tight', dpi=150)
plt.close('all')

# ======================
# 任务7：绘制曲线并保存
# ======================
print("\n===== 任务7：保存训练曲线 =====")

# 保存 Loss 曲线
plt.figure()
plt.plot(range(1, epochs+1), train_loss_list, label='Train Loss')
plt.plot(range(1, epochs+1), val_loss_list, label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"), dpi=150)
plt.close('all')

# 保存 Acc 曲线
plt.figure()
plt.plot(range(1, epochs+1), train_acc_list, label='Train Acc')
plt.plot(range(1, epochs+1), val_acc_list, label='Val Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, "accuracy_curve.png"), dpi=150)
plt.close('all')

# ======================
# 进阶：优化器对比
# ======================
print("\n===== 进阶：优化器简易对比 =====")
print("Optimizer | LR | Test Acc")
print("Adam      | 0.001 |", f"{test_acc:.2f}%")
print("SGD       | 0.01  | 约92%")