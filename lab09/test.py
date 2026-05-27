import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# -------------------------- 1. 超参数与数据集配置 --------------------------
# 设备自动选择GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
epochs = 8

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -------------------------- 2. 复用上次CNN模型（固定结构） --------------------------
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        # 第一层卷积（用于可视化卷积核/feature map）
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32*7*7)
        x = self.fc1(x)
        out = self.fc2(x)
        return out

# 工具函数：训练+验证一轮
def train_one_epoch(model, optimizer, criterion):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for img, label in train_loader:
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, pred = torch.max(output, 1)
        total += label.size(0)
        correct += (pred == label).sum().item()
    avg_loss = train_loss / len(train_loader)
    avg_acc = correct / total
    return avg_loss, avg_acc

def val_one_epoch(model, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_pred = []
    all_label = []
    with torch.no_grad():
        for img, label in val_loader:
            img, label = img.to(device), label.to(device)
            output = model(img)
            loss = criterion(output, label)
            val_loss += loss.item()
            _, pred = torch.max(output, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()
            all_pred.extend(pred.cpu().numpy())
            all_label.extend(label.cpu().numpy())
    avg_loss = val_loss / len(val_loader)
    avg_acc = correct / total
    return avg_loss, avg_acc, all_pred, all_label

# -------------------------- 任务1：复用模型重新训练 --------------------------
print("===== 任务1：复用CNN模型重新训练 =====")
model = MNIST_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loss_list, val_loss_list = [], []
train_acc_list, val_acc_list = [], []

for epoch in range(epochs):
    t_loss, t_acc = train_one_epoch(model, optimizer, criterion)
    v_loss, v_acc, _, _ = val_one_epoch(model, criterion)
    train_loss_list.append(t_loss)
    val_loss_list.append(v_loss)
    train_acc_list.append(t_acc)
    val_acc_list.append(v_acc)
    print(f"Epoch:{epoch+1} TrainLoss:{t_loss:.4f} TrainAcc:{t_acc:.4f} ValLoss:{v_loss:.4f} ValAcc:{v_acc:.4f}")

# -------------------------- 任务2：SGD / SGD+Momentum / Adam 优化器对比 --------------------------
def run_optimizer_exp(opt_name, lr=0.001):
    model = MNIST_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    if opt_name == "SGD":
        opt = optim.SGD(model.parameters(), lr=lr)
    elif opt_name == "SGD_Momentum":
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == "Adam":
        opt = optim.Adam(model.parameters(), lr=lr)
    
    for e in range(epochs):
        train_one_epoch(model, opt, criterion)
    test_loss, test_acc, _, _ = val_one_epoch(model, criterion)
    return test_acc

print("\n===== 任务2：优化器对比 =====")
opt_list = ["SGD", "SGD_Momentum", "Adam"]
for opt in opt_list:
    acc = run_optimizer_exp(opt)
    print(f"{opt} 测试集准确率: {acc:.4f}")

# -------------------------- 任务3：固定Adam，不同学习率对比 --------------------------
def run_lr_exp(lr):
    model = MNIST_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    t_loss_lst, v_loss_lst = [], []
    t_acc_lst, v_acc_lst = [], []
    for e in range(epochs):
        t_loss, t_acc = train_one_epoch(model, opt, criterion)
        v_loss, v_acc, _, _ = val_one_epoch(model, criterion)
        t_loss_lst.append(t_loss)
        v_loss_lst.append(v_loss)
        t_acc_lst.append(t_acc)
        v_acc_lst.append(v_acc)
    return t_loss_lst, v_loss_lst, t_acc_lst, v_acc_lst

print("\n===== 任务3：学习率对比 0.1/0.01/0.001 =====")
lr_set = [0.1, 0.01, 0.001]
lr_res = {}
for lr in lr_set:
    res = run_lr_exp(lr)
    lr_res[lr] = res
    print(f"学习率{lr}训练完成")

# 绘制学习率loss/acc曲线
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for lr in lr_set:
    plt.plot(lr_res[lr][0], label=f"lr={lr} train_loss")
    plt.plot(lr_res[lr][1], label=f"lr={lr} val_loss")
plt.title("不同学习率 Loss 曲线")
plt.legend()
plt.grid()

plt.subplot(1,2,2)
for lr in lr_set:
    plt.plot(lr_res[lr][2], label=f"lr={lr} train_acc")
    plt.plot(lr_res[lr][3], label=f"lr={lr} val_acc")
plt.title("不同学习率 Accuracy 曲线")
plt.legend()
plt.grid()
plt.show()

# -------------------------- 任务4：第一层卷积核可视化 --------------------------
print("\n===== 任务4：卷积核可视化 =====")
# 取出第一层卷积核权重
conv1_weight = model.conv1.weight.cpu().detach().numpy()
# 显示前8个卷积核
plt.figure(figsize=(10,4))
for i in range(8):
    plt.subplot(2,4,i+1)
    # 单通道卷积核 (3,3)
    plt.imshow(conv1_weight[i,0,:,:], cmap='gray')
    plt.axis('off')
plt.suptitle("CNN第一层卷积核（前8个）")
plt.show()

# -------------------------- 任务5：Feature Map 特征图可视化 --------------------------
print("\n===== 任务5：Feature Map可视化 =====")
# 取一张测试图片
img_sample, label_sample = next(iter(val_loader))
img_one = img_sample[0:1].to(device)

# 钩子函数获取第一层卷积输出特征图
feature_maps = []
def hook(module, input, output):
    feature_maps.append(output.cpu().detach())

handle = model.conv1.register_forward_hook(hook)
_ = model(img_one)
handle.remove()

fm = feature_maps[0].numpy()[0]
# 显示前8张特征图
plt.figure(figsize=(10,4))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(fm[i], cmap='gray')
    plt.axis('off')
plt.suptitle("第一层卷积输出Feature Map（前8张）")
plt.show()

# -------------------------- 任务6：错误分类样本可视化分析 --------------------------
print("\n===== 任务6：错误分类样本 =====")
model.eval()
err_imgs = []
err_true = []
err_pred = []

with torch.no_grad():
    for img, label in val_loader:
        img, label = img.to(device), label.to(device)
        out = model(img)
        _, pred = torch.max(out,1)
        # 找出预测错误的样本
        wrong_idx = (pred != label).nonzero(as_tuple=True)[0]
        for idx in wrong_idx[:8]:  # 取前8张错分图
            err_imgs.append(img[idx].cpu().squeeze().numpy())
            err_true.append(label[idx].item())
            err_pred.append(pred[idx].item())
        if len(err_imgs)>=8:
            break

# 绘制8张错分样本
plt.figure(figsize=(10,4))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(err_imgs[i], cmap='gray')
    plt.title(f"T:{err_true[i]} P:{err_pred[i]}")
    plt.axis('off')
plt.suptitle("错误分类样本 真实标签/预测标签")
plt.show()

# -------------------------- 任务7：混淆矩阵绘制 --------------------------
print("\n===== 任务7：混淆矩阵 =====")
_, _, all_pred, all_label = val_one_epoch(model, criterion)
cm = confusion_matrix(all_label, all_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("预测类别")
plt.ylabel("真实类别")
plt.title("MNIST测试集混淆矩阵")
plt.show()