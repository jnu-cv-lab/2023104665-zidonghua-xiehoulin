import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 超参（路径适配lab11子文件夹）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 132
TARGET_FRAMES = 30
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FF = 256
NUM_CLASS = 6
DROPOUT = 0.1
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
SAVE_CKPT = "../ckpt/badminton_transformer.pth"
DATA_CACHE = "../data_cache"
os.makedirs("../ckpt", exist_ok=True)

class SkeletonDataset(Dataset):
    def __init__(self, x_npy, y_npy):
        self.x = np.load(x_npy)
        self.y = np.load(y_npy)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        seq = torch.from_numpy(self.x[idx]).float()
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return seq, label

class SkeletonTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(INPUT_DIM, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, TARGET_FRAMES, D_MODEL))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FF,
            dropout=DROPOUT,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.drop = nn.Dropout(DROPOUT)
        self.cls_head = nn.Linear(D_MODEL, NUM_CLASS)

    def forward(self, x):
        b, t, _ = x.shape
        x = self.embed(x)
        x = x + self.pos_emb[:, :t, :]
        x = self.encoder(x)
        x = torch.mean(x, dim=1)
        x = self.drop(x)
        logits = self.cls_head(x)
        return logits

def train_epoch(model, loader, loss_fn, opt):
    model.train()
    total_loss = 0
    all_pred = []
    all_true = []
    for seq, lab in loader:
        seq, lab = seq.to(DEVICE), lab.to(DEVICE)
        opt.zero_grad()
        logits = model(seq)
        loss = loss_fn(logits, lab)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        all_pred.extend(pred)
        all_true.extend(lab.cpu().numpy())
    acc = accuracy_score(all_true, all_pred)
    avg_loss = total_loss / len(loader)
    return avg_loss, acc

def val_epoch(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    all_pred = []
    all_true = []
    with torch.no_grad():
        for seq, lab in loader:
            seq, lab = seq.to(DEVICE), lab.to(DEVICE)
            logits = model(seq)
            loss = loss_fn(logits, lab)
            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            all_pred.extend(pred)
            all_true.extend(lab.cpu().numpy())
    acc = accuracy_score(all_true, all_pred)
    avg_loss = total_loss / len(loader)
    return avg_loss, acc, all_true, all_pred

if __name__ == "__main__":
    train_ds = SkeletonDataset(
        x_npy=os.path.join(DATA_CACHE, "X_train.npy"),
        y_npy=os.path.join(DATA_CACHE, "y_train.npy")
    )
    test_ds = SkeletonDataset(
        x_npy=os.path.join(DATA_CACHE, "X_test.npy"),
        y_npy=os.path.join(DATA_CACHE, "y_test.npy")
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"训练集：{len(train_ds)} 样本 | 测试集：{len(test_ds)} 样本")
    model = SkeletonTransformer().to(DEVICE)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []
    for ep in range(EPOCHS):
        tr_loss, tr_acc = train_epoch(model, train_loader, loss_func, optimizer)
        va_loss, va_acc, y_true, y_pred = val_epoch(model, test_loader, loss_func)
        train_loss_list.append(tr_loss)
        train_acc_list.append(tr_acc)
        val_loss_list.append(va_loss)
        val_acc_list.append(va_acc)
        print(f"Epoch [{ep+1:2d}/{EPOCHS}] | Train Loss:{tr_loss:.4f} Acc:{tr_acc:.4f} | Val Loss:{va_loss:.4f} Acc:{va_acc:.4f}")
    torch.save(model.state_dict(), SAVE_CKPT)
    print(f"模型权重已保存至 {SAVE_CKPT}")
    with open(os.path.join(DATA_CACHE, "label_map.json"), "r") as f:
        label_map = json.load(f)
    cls_names = list(label_map.values())
    print("\n===== 混淆矩阵 =====")
    print(confusion_matrix(y_true, y_pred))
    print("\n===== 分类报告 =====")
    print(classification_report(y_true, y_pred, target_names=cls_names))
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.subplot(1,2,2)
    plt.plot(train_acc_list, label="Train Acc")
    plt.plot(val_acc_list, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.tight_layout()
    plt.savefig("../train_curve.png", dpi=200)
    print("训练曲线图已保存到根目录 train_curve.png")