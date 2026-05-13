
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

SAVE_DIR = r"C:\cv-course\lab07"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 任务1：数据准备与样本图 =====================
digits = load_digits()
X = digits.data
y = digits.target
images = digits.images

print("=== 任务1：数据基本信息 ===")
print(f"总样本数：{len(images)}")
print(f"单张图像大小：{images[0].shape}")
print(f"类别标签：{np.unique(y)}")

# 生成样本图片（真正的图片数据）
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"{y[i]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "digits_samples.png"), dpi=150)
plt.close()

# ===================== 任务2：数据划分 =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print("\n=== 任务2：数据划分 ===")
print(f"训练集：{X_train.shape}")
print(f"测试集：{X_test.shape}")

# ===================== 任务3：模型训练 =====================
models = {
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}
y_pred_dict = {}

print("\n=== 任务4：模型准确率 ===")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    y_pred_dict[name] = y_pred
    print(f"{name:18s}：{acc:.4f}")

# ===================== 任务5：结果对比 =====================
print("\n=== 模型准确率表格 ===")
print("| 模型                | 测试准确率 |")
print("|---------------------|------------|")
for name, acc in results.items():
    print(f"| {name:18s} | {acc:.4f}    |")

best = max(results, key=results.get)
worst = min(results, key=results.get)
print(f"\n最高准确率：{best} ({results[best]:.4f})")
print(f"最低准确率：{worst} ({results[worst]:.4f})")

# ===================== 任务6：SVM混淆矩阵与错误样本 =====================
y_pred_svm = y_pred_dict["SVM"]
cm = confusion_matrix(y_test, y_pred_svm)

# 生成混淆矩阵图片
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(cm, display_labels=digits.target_names).plot(cmap='Blues', ax=plt.gca())
plt.title("SVM Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "svm_confusion_matrix.png"), dpi=150)
plt.close()

# 生成错误样本图片
error_idx = np.where(y_pred_svm != y_test)[0]
print(f"\n错误样本数：{len(error_idx)}")

plt.figure(figsize=(10, 3))
for i, idx in enumerate(error_idx[:5]):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[idx].reshape(8,8), cmap='gray')
    plt.title(f"True:{y_test[idx]}\nPred:{y_pred_svm[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "error_samples.png"), dpi=150)
plt.close()

# ===================== 生成结果文本 =====================
with open(os.path.join(SAVE_DIR, "实验结果.txt"), "w", encoding="utf-8") as f:
    f.write("===== 计算机视觉作业 实验结果 =====\n")
    f.write(f"姓名：谢厚麟\n")
    f.write(f"学号：2023104665\n\n")
    f.write("各模型准确率：\n")
    for name, acc in results.items():
        f.write(f"{name}: {acc:.4f}\n")
    f.write(f"\n最优模型：{best}\n")
    f.write(f"最差模型：{worst}\n")
print("1. 实验结果.txt")
print("2. digits_samples.png（可打开的样本图）")
print("3. svm_confusion_matrix.png（可打开的混淆矩阵）")
print("4. error_samples.png（可打开的错误样本）")