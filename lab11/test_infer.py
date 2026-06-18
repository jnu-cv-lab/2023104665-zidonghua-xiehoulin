import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader
from preprocess import extract_video_skeleton, CLASS_MAP, TARGET_FRAMES, FRAME_DIM
from train import SkeletonTransformer, SkeletonDataset, DEVICE, BATCH_SIZE

CKPT_PATH = "../ckpt/badminton_transformer.pth"
DATA_CACHE = "../data_cache"

def load_model():
    model = SkeletonTransformer().to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()
    return model

def evaluate_full_testset(model):
    test_ds = SkeletonDataset(
        x_npy=os.path.join(DATA_CACHE, "X_test.npy"),
        y_npy=os.path.join(DATA_CACHE, "y_test.npy")
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    all_pred = []
    all_true = []
    with torch.no_grad():
        for seq, lab in test_loader:
            seq, lab = seq.to(DEVICE), lab.to(DEVICE)
            logits = model(seq)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            all_pred.extend(pred)
            all_true.extend(lab.cpu().numpy())
    acc = accuracy_score(all_true, all_pred)
    cls_names = list(CLASS_MAP.values())
    print("======== 完整测试集评估 ========")
    print(f"测试集准确率: {acc:.4f}")
    print("混淆矩阵：")
    print(confusion_matrix(all_true, all_pred))
    print(classification_report(all_true, all_pred, target_names=cls_names))
    return acc

def single_video_infer(model, video_path):
    sk_seq = extract_video_skeleton(video_path)
    tensor_seq = torch.from_numpy(sk_seq).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor_seq)
        prob = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(prob, dim=1).item()
        conf = prob[0, pred_idx].item()
    pred_class_name = CLASS_MAP[pred_idx]
    print("\n======== 单视频推理结果 ========")
    print(f"预测动作类别: {pred_class_name}")
    print(f"置信度: {conf:.2f}")
    return pred_class_name, conf

if __name__ == "__main__":
    model = load_model()
    evaluate_full_testset(model)
    # 替换成你archive里任意视频路径
    demo_video = "archive/forehand_clear/你的视频.mp4"
    if os.path.exists(demo_video):
        single_video_infer(model, demo_video)
    else:
        print("\n修改demo_video为真实视频路径后可单独推理单条视频")