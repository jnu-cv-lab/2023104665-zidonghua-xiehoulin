import os
import cv2
import json
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ===================== 超参配置 =====================
TARGET_FRAMES = 30
KEYPOINT_NUM = 33
FEAT_PER_KPT = 4
FRAME_DIM = KEYPOINT_NUM * FEAT_PER_KPT
DATA_ROOT = "archive"
SAVE_DIR = "../data_cache"
TEST_SPLIT = 0.2

# 匹配你本地无数字的文件夹名称
CLASS_MAP = {
    0: "forehand_drive",
    1: "forehand_lift",
    2: "forehand_net_shot",
    3: "forehand_clear",
    4: "backhand_drive",
    5: "backhand_net_shot"
}
os.makedirs(SAVE_DIR, exist_ok=True)

# 新版mediapipe姿态检测器
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)

def uniform_resample(seq, target_len):
    n = len(seq)
    if n == target_len:
        return seq
    indices = np.linspace(0, n-1, target_len, dtype=int)
    return seq[indices]

def normalize_skeleton(frame_kpts):
    kpts = frame_kpts.reshape(KEYPOINT_NUM, FEAT_PER_KPT)
    hip_left = kpts[23, :3]
    hip_right = kpts[24, :3]
    hip_center = (hip_left + hip_right) / 2.0
    kpts[:, :3] -= hip_center
    shoulder_left = kpts[11, :3]
    shoulder_right = kpts[12, :3]
    shoulder_width = np.linalg.norm(shoulder_left - shoulder_right)
    if shoulder_width > 1e-6:
        kpts[:, :3] /= shoulder_width
    return kpts.flatten()

def extract_video_skeleton(vid_path):
    cap = cv2.VideoCapture(vid_path)
    seq = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = detector.detect(mp_image)
        if detection_result.pose_landmarks:
            lm_list = detection_result.pose_landmarks[0]
            arr = []
            for landmark in lm_list:
                arr.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            arr = np.array(arr, dtype=np.float32)
            arr = normalize_skeleton(arr)
        else:
            arr = np.zeros(FRAME_DIM, dtype=np.float32)
        seq.append(arr)
    cap.release()
    seq = np.array(seq, dtype=np.float32)
    seq = uniform_resample(seq, TARGET_FRAMES)
    return seq

if __name__ == "__main__":
    all_data = []
    all_label = []
    for cls_id, folder_name in CLASS_MAP.items():
        cls_folder = os.path.join(DATA_ROOT, folder_name)
        if not os.path.exists(cls_folder):
            print(f"跳过不存在文件夹：{cls_folder}")
            continue
        video_list = [f for f in os.listdir(cls_folder) if f.endswith((".mp4",".avi",".mov",".mkv"))]
        print(f"\n处理类别 {cls_id}-{folder_name}，视频数量：{len(video_list)}")
        for vid in tqdm(video_list):
            full_path = os.path.join(cls_folder, vid)
            sk_seq = extract_video_skeleton(full_path)
            all_data.append(sk_seq)
            all_label.append(cls_id)
    X = np.array(all_data, dtype=np.float32)
    y = np.array(all_label, dtype=np.int32)
    print(f"\n总样本数：{X.shape[0]}，单样本尺寸：{X.shape[1:]}")

    # 空样本拦截
    if len(X) == 0:
        print("【致命错误】未读取到任何视频样本！检查archive内文件夹名称")
        exit()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=42, stratify=y)
    np.save(os.path.join(SAVE_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(SAVE_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(SAVE_DIR, "y_test.npy"), y_test)
    with open(os.path.join(SAVE_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(CLASS_MAP, f, ensure_ascii=False, indent=2)
    print("预处理完成，数据存放在上级目录 data_cache 文件夹")
    print(f"训练集：{X_train.shape[0]} 条 | 测试集：{X_test.shape[0]} 条")