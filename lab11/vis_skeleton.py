import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 姿态检测器，降低置信度，提升识别成功率
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    min_pose_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
detector = vision.PoseLandmarker.create_from_options(options)

# 人体33关节标准连线
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),
    (0,9),(9,10),(11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)
]

def draw_skeleton(video_path, save_img_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"警告：无法打开视频 {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            print(f"警告：读取帧失败 {video_path}")
            cap.release()
            return

    # 裁剪画面，切掉左右场外裁判观众
    h_full, w_full = frame.shape[:2]
    frame = frame[:, 200 : w_full-200, :]
    h,w = frame.shape[:2]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = detector.detect(mp_img)

    max_area = 0
    best_landmarks = None
    for landmarks in res.pose_landmarks:
        # 修复语法错误：lm.y for lm.y → lm.y for lm
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        area = (max(xs)-min(xs)) * (max(ys)-min(ys))
        if area > max_area:
            max_area = area
            best_landmarks = landmarks

    if best_landmarks is not None:
        pts = []
        for lm in best_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            pts.append((x,y))
        # 绿色骨骼连线
        for a,b in POSE_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0,255,0), 3)
        # 红色关节点
        for (x,y) in pts:
            cv2.circle(frame, (x,y), 5, (0,0,255), -1)
        print(f"成功识别运动员骨骼：{video_path}")
    else:
        print(f"提示：{video_path} 未检测到场上运动员")

    cv2.imwrite(save_img_path, frame)
    print(f"图片已保存：{save_img_path}")
    cap.release()

if __name__ == "__main__":
    draw_skeleton("archive/forehand_clear/005.mp4", "vis_forehand_clear_fixed.png")
    draw_skeleton("archive/forehand_lift/001.mp4", "vis_forehand_lift.png")
    print("两张有效骨骼可视化图片生成完成")