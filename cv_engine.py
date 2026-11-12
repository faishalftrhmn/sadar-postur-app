import os
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from ultralytics import YOLO


class SadarPosturHybridEngine:

  def __init__(
      self,
      model_path="models/best.pt",
      landmarker_path="models/pose_landmarker_lite.task",
  ):
    if not os.path.exists(model_path):
      raise FileNotFoundError(f"Model YOLO tidak ditemukan di: {model_path}")

    self.model = YOLO(model_path)
    self.class_map = {
        0: {"name": "Good Posture", "color": (22, 163, 74)},  # Hijau (RGB)
        1: {"name": "Bad Posture", "color": (225, 29, 72)},  # Merah (RGB)
    }

    # Muat MediaPipe PoseLandmarker untuk garis visualisasi & sudut biomekanik
    self.landmarker = None
    try:
      if not os.path.exists(landmarker_path):
        os.makedirs(os.path.dirname(landmarker_path), exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        urllib.request.urlretrieve(url, landmarker_path)

      base_options = python.BaseOptions(model_asset_path=landmarker_path)
      options = vision.PoseLandmarkerOptions(
          base_options=base_options,
          min_pose_detection_confidence=0.25,
          min_tracking_confidence=0.25,
          output_segmentation_masks=False,
      )
      self.landmarker = vision.PoseLandmarker.create_from_options(options)
    except Exception as e:
      print(f"[PERINGATAN] MediaPipe PoseLandmarker dilewati: {e}")

  def _calculate_angle(self, p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    radians = np.arctan2(abs(dx), abs(dy) + 1e-6)
    return float(np.degrees(radians))

  def _get_score(self, point):
    return max(
        getattr(point, "presence", 0.0), getattr(point, "visibility", 0.0)
    )

  def _draw_biomechanical_lines(self, img_rgb, lm, is_bad_posture):
    h, w, _ = img_rgb.shape

    vis_left = self._get_score(lm[7]) + self._get_score(lm[11])
    vis_right = self._get_score(lm[8]) + self._get_score(lm[12])

    if vis_left >= vis_right:
      ear, shoulder, hip = lm[7], lm[11], lm[23]
    else:
      ear, shoulder, hip = lm[8], lm[12], lm[24]

    head = ear if self._get_score(ear) > 0.25 else lm[0]

    if self._get_score(shoulder) < 0.25 or self._get_score(head) < 0.25:
      return None

    pt_head = (int(head.x * w), int(head.y * h))
    pt_shoulder = (int(shoulder.x * w), int(shoulder.y * h))

    line_color = (225, 29, 72) if is_bad_posture else (22, 163, 74)
    joint_color = (250, 204, 21)

    # 1. Garis Telinga/Kepala ke Bahu
    cv2.line(img_rgb, pt_head, pt_shoulder, line_color, 4, cv2.LINE_AA)

    # Garis bantu vertikal tegak lurus dari bahu ke atas (referensi 0°)
    cv2.line(
        img_rgb,
        pt_shoulder,
        (pt_shoulder[0], max(0, pt_shoulder[1] - 90)),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    neck_angle = self._calculate_angle(
        [head.x, head.y], [shoulder.x, shoulder.y]
    )
    torso_angle = neck_angle

    # 2. Garis Bahu ke Pinggul/Punggung Bawah
    if self._get_score(hip) > 0.25:
      pt_hip = (int(hip.x * w), int(hip.y * h))
      cv2.line(img_rgb, pt_shoulder, pt_hip, line_color, 4, cv2.LINE_AA)
      cv2.circle(img_rgb, pt_hip, 7, joint_color, -1)
      cv2.circle(img_rgb, pt_hip, 9, (15, 23, 42), 2)
      torso_angle = self._calculate_angle(
          [shoulder.x, shoulder.y], [hip.x, hip.y]
      )
    else:
      pt_back_est = (pt_shoulder[0], min(h - 10, pt_shoulder[1] + 130))
      cv2.line(img_rgb, pt_shoulder, pt_back_est, line_color, 3, cv2.LINE_AA)

    # Titik lingkaran sendi pada Telinga & Bahu
    for pt in [pt_head, pt_shoulder]:
      cv2.circle(img_rgb, pt, 7, joint_color, -1)
      cv2.circle(img_rgb, pt, 9, (15, 23, 42), 2)

    return {
        "neck_angle": neck_angle,
        "torso_angle": torso_angle,
        "hip_detected": self._get_score(hip) > 0.25,
    }

  def analyze_frame(self, frame_bgr, conf_threshold=0.35):
    start_time = time.time()

    # 1. Deteksi & Klasifikasi Visual menggunakan YOLO11n
    results = self.model.predict(
        source=frame_bgr, conf=conf_threshold, verbose=False
    )
    annotated_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detections = []

    for r in results:
      for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({
            "class_id": cls_id,
            "confidence": conf * 100,
            "bbox": (x1, y1, x2, y2),
        })

    detections.sort(key=lambda x: x["confidence"], reverse=True)

    # 2. Ekstraksi Titik Sendi & Sudut Biomekanik (MediaPipe)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
    )
    pose_res = (
        self.landmarker.detect(mp_image)
        if self.landmarker is not None
        else None
    )

    biomechanics = None
    if detections:
      utama = detections[0]

      # Evaluasi sudut biomekanik sebelum menggambar kotak
      if pose_res and pose_res.pose_landmarks:
        lm = pose_res.pose_landmarks[0]
        vis_left = self._get_score(lm[7]) + self._get_score(lm[11])
        vis_right = self._get_score(lm[8]) + self._get_score(lm[12])
        ear, shoulder, hip = (
            (lm[7], lm[11], lm[23])
            if vis_left >= vis_right
            else (lm[8], lm[12], lm[24])
        )
        head = ear if self._get_score(ear) > 0.25 else lm[0]

        if self._get_score(shoulder) >= 0.25 and self._get_score(head) >= 0.25:
          neck_ang = self._calculate_angle(
              [head.x, head.y], [shoulder.x, shoulder.y]
          )
          torso_ang = (
              self._calculate_angle([shoulder.x, shoulder.y], [hip.x, hip.y])
              if self._get_score(hip) > 0.25
              else neck_ang
          )

          # Hybrid Decision Fusion: koreksi bias low-light YOLO jika sudut tulang belakang terbukti tegap
          if (
              utama["class_id"] == 1
              and neck_ang < 18.0
              and torso_ang < 15.0
              and utama["confidence"] < 88.0
          ):
            utama["class_id"] = 0
          elif utama["class_id"] == 0 and neck_ang > 24.0:
            utama["class_id"] = 1

      # 3. Gambar Bounding Box & Garis sesuai Keputusan Akhir
      info = self.class_map.get(
          utama["class_id"],
          {"name": f"Class {utama['class_id']}", "color": (59, 130, 246)},
      )
      utama["label"] = info["name"]
      color = info["color"]
      x1, y1, x2, y2 = utama["bbox"]
      label_text = f"{info['name']} {utama['confidence']:.1f}%"

      cv2.rectangle(annotated_rgb, (x1, y1), (x2, y2), color, 2)
      (tw, th), _ = cv2.getTextSize(
          label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
      )
      cv2.rectangle(
          annotated_rgb,
          (x1, max(0, y1 - th - 12)),
          (x1 + tw + 12, y1),
          color,
          -1,
      )
      cv2.putText(
          annotated_rgb,
          label_text,
          (x1 + 6, max(16, y1 - 6)),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.65,
          (255, 255, 255),
          2,
          cv2.LINE_AA,
      )

      if pose_res and pose_res.pose_landmarks:
        biomechanics = self._draw_biomechanical_lines(
            annotated_rgb,
            pose_res.pose_landmarks[0],
            is_bad_posture=(utama["class_id"] == 1),
        )

    inference_ms = (time.time() - start_time) * 1000
    return annotated_rgb, detections, biomechanics, inference_ms