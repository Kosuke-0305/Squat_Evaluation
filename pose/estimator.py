"""MediaPipe Pose ラッパーと両足関節角度計算モジュール。"""

import numpy as np
import cv2
import mediapipe as mp


def calc_angle(a: tuple, b: tuple, c: tuple) -> float:
    """b を頂点として、ベクトル b→a と b→c のなす角度を度数法で返す。

    Args:
        a: 始点の (x, y) 座標
        b: 頂点の (x, y) 座標
        c: 終点の (x, y) 座標

    Returns:
        角度（度数法、0〜180°）
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


class PoseEstimator:
    """MediaPipe Pose を使って左右両足の骨格推定と関節角度計算を行うクラス。"""

    VISIBILITY_THRESHOLD = 0.5

    def __init__(self):
        """MediaPipe Pose を初期化する。"""
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=self.VISIBILITY_THRESHOLD,
            min_tracking_confidence=self.VISIBILITY_THRESHOLD,
        )
        self._L = self._mp_pose.PoseLandmark

    def _process_side(self, lm, hip_lm, knee_lm, ankle_lm, shoulder_lm) -> dict | None:
        """片足の角度を計算する。HIP/KNEE/ANKLE/SHOULDER の visibility 不足なら None を返す。"""
        hip = lm[hip_lm]
        knee = lm[knee_lm]
        ankle = lm[ankle_lm]
        shoulder = lm[shoulder_lm]

        visibility = min(hip.visibility, knee.visibility, ankle.visibility, shoulder.visibility)
        if visibility < self.VISIBILITY_THRESHOLD:
            return None

        knee_angle = calc_angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))
        hip_angle = calc_angle((shoulder.x, shoulder.y), (hip.x, hip.y), (knee.x, knee.y))
        ankle_angle = calc_angle((knee.x, knee.y), (ankle.x, ankle.y), (ankle.x, ankle.y + 0.1))

        return {
            "knee_angle":  knee_angle,
            "hip_angle":   hip_angle,
            "ankle_angle": ankle_angle,
            "knee_x":      knee.x,
            "ankle_x":     ankle.x,
            "knee_y":      knee.y,
            "hip_y":       hip.y,
            "shoulder_x":  shoulder.x,
            "shoulder_y":  shoulder.y,
            "hip_x":       hip.x,
            "visibility":  float(visibility),
        }

    def process(self, frame: np.ndarray) -> dict | None:
        """フレームから両足の骨格推定を行い角度情報を返す。

        片足のみ検出できた場合も dict を返す（未検出側は None）。
        両足とも visibility 不足のとき None を返す。

        Args:
            frame: BGR 形式の画像フレーム

        Returns:
            {"left": dict|None, "right": dict|None, "landmarks": pose_landmarks} または None
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        if not results.pose_landmarks:
            return None

        lm = results.pose_landmarks.landmark
        L = self._L

        left = self._process_side(lm, L.LEFT_HIP,  L.LEFT_KNEE,  L.LEFT_ANKLE,  L.LEFT_SHOULDER)
        right = self._process_side(lm, L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE, L.RIGHT_SHOULDER)

        if left is None and right is None:
            return None

        return {
            "left": left,
            "right": right,
            "landmarks": results.pose_landmarks,
        }

    def process_video(self, path: str) -> list:
        """動画ファイルの全フレームを処理し、結果リストを返す。"""
        cap = cv2.VideoCapture(path)
        frame_results = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_results.append(self.process(frame))
        cap.release()
        return frame_results

    def __del__(self):
        self._pose.close()
