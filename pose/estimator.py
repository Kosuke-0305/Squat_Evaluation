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

    # クロップフィルタのデフォルト範囲（正規化X座標 0.0〜1.0）
    CROP_MIN_DEFAULT = 0.25
    CROP_MAX_DEFAULT = 0.75

    # 足首スムージングのデフォルト値
    ANKLE_JUMP_THRESHOLD_DEFAULT = 0.08  # この値以上の1フレーム変化を異常とみなす（正規化X座標）
    ANKLE_ALPHA_DEFAULT          = 0.2   # EMA の重み（小さいほど変化が緩やか）

    def __init__(self, crop_filter: bool = False,
                 crop_min: float = None,
                 crop_max: float = None,
                 smooth_ankle: bool = False,
                 ankle_jump: float = None,
                 ankle_alpha: float = None):
        """MediaPipe Pose を初期化する。

        Args:
            crop_filter:  True にするとフレームを横方向にクロップしてから
                          MediaPipe に渡す。補助員が端に位置する場合の
                          誤検出を抑制できる。
            crop_min:     クロップの左端（正規化X座標 0.0〜1.0）。
                          省略時は CROP_MIN_DEFAULT（0.25）を使用。
            crop_max:     クロップの右端（正規化X座標 0.0〜1.0）。
                          省略時は CROP_MAX_DEFAULT（0.75）を使用。
            smooth_ankle: True にすると足首X座標に急峻変化抑制 + EMA スムージングを適用する。
                          補助員との揺れ挙動が多い場合に有効にする。
            ankle_jump:   1フレームあたりの足首X座標変化がこの値以上なら異常とみなし
                          前フレームの値で置き換える（正規化座標 0.0〜1.0）。
                          省略時は ANKLE_JUMP_THRESHOLD_DEFAULT（0.08）を使用。
            ankle_alpha:  EMA の重み係数（0.0〜1.0）。大きいほど現在値の影響が強くなる。
                          省略時は ANKLE_ALPHA_DEFAULT（0.2）を使用。
        """
        self._crop_filter = crop_filter
        self._crop_min = crop_min if crop_min is not None else self.CROP_MIN_DEFAULT
        self._crop_max = crop_max if crop_max is not None else self.CROP_MAX_DEFAULT
        self._smooth_ankle     = smooth_ankle
        self._ankle_jump       = ankle_jump  if ankle_jump  is not None else self.ANKLE_JUMP_THRESHOLD_DEFAULT
        self._ankle_alpha      = ankle_alpha if ankle_alpha is not None else self.ANKLE_ALPHA_DEFAULT
        self._prev_left_ankle_x:  float | None = None
        self._prev_right_ankle_x: float | None = None
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=self.VISIBILITY_THRESHOLD,
            min_tracking_confidence=self.VISIBILITY_THRESHOLD,
        )
        self._L = self._mp_pose.PoseLandmark

    def _apply_crop(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        """フレームを横方向にクロップする。

        Args:
            frame: BGR 形式の元フレーム

        Returns:
            (クロップ済みフレーム, クロップ左端の正規化X座標, クロップ右端の正規化X座標)
            crop_filter=False のときは元フレームをそのまま返し、範囲は (0.0, 1.0)。
        """
        if not self._crop_filter:
            return frame, 0.0, 1.0
        h, w = frame.shape[:2]
        x1 = int(w * self._crop_min)
        x2 = int(w * self._crop_max)
        return frame[:, x1:x2], self._crop_min, self._crop_max

    @staticmethod
    def _restore_x(norm_x: float, crop_min: float, crop_max: float) -> float:
        """クロップ後の正規化X座標を元フレームの正規化X座標に逆変換する。

        クロップ後のX座標は 0.0〜1.0 でクロップ範囲に対する相対値になっているため、
        元フレームの座標系に戻す必要がある。

        Args:
            norm_x:   クロップ後の正規化X座標（0.0〜1.0）
            crop_min: クロップ左端の元フレーム上の正規化X座標
            crop_max: クロップ右端の元フレーム上の正規化X座標

        Returns:
            元フレームの正規化X座標
        """
        return crop_min + norm_x * (crop_max - crop_min)

    def _smooth_ankle_x(self, current_x: float, prev_x: float | None) -> float:
        """足首X座標に急峻変化抑制 + EMA スムージングを適用する。

        1. 前フレームの値がない（初期フレーム）場合は current_x をそのまま返す。
        2. 前フレームとの差が ankle_jump 以上なら補助員への切り替わりとみなし、
           current_x を採用せず prev_x を返す（急峻な変化を無効化）。
        3. 正常範囲内なら EMA でスムージングした値を返す。

        Args:
            current_x: 現フレームの足首X座標（正規化）
            prev_x:    前フレームのスムージング済み足首X座標（None = 初期状態）

        Returns:
            スムージング後の足首X座標
        """
        if prev_x is None:
            return current_x
        if abs(current_x - prev_x) >= self._ankle_jump:
            return prev_x
        return self._ankle_alpha * current_x + (1 - self._ankle_alpha) * prev_x

    def reset_ankle_smoothing(self) -> None:
        """足首スムージングの状態をリセットする。

        レップが変わるタイミング（新しい試技の開始時）に呼ぶこと。
        リセットしないと前の試技の足首位置が次の試技の初期値として引き継がれる。
        """
        self._prev_left_ankle_x  = None
        self._prev_right_ankle_x = None

    def _process_side(self, lm, hip_lm, knee_lm, ankle_lm, shoulder_lm,
                      prev_ankle_x: float | None = None) -> dict | None:
        """片足の角度を計算する。HIP/KNEE/ANKLE/SHOULDER の visibility 不足なら None を返す。

        Args:
            lm:           ランドマークリスト
            hip_lm:       腰のランドマーク定数
            knee_lm:      膝のランドマーク定数
            ankle_lm:     足首のランドマーク定数
            shoulder_lm:  肩のランドマーク定数
            prev_ankle_x: 前フレームのスムージング済み足首X座標。
                          smooth_ankle=True のとき PoseEstimator から渡される。
        """
        hip = lm[hip_lm]
        knee = lm[knee_lm]
        ankle = lm[ankle_lm]
        shoulder = lm[shoulder_lm]

        visibility = min(hip.visibility, knee.visibility, ankle.visibility, shoulder.visibility)
        if visibility < self.VISIBILITY_THRESHOLD:
            return None

        ankle_x = ankle.x
        if self._smooth_ankle:
            ankle_x = self._smooth_ankle_x(ankle.x, prev_ankle_x)

        knee_angle  = calc_angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))
        hip_angle   = calc_angle((shoulder.x, shoulder.y), (hip.x, hip.y), (knee.x, knee.y))
        ankle_angle = calc_angle((knee.x, knee.y), (ankle.x, ankle.y), (ankle.x, ankle.y + 0.1))

        return {
            "knee_angle":  knee_angle,
            "hip_angle":   hip_angle,
            "ankle_angle": ankle_angle,
            "knee_x":      knee.x,
            "ankle_x":     ankle_x,
            "knee_y":      knee.y,
            "hip_y":       hip.y,
            "shoulder_x":  shoulder.x,
            "shoulder_y":  shoulder.y,
            "hip_x":       hip.x,
            "visibility":  float(visibility),
        }

    def process(self, frame: np.ndarray) -> dict | None:
        """フレームから両足の骨格推定を行い角度情報を返す。

        クロップフィルタが有効な場合、フレームを横方向に切り取ってから MediaPipe に渡し、
        検出後にランドマークのX座標を元フレームの座標系に逆変換する。
        片足のみ検出できた場合も dict を返す（未検出側は None）。
        両足とも visibility 不足のとき None を返す。

        Args:
            frame: BGR 形式の画像フレーム

        Returns:
            {"left": dict|None, "right": dict|None, "landmarks": pose_landmarks} または None
        """
        cropped, crop_min, crop_max = self._apply_crop(frame)
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        if not results.pose_landmarks:
            return None

        if self._crop_filter:
            for lm in results.pose_landmarks.landmark:
                lm.x = self._restore_x(lm.x, crop_min, crop_max)

        lm = results.pose_landmarks.landmark
        L  = self._L

        left = self._process_side(
            lm, L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE, L.LEFT_SHOULDER,
            prev_ankle_x=self._prev_left_ankle_x,
        )
        right = self._process_side(
            lm, L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE, L.RIGHT_SHOULDER,
            prev_ankle_x=self._prev_right_ankle_x,
        )

        if self._smooth_ankle:
            if left  is not None:
                self._prev_left_ankle_x  = left["ankle_x"]
            if right is not None:
                self._prev_right_ankle_x = right["ankle_x"]

        if left is None and right is None:
            return None

        return {
            "left":      left,
            "right":     right,
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
