"""有効試技判定用特徴量と baseline を計算するモジュール。"""


def compute_baseline(frame_result: dict | None) -> dict | None:
    """立位フレームから baseline を計算する。

    Returns:
        {left_knee_angle, right_knee_angle, left_ankle_x, right_ankle_x,
         left_shoulder_y, right_shoulder_y} または None
    """
    if frame_result is None:
        return None
    left  = frame_result.get("left")
    right = frame_result.get("right")
    if left is None and right is None:
        return None

    # 片側欠損時は存在する側で代用（delta が 0 になり無影響）
    l = left  if left  is not None else right
    r = right if right is not None else left

    return {
        "left_knee_angle":  l["knee_angle"],
        "right_knee_angle": r["knee_angle"],
        "left_ankle_x":     l["ankle_x"],
        "right_ankle_x":    r["ankle_x"],
        "left_shoulder_y":  l["shoulder_y"],
        "right_shoulder_y": r["shoulder_y"],
    }


def extract_form_features(frame_result: dict | None, baseline: dict | None) -> dict:
    """フレーム単位の有効試技判定用特徴量（D=11）を計算する。

    Returns:
        11 要素の特徴量 dict。frame_result が None の場合はゼロベクトル。
    """
    _zero = {
        "left_knee_angle":        0.0,
        "right_knee_angle":       0.0,
        "left_hip_angle":         0.0,
        "right_hip_angle":        0.0,
        "left_shoulder_y_delta":  0.0,
        "right_shoulder_y_delta": 0.0,
        "left_ankle_x_delta":     0.0,
        "right_ankle_x_delta":    0.0,
        "lr_knee_diff":           0.0,
        "left_visibility":        0.0,
        "right_visibility":       0.0,
    }
    if frame_result is None:
        return _zero

    left  = frame_result.get("left")
    right = frame_result.get("right")

    left_knee  = left["knee_angle"]   if left  is not None else 0.0
    right_knee = right["knee_angle"]  if right is not None else 0.0
    left_hip   = left["hip_angle"]    if left  is not None else 0.0
    right_hip  = right["hip_angle"]   if right is not None else 0.0
    left_vis   = left["visibility"]   if left  is not None else 0.0
    right_vis  = right["visibility"]  if right is not None else 0.0
    lr_knee_diff = abs(left_knee - right_knee) if left is not None and right is not None else 0.0

    if baseline is None:
        return {
            "left_knee_angle":        left_knee,
            "right_knee_angle":       right_knee,
            "left_hip_angle":         left_hip,
            "right_hip_angle":        right_hip,
            "left_shoulder_y_delta":  0.0,
            "right_shoulder_y_delta": 0.0,
            "left_ankle_x_delta":     0.0,
            "right_ankle_x_delta":    0.0,
            "lr_knee_diff":           lr_knee_diff,
            "left_visibility":        left_vis,
            "right_visibility":       right_vis,
        }

    return {
        "left_knee_angle":        left_knee,
        "right_knee_angle":       right_knee,
        "left_hip_angle":         left_hip,
        "right_hip_angle":        right_hip,
        "left_shoulder_y_delta":  (left["shoulder_y"]  - baseline["left_shoulder_y"])  if left  is not None else 0.0,
        "right_shoulder_y_delta": (right["shoulder_y"] - baseline["right_shoulder_y"]) if right is not None else 0.0,
        "left_ankle_x_delta":     (left["ankle_x"]     - baseline["left_ankle_x"])     if left  is not None else 0.0,
        "right_ankle_x_delta":    (right["ankle_x"]    - baseline["right_ankle_x"])    if right is not None else 0.0,
        "lr_knee_diff":           lr_knee_diff,
        "left_visibility":        left_vis,
        "right_visibility":       right_vis,
    }
