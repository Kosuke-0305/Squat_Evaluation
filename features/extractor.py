"""フォーム崩れ検出用特徴量と baseline を計算するモジュール。"""

import math


def _average_point(points: list[tuple[float, float]]) -> tuple[float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _compute_side_values(frame_result: dict | None) -> tuple[list[float], list[tuple[float, float]], list[tuple[float, float]]]:
    knee_foot_diffs = []
    shoulders = []
    hips = []

    if frame_result is None:
        return knee_foot_diffs, shoulders, hips

    for side in (frame_result.get("left"), frame_result.get("right")):
        if side is None:
            continue
        knee_foot_diffs.append(side["knee_x"] - side["ankle_x"])
        shoulders.append((side["shoulder_x"], side["shoulder_y"]))
        hips.append((side["hip_x"], side["hip_y"]))

    return knee_foot_diffs, shoulders, hips


def compute_baseline(frame_result: dict | None) -> dict | None:
    """1フレームの推定結果から baseline を計算する。"""
    knee_foot_diffs, shoulders, hips = _compute_side_values(frame_result)
    if not knee_foot_diffs or not shoulders or not hips:
        return None

    knee_foot_diff = float(sum(knee_foot_diffs) / len(knee_foot_diffs))
    shoulder_mid = _average_point(shoulders)
    hip_mid = _average_point(hips)

    dx = hip_mid[0] - shoulder_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    trunk_angle = math.degrees(math.atan2(dx, dy)) if dy != 0 else 90.0
    shoulder_hip_dist = math.hypot(shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1])

    return {
        "knee_foot_diff": knee_foot_diff,
        "trunk_angle": float(trunk_angle),
        "shoulder_hip_dist": float(shoulder_hip_dist),
    }


def extract_form_features(frame_result: dict | None, baseline: dict | None) -> dict:
    """フレーム単位のフォーム特徴量を計算する。"""
    if frame_result is None or baseline is None:
        return {
            "knee_forward_ratio": 0.0,
            "trunk_lean_delta": 0.0,
            "back_round_ratio": 0.0,
            "left_knee_angle": 0.0,
            "right_knee_angle": 0.0,
            "left_hip_angle": 0.0,
            "right_hip_angle": 0.0,
            "left_ankle_angle": 0.0,
            "right_ankle_angle": 0.0,
            "lr_knee_diff": 0.0,
            "left_visibility": 0.0,
            "right_visibility": 0.0,
        }

    knee_foot_diffs, _, _ = _compute_side_values(frame_result)
    current_diff = float(sum(knee_foot_diffs) / len(knee_foot_diffs)) if knee_foot_diffs else 0.0
    knee_forward_ratio = current_diff / baseline["knee_foot_diff"] if baseline["knee_foot_diff"] != 0 else 0.0

    shoulder_hip_dist = 0.0
    trunk_angle = 0.0
    if frame_result.get("left") is not None or frame_result.get("right") is not None:
        _, shoulders, hips = _compute_side_values(frame_result)
        if shoulders and hips:
            shoulder_mid = _average_point(shoulders)
            hip_mid = _average_point(hips)
            dx = hip_mid[0] - shoulder_mid[0]
            dy = shoulder_mid[1] - hip_mid[1]
            trunk_angle = math.degrees(math.atan2(dx, dy)) if dy != 0 else 90.0
            shoulder_hip_dist = math.hypot(shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1])

    trunk_lean_delta = float(trunk_angle - baseline["trunk_angle"])
    back_round_ratio = shoulder_hip_dist / baseline["shoulder_hip_dist"] if baseline["shoulder_hip_dist"] != 0 else 0.0

    left = frame_result.get("left")
    right = frame_result.get("right")

    left_knee_angle = left["knee_angle"] if left is not None else 0.0
    right_knee_angle = right["knee_angle"] if right is not None else 0.0
    left_hip_angle = left["hip_angle"] if left is not None else 0.0
    right_hip_angle = right["hip_angle"] if right is not None else 0.0
    left_ankle_angle = left["ankle_angle"] if left is not None else 0.0
    right_ankle_angle = right["ankle_angle"] if right is not None else 0.0
    left_visibility = left["visibility"] if left is not None else 0.0
    right_visibility = right["visibility"] if right is not None else 0.0
    lr_knee_diff = abs(left_knee_angle - right_knee_angle) if left is not None and right is not None else 0.0

    return {
        "knee_forward_ratio": knee_forward_ratio,
        "trunk_lean_delta": trunk_lean_delta,
        "back_round_ratio": back_round_ratio,
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "left_hip_angle": left_hip_angle,
        "right_hip_angle": right_hip_angle,
        "left_ankle_angle": left_ankle_angle,
        "right_ankle_angle": right_ankle_angle,
        "lr_knee_diff": lr_knee_diff,
        "left_visibility": left_visibility,
        "right_visibility": right_visibility,
    }
