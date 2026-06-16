"""OpenCV によるスクワット有効試技判定結果のオーバーレイ描画モジュール。"""

import cv2
import mediapipe as mp
import numpy as np

from classifier import THRESHOLDS, INVALID_MAJORITY

_MP_POSE = mp.solutions.pose
_WHITE = (255, 255, 255)
_RED   = (0,   0, 255)

# 試技判定表示色
_COLOR_VALID    = (0,  200,  80)   # 緑
_COLOR_INVALID  = (0,   60, 220)   # 赤
_COLOR_JUDGING  = (180, 180, 180)  # グレー

# 深さインジケータの角度範囲
_ANGLE_MAX = 180.0  # 立位（バーが空）
_ANGLE_MIN =  60.0  # 最深（バーが満杯）

_FAIL_LABELS = {
    "depth_score":       "DEPTH FAIL",
    "lockout_score":     "LOCKOUT FAIL",
    "bar_descent_score": "BAR DESCENT",
    "bounce_score":      "BOUNCE",
    "foot_shift_score":  "FOOT SHIFT",
}


def _to_px(lm, h: int, w: int) -> tuple[int, int]:
    return (int(lm.x * w), int(lm.y * h))


def draw_overlay(
    frame: np.ndarray,
    result: dict | None,
    form_result: dict,
    features: dict,
    rep_count: int,
    rep_total: int,
    current_frame: int = 0,
    rep_start: int = 0,
    rep_end: int = 0,
    judging: bool = False,
) -> np.ndarray:
    """フレームに有効試技判定のオーバーレイを描画する。"""
    h, w = frame.shape[:2]
    L = _MP_POSE.PoseLandmark

    if result is None:
        text = "Pose not detected"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(frame, text, ((w - tw) // 2, (h + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, _RED, 2, cv2.LINE_AA)
        _draw_rep_counter(frame, w, rep_count, rep_total)
        return frame

    # ---- 骨格描画 ----
    lm = result["landmarks"].landmark
    for hip_lm, knee_lm, ankle_lm in [
        (L.LEFT_HIP,  L.LEFT_KNEE,  L.LEFT_ANKLE),
        (L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE),
    ]:
        hip   = _to_px(lm[hip_lm],   h, w)
        knee  = _to_px(lm[knee_lm],  h, w)
        ankle = _to_px(lm[ankle_lm], h, w)
        cv2.line(frame, hip, knee,   _WHITE, 3, cv2.LINE_AA)
        cv2.line(frame, knee, ankle, _WHITE, 3, cv2.LINE_AA)
        for pt in (hip, knee, ankle):
            cv2.circle(frame, pt, 6, _WHITE, -1, cv2.LINE_AA)

    # ---- 膝角度テキスト ----
    left_angle  = (result.get("left")  or {}).get("knee_angle")
    right_angle = (result.get("right") or {}).get("knee_angle")
    left_knee   = _to_px(lm[L.LEFT_KNEE], h, w)
    l_str = f"L:{left_angle:.1f}\xb0"  if left_angle  is not None else "L:--"
    r_str = f"R:{right_angle:.1f}\xb0" if right_angle is not None else "R:--"
    cv2.putText(frame, f"{l_str} {r_str}",
                (left_knee[0] + 10, left_knee[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 2, cv2.LINE_AA)

    # ---- 試技判定表示 ----
    y = 40
    if judging:
        cv2.putText(frame, "JUDGING...", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, _COLOR_JUDGING, 2, cv2.LINE_AA)
    elif form_result.get("valid"):
        cv2.putText(frame, "VALID", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, _COLOR_VALID, 3, cv2.LINE_AA)
    else:
        cv2.putText(frame, "INVALID", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, _COLOR_INVALID, 3, cv2.LINE_AA)
        y += 36
        for score_key, label in _FAIL_LABELS.items():
            if (form_result.get(score_key) or 0) >= INVALID_MAJORITY:
                cv2.putText(frame, label, (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, _COLOR_INVALID, 2, cv2.LINE_AA)
                y += 26

    # ---- レップカウンタ ----
    _draw_rep_counter(frame, w, rep_count, rep_total)

    # ---- 深さインジケータバー ----
    bar_h  = 16
    bar_y0 = h - bar_h - 10
    bar_y1 = h - 10

    # 背景（グレー）
    cv2.rectangle(frame, (0, bar_y0), (w, bar_y1), (60, 60, 60), -1)

    # 現在の深さ（代表膝角度）
    repr_angle = features.get("left_knee_angle", 0.0)
    if features.get("right_knee_angle", 0.0) > 0:
        repr_angle = (repr_angle + features["right_knee_angle"]) / 2.0

    fill_ratio = float(np.clip(
        (_ANGLE_MAX - repr_angle) / (_ANGLE_MAX - _ANGLE_MIN), 0.0, 1.0
    ))
    fill_w = int(w * fill_ratio)
    depth_reached = repr_angle <= THRESHOLDS["depth"] and repr_angle > 0
    bar_color = _COLOR_VALID if depth_reached else (100, 180, 255)
    if fill_w > 0:
        cv2.rectangle(frame, (0, bar_y0), (fill_w, bar_y1), bar_color, -1)

    # 深さ閾値ライン
    threshold_x = int(w * (_ANGLE_MAX - THRESHOLDS["depth"]) / (_ANGLE_MAX - _ANGLE_MIN))
    line_color = _COLOR_VALID if depth_reached else _WHITE
    cv2.line(frame, (threshold_x, bar_y0 - 2), (threshold_x, bar_y1 + 2), line_color, 2)

    # レップ進行プログレス（バー上）
    if rep_end > rep_start:
        progress = float(np.clip(
            (current_frame - rep_start) / max(rep_end - rep_start, 1), 0.0, 1.0
        ))
        progress_w = int(w * progress)
        cv2.rectangle(frame, (0, bar_y0 - 6), (progress_w, bar_y0 - 1), _WHITE, -1)

    return frame


def _draw_rep_counter(frame: np.ndarray, w: int, rep_count: int, rep_total: int) -> None:
    text = f"Rep: {rep_count} / {rep_total}" if rep_total > 0 else f"Rep: {rep_count}"
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(frame, text, (w - tw - 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, _WHITE, 2, cv2.LINE_AA)
