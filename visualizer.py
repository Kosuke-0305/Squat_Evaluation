"""OpenCV によるスクワットフォーム判定結果のオーバーレイ描画モジュール。"""

import cv2
import mediapipe as mp
import numpy as np

_MP_POSE = mp.solutions.pose
_WHITE = (255, 255, 255)
_RED = (0, 0, 255)
_YELLOW = (0, 220, 220)

FORM_COLORS = {
    "knee_forward": (0, 60, 220),
    "trunk_lean":   (0, 140, 255),
    "back_round":   (180, 60, 200),
    "good_form":    (0, 200, 80),
}

LR_DIFF_THRESHOLD = 15.0


def _to_px(lm, h: int, w: int) -> tuple[int, int]:
    """正規化座標をピクセル座標に変換する。"""
    return (int(lm.x * w), int(lm.y * h))


def _light_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(int(c * 0.35 + 255 * 0.65) for c in color)


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
) -> np.ndarray:
    """フレームにフォーム判定のオーバーレイを描画する。"""
    h, w = frame.shape[:2]
    L = _MP_POSE.PoseLandmark

    if result is None:
        text = "Pose not detected"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(frame, text, ((w - tw) // 2, (h + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, _RED, 2, cv2.LINE_AA)
        _draw_rep_counter(frame, w, rep_count, rep_total)
        return frame

    lm = result["landmarks"].landmark
    for hip_lm, knee_lm, ankle_lm in [
        (L.LEFT_HIP,  L.LEFT_KNEE,  L.LEFT_ANKLE),
        (L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE),
    ]:
        hip = _to_px(lm[hip_lm], h, w)
        knee = _to_px(lm[knee_lm], h, w)
        ankle = _to_px(lm[ankle_lm], h, w)
        cv2.line(frame, hip, knee, _WHITE, 3, cv2.LINE_AA)
        cv2.line(frame, knee, ankle, _WHITE, 3, cv2.LINE_AA)
        for pt in (hip, knee, ankle):
            cv2.circle(frame, pt, 6, _WHITE, -1, cv2.LINE_AA)

    left_angle = (result.get("left") or {}).get("knee_angle")
    right_angle = (result.get("right") or {}).get("knee_angle")
    left_knee = _to_px(lm[L.LEFT_KNEE], h, w)
    l_str = f"L:{left_angle:.1f}°" if left_angle is not None else "L:--"
    r_str = f"R:{right_angle:.1f}°" if right_angle is not None else "R:--"
    cv2.putText(frame, f"{l_str} {r_str}",
                (left_knee[0] + 10, left_knee[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 2, cv2.LINE_AA)

    y = 40
    if form_result["any_error"]:
        if form_result["knee_forward"]:
            cv2.putText(frame, "KNEE FORWARD", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, FORM_COLORS["knee_forward"], 2, cv2.LINE_AA)
            y += 30
        if form_result["trunk_lean"]:
            cv2.putText(frame, "TRUNK LEAN", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, FORM_COLORS["trunk_lean"], 2, cv2.LINE_AA)
            y += 30
        if form_result["back_round"]:
            cv2.putText(frame, "BACK ROUND", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, FORM_COLORS["back_round"], 2, cv2.LINE_AA)
            y += 30
    else:
        cv2.putText(frame, "GOOD FORM", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, FORM_COLORS["good_form"], 2, cv2.LINE_AA)
        y += 30

    score_text = f"kf:{features['knee_forward_ratio']:.2f} tl:{features['trunk_lean_delta']:.1f}° br:{features['back_round_ratio']:.2f}"
    cv2.putText(frame, score_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _WHITE, 2, cv2.LINE_AA)

    if left_angle is not None and right_angle is not None:
        diff = abs(left_angle - right_angle)
        if diff >= LR_DIFF_THRESHOLD:
            cv2.putText(frame, f"L/R diff: {diff:.1f}°", (20, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _YELLOW, 2, cv2.LINE_AA)

    _draw_rep_counter(frame, w, rep_count, rep_total)

    bar_height = 14
    bar_y0 = h - bar_height - 10
    bar_y1 = h - 10
    section_w = w // 3
    for i, key in enumerate(["knee_forward", "trunk_lean", "back_round"]):
        x0 = i * section_w
        x1 = (i + 1) * section_w if i < 2 else w
        color = FORM_COLORS[key]
        fill = color if form_result[key] else _light_color(color)
        cv2.rectangle(frame, (x0, bar_y0), (x1, bar_y1), fill, -1)

    if rep_end > rep_start:
        progress = float(np.clip((current_frame - rep_start) / max((rep_end - rep_start), 1), 0.0, 1.0))
        progress_w = int(w * progress)
        cv2.rectangle(frame, (0, bar_y0 - 6), (progress_w, bar_y0), _WHITE, -1)

    return frame


def _draw_rep_counter(frame: np.ndarray, w: int, rep_count: int, rep_total: int) -> None:
    text = f"Rep: {rep_count} / {rep_total}" if rep_total > 0 else f"Rep: {rep_count}"
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(frame, text, (w - tw - 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, _WHITE, 2, cv2.LINE_AA)
