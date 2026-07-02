"""レップ回数・最深点フレーム検出モジュール。"""

import numpy as np
from scipy.signal import find_peaks

NONE_RUN_LIMIT = 5    # 連続 None フレームの無効判定閾値
PROMINENCE     = 20.0  # find_peaks の prominence（谷の最小深さ）
DISTANCE       = 30    # find_peaks の distance（最小フレーム間隔）


def get_representative_angle(frame_result: dict | None) -> float | None:
    """左右の膝角度から代表角度を返す。

    - 左右両足が有効 → 平均値
    - 片足のみ有効 → その足の角度
    - 両足 None → None
    """
    if frame_result is None:
        return None
    left  = frame_result.get("left")
    right = frame_result.get("right")
    if left is not None and right is not None:
        return (left["knee_angle"] + right["knee_angle"]) / 2.0
    if left is not None:
        return left["knee_angle"]
    if right is not None:
        return right["knee_angle"]
    return None


def frame_to_sec(frame_idx: int, fps: float) -> float:
    """フレームインデックスを秒数に変換する。"""
    return round(frame_idx / fps, 2)


def frame_to_mmss(frame_idx: int, fps: float) -> str:
    """フレームインデックスを m:ss.ff 形式の文字列に変換する。"""
    total_sec = frame_idx / fps
    m = int(total_sec) // 60
    s = total_sec - m * 60
    return f"{m}:{s:05.2f}"


def _linear_interp(angles: list) -> np.ndarray:
    """None を線形補間して float 配列を返す。端の None は最近傍値でクランプ。"""
    arr = np.array([a if a is not None else np.nan for a in angles], dtype=float)
    nans = np.isnan(arr)
    if nans.all() or not nans.any():
        return arr
    x = np.arange(len(arr))
    arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return arr


def _has_long_none_run(frame_results: list, start: int, end: int) -> bool:
    """区間内に NONE_RUN_LIMIT フレーム以上連続する None があるか判定する。"""
    run = 0
    for i in range(start, min(end + 1, len(frame_results))):
        if frame_results[i] is None:
            run += 1
            if run >= NONE_RUN_LIMIT:
                return True
        else:
            run = 0
    return False


def detect_reps(
    frame_results: list,
    fps: float | None = None,
    ignore_before_frame: int = 0,
    ignore_after_frame: int | None = None,
) -> list:
    """全フレームの代表角度列からレップを検出する。

    Args:
        frame_results: PoseEstimator.process() の結果リスト
        fps: 動画のフレームレート（時間情報の付加に使用）
        ignore_before_frame: この前のフレームを検出対象外にする
        ignore_after_frame: この後のフレームを検出対象外にする（None = 末尾まで）

    Returns:
        各レップの情報 dict のリスト。無効レップは除外済み。
    """
    repr_angles = [get_representative_angle(r) for r in frame_results]
    n = len(repr_angles)
    if n == 0:
        return []

    interpolated = _linear_interp(repr_angles)
    if np.isnan(interpolated).all():
        return []

    valleys, _ = find_peaks(-interpolated, prominence=PROMINENCE, distance=DISTANCE)
    if len(valleys) == 0:
        return []

    # アノテーション対象外フレームの除外
    eff_after = (n - 1) if ignore_after_frame is None else ignore_after_frame
    valleys = valleys[(valleys >= ignore_before_frame) & (valleys <= eff_after)]
    if len(valleys) == 0:
        return []

    reps = []
    for i, v in enumerate(valleys):
        start = int((valleys[i - 1] + v) // 2) if i > 0 else ignore_before_frame
        end   = int((v + valleys[i + 1]) // 2) if i < len(valleys) - 1 else eff_after
        start = max(start, ignore_before_frame)
        end   = min(end,   eff_after)

        if _has_long_none_run(frame_results, start, end):
            continue

        r = frame_results[v]
        left_angle  = r["left"]["knee_angle"]  if r and r.get("left")  else None
        right_angle = r["right"]["knee_angle"] if r and r.get("right") else None
        repr_val    = get_representative_angle(r) or float(interpolated[v])

        rep_dict = {
            "valley_frame":    int(v),
            "start_frame":     start,
            "end_frame":       end,
            "min_angle_left":  left_angle,
            "min_angle_right": right_angle,
            "min_angle_repr":  repr_val,
        }

        if fps is not None and fps > 0:
            rep_dict.update({
                "start_time_sec":   frame_to_sec(start,  fps),
                "end_time_sec":     frame_to_sec(end,    fps),
                "deepest_time_sec": frame_to_sec(int(v), fps),
                "start_mmss":       frame_to_mmss(start,  fps),
                "end_mmss":         frame_to_mmss(end,    fps),
                "deepest_mmss":     frame_to_mmss(int(v), fps),
            })

        reps.append(rep_dict)

    return reps
