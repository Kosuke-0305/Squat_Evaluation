"""無効試技の時系列データを拡張して training_data/ に保存するスクリプト。

拡張手法（無効試技のみ対象、有効試技はスキップ）:
  time_warp: 時系列を 80% / 90% / 110% / 120% の長さにリサンプル  (×4)
  noise:     各特徴量にガウスノイズを加算                          (×2)
  flip:      左右対称に反転                                        (×1)
                                              合計 7倍 → 113件→904件

使用例:
    python augment_data.py --dry-run        # 件数確認のみ
    python augment_data.py                  # 実際に保存
"""

import argparse
import csv
import json
import os

import numpy as np
from scipy.interpolate import interp1d

from model import FEATURE_KEYS

# --------------------------------------------------------------------------
# 左右反転の定義
# --------------------------------------------------------------------------
_LR_SWAP = [
    ("left_knee_angle",       "right_knee_angle"),
    ("left_hip_angle",        "right_hip_angle"),
    ("left_shoulder_y_delta", "right_shoulder_y_delta"),
    ("left_visibility",       "right_visibility"),
]
# x 方向デルタは左右交換 + 符号反転
_LR_NEGATE_SWAP = [
    ("left_ankle_x_delta", "right_ankle_x_delta"),
]

# 特徴量ごとのノイズ標準偏差
_NOISE_STD = {
    "left_knee_angle":        1.5,
    "right_knee_angle":       1.5,
    "left_hip_angle":         1.5,
    "right_hip_angle":        1.5,
    "left_shoulder_y_delta":  0.002,
    "right_shoulder_y_delta": 0.002,
    "left_ankle_x_delta":     0.002,
    "right_ankle_x_delta":    0.002,
    "lr_knee_diff":           1.0,
    "left_visibility":        0.005,
    "right_visibility":       0.005,
}

_KEY_IDX = {k: i for i, k in enumerate(FEATURE_KEYS)}


# --------------------------------------------------------------------------
# 拡張関数
# --------------------------------------------------------------------------

def _to_array(features: list) -> np.ndarray:
    return np.array([[f.get(k, 0.0) for k in FEATURE_KEYS] for f in features],
                    dtype=np.float32)


def _to_features(arr: np.ndarray) -> list:
    return [{k: float(arr[t, i]) for i, k in enumerate(FEATURE_KEYS)}
            for t in range(len(arr))]


def time_warp(arr: np.ndarray, ratio: float) -> np.ndarray:
    T, D = arr.shape
    new_T = max(2, int(round(T * ratio)))
    x_old = np.linspace(0.0, 1.0, T)
    x_new = np.linspace(0.0, 1.0, new_T)
    out = np.empty((new_T, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = interp1d(x_old, arr[:, d], kind="linear")(x_new)
    return out


def add_noise(arr: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = arr.copy()
    for i, k in enumerate(FEATURE_KEYS):
        out[:, i] += rng.normal(0.0, _NOISE_STD[k], size=len(arr)).astype(np.float32)
    for i, k in enumerate(FEATURE_KEYS):
        if "visibility" in k:
            out[:, i] = np.clip(out[:, i], 0.0, 1.0)
    return out


def lr_flip(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    for lk, rk in _LR_SWAP:
        li, ri = _KEY_IDX[lk], _KEY_IDX[rk]
        out[:, li], out[:, ri] = arr[:, ri].copy(), arr[:, li].copy()
    for lk, rk in _LR_NEGATE_SWAP:
        li, ri = _KEY_IDX[lk], _KEY_IDX[rk]
        out[:, li] = -arr[:, ri]
        out[:, ri] = -arr[:, li]
    if "lr_knee_diff" in _KEY_IDX:
        ki = _KEY_IDX["lr_knee_diff"]
        out[:, ki] = -arr[:, ki]
    return out


def _all_augmentations(arr: np.ndarray) -> list:
    """(tag, augmented_array) のリストを返す（7種）。"""
    return [
        ("tw80",  time_warp(arr, 0.80)),
        ("tw90",  time_warp(arr, 0.90)),
        ("tw110", time_warp(arr, 1.10)),
        ("tw120", time_warp(arr, 1.20)),
        ("ns0",   add_noise(arr, seed=0)),
        ("ns1",   add_noise(arr, seed=1)),
        ("flip",  lr_flip(arr)),
    ]


# --------------------------------------------------------------------------
# ファイル保存
# --------------------------------------------------------------------------

def _save(tag: str, base: str, aug_arr: np.ndarray, rep: dict,
          csv_row: dict, out_dir: str) -> None:
    new_base = f"aug{tag}_{base}"
    # JSON
    new_rep = {k: v for k, v in rep.items() if k != "features"}
    new_rep["features"] = _to_features(aug_arr)
    seq_data = {"video_id": f"aug_{rep.get('rep_id', 0)}", "reps": [new_rep]}
    with open(os.path.join(out_dir, f"seq_{new_base}.json"), "w", encoding="utf-8") as f:
        json.dump(seq_data, f, ensure_ascii=False)
    # CSV
    fields = list(csv_row.keys())
    with open(os.path.join(out_dir, f"anno_{new_base}.csv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(csv_row)


# --------------------------------------------------------------------------
# メイン
# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="無効試技データ拡張")
    p.add_argument("--data-dir", default="training_data")
    p.add_argument("--dry-run",  action="store_true", help="件数確認のみ（保存しない）")
    args = p.parse_args()

    td = args.data_dir
    invalid_count = 0
    generated = 0

    for fname in sorted(os.listdir(td)):
        if not fname.startswith("anno_") or not fname.endswith(".csv"):
            continue
        # augで生成済みのファイルは対象外
        if fname.startswith("anno_aug"):
            continue
        base = fname[5:-4]
        seq_path = os.path.join(td, f"seq_{base}.json")
        if not os.path.exists(seq_path):
            continue

        with open(os.path.join(td, fname), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        row = rows[0]
        v = row.get("valid", "").strip()
        if not v or int(float(v)) >= 1:
            continue  # 有効試技はスキップ

        invalid_count += 1
        with open(seq_path, encoding="utf-8") as f:
            data = json.load(f)

        for rep in data["reps"]:
            if not rep.get("features"):
                continue
            arr = _to_array(rep["features"])
            for tag, aug_arr in _all_augmentations(arr):
                if not args.dry_run:
                    _save(tag, base, aug_arr, rep, row, td)
                generated += 1

    print(f"元の無効試技: {invalid_count}件")
    print(f"拡張データ:   {generated}件  ({invalid_count}件 × 7手法)")
    print(f"拡張後の合計無効試技: {invalid_count + generated}件")
    if args.dry_run:
        print("※ dry-run のため保存していません")
    else:
        print(f"保存先: {td}/")


if __name__ == "__main__":
    main()
