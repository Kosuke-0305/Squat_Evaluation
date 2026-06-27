"""ルールベース分類器の閾値最適化実験。

1. 現状の問題診断（0°欠損値バグの確認）
2. 修正後の各閾値スウィープ（AUC・Precision・Recall）
3. 最適閾値の提案

使用例:
    python threshold_sweep.py --data-dir training_data
"""

import argparse
import csv
import json
import os
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

# ──────────────────────────────────────
# データ読み込み
# ──────────────────────────────────────

def load_all_reps(data_dir: str):
    """全元データの (features_seq, label) を返す。"""
    reps = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.startswith("anno_") or fname.startswith("anno_aug") or not fname.endswith(".csv"):
            continue
        base = fname[5:-4]
        seq_path  = os.path.join(data_dir, f"seq_{base}.json")
        anno_path = os.path.join(data_dir, fname)
        if not os.path.exists(seq_path):
            continue

        labels = {}
        with open(anno_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                v = row.get("valid", "").strip()
                if not v:
                    continue
                try:
                    labels[int(row["rep_id"])] = int(float(v) >= 1.0)
                except (KeyError, ValueError):
                    continue

        with open(seq_path, encoding="utf-8") as f:
            data = json.load(f)
        for rep in data["reps"]:
            if rep["rep_id"] not in labels:
                continue
            reps.append((rep["features"], labels[rep["rep_id"]]))

    return reps


# ──────────────────────────────────────
# _repr_knee（バグなし版: 0を除外）
# ──────────────────────────────────────

def repr_knee(f: dict) -> float:
    l = f.get("left_knee_angle",  0.0)
    r = f.get("right_knee_angle", 0.0)
    if l > 0 and r > 0:
        return (l + r) / 2.0
    return max(l, r)

def repr_knee_nonzero(f: dict) -> float:
    """0°（未検出フレーム）を除外した代表膝角度。"""
    l = f.get("left_knee_angle",  0.0)
    r = f.get("right_knee_angle", 0.0)
    if l > 0 and r > 0:
        return (l + r) / 2.0
    if l > 0:
        return l
    if r > 0:
        return r
    return None  # 両方0 = 未検出


# ──────────────────────────────────────
# ルール判定（閾値パラメータ化）
# ──────────────────────────────────────

def classify_rep(
    features_seq: list,
    depth_thr: float = 90.0,
    lockout_thr: float = 165.0,
    bar_descent_thr: float = 0.03,
    bounce_delta_thr: float = 10.0,
    foot_shift_thr: float = 0.05,
    invalid_majority: int = 2,
    fix_zero_bug: bool = True,
) -> dict:
    n = len(features_seq)
    if n == 0:
        return {"valid": True, "depth": 0, "lockout": 0, "bar_descent": 0, "bounce": 0, "foot": 0}

    if fix_zero_bug:
        knee_angles = [repr_knee_nonzero(f) for f in features_seq]
        knee_angles_valid = [ka for ka in knee_angles if ka is not None]
    else:
        knee_angles = [repr_knee(f) for f in features_seq]
        knee_angles_valid = knee_angles

    # 最深点（最小膝角度フレーム）
    if knee_angles_valid:
        # fix_zero_bug時はNoneを除外してインデックスを探す
        if fix_zero_bug:
            valid_indices = [i for i, ka in enumerate(knee_angles) if ka is not None]
            valley_idx = min(valid_indices, key=lambda i: knee_angles[i]) if valid_indices else n // 2
        else:
            valley_idx = int(np.argmin(knee_angles_valid))
    else:
        valley_idx = n // 2

    # depth_score
    if knee_angles_valid:
        depth_score = 3 if not any(ka <= depth_thr for ka in knee_angles_valid) else 0
    else:
        depth_score = 0  # 検出不可 → 判定しない

    # lockout_score
    all_knees = knee_angles if not fix_zero_bug else [ka if ka is not None else 999.0 for ka in knee_angles]
    start_frames = all_knees[:min(3, n)]
    end_frames   = all_knees[max(0, n - 3):]
    lockout_ok = (
        any(ka >= lockout_thr for ka in start_frames) and
        any(ka >= lockout_thr for ka in end_frames)
    )
    lockout_score = 0 if lockout_ok else 3

    # bar_descent_score
    ascending = features_seq[valley_idx:]
    bar_descent_score = 0
    for i in range(1, len(ascending)):
        dl = (ascending[i].get("left_shoulder_y_delta",  0.0)
              - ascending[i - 1].get("left_shoulder_y_delta",  0.0))
        dr = (ascending[i].get("right_shoulder_y_delta", 0.0)
              - ascending[i - 1].get("right_shoulder_y_delta", 0.0))
        if dl >= bar_descent_thr or dr >= bar_descent_thr:
            bar_descent_score = 3
            break

    # bounce_score
    bounce_score = 0
    hip_seq = [
        (f.get("left_hip_angle", 0.0) + f.get("right_hip_angle", 0.0)) / 2.0
        for f in features_seq[:valley_idx + 1]
    ]
    if len(hip_seq) >= 3:
        reversals = 0
        for i in range(1, len(hip_seq) - 1):
            prev_d = hip_seq[i]     - hip_seq[i - 1]
            next_d = hip_seq[i + 1] - hip_seq[i]
            if abs(prev_d) >= bounce_delta_thr and prev_d * next_d < 0:
                reversals += 1
        if reversals >= 2:
            bounce_score = 3

    # foot_shift_score
    foot_shift_score = 3 if any(
        abs(f.get("left_ankle_x_delta",  0.0)) >= foot_shift_thr or
        abs(f.get("right_ankle_x_delta", 0.0)) >= foot_shift_thr
        for f in features_seq
    ) else 0

    scores = [depth_score, lockout_score, bar_descent_score, bounce_score, foot_shift_score]
    valid = not any(s >= invalid_majority for s in scores)
    return {
        "valid": valid,
        "depth": depth_score,
        "lockout": lockout_score,
        "bar_descent": bar_descent_score,
        "bounce": bounce_score,
        "foot": foot_shift_score,
    }


def evaluate(reps, **kwargs):
    labels = [lab for _, lab in reps]
    preds  = [int(classify_rep(fs, **kwargs)["valid"]) for fs, _ in reps]
    # AUC: valid=1が正例 → 予測も valid=1
    try:
        auc = roc_auc_score(labels, preds)
    except Exception:
        auc = float("nan")
    # invalid recall/precision: label=0 が invalid
    inv_labels = [1 - l for l in labels]
    inv_preds  = [1 - p for p in preds]
    prec, rec, f1, _ = precision_recall_fscore_support(
        inv_labels, inv_preds, pos_label=1, average="binary", zero_division=0
    )
    n_invalid_detected = sum(p == 1 for p in inv_preds)
    return {"auc": auc, "inv_prec": prec, "inv_rec": rec, "inv_f1": f1,
            "n_pred_invalid": n_invalid_detected}


# ──────────────────────────────────────
# メイン
# ──────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="training_data")
    args = p.parse_args()

    reps = load_all_reps(args.data_dir)
    n_valid   = sum(1 for _, l in reps if l == 1)
    n_invalid = sum(1 for _, l in reps if l == 0)
    print(f"データ: {len(reps)}件（有効={n_valid}, 無効={n_invalid}）\n")

    # ──────────────────────────────────
    # 診断1: 膝角度の分布（0除外前後）
    # ──────────────────────────────────
    print("=" * 60)
    print("【診断1】膝角度の分布（レップ単位の最小値）")
    print("=" * 60)

    min_knees_with_zero   = []  # 0を含む
    min_knees_without_zero = []  # 0を除外

    for features_seq, _ in reps:
        kas_all = [repr_knee(f) for f in features_seq]
        kas_nz  = [ka for ka in kas_all if ka > 0]
        min_knees_with_zero.append(min(kas_all))
        min_knees_without_zero.append(min(kas_nz) if kas_nz else float("nan"))

    for label_name, arr_valid, arr_invalid in [
        ("0含む（現状）",
         [v for v, (_, l) in zip(min_knees_with_zero, reps) if l == 1],
         [v for v, (_, l) in zip(min_knees_with_zero, reps) if l == 0]),
        ("0除外（修正後）",
         [v for v, (_, l) in zip(min_knees_without_zero, reps) if l == 1],
         [v for v, (_, l) in zip(min_knees_without_zero, reps) if l == 0]),
    ]:
        av, ai = np.array(arr_valid), np.array(arr_invalid)
        print(f"\n--- {label_name} ---")
        print(f"  有効: 平均={np.nanmean(av):.1f}°  std={np.nanstd(av):.1f}°  "
              f"90°以下の割合={np.mean(av <= 90):.1%}  ({np.sum(av <= 90)}件)")
        print(f"  無効: 平均={np.nanmean(ai):.1f}°  std={np.nanstd(ai):.1f}°  "
              f"90°以下の割合={np.mean(ai <= 90):.1%}  ({np.sum(ai <= 90)}件)")

    # ──────────────────────────────────
    # 診断2: 現状（バグあり）vs 修正後 の全体性能
    # ──────────────────────────────────
    print("\n" + "=" * 60)
    print("【診断2】現状（0バグあり）vs 修正後の全体性能")
    print("=" * 60)

    for fix, label in [(False, "バグあり（現状）"), (True, "バグ修正後")]:
        r = evaluate(reps, fix_zero_bug=fix)
        # 各ルールの発火件数
        cnt = {"depth": 0, "lockout": 0, "bar_descent": 0, "bounce": 0, "foot": 0}
        for fs, _ in reps:
            res = classify_rep(fs, fix_zero_bug=fix)
            for k in cnt:
                if res[k] == 3:
                    cnt[k] += 1
        print(f"\n{label}:")
        print(f"  AUC={r['auc']:.3f}  inv_Prec={r['inv_prec']:.2f}  "
              f"inv_Rec={r['inv_rec']:.2f}  inv_F1={r['inv_f1']:.2f}  "
              f"予測invalid={r['n_pred_invalid']}件")
        print(f"  ルール発火件数: depth={cnt['depth']}  lockout={cnt['lockout']}  "
              f"bar_descent={cnt['bar_descent']}  bounce={cnt['bounce']}  foot={cnt['foot']}")

    # ──────────────────────────────────
    # 閾値スウィープ（バグ修正後）
    # ──────────────────────────────────
    print("\n" + "=" * 60)
    print("【閾値スウィープ】各ルールの閾値を単独で変化させたときのAUC")
    print("（他の閾値はデフォルト値を使用、0バグ修正後）")
    print("=" * 60)

    # depth threshold sweep
    print("\n--- depth 閾値（現在: 90°） ---")
    print(f"{'閾値':>8}  {'AUC':>6}  {'inv_Prec':>8}  {'inv_Rec':>7}  {'inv_F1':>6}  {'発火件数':>7}")
    best_depth = (0.0, 90.0)
    for thr in [75, 80, 85, 88, 90, 92, 95, 100, 105, 110, 115, 120]:
        r = evaluate(reps, depth_thr=thr, fix_zero_bug=True)
        n_fire = sum(1 for fs, _ in reps
                     if classify_rep(fs, depth_thr=thr, fix_zero_bug=True)["depth"] == 3)
        print(f"  {thr:>5}°   {r['auc']:>6.3f}  {r['inv_prec']:>8.2f}  "
              f"{r['inv_rec']:>7.2f}  {r['inv_f1']:>6.2f}  {n_fire:>7}")
        if r["auc"] > best_depth[0]:
            best_depth = (r["auc"], thr)
    print(f"  → 最適: {best_depth[1]}°  (AUC={best_depth[0]:.3f})")

    # lockout threshold sweep
    print("\n--- lockout 閾値（現在: 165°） ---")
    print(f"{'閾値':>8}  {'AUC':>6}  {'inv_Prec':>8}  {'inv_Rec':>7}  {'inv_F1':>6}  {'発火件数':>7}")
    best_lockout = (0.0, 165.0)
    for thr in [150, 155, 160, 163, 165, 167, 170]:
        r = evaluate(reps, lockout_thr=thr, fix_zero_bug=True)
        n_fire = sum(1 for fs, _ in reps
                     if classify_rep(fs, lockout_thr=thr, fix_zero_bug=True)["lockout"] == 3)
        print(f"  {thr:>5}°   {r['auc']:>6.3f}  {r['inv_prec']:>8.2f}  "
              f"{r['inv_rec']:>7.2f}  {r['inv_f1']:>6.2f}  {n_fire:>7}")
        if r["auc"] > best_lockout[0]:
            best_lockout = (r["auc"], thr)
    print(f"  → 最適: {best_lockout[1]}°  (AUC={best_lockout[0]:.3f})")

    # bar_descent threshold sweep
    print("\n--- bar_descent 閾値（現在: 0.03） ---")
    print(f"{'閾値':>8}  {'AUC':>6}  {'inv_Prec':>8}  {'inv_Rec':>7}  {'inv_F1':>6}  {'発火件数':>7}")
    best_bar = (0.0, 0.03)
    for thr in [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06]:
        r = evaluate(reps, bar_descent_thr=thr, fix_zero_bug=True)
        n_fire = sum(1 for fs, _ in reps
                     if classify_rep(fs, bar_descent_thr=thr, fix_zero_bug=True)["bar_descent"] == 3)
        print(f"  {thr:>6.3f}   {r['auc']:>6.3f}  {r['inv_prec']:>8.2f}  "
              f"{r['inv_rec']:>7.2f}  {r['inv_f1']:>6.2f}  {n_fire:>7}")
        if r["auc"] > best_bar[0]:
            best_bar = (r["auc"], thr)
    print(f"  → 最適: {best_bar[1]:.3f}  (AUC={best_bar[0]:.3f})")

    # foot_shift threshold sweep
    print("\n--- foot_shift 閾値（現在: 0.05） ---")
    print(f"{'閾値':>8}  {'AUC':>6}  {'inv_Prec':>8}  {'inv_Rec':>7}  {'inv_F1':>6}  {'発火件数':>7}")
    best_foot = (0.0, 0.05)
    for thr in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
        r = evaluate(reps, foot_shift_thr=thr, fix_zero_bug=True)
        n_fire = sum(1 for fs, _ in reps
                     if classify_rep(fs, foot_shift_thr=thr, fix_zero_bug=True)["foot"] == 3)
        print(f"  {thr:>6.3f}   {r['auc']:>6.3f}  {r['inv_prec']:>8.2f}  "
              f"{r['inv_rec']:>7.2f}  {r['inv_f1']:>6.2f}  {n_fire:>7}")
        if r["auc"] > best_foot[0]:
            best_foot = (r["auc"], thr)
    print(f"  → 最適: {best_foot[1]:.3f}  (AUC={best_foot[0]:.3f})")

    # ──────────────────────────────────
    # 最適閾値の組み合わせ評価
    # ──────────────────────────────────
    print("\n" + "=" * 60)
    print("【最終比較】現状 / バグ修正のみ / 最適閾値")
    print("=" * 60)

    configs = [
        ("現状（バグあり、閾値変更なし）",
         dict(depth_thr=90, lockout_thr=165, bar_descent_thr=0.03,
              foot_shift_thr=0.05, fix_zero_bug=False)),
        ("バグ修正のみ（閾値はデフォルト）",
         dict(depth_thr=90, lockout_thr=165, bar_descent_thr=0.03,
              foot_shift_thr=0.05, fix_zero_bug=True)),
        (f"バグ修正 + 最適depth={best_depth[1]}°",
         dict(depth_thr=best_depth[1], lockout_thr=165, bar_descent_thr=0.03,
              foot_shift_thr=0.05, fix_zero_bug=True)),
        (f"バグ修正 + 最適lockout={best_lockout[1]}°",
         dict(depth_thr=90, lockout_thr=best_lockout[1], bar_descent_thr=0.03,
              foot_shift_thr=0.05, fix_zero_bug=True)),
        (f"バグ修正 + 最適bar_descent={best_bar[1]:.3f}",
         dict(depth_thr=90, lockout_thr=165, bar_descent_thr=best_bar[1],
              foot_shift_thr=0.05, fix_zero_bug=True)),
        ("バグ修正 + 全閾値最適",
         dict(depth_thr=best_depth[1], lockout_thr=best_lockout[1],
              bar_descent_thr=best_bar[1], foot_shift_thr=best_foot[1],
              fix_zero_bug=True)),
    ]

    print(f"\n{'設定':<42}  {'AUC':>6}  {'inv_Prec':>8}  {'inv_Rec':>7}  {'inv_F1':>6}")
    for name, kw in configs:
        r = evaluate(reps, **kw)
        print(f"  {name:<40}  {r['auc']:>6.3f}  {r['inv_prec']:>8.2f}  "
              f"{r['inv_rec']:>7.2f}  {r['inv_f1']:>6.2f}")


if __name__ == "__main__":
    main()
