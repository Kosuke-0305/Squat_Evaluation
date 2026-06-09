"""スクワットフォーム崩れ検出デモのエントリポイント。"""

import argparse
import csv
import json
import os
import queue as _queue
import sys
import threading

import cv2
import numpy as np

from pose.estimator import PoseEstimator
from pose.rep_detector import detect_reps, get_representative_angle, frame_to_sec, frame_to_mmss
from features.extractor import compute_baseline, extract_form_features
from classifier import create_classifier
from visualizer import draw_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="スクワットフォーム崩れ検出デモ")
    parser.add_argument("--input",              type=str,   default=None,  help="動画ファイルパス（省略時はWebカメラ）")
    parser.add_argument("--save",               action="store_true",       help="処理済み動画を output.mp4 に保存")
    parser.add_argument("--save-frames",        action="store_true",       help="最深点フレームを PNG で保存")
    parser.add_argument("--style",              type=str,   default=None,  help="スタイル名（例: HB_NS）をサマリに記録")
    parser.add_argument("--export-annotation",  action="store_true",       help="アノテーション候補 CSV と特徴量シーケンス JSON を出力")
    parser.add_argument("--classifier",         choices=["rule", "lstm"],  default="rule",
                        help="判定モード: rule=ルールベース（デフォルト）/ lstm=学習済みモデル")
    parser.add_argument("--model",              type=str,   default=None,  help="LSTM モデルパス（--classifier lstm のとき必須）")
    parser.add_argument("--ignore-before-sec",  type=float, default=None,  help="この秒数より前のフレームをレップ検出対象外にする（ラックアップ前の除外）")
    parser.add_argument("--ignore-after-sec",   type=float, default=None,  help="この秒数より後のフレームをレップ検出対象外にする（後処理動作の除外）")
    parser.add_argument("--review-annotation",  action="store_true",       help="検出したレップ境界を対話的に確認・修正するモード")
    return parser.parse_args()


def open_capture(input_path: str | None) -> cv2.VideoCapture:
    source = 0 if input_path is None else input_path
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: ソースを開けませんでした: {source}", file=sys.stderr)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def create_writer(cap: cv2.VideoCapture, path: str) -> cv2.VideoWriter:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (w, h))


def _compute_max_lr_diff(frame_results: list, start: int, end: int) -> float:
    """レップ区間内の左右膝角度差の最大値を返す。"""
    max_diff = 0.0
    for r in frame_results[start:end + 1]:
        if r and r.get("left") and r.get("right"):
            diff = abs(r["left"]["knee_angle"] - r["right"]["knee_angle"])
            max_diff = max(max_diff, diff)
    return max_diff


def _format_error_labels(form_flags: dict) -> str:
    labels = [k for k in ("knee_forward", "trunk_lean", "back_round") if form_flags[k]]
    return "_".join(labels) if labels else "good_form"


def _rep_features_seq(rep: dict, frame_results: list, baseline: dict | None) -> list:
    """レップ区間のフレーム特徴量リストを返す。"""
    start = rep["start_frame"]
    end   = rep["end_frame"]
    return [
        extract_form_features(frame_results[i], baseline)
        for i in range(start, min(end + 1, len(frame_results)))
    ]


def build_summary(reps: list, frame_results: list, squat_style: str | None, classifier) -> dict:
    """サマリ dict を構築する。"""
    rep_list = []
    for n, rep in enumerate(reps, 1):
        start_frame = rep["start_frame"]
        baseline_frame = frame_results[start_frame] if 0 <= start_frame < len(frame_results) else None
        baseline = compute_baseline(baseline_frame)

        deepest_frame  = rep["valley_frame"]
        deepest_result = frame_results[deepest_frame] if 0 <= deepest_frame < len(frame_results) else None
        features_at_deepest = extract_form_features(deepest_result, baseline)

        # 後処理はレップ全体シーケンスで判定（LSTM/ルールベース共通インターフェース）
        features_seq = _rep_features_seq(rep, frame_results, baseline)
        form_flags   = classifier.classify_sequence(features_seq)

        rep_list.append({
            "rep_id":           n,
            "start_frame":      start_frame,
            "end_frame":        rep["end_frame"],
            "deepest_frame":    deepest_frame,
            "start_time_sec":   rep.get("start_time_sec"),
            "end_time_sec":     rep.get("end_time_sec"),
            "deepest_time_sec": rep.get("deepest_time_sec"),
            "start_mmss":       rep.get("start_mmss"),
            "end_mmss":         rep.get("end_mmss"),
            "deepest_mmss":     rep.get("deepest_mmss"),
            "min_angle_repr": round(rep["min_angle_repr"], 1),
            "max_lr_diff":   round(_compute_max_lr_diff(frame_results, start_frame, rep["end_frame"]), 1),
            "baseline": {
                "knee_foot_diff":    round(baseline["knee_foot_diff"],    3) if baseline else None,
                "trunk_angle":       round(baseline["trunk_angle"],       3) if baseline else None,
                "shoulder_hip_dist": round(baseline["shoulder_hip_dist"], 3) if baseline else None,
            },
            "features_at_deepest": {
                "knee_forward_ratio": round(features_at_deepest["knee_forward_ratio"], 3),
                "trunk_lean_delta":   round(features_at_deepest["trunk_lean_delta"],   3),
                "back_round_ratio":   round(features_at_deepest["back_round_ratio"],   3),
            },
            "form_labels": {
                "knee_forward": int(form_flags["knee_forward"]),
                "trunk_lean":   int(form_flags["trunk_lean"]),
                "back_round":   int(form_flags["back_round"]),
            },
        })
    return {
        "total_reps":  len(reps),
        "squat_style": squat_style,
        "classifier":  "lstm" if classifier.__class__.__name__ == "LSTMClassifier" else "rule",
        "reps":        rep_list,
    }


def export_annotation_csv(
    reps: list, frame_results: list,
    squat_style: str | None, video_id: str,
    classifier,
) -> None:
    """アノテーション候補 CSV を出力する。"""
    fieldnames = [
        "video_id", "rep_id", "squat_style",
        "start_frame", "end_frame", "deepest_frame",
        "start_time_sec", "end_time_sec", "deepest_time_sec",
        "start_mmss", "end_mmss", "deepest_mmss",
        "knee_forward_flag", "trunk_lean_flag", "back_round_flag", "needs_review",
        "knee_forward_ratio_max", "trunk_lean_delta_max", "back_round_ratio_min",
    ]
    with open("annotation_candidates.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for n, rep in enumerate(reps, 1):
            start_frame = rep["start_frame"]
            baseline_frame = frame_results[start_frame] if 0 <= start_frame < len(frame_results) else None
            baseline = compute_baseline(baseline_frame)

            features_seq = _rep_features_seq(rep, frame_results, baseline)

            # フォーム判定はシーケンス単位で実施
            form_flags = classifier.classify_sequence(features_seq)

            # スコア集計はフレーム単位で行う（CSV の ratio 列はルールベース側の連続値）
            kf_ratios = [f["knee_forward_ratio"] for f in features_seq]
            tl_deltas = [f["trunk_lean_delta"]   for f in features_seq]
            br_ratios = [f["back_round_ratio"]   for f in features_seq]

            writer.writerow({
                "video_id":               video_id,
                "rep_id":                 n,
                "squat_style":            squat_style or "",
                "start_frame":            start_frame,
                "end_frame":              rep["end_frame"],
                "deepest_frame":          rep["valley_frame"],
                "start_time_sec":         rep.get("start_time_sec", ""),
                "end_time_sec":           rep.get("end_time_sec", ""),
                "deepest_time_sec":       rep.get("deepest_time_sec", ""),
                "start_mmss":             rep.get("start_mmss", ""),
                "end_mmss":               rep.get("end_mmss", ""),
                "deepest_mmss":           rep.get("deepest_mmss", ""),
                "knee_forward_flag":      int(form_flags["knee_forward"]),
                "trunk_lean_flag":        int(form_flags["trunk_lean"]),
                "back_round_flag":        int(form_flags["back_round"]),
                "needs_review":           int(form_flags["any_error"]),
                "knee_forward_ratio_max": round(max(kf_ratios) if kf_ratios else 0.0, 3),
                "trunk_lean_delta_max":   round(max(tl_deltas)  if tl_deltas  else 0.0, 3),
                "back_round_ratio_min":   round(min(br_ratios)  if br_ratios  else 0.0, 3),
            })


def save_feature_sequences(reps: list, frame_results: list, video_id: str) -> None:
    """各レップの特徴量シーケンスを JSON に保存する（LSTM 学習用）。

    正式 baseline（各レップ start_frame）を使って特徴量を再計算する。
    """
    all_reps = []
    for n, rep in enumerate(reps, 1):
        start_frame = rep["start_frame"]
        baseline_frame = frame_results[start_frame] if 0 <= start_frame < len(frame_results) else None
        baseline = compute_baseline(baseline_frame)
        features_seq = _rep_features_seq(rep, frame_results, baseline)
        all_reps.append({
            "rep_id":           n,
            "start_frame":      start_frame,
            "end_frame":        rep["end_frame"],
            "deepest_frame":    rep["valley_frame"],
            "start_time_sec":   rep.get("start_time_sec"),
            "end_time_sec":     rep.get("end_time_sec"),
            "deepest_time_sec": rep.get("deepest_time_sec"),
            "start_mmss":       rep.get("start_mmss"),
            "end_mmss":         rep.get("end_mmss"),
            "deepest_mmss":     rep.get("deepest_mmss"),
            "features":         features_seq,
        })
    data = {"video_id": video_id, "reps": all_reps}
    path = "feature_sequences.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"特徴量シーケンス保存: {path}")


def _save_valley_frame(n: int, vf: int, valley_orig, rep: dict,
                       result_buffer: list, classifier) -> None:
    """valley_frame を分類してラベル付きファイル名で PNG に保存する。"""
    baseline_frame = result_buffer[rep["start_frame"]] if 0 <= rep["start_frame"] < len(result_buffer) else None
    baseline = compute_baseline(baseline_frame)
    features_seq = _rep_features_seq(rep, result_buffer, baseline)
    form_flags   = classifier.classify_sequence(features_seq)
    labels = _format_error_labels(form_flags)
    fname = f"rep{n:02d}_frame{vf:04d}_{labels}.png"
    cv2.imwrite(fname, valley_orig)
    print(f"保存: {fname}")


def review_annotation(reps: list, fps: float, total_frames: int) -> list:
    """レップ境界を対話的に確認・修正する。

    各レップの時間情報を表示し、ユーザが開始・終了秒数を入力することで
    start_frame / end_frame を上書きできる。
    """
    if not reps:
        return reps

    print("\n" + "=" * 60)
    print("レップ境界レビュー")
    print("=" * 60)
    print("修正する場合は「開始秒 終了秒」を入力（例: 1.5 3.2）")
    print("変更なしの場合は Enter を押してください\n")

    corrected = []
    for i, rep in enumerate(reps, 1):
        start_f = rep["start_frame"]
        end_f   = rep["end_frame"]
        deep_f  = rep["valley_frame"]

        start_t = rep.get("start_mmss")  or f"{frame_to_sec(start_f, fps)}s"
        end_t   = rep.get("end_mmss")    or f"{frame_to_sec(end_f,   fps)}s"
        deep_t  = rep.get("deepest_mmss") or f"{frame_to_sec(deep_f, fps)}s"

        print(f"Rep {i}:")
        print(f"  開始   : frame {start_f:5d}  ({start_t})")
        print(f"  終了   : frame {end_f:5d}  ({end_t})")
        print(f"  最深点 : frame {deep_f:5d}  ({deep_t})  膝角度={rep['min_angle_repr']:.1f}°")

        try:
            user_input = input("  修正 [開始秒 終了秒 / Enter でスキップ]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            corrected.append(rep)
            continue

        if not user_input:
            corrected.append(rep)
            continue

        parts = user_input.split()
        if len(parts) != 2:
            print("  ※ 入力形式が不正です（例: 1.5 3.2）。変更なしで続行します。")
            corrected.append(rep)
            continue

        try:
            new_start_sec = float(parts[0])
            new_end_sec   = float(parts[1])
        except ValueError:
            print("  ※ 数値として読み取れません。変更なしで続行します。")
            corrected.append(rep)
            continue

        new_start_f = max(0, min(int(round(new_start_sec * fps)), total_frames - 1))
        new_end_f   = max(0, min(int(round(new_end_sec   * fps)), total_frames - 1))

        if new_start_f >= new_end_f:
            print("  ※ 開始フレーム ≥ 終了フレームになります。変更なしで続行します。")
            corrected.append(rep)
            continue

        updated = dict(rep)
        updated["start_frame"]     = new_start_f
        updated["end_frame"]       = new_end_f
        updated["start_time_sec"]  = frame_to_sec(new_start_f, fps)
        updated["end_time_sec"]    = frame_to_sec(new_end_f,   fps)
        updated["start_mmss"]      = frame_to_mmss(new_start_f, fps)
        updated["end_mmss"]        = frame_to_mmss(new_end_f,   fps)

        print(f"  → 更新: frames {new_start_f}–{new_end_f}"
              f"  ({updated['start_mmss']} – {updated['end_mmss']})")
        corrected.append(updated)

    print()
    return corrected


def main() -> None:
    args = parse_args()

    # 判定器を生成（rule / lstm）
    try:
        classifier = create_classifier(args.classifier, args.model)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    cap = open_capture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    estimator = PoseEstimator()
    writer: cv2.VideoWriter | None = None
    if args.save:
        writer = create_writer(cap, "output.mp4")
        if not writer.isOpened():
            print("ERROR: VideoWriter の初期化に失敗しました。録画を無効化します。", file=sys.stderr)
            writer = None
        else:
            print("出力先: output.mp4")

    result_buffer: list = []
    frame_candidates: dict = {}  # {frame_idx: ndarray} — webcam 用スパースバッファ
    frame_idx = 0
    prov_count = 0
    prov_in_squat = False
    current_squat_start = 0
    latest_stand_frame_result = None

    print("モデル初期化中 ...")
    estimator.process(np.zeros((480, 640, 3), dtype=np.uint8))

    _frame_q: _queue.Queue = _queue.Queue(maxsize=1)
    _latest_result: list = [None]
    _result_lock = threading.Lock()
    _stop_worker = threading.Event()

    def _pose_worker() -> None:
        while not _stop_worker.is_set():
            try:
                frm = _frame_q.get(timeout=0.1)
            except _queue.Empty:
                continue
            res = estimator.process(frm)
            with _result_lock:
                _latest_result[0] = res

    _worker_thread = threading.Thread(target=_pose_worker, daemon=True)
    _worker_thread.start()

    print(f"スタート [{args.classifier.upper()} モード] — 終了するには 'q' キーを押してください")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            _frame_q.put_nowait(frame.copy())
        except _queue.Full:
            pass

        with _result_lock:
            result = _latest_result[0]

        repr_angle = get_representative_angle(result)
        if args.save_frames and args.input is None and repr_angle is not None and repr_angle < 150.0:
            frame_candidates[frame_idx] = frame.copy()

        if repr_angle is not None and repr_angle >= 170.0 and result is not None:
            latest_stand_frame_result = result

        baseline = compute_baseline(latest_stand_frame_result)
        features = extract_form_features(result, baseline)
        form_result = classifier.classify(features)  # リアルタイム判定

        if repr_angle is not None:
            if not prov_in_squat and repr_angle < 90.0:
                prov_in_squat = True
                current_squat_start = frame_idx
                classifier.reset()  # レップ開始時にウィンドウリセット
            elif prov_in_squat and repr_angle >= 90.0:
                prov_in_squat = False
                prov_count += 1

        rep_start = current_squat_start if prov_in_squat else 0
        rep_end   = frame_idx if prov_in_squat else 0

        frame = draw_overlay(
            frame, result, form_result, features,
            rep_count=prov_count, rep_total=0,
            current_frame=frame_idx, rep_start=rep_start, rep_end=rep_end,
        )

        cv2.imshow("Squat Form Demo", frame)
        if writer:
            writer.write(frame)
        result_buffer.append(result)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    _stop_worker.set()
    _worker_thread.join(timeout=2.0)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    ignore_before_frame = int(args.ignore_before_sec * fps) if args.ignore_before_sec is not None else 0
    ignore_after_frame  = int(args.ignore_after_sec  * fps) if args.ignore_after_sec  is not None else None
    reps = detect_reps(
        result_buffer,
        fps=fps,
        ignore_before_frame=ignore_before_frame,
        ignore_after_frame=ignore_after_frame,
    )

    if args.review_annotation and reps:
        reps = review_annotation(reps, fps, total_frames=frame_idx)

    if args.save_frames and reps:
        if args.input is not None:
            cap2 = cv2.VideoCapture(args.input)
            for n, rep in enumerate(reps, 1):
                vf = rep["valley_frame"]
                cap2.set(cv2.CAP_PROP_POS_FRAMES, vf)
                ret, valley_orig = cap2.read()
                if not ret:
                    print(f"警告: rep{n} valley_frame={vf} を読み込めませんでした", file=sys.stderr)
                    continue
                _save_valley_frame(n, vf, valley_orig, rep, result_buffer, classifier)
            cap2.release()
        else:
            for n, rep in enumerate(reps, 1):
                vf = rep["valley_frame"]
                valley_orig = frame_candidates.get(vf)
                if valley_orig is None:
                    print(f"警告: rep{n} valley_frame={vf} はバッファに存在しません", file=sys.stderr)
                    continue
                _save_valley_frame(n, vf, valley_orig, rep, result_buffer, classifier)

    video_id = "webcam" if args.input is None else os.path.splitext(os.path.basename(args.input))[0]

    summary = build_summary(reps, result_buffer, args.style, classifier)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    with open("summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("サマリ保存: summary.json")

    if args.export_annotation:
        export_annotation_csv(reps, result_buffer, args.style, video_id, classifier)
        print("アノテーション候補 CSV 保存: annotation_candidates.csv")
        if reps:
            save_feature_sequences(reps, result_buffer, video_id)


if __name__ == "__main__":
    main()
