"""スクワット有効試技判定 LSTM の学習スクリプト（可変長・多出力対応）。

GPU（CUDA）が使えれば自動選択し、なければ CPU にフォールバックする。

モデルは valid + 5 失敗要素スコアの 6 出力を持つ。
失敗スコアが未記入のレップは損失マスクで除外される。
unanimous=0 のレップは sample_weight=0.5 として損失を半分に割り引く。

使用例:
    # 単一動画のデータで学習
    python train.py --sequences feature_sequences.json --annotations annotation_candidates.csv

    # 複数動画のデータを結合して学習
    python train.py \\
        --sequences seq1.json seq2.json \\
        --annotations anno1.csv anno2.csv \\
        --epochs 100 --output model.pt
"""

import argparse
import csv as csv_module
import json
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from classifier import INVALID_MAJORITY
from model import SquatLSTM, FEATURE_KEYS, INPUT_DIM, LABEL_KEYS, OUTPUT_DIM
from dataset import SquatRepDataset, collate_fn

_SCORE_KEYS = [
    "depth_score", "lockout_score", "bar_descent_score",
    "bounce_score", "foot_shift_score",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="スクワット有効試技判定 LSTM 学習")
    p.add_argument("--sequences",    nargs="+", required=True, metavar="PATH",
                   help="feature_sequences.json のパス（複数可）")
    p.add_argument("--annotations",  nargs="+", required=True, metavar="PATH",
                   help="annotation_candidates.csv のパス（--sequences と順序対応）")
    p.add_argument("--output",       default="model.pt",  help="モデル保存先 (default: model.pt)")
    p.add_argument("--max-seq-len",  type=int,   default=None,
                   help="シーケンス最大長。超えた場合は末尾フレームを使用（default: 制限なし）")
    p.add_argument("--epochs",       type=int,   default=50,   help="エポック数 (default: 50)")
    p.add_argument("--batch-size",   type=int,   default=32,   help="バッチサイズ (default: 32)")
    p.add_argument("--lr",           type=float, default=1e-3, help="学習率 (default: 0.001)")
    p.add_argument("--hidden-size",  type=int,   default=64,   help="LSTM 隠れ層サイズ (default: 64)")
    p.add_argument("--num-layers",   type=int,   default=2,    help="LSTM 層数 (default: 2)")
    p.add_argument("--dropout",      type=float, default=0.3,  help="ドロップアウト率 (default: 0.3)")
    p.add_argument("--val-ratio",    type=float, default=0.2,  help="検証データ割合 (default: 0.2)")
    return p.parse_args()


def _compute_pos_weight(dataset: SquatRepDataset, device: torch.device) -> torch.Tensor:
    """クラス不均衡対策用の pos_weight を出力次元ごとに算出する。

    Returns:
        (OUTPUT_DIM,) FloatTensor — 各出力の pos_weight
    """
    ys = np.stack([dataset[i][2] for i in range(len(dataset))])  # (N, 6)
    mask = (ys >= 0)  # bool (N, 6)
    pos = np.where(mask, ys,       0.0).sum(0)  # (6,)
    neg = np.where(mask, 1.0 - ys, 0.0).sum(0)  # (6,)
    weight = np.where(pos > 0, neg / np.maximum(pos, 1e-6), 1.0)
    return torch.tensor(np.clip(weight, 0.0, 10.0), dtype=torch.float32).to(device)


def _evaluate(
    model: SquatLSTM,
    loader: DataLoader,
    criterion_none: nn.BCEWithLogitsLoss,
    device: torch.device,
) -> tuple[float, dict]:
    """検証ループ。有効/無効（index 0）を主指標として loss・accuracy・recall・F1 を返す。"""
    model.eval()
    total_loss    = 0.0
    total_mask_n  = 0.0
    all_preds:   list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            x, lengths, y, _ = (b.to(device) for b in batch)
            logits = model(x, lengths)                         # (batch, 6)
            mask   = (y >= 0).float()                          # (batch, 6)
            loss_per = criterion_none(logits, y.clamp(min=0))  # (batch, 6)
            total_loss   += (loss_per * mask).sum().item()
            total_mask_n += mask.sum().item()

            # valid（index 0）のみで精度指標を計算
            valid_mask = mask[:, 0].bool()
            if valid_mask.any():
                preds   = (torch.sigmoid(logits[valid_mask, 0]) > 0.5).float().cpu()
                targets = y[valid_mask, 0].cpu()
                all_preds.append(preds)
                all_targets.append(targets)

    avg_loss = total_loss / max(total_mask_n, 1.0)

    if not all_preds:
        return avg_loss, {"accuracy": 0.0, "f1": 0.0}

    preds_cat   = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)

    accuracy  = float((preds_cat == targets_cat).float().mean())
    tp        = float((preds_cat * targets_cat).sum())
    fp        = float((preds_cat * (1 - targets_cat)).sum())
    fn        = float(((1 - preds_cat) * targets_cat).sum())
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    metrics = {"accuracy": accuracy, "recall": recall, "f1": f1}
    try:
        from sklearn.metrics import roc_auc_score
        metrics["auc"] = float(roc_auc_score(
            targets_cat.numpy().flatten(), preds_cat.numpy().flatten()
        ))
    except (ImportError, ValueError):
        pass

    return avg_loss, metrics


def _evaluate_failure_recall(
    model: SquatLSTM,
    seq_paths: list,
    csv_paths: list,
    device: torch.device,
    max_seq_len: int | None = None,
) -> None:
    """失敗要素ごとの recall と際どい試技の精度を表示する。

    annotation_candidates.csv のスコア列が記入済みのレップのみ対象とする。
    """
    reps_data: list = []  # [(x_arr, length, flags)]

    for sp, cp in zip(seq_paths, csv_paths):
        flags_by_rep: dict = {}
        with open(cp, encoding="utf-8") as f:
            for row in csv_module.DictReader(f):
                try:
                    rep_id = int(row["rep_id"])
                except (KeyError, ValueError):
                    continue
                flags: dict = {}
                for col in _SCORE_KEYS:
                    val = row.get(col, "").strip()
                    flags[col] = int(val) if val.lstrip("-").isdigit() else None

                v_str = row.get("valid", "").strip()
                flags["_valid"] = int(v_str) if v_str.isdigit() else None

                u_str = row.get("unanimous", "").strip()
                flags["_unanimous"] = (u_str != "0") if u_str else True

                flags_by_rep[rep_id] = flags

        with open(sp, encoding="utf-8") as f:
            data = json.load(f)

        for rep in data["reps"]:
            rep_id = rep["rep_id"]
            if rep_id not in flags_by_rep:
                continue
            seq = [[frame.get(k, 0.0) for k in FEATURE_KEYS] for frame in rep["features"]]
            if not seq:
                continue
            arr = np.array(seq, dtype=np.float32)
            if max_seq_len is not None and len(arr) > max_seq_len:
                arr = arr[-max_seq_len:]
            reps_data.append((arr, len(arr), flags_by_rep[rep_id]))

    if not reps_data:
        return

    model.eval()
    print(f"\n--- 失敗要素別 recall（スコア >= {INVALID_MAJORITY} を失敗ありとみなした割合）---")
    with torch.no_grad():
        for col in _SCORE_KEYS:
            subset = [(arr, l, flags) for arr, l, flags in reps_data
                      if (flags.get(col) or 0) >= INVALID_MAJORITY]
            if not subset:
                continue
            correct = 0
            for arr, l, _ in subset:
                x       = torch.from_numpy(arr).unsqueeze(0).to(device)
                lengths = torch.tensor([l], dtype=torch.long)
                logits  = model(x, lengths)                    # (1, 6)
                if torch.sigmoid(logits[0, 0]).item() <= 0.5:  # invalid と判定
                    correct += 1
            print(f"  {col:<22} recall: {correct / len(subset):.3f}  (n={len(subset)})")

        # 際どい試技（unanimous=False）の精度
        borderline = [
            (arr, l, flags) for arr, l, flags in reps_data
            if not flags.get("_unanimous", True) and flags.get("_valid") is not None
        ]
        if borderline:
            correct = 0
            for arr, l, flags in borderline:
                x       = torch.from_numpy(arr).unsqueeze(0).to(device)
                lengths = torch.tensor([l], dtype=torch.long)
                logits  = model(x, lengths)
                pred    = int(torch.sigmoid(logits[0, 0]).item() > 0.5)
                if pred == flags["_valid"]:
                    correct += 1
            print(f"  際どい試技の精度: {correct / len(borderline):.3f}  (n={len(borderline)})")


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    dataset = SquatRepDataset(
        args.sequences, args.annotations, max_seq_len=args.max_seq_len
    )
    if len(dataset) == 0:
        print("ERROR: 有効なサンプルがありません。valid 列が記入済みの CSV と JSON を確認してください。",
              file=sys.stderr)
        sys.exit(1)
    print(f"総サンプル数: {len(dataset)}")

    val_size   = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    if train_size < 1:
        print("ERROR: 学習サンプルが不足しています（最低 2 サンプル必要）。", file=sys.stderr)
        sys.exit(1)
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"  学習: {train_size}  検証: {val_size}")

    # 可変長バッチ化には collate_fn が必須
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              drop_last=False, collate_fn=collate_fn)

    model = SquatLSTM(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_labels=OUTPUT_DIM,
    ).to(device)

    pos_weight    = _compute_pos_weight(dataset, device)           # (6,)
    criterion_none = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    optimizer      = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler      = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    print()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss   = 0.0
        train_mask_n = 0.0

        for x, lengths, y, weights in train_loader:
            x       = x.to(device)
            lengths = lengths.to(device)
            y       = y.to(device)
            weights = weights.to(device)

            optimizer.zero_grad()
            logits   = model(x, lengths)                         # (batch, 6)
            mask     = (y >= 0).float()                          # (batch, 6)
            loss_per = criterion_none(logits, y.clamp(min=0))    # (batch, 6)
            # 重み付き損失: sample_weight × マスク済み BCE の平均
            loss = (loss_per * mask * weights.unsqueeze(1)).sum() / mask.sum().clamp(min=1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss   += (loss_per * mask).sum().item()
            train_mask_n += mask.sum().item()

        train_loss /= max(train_mask_n, 1.0)

        val_loss, val_metrics = _evaluate(model, val_loader, criterion_none, device)
        scheduler.step(val_loss)

        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "hidden_size":      args.hidden_size,
                "num_layers":       args.num_layers,
                "num_labels":       OUTPUT_DIM,
                "max_seq_len":      args.max_seq_len,
            }, args.output)
            saved = f"  -> saved {args.output}"

        metrics_str = "  ".join(f"{k}={v:.3f}" for k, v in val_metrics.items())
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  [{metrics_str}]{saved}")

    print(f"\n学習完了。最良モデル: {args.output}  (best val_loss={best_val_loss:.4f})")

    _evaluate_failure_recall(
        model, args.sequences, args.annotations, device, args.max_seq_len
    )


if __name__ == "__main__":
    main()
