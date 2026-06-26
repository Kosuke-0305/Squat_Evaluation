"""有効/無効 2値分類に特化した LSTM 学習スクリプト。

- 出力は valid 1次元のみ（失敗スコアは無視）
- --balance フラグで有効/無効を 1:1 にダウンサンプリング
- valid 列が記入済みであれば全サンプルを損失に使用（マスク不要）

使用例:
    python train_binary.py --data-dir training_data --balance --epochs 100
"""

import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from model import SquatLSTM, FEATURE_KEYS, INPUT_DIM


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _load_samples(seq_path: str, csv_path: str) -> list[tuple]:
    """1ファイルペアから (x_arr, length, label, weight) のリストを返す。"""
    labels: dict[int, tuple[float, float]] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = row.get("valid", "").strip()
            if not v:
                continue
            try:
                label = 1.0 if float(v) >= 1.0 else 0.0
            except ValueError:
                continue
            u = row.get("unanimous", "").strip()
            weight = 0.5 if u == "0" else 1.0
            try:
                labels[int(row["rep_id"])] = (label, weight)
            except (KeyError, ValueError):
                continue

    samples = []
    with open(seq_path, encoding="utf-8") as f:
        data = json.load(f)
    for rep in data["reps"]:
        rep_id = rep["rep_id"]
        if rep_id not in labels:
            continue
        seq = [[frame.get(k, 0.0) for k in FEATURE_KEYS] for frame in rep["features"]]
        if not seq:
            continue
        x = np.array(seq, dtype=np.float32)
        label, weight = labels[rep_id]
        samples.append((x, len(x), label, weight))
    return samples


class BinarySquatDataset(Dataset):
    def __init__(self, seq_paths: list, csv_paths: list):
        self.samples: list = []
        for sp, cp in zip(seq_paths, csv_paths):
            self.samples.extend(_load_samples(sp, cp))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    xs, lengths, labels, weights = zip(*batch)
    max_len = max(lengths)
    x_padded = torch.zeros(len(xs), max_len, INPUT_DIM, dtype=torch.float32)
    for i, (x, l) in enumerate(zip(xs, lengths)):
        x_padded[i, :l] = torch.from_numpy(x)
    return (
        x_padded,
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(labels,  dtype=torch.float32).unsqueeze(1),  # (batch, 1)
        torch.tensor(weights, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_pairs(data_dir: str) -> tuple[list, list, list, list]:
    """有効/無効・元データ/拡張データに分類しファイルペアリストを返す。

    Returns:
        orig_valid, orig_invalid  : 元データ（val にも使う）
        aug_valid,  aug_invalid   : 拡張データ（train のみ）
    """
    orig_valid, orig_invalid = [], []
    aug_valid,  aug_invalid  = [], []

    for fname in sorted(os.listdir(data_dir)):
        if not fname.startswith("anno_") or not fname.endswith(".csv"):
            continue
        base = fname[5:-4]
        if not os.path.exists(os.path.join(data_dir, f"seq_{base}.json")):
            continue
        csv_path = os.path.join(data_dir, fname)
        with open(csv_path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        v = rows[0].get("valid", "").strip()
        if not v:
            continue
        is_aug   = base.startswith("aug")
        is_valid = int(float(v)) >= 1
        if is_aug:
            (aug_valid if is_valid else aug_invalid).append(base)
        else:
            (orig_valid if is_valid else orig_invalid).append(base)

    return orig_valid, orig_invalid, aug_valid, aug_invalid


def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    tp = fp = fn = 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for x, lengths, y, weights in loader:
            x, lengths, y, weights = x.to(device), lengths.to(device), y.to(device), weights.to(device)
            logits = model(x, lengths)          # (batch, 1)
            loss = (criterion(logits, y) * weights.unsqueeze(1)).mean()
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total   += y.numel()
            tp += (preds * y).sum().item()
            fp += (preds * (1 - y)).sum().item()
            fn += ((1 - preds) * y).sum().item()
            preds_all.append(preds.cpu())
            labels_all.append(y.cpu())

    accuracy  = correct / max(total, 1)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    metrics = {"accuracy": accuracy, "recall": recall, "f1": f1}
    try:
        from sklearn.metrics import roc_auc_score
        p_cat = torch.cat(preds_all).numpy().flatten()
        l_cat = torch.cat(labels_all).numpy().flatten()
        metrics["auc"] = float(roc_auc_score(l_cat, p_cat))
    except (ImportError, ValueError):
        pass

    return total_loss / max(len(loader), 1), metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="有効/無効 2値分類 LSTM 学習")
    p.add_argument("--data-dir",    default="training_data")
    p.add_argument("--balance",     action="store_true", help="有効/無効を 1:1 にダウンサンプル")
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--output",      default="model_binary.pt")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch-size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--hidden-size", type=int,   default=64)
    p.add_argument("--num-layers",  type=int,   default=2)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--val-ratio",   type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    orig_valid, orig_invalid, aug_valid, aug_invalid = collect_pairs(args.data_dir)

    if args.balance:
        n = min(len(orig_valid), len(orig_invalid))
        rng = random.Random(args.seed)
        orig_valid   = rng.sample(orig_valid,   n)
        orig_invalid = rng.sample(orig_invalid, n)

    print(f"元データ  — 有効: {len(orig_valid)}  無効: {len(orig_invalid)}")
    print(f"拡張データ — 有効: {len(aug_valid)}  無効: {len(aug_invalid)}")

    # val は元データのみから分割（データリーク防止）
    rng = random.Random(args.seed)
    def split_val(bases):
        k = max(1, int(len(bases) * args.val_ratio))
        val = rng.sample(bases, k)
        val_set = set(val)
        train = [b for b in bases if b not in val_set]
        return train, val

    train_valid_orig, val_valid   = split_val(orig_valid)
    train_invalid_orig, val_invalid = split_val(orig_invalid)

    train_bases = sorted(train_valid_orig + train_invalid_orig + aug_valid + aug_invalid)
    val_bases   = sorted(val_valid + val_invalid)

    def make_dataset(bases):
        seqs  = [os.path.join(args.data_dir, f"seq_{b}.json")  for b in bases]
        annos = [os.path.join(args.data_dir, f"anno_{b}.csv") for b in bases]
        return BinarySquatDataset(seqs, annos)

    train_ds = make_dataset(train_bases)
    val_ds   = make_dataset(val_bases)
    print(f"学習サンプル: {len(train_ds)}  検証サンプル（元データのみ）: {len(val_ds)}\n")

    if len(train_ds) == 0:
        print("ERROR: 学習サンプルがありません", file=sys.stderr)
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = SquatLSTM(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_labels=1,
    ).to(device)

    # クラス不均衡対策（train のみで算出）
    n_pos = sum(1 for s in train_ds.samples if s[2] == 1.0)
    n_neg = len(train_ds) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=8, factor=0.5)

    best_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, lengths, y, weights in train_loader:
            x, lengths, y, weights = x.to(device), lengths.to(device), y.to(device), weights.to(device)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = (criterion(logits, y) * weights.unsqueeze(1)).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)
        val_loss, metrics = _evaluate(model, val_loader, criterion, device)
        auc = metrics.get("auc", 0.0)
        scheduler.step(auc)

        saved = ""
        if auc > best_auc:
            best_auc = auc
            torch.save({
                "model_state_dict": model.state_dict(),
                "hidden_size":      args.hidden_size,
                "num_layers":       args.num_layers,
                "num_labels":       1,
                "max_seq_len":      None,
                "val_bases":        val_bases,  # 検証セット再現用
            }, args.output)
            saved = f"  -> saved {args.output}"

        metrics_str = "  ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  [{metrics_str}]{saved}")

    print(f"\n学習完了。最良モデル: {args.output}  (best AUC={best_auc:.4f})")


if __name__ == "__main__":
    main()
