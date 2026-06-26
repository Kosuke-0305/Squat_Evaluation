"""5-fold 層化交差検証で LSTM 二値分類器の AUC を推定するスクリプト。

- 元データのみで fold 分割（データリーク防止）
- 拡張データは各 fold の train にのみ追加
- モデル保存基準は val AUC（pos_weight による損失歪みを回避）

使用例:
    python cv_binary.py --data-dir training_data --epochs 80
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from model import SquatLSTM, INPUT_DIM
from train_binary import BinarySquatDataset, collate_fn, collect_pairs, _evaluate


def parse_args():
    p = argparse.ArgumentParser(description="5-fold CV AUC 評価")
    p.add_argument("--data-dir",    default="training_data")
    p.add_argument("--epochs",      type=int,   default=80)
    p.add_argument("--batch-size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--hidden-size", type=int,   default=64)
    p.add_argument("--num-layers",  type=int,   default=2)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--n-splits",    type=int,   default=5)
    p.add_argument("--seed",        type=int,   default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    orig_valid, orig_invalid, aug_valid, aug_invalid = collect_pairs(args.data_dir)
    print(f"元データ  — 有効: {len(orig_valid)}  無効: {len(orig_invalid)}")
    print(f"拡張データ — 有効: {len(aug_valid)}  無効: {len(aug_invalid)}\n")

    orig_all = orig_valid + orig_invalid
    labels   = [1] * len(orig_valid) + [0] * len(orig_invalid)

    skf  = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(orig_all, labels), 1):
        train_orig = [orig_all[i] for i in train_idx]
        val_bases  = [orig_all[i] for i in val_idx]
        train_bases = sorted(train_orig + aug_valid + aug_invalid)

        def make_ds(bases):
            seqs  = [os.path.join(args.data_dir, f"seq_{b}.json")  for b in bases]
            annos = [os.path.join(args.data_dir, f"anno_{b}.csv") for b in bases]
            return BinarySquatDataset(seqs, annos)

        train_ds = make_ds(train_bases)
        val_ds   = make_ds(val_bases)

        n_pos      = sum(1 for s in train_ds.samples if s[2] == 1.0)
        n_neg      = len(train_ds) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
        criterion  = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        model = SquatLSTM(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_labels=1,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=8, factor=0.5
        )

        best_auc = 0.0
        n_inv_val = sum(1 for b in val_bases if b in set(orig_invalid))

        print(f"--- Fold {fold}/{args.n_splits}  train={len(train_ds)}  val={len(val_ds)} (無効={n_inv_val}) ---")

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            for x, lengths, y, weights in train_loader:
                x, lengths, y, weights = (t.to(device) for t in (x, lengths, y, weights))
                optimizer.zero_grad()
                logits = model(x, lengths)
                loss   = (criterion(logits, y) * weights.unsqueeze(1)).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            _, metrics = _evaluate(model, val_loader, criterion, device)
            auc = metrics.get("auc", 0.0)
            scheduler.step(auc)

            if auc > best_auc:
                best_auc = auc

            if epoch % 10 == 0 or epoch == args.epochs:
                print(f"  Epoch {epoch:3d}/{args.epochs}  "
                      f"train={train_loss/len(train_loader):.4f}  "
                      f"auc={auc:.3f}  best={best_auc:.3f}")

        aucs.append(best_auc)
        print(f"  → Fold {fold} best AUC: {best_auc:.3f}\n")

    print("=" * 40)
    print(f"5-fold CV AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    for i, a in enumerate(aucs, 1):
        print(f"  Fold {i}: {a:.3f}")


if __name__ == "__main__":
    main()
