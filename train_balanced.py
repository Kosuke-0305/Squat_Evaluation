"""有効/無効を 1:1 にダウンサンプリングして LSTM 学習を行うラッパースクリプト。

使用例:
    python train_balanced.py --data-dir training_data --epochs 100 --output model_balanced.pt
    python train_balanced.py --data-dir training_data --seed 42 --epochs 100
"""

import argparse
import csv
import os
import random
import subprocess
import sys


def collect_pairs(data_dir: str) -> tuple[list[str], list[str]]:
    """(base_id, valid_flag) のリストを返す。seq/anno ペアが揃いかつ valid 記入済みのみ。"""
    valid_bases, invalid_bases = [], []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.startswith("anno_") or not fname.endswith(".csv"):
            continue
        base = fname[5:-4]
        if not os.path.exists(os.path.join(data_dir, f"seq_{base}.json")):
            continue
        with open(os.path.join(data_dir, fname), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        v = rows[0].get("valid", "").strip()
        if not v:
            continue
        if int(float(v)) >= 1:
            valid_bases.append(base)
        else:
            invalid_bases.append(base)
    return valid_bases, invalid_bases


def main() -> None:
    p = argparse.ArgumentParser(description="バランス調整付き LSTM 学習")
    p.add_argument("--data-dir",   default="training_data", help="training_data ディレクトリ")
    p.add_argument("--seed",       type=int, default=0,    help="乱数シード (default: 0)")
    p.add_argument("--output",     default="model_balanced.pt", help="モデル保存先")
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--hidden-size",type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--val-ratio",  type=float, default=0.2)
    args = p.parse_args()

    valid_bases, invalid_bases = collect_pairs(args.data_dir)
    n = min(len(valid_bases), len(invalid_bases))

    rng = random.Random(args.seed)
    sampled_valid   = rng.sample(valid_bases,   n)
    sampled_invalid = rng.sample(invalid_bases, n)
    selected = sorted(sampled_valid + sampled_invalid)

    print(f"有効試技: {len(valid_bases)} → {n} にダウンサンプル")
    print(f"無効試技: {len(invalid_bases)}")
    print(f"合計サンプル数: {len(selected)}")
    print()

    seqs  = [os.path.join(args.data_dir, f"seq_{b}.json")  for b in selected]
    annos = [os.path.join(args.data_dir, f"anno_{b}.csv") for b in selected]

    cmd = [
        sys.executable, "train.py",
        "--sequences",  *seqs,
        "--annotations", *annos,
        "--epochs",      str(args.epochs),
        "--batch-size",  str(args.batch_size),
        "--lr",          str(args.lr),
        "--hidden-size", str(args.hidden_size),
        "--num-layers",  str(args.num_layers),
        "--dropout",     str(args.dropout),
        "--val-ratio",   str(args.val_ratio),
        "--output",      args.output,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
