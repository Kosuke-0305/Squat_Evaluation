"""スクワットフォーム LSTM の学習スクリプト。

GPU（CUDA）が使えれば自動選択し、なければ CPU にフォールバックする。

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
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model import SquatLSTM, LABEL_KEYS
from dataset import SquatRepDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="スクワットフォーム LSTM 学習")
    p.add_argument("--sequences",   nargs="+", required=True, metavar="PATH",
                   help="feature_sequences.json のパス（複数可）")
    p.add_argument("--annotations", nargs="+", required=True, metavar="PATH",
                   help="annotation_candidates.csv のパス（--sequences と順序対応）")
    p.add_argument("--output",      default="model.pt",  help="モデル保存先 (default: model.pt)")
    p.add_argument("--seq-len",     type=int,   default=90,   help="シーケンス長フレーム数 (default: 90)")
    p.add_argument("--epochs",      type=int,   default=50,   help="エポック数 (default: 50)")
    p.add_argument("--batch-size",  type=int,   default=32,   help="バッチサイズ (default: 32)")
    p.add_argument("--lr",          type=float, default=1e-3, help="学習率 (default: 0.001)")
    p.add_argument("--hidden-size", type=int,   default=64,   help="LSTM 隠れ層サイズ (default: 64)")
    p.add_argument("--num-layers",  type=int,   default=2,    help="LSTM 層数 (default: 2)")
    p.add_argument("--dropout",     type=float, default=0.3,  help="ドロップアウト率 (default: 0.3)")
    p.add_argument("--val-ratio",   type=float, default=0.2,  help="検証データ割合 (default: 0.2)")
    return p.parse_args()


def _compute_pos_weight(dataset: SquatRepDataset, device: torch.device) -> torch.Tensor:
    """クラス不均衡対策用の pos_weight を算出する。

    崩れあり（正例）の数が少ないとき、損失関数で正例を重くする。
    """
    all_y = torch.stack([dataset[i][1] for i in range(len(dataset))])
    pos = all_y.sum(0)
    neg = len(dataset) - pos
    return (neg / (pos + 1e-6)).clamp(max=10.0).to(device)


def _evaluate(model: SquatLSTM, loader: DataLoader,
              criterion, device: torch.device) -> tuple[float, dict]:
    """検証ループ。loss と各ラベルの accuracy を返す。"""
    model.eval()
    total_loss = 0.0
    correct = torch.zeros(len(LABEL_KEYS))
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * len(x)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).float().sum(0).cpu()
            n += len(x)
    acc = {k: float(correct[i] / n) for i, k in enumerate(LABEL_KEYS)}
    return total_loss / max(n, 1), acc


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # データセット構築
    dataset = SquatRepDataset(args.sequences, args.annotations, seq_len=args.seq_len)
    if len(dataset) == 0:
        print("ERROR: 有効なサンプルがありません。CSV と JSON の内容を確認してください。",
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)

    # モデル・損失・最適化
    model = SquatLSTM(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    pos_weight = _compute_pos_weight(dataset, device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    print()

    for epoch in range(1, args.epochs + 1):
        # 学習
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= train_size

        # 検証
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "hidden_size":      args.hidden_size,
                "num_layers":       args.num_layers,
                "seq_len":          args.seq_len,
            }, args.output)
            saved = f"  -> saved {args.output}"

        acc_str = "  ".join(f"{k}={v:.2f}" for k, v in val_acc.items())
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  [{acc_str}]{saved}")

    print(f"\n学習完了。最良モデル: {args.output}  (best val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    main()
