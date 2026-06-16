"""LSTM 学習用データセット（可変長対応）。

feature_sequences.json（main.py --export-annotation で生成）と
annotation_candidates.csv を組み合わせてシーケンスデータを構築する。

各レップのシーケンスは実際の長さのまま保持し、バッチ化時に collate_fn で
動的パディングを行う（pack_padded_sequence 対応）。

出力ラベルは 6 次元:
    [valid, depth_score, lockout_score, bar_descent_score, bounce_score, foot_shift_score]
失敗スコアが未記入のレップは -1 を格納し、学習時に損失マスクで除外する。
"""

import csv
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from classifier import INVALID_MAJORITY
from model import FEATURE_KEYS, INPUT_DIM, LABEL_KEYS

# LABEL_KEYS の順序に対応するCSV列名（index 0 = valid は別途読み込み）
_SCORE_CSV_COLS = [
    "depth_score",
    "lockout_score",
    "bar_descent_score",
    "bounce_score",
    "foot_shift_score",
]


def collate_fn(batch):
    """可変長シーケンスをパディングしてバッチ化する。

    DataLoader の collate_fn として使用する。

    Args:
        batch: list of (x_arr, length, y_arr, weight)
            x_arr:  (T_i, 11) numpy array（パディングなし）
            length: T_i (int)
            y_arr:  (6,) numpy array (float32)、未記入は -1.0
            weight: float

    Returns:
        x_padded: (batch, T_max, 11) FloatTensor
        lengths:  (batch,) LongTensor
        y:        (batch, 6) FloatTensor
        weights:  (batch,) FloatTensor
    """
    xs, lengths, ys, weights = zip(*batch)

    max_len = max(lengths)
    x_padded = torch.zeros(len(xs), max_len, INPUT_DIM, dtype=torch.float32)
    for i, (x, l) in enumerate(zip(xs, lengths)):
        x_padded[i, :l] = torch.from_numpy(x)

    return (
        x_padded,
        torch.tensor(lengths, dtype=torch.long),
        torch.stack([torch.from_numpy(y) for y in ys]),
        torch.tensor(weights, dtype=torch.float32),
    )


class SquatRepDataset(Dataset):
    """レップ単位の特徴量シーケンスとラベルを返すデータセット（可変長対応）。

    使用するファイル:
        feature_sequences.json — main.py --export-annotation で生成されるフレーム単位の特徴量
        annotation_candidates.csv — 同じく --export-annotation で生成されるレップ単位のラベル

    複数の動画ファイルのデータを結合する場合は、それぞれのパスをリストで渡す。

    __getitem__ は (x_arr, length, y_arr, weight) の 4 要素タプルを返す。
    DataLoader には collate_fn を必ず指定すること。
    """

    def __init__(
        self,
        seq_paths: list,
        csv_paths: list,
        max_seq_len: int | None = None,
    ):
        """
        Args:
            seq_paths:   feature_sequences.json のパスリスト
            csv_paths:   annotation_candidates.csv のパスリスト（seq_paths と順序対応）
            max_seq_len: シーケンス最大長。超えた場合は末尾 max_seq_len フレームを使用。
                         None の場合は制限なし。
        """
        if len(seq_paths) != len(csv_paths):
            raise ValueError("seq_paths と csv_paths の数が一致しません")
        self.max_seq_len = max_seq_len
        self.samples: list = []  # [(x_arr, length, y_arr, weight), ...]
        for sp, cp in zip(seq_paths, csv_paths):
            self._load_one(sp, cp)

    def _load_one(self, seq_path: str, csv_path: str) -> None:
        # CSV からレップごとのラベルを読み込む（valid 空欄はスキップ）
        # valid: 1=有効試技 / 0=無効試技（二値）
        # 失敗スコア: INVALID_MAJORITY 以上 → 1.0, 未満 → 0.0, 空欄 → -1.0（マスク）
        # unanimous: 空欄または "1" → weight=1.0, "0" → weight=0.5
        labels_by_rep: dict = {}  # rep_id -> (y_arr, weight)
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                valid_str = row.get("valid", "").strip()
                if not valid_str:
                    continue
                try:
                    v = float(valid_str)
                    y_valid = 1.0 if v >= 1.0 else 0.0
                except ValueError:
                    continue

                # 失敗スコア列（未記入は -1 でマスク）
                y_scores = []
                for col in _SCORE_CSV_COLS:
                    s = row.get(col, "").strip()
                    if s and s.lstrip("-").isdigit():
                        score = int(s)
                        y_scores.append(1.0 if score >= INVALID_MAJORITY else 0.0)
                    else:
                        y_scores.append(-1.0)  # 未アノテーション → 損失マスクで除外

                u_str = row.get("unanimous", "").strip()
                weight = 0.5 if u_str == "0" else 1.0

                try:
                    rep_id = int(row["rep_id"])
                except (ValueError, KeyError):
                    continue

                y_arr = np.array([y_valid] + y_scores, dtype=np.float32)  # (6,)
                labels_by_rep[rep_id] = (y_arr, weight)

        # feature_sequences.json を読み込む
        with open(seq_path, encoding="utf-8") as f:
            data = json.load(f)

        for rep in data["reps"]:
            rep_id = rep["rep_id"]
            if rep_id not in labels_by_rep:
                continue
            seq = [[frame[k] for k in FEATURE_KEYS] for frame in rep["features"]]
            if not seq:
                continue

            x = np.array(seq, dtype=np.float32)  # (T, 11) — パディングなし
            if self.max_seq_len is not None and len(x) > self.max_seq_len:
                x = x[-self.max_seq_len:]  # 末尾を使用
            length = len(x)

            y_arr, weight = labels_by_rep[rep_id]
            self.samples.append((x, length, y_arr, float(weight)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, length, y, w = self.samples[idx]
        # x は numpy array のまま返す（collate_fn でパディング）
        return x, length, y, w
