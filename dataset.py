"""LSTM 学習用データセット。

feature_sequences.json（main.py --export-annotation で生成）と
annotation_candidates.csv を組み合わせてシーケンスデータを構築する。
"""

import csv
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from model import FEATURE_KEYS, INPUT_DIM


def pad_or_trim(seq: list, seq_len: int) -> np.ndarray:
    """シーケンスを seq_len フレームに揃える。

    短ければ先頭にゼロパディング、長ければ末尾 seq_len フレームを使用。
    """
    arr = np.array(seq, dtype=np.float32)
    if len(arr) == 0:
        return np.zeros((seq_len, INPUT_DIM), dtype=np.float32)
    if len(arr) >= seq_len:
        return arr[-seq_len:]
    pad = np.zeros((seq_len - len(arr), INPUT_DIM), dtype=np.float32)
    return np.concatenate([pad, arr], axis=0)


class SquatRepDataset(Dataset):
    """レップ単位の特徴量シーケンスとラベルを返すデータセット。

    使用するファイル:
        feature_sequences.json — main.py --export-annotation で生成されるフレーム単位の特徴量
        annotation_candidates.csv — 同じく --export-annotation で生成されるレップ単位のラベル

    複数の動画ファイルのデータを結合する場合は、それぞれのパスをリストで渡す。
    """

    def __init__(self, seq_paths: list, csv_paths: list, seq_len: int = 90):
        """
        Args:
            seq_paths: feature_sequences.json のパスリスト
            csv_paths: annotation_candidates.csv のパスリスト（seq_paths と順序対応）
            seq_len:   固定シーケンス長（フレーム数）
        """
        if len(seq_paths) != len(csv_paths):
            raise ValueError("seq_paths と csv_paths の数が一致しません")
        self.seq_len = seq_len
        self.samples: list = []
        for sp, cp in zip(seq_paths, csv_paths):
            self._load_one(sp, cp)

    def _load_one(self, seq_path: str, csv_path: str) -> None:
        # CSV からレップごとのラベルを読み込む
        labels_by_rep: dict = {}
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                labels_by_rep[int(row["rep_id"])] = [
                    int(row["knee_forward_flag"]),
                    int(row["trunk_lean_flag"]),
                    int(row["back_round_flag"]),
                ]

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
            x = pad_or_trim(seq, self.seq_len)
            y = np.array(labels_by_rep[rep_id], dtype=np.float32)
            self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)
