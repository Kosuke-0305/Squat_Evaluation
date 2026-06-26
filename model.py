"""スクワット有効試技判定 LSTM モデル定義。

可変長シーケンスを pack_padded_sequence で処理し、
有効/無効の 2 値分類（デフォルト）または任意次元の出力を返す。
"""

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

FEATURE_KEYS = [
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_shoulder_y_delta",
    "right_shoulder_y_delta",
    "left_ankle_x_delta",
    "right_ankle_x_delta",
    "lr_knee_diff",
    "left_visibility",
    "right_visibility",
]

INPUT_DIM = len(FEATURE_KEYS)  # 11


class SquatLSTM(nn.Module):
    """スクワット有効試技判定 LSTM 分類器。

    入力: (batch, T, input_size) + lengths (batch,) の可変長シーケンス
    出力: (batch, num_labels) の生ロジット（sigmoid 前）
    """

    def __init__(
        self,
        input_size: int = INPUT_DIM,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_labels: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, num_labels)

    def forward(self, x, lengths=None):
        if lengths is not None:
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])  # (batch, num_labels)
