"""スクワット有効試技判定 LSTM モデル定義。

可変長シーケンスを pack_padded_sequence で処理し、
valid + 5 失敗要素スコアの 6 出力を返す多出力設計。
"""

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

# extract_form_features が返す11キーと順序を固定する
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

# 出力ラベルの順序（インデックス 0 = valid、1〜5 = 失敗要素スコア）
LABEL_KEYS = [
    "valid",
    "depth_score",
    "lockout_score",
    "bar_descent_score",
    "bounce_score",
    "foot_shift_score",
]

INPUT_DIM  = len(FEATURE_KEYS)  # 11
OUTPUT_DIM = len(LABEL_KEYS)    # 6


class SquatLSTM(nn.Module):
    """スクワット有効試技判定 LSTM 分類器。

    入力: (batch, T, input_size) + lengths (batch,) の可変長シーケンス
    出力: (batch, num_labels) の生ロジット（sigmoid 前）

    可変長対応:
        forward(x, lengths) を呼ぶと pack_padded_sequence で正確な最終隠れ状態を取得する。
        lengths=None のときは固定長入力として扱う（後方互換）。
    """

    def __init__(
        self,
        input_size: int = INPUT_DIM,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_labels: int = OUTPUT_DIM,
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
        """
        Args:
            x:       (batch, T, input_size) パディング済み（または固定長）シーケンス
            lengths: (batch,) 各サンプルの実際の長さ（LongTensor）。
                     None の場合は固定長入力として扱う（後方互換）。

        Returns:
            logits: (batch, num_labels) — BCEWithLogitsLoss に渡す前の生スコア
        """
        if lengths is not None:
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, batch, hidden_size) — 最終層の最終タイムステップ隠れ状態
        return self.head(h_n[-1])  # (batch, num_labels)
