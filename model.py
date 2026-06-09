"""スクワットフォーム分類 LSTM モデル定義。"""

import torch.nn as nn

# extractor.py の extract_form_features が返す12キーと順序を固定する
FEATURE_KEYS = [
    "knee_forward_ratio",
    "trunk_lean_delta",
    "back_round_ratio",
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_ankle_angle",
    "right_ankle_angle",
    "lr_knee_diff",
    "left_visibility",
    "right_visibility",
]

LABEL_KEYS = ["knee_forward", "trunk_lean", "back_round"]

INPUT_DIM  = len(FEATURE_KEYS)   # 12
OUTPUT_DIM = len(LABEL_KEYS)     # 3


class SquatLSTM(nn.Module):
    """スクワットフォーム崩れのマルチラベル LSTM 分類器。

    入力: (batch, seq_len, INPUT_DIM) の特徴量シーケンス
    出力: (batch, OUTPUT_DIM) の生ロジット（sigmoid 前）
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            INPUT_DIM,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, OUTPUT_DIM)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, INPUT_DIM)
        Returns:
            logits: (batch, OUTPUT_DIM) — BCEWithLogitsLoss に渡す前の生スコア
        """
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # 最終タイムステップの出力を使用
