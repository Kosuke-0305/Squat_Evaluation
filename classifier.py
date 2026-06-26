"""スクワット有効試技判定モジュール。

ルールベース（デフォルト）と LSTM モデルの 2 モードを
`create_classifier()` ファクトリで切り替えられる。

各失敗要素は審判3人のうち何人が抵触と判断するかを 0〜3 で表現する。
INVALID_MAJORITY 以上のスコアを持つ要素が1つでもあれば無効試技と判定する。
"""

import collections

THRESHOLDS = {
    "depth":        90.0,   # 代表膝角度がこの値以下なら深さ達成（度）
    "lockout":     165.0,   # 代表膝角度がこの値以上でロックアウト達成（度）
    "bar_descent":   0.03,  # 肩 Y delta の前フレーム比増加でバー下降検出（正規化座標）
    "bounce_delta": 10.0,   # 反復動作検出用の角度変化閾値（度）
    "foot_shift":    0.05,  # 足首 X delta がこの値以上でずれ検出（正規化座標）
}

# 有効試技の判定閾値（何人以上が抵触と判断したら無効か）
INVALID_MAJORITY = 2  # 2人以上 = 過半数

_EMPTY_RESULT = {
    "valid":             True,
    "unanimous":         True,
    "depth_score":       0,
    "lockout_score":     0,
    "bar_descent_score": 0,
    "bounce_score":      0,
    "foot_shift_score":  0,
}


def _repr_knee(f: dict) -> float:
    """代表膝角度を返す（左右平均、片側 0 の場合は有効側のみ）。"""
    l = f.get("left_knee_angle",  0.0)
    r = f.get("right_knee_angle", 0.0)
    if l > 0 and r > 0:
        return (l + r) / 2.0
    return max(l, r)


# ---------------------------------------------------------------------------
# シーケンス単位判定（モジュールレベル関数 API）
# ---------------------------------------------------------------------------

def classify_attempt(
    features_seq: list[dict],
    valley_frame_idx: int | None = None,
) -> dict:
    """1レップ分の特徴量シーケンスから有効試技かどうかを判定する。

    Args:
        features_seq:     extract_form_features が返す dict のリスト
        valley_frame_idx: シーケンス内の最深点フレームのインデックス。
                          None の場合はシーケンス内で自動検出する。

    Returns:
        {valid, unanimous, depth_score, lockout_score, bar_descent_score,
         bounce_score, foot_shift_score}

        各スコアはルールベースでは 0（問題なし）か 3（全員抵触）の二択。
        人間アノテーション時は 0〜3 の値をとる。
    """
    if not features_seq:
        return dict(_EMPTY_RESULT)

    n = len(features_seq)
    knee_angles = [_repr_knee(f) for f in features_seq]

    # 最深点フレームを特定
    if valley_frame_idx is None or not (0 <= valley_frame_idx < n):
        valley_idx = int(min(range(n), key=lambda i: knee_angles[i]))
    else:
        valley_idx = valley_frame_idx

    # ---- depth_score ----
    depth_score = 3 if not any(ka <= THRESHOLDS["depth"] for ka in knee_angles) else 0

    # ---- lockout_score ----
    # 先頭3フレームと末尾3フレームのいずれかが lockout 閾値を達成しているかチェック
    start_frames = knee_angles[:min(3, n)]
    end_frames   = knee_angles[max(0, n - 3):]
    lockout_ok   = (
        any(ka >= THRESHOLDS["lockout"] for ka in start_frames) and
        any(ka >= THRESHOLDS["lockout"] for ka in end_frames)
    )
    lockout_score = 0 if lockout_ok else 3

    # ---- bar_descent_score: 上昇フェーズ（最深点以降）で肩が前フレームより下降 ----
    ascending = features_seq[valley_idx:]
    bar_descent_score = 0
    for i in range(1, len(ascending)):
        dl = (ascending[i].get("left_shoulder_y_delta",  0.0)
              - ascending[i - 1].get("left_shoulder_y_delta",  0.0))
        dr = (ascending[i].get("right_shoulder_y_delta", 0.0)
              - ascending[i - 1].get("right_shoulder_y_delta", 0.0))
        if dl >= THRESHOLDS["bar_descent"] or dr >= THRESHOLDS["bar_descent"]:
            bar_descent_score = 3
            break

    # ---- bounce_score: 下降フェーズで股関節角度の方向逆転が2回以上 ----
    bounce_score = 0
    hip_seq = [
        (f.get("left_hip_angle", 0.0) + f.get("right_hip_angle", 0.0)) / 2.0
        for f in features_seq[: valley_idx + 1]
    ]
    if len(hip_seq) >= 3:
        reversals = 0
        for i in range(1, len(hip_seq) - 1):
            prev_d = hip_seq[i]     - hip_seq[i - 1]
            next_d = hip_seq[i + 1] - hip_seq[i]
            if abs(prev_d) >= THRESHOLDS["bounce_delta"] and prev_d * next_d < 0:
                reversals += 1
        if reversals >= 2:
            bounce_score = 3

    # ---- foot_shift_score ----
    foot_shift_score = 3 if any(
        abs(f.get("left_ankle_x_delta",  0.0)) >= THRESHOLDS["foot_shift"] or
        abs(f.get("right_ankle_x_delta", 0.0)) >= THRESHOLDS["foot_shift"]
        for f in features_seq
    ) else 0

    scores = [depth_score, lockout_score, bar_descent_score, bounce_score, foot_shift_score]
    valid     = not any(s >= INVALID_MAJORITY for s in scores)
    # ルールベースでは全スコアが 0 か 3 のみなので常に unanimous=True
    unanimous = all(s in (0, 3) for s in scores)

    return {
        "valid":             valid,
        "unanimous":         unanimous,
        "depth_score":       depth_score,
        "lockout_score":     lockout_score,
        "bar_descent_score": bar_descent_score,
        "bounce_score":      bounce_score,
        "foot_shift_score":  foot_shift_score,
    }


def classify_attempt_lstm(
    features_seq: list[dict],
    model,
    seq_len: int | None = None,
    valley_frame_idx: int | None = None,
) -> dict:
    """学習済み LSTM モデルで有効試技を判定する（可変長・多出力対応）。

    モデルが 6 出力（valid + 5 失敗スコア）の場合:
        - valid: LSTM 出力（index 0）
        - 失敗スコア: ルールベースを基準に LSTM の予測で補完
          * ルールベース=3 → 3 を維持（ルールは確定的）
          * ルールベース=0 かつ LSTM が失敗予測 → 2（LSTM が検出した際どい失敗）
    モデルが 1 出力（旧形式 checkpoint）の場合:
        - valid のみ LSTM 出力、失敗スコアはルールベースのまま

    Args:
        features_seq:     extract_form_features が返す dict のリスト
        model:            SquatLSTM インスタンス（eval 済み）
        seq_len:          最大シーケンス長（超えた場合末尾を使用）。None で制限なし。
        valley_frame_idx: 最深点フレームのインデックス（None なら自動検出）

    Returns:
        classify_attempt() と同じ構造の dict
    """
    import numpy as np
    import torch
    from model import FEATURE_KEYS

    result = classify_attempt(features_seq, valley_frame_idx)

    seq = [[float(f.get(k, 0.0)) for k in FEATURE_KEYS] for f in features_seq]
    if not seq:
        return result

    arr = np.array(seq, dtype=np.float32)
    if seq_len is not None and len(arr) > seq_len:
        arr = arr[-seq_len:]

    device  = next(model.parameters()).device
    x       = torch.from_numpy(arr).unsqueeze(0).to(device)       # (1, T, 11)
    lengths = torch.tensor([len(arr)], dtype=torch.long)

    with torch.no_grad():
        logits = model(x, lengths)                                  # (1, num_labels)
        probs  = torch.sigmoid(logits[0])                          # (num_labels,)

    result["valid"] = bool(probs[0].item() > 0.5)
    return result


# ---------------------------------------------------------------------------
# オブジェクト API — main.py は create_classifier() を通じてこちらを使う
# ---------------------------------------------------------------------------

class RuleBasedClassifier:
    """ルールベース判定器のオブジェクトラッパー。"""

    def classify(self, features: dict) -> dict:
        """単一フレームの概況を返す（リアルタイムループ用）。

        深さ・ロックアウト・反復動作はシーケンス単位でしか判定できないため、
        バー下降・足ずれのみをリアルタイムで検出する。
        """
        bar_descent = (
            features.get("left_shoulder_y_delta",  0.0) >= THRESHOLDS["bar_descent"] or
            features.get("right_shoulder_y_delta", 0.0) >= THRESHOLDS["bar_descent"]
        )
        foot_shift = (
            abs(features.get("left_ankle_x_delta",  0.0)) >= THRESHOLDS["foot_shift"] or
            abs(features.get("right_ankle_x_delta", 0.0)) >= THRESHOLDS["foot_shift"]
        )
        return {
            "valid":             not (bar_descent or foot_shift),
            "unanimous":         True,
            "depth_score":       0,
            "lockout_score":     0,
            "bar_descent_score": 3 if bar_descent else 0,
            "bounce_score":      0,
            "foot_shift_score":  3 if foot_shift  else 0,
        }

    def classify_sequence(self, features_seq: list) -> dict:
        """レップ全体のシーケンスを判定する（後処理用）。"""
        return classify_attempt(features_seq)

    def reset(self) -> None:
        pass


class LSTMClassifier:
    """学習済み LSTM モデルによる判定器。"""

    INFER_THRESHOLD  = 0.5
    MIN_WINDOW_RATIO = 0.25

    def __init__(self, model_path: str) -> None:
        import torch
        import numpy as np
        from model import SquatLSTM, FEATURE_KEYS, INPUT_DIM

        self._torch        = torch
        self._np           = np
        self._FEATURE_KEYS = FEATURE_KEYS
        self._INPUT_DIM    = INPUT_DIM

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt   = torch.load(model_path, map_location=device, weights_only=True)

        # num_labels: checkpoint から取得（旧形式 = 1 出力との後方互換）
        head_bias = ckpt["model_state_dict"].get("head.bias")
        num_labels = int(head_bias.shape[0]) if head_bias is not None else ckpt.get("num_labels", 1)

        self._model = SquatLSTM(
            hidden_size=ckpt.get("hidden_size", 64),
            num_layers=ckpt.get("num_layers",   2),
            num_labels=num_labels,
        ).to(device)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()

        self._device      = device
        self._num_labels  = num_labels
        # max_seq_len: 可変長対応後は None（制限なし）、旧 checkpoint は seq_len を保持
        self._max_seq_len = ckpt.get("max_seq_len") or None
        self._window: collections.deque = collections.deque(maxlen=256)

        print(f"LSTMClassifier: {model_path} をロード "
              f"(device={device}, num_labels={num_labels})")

    def _to_vec(self, features: dict) -> list:
        return [float(features.get(k, 0.0)) for k in self._FEATURE_KEYS]

    def classify(self, features: dict) -> dict:
        """スライディングウィンドウを更新しつつリアルタイム指標を返す。"""
        self._window.append(self._to_vec(features))
        bar_descent = (
            features.get("left_shoulder_y_delta",  0.0) >= THRESHOLDS["bar_descent"] or
            features.get("right_shoulder_y_delta", 0.0) >= THRESHOLDS["bar_descent"]
        )
        foot_shift = (
            abs(features.get("left_ankle_x_delta",  0.0)) >= THRESHOLDS["foot_shift"] or
            abs(features.get("right_ankle_x_delta", 0.0)) >= THRESHOLDS["foot_shift"]
        )
        return {
            "valid":             not (bar_descent or foot_shift),
            "unanimous":         True,
            "depth_score":       0,
            "lockout_score":     0,
            "bar_descent_score": 3 if bar_descent else 0,
            "bounce_score":      0,
            "foot_shift_score":  3 if foot_shift  else 0,
        }

    def classify_sequence(self, features_seq: list) -> dict:
        """レップ全体シーケンスを LSTM で推論する（後処理用）。"""
        if not features_seq:
            return dict(_EMPTY_RESULT)
        return classify_attempt_lstm(features_seq, self._model, self._max_seq_len)

    def reset(self) -> None:
        self._window.clear()


# ---------------------------------------------------------------------------
# ファクトリ
# ---------------------------------------------------------------------------

def create_classifier(mode: str = "rule", model_path: str | None = None):
    """判定器を生成するファクトリ関数。

    Args:
        mode:       "rule" または "lstm"
        model_path: LSTM モデルのパス（mode="lstm" のとき必須）
    """
    if mode == "lstm":
        if not model_path:
            raise ValueError(
                "--classifier lstm を指定する場合は --model でモデルパスを指定してください"
            )
        return LSTMClassifier(model_path)
    return RuleBasedClassifier()
