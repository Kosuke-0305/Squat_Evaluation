"""スクワットフォーム崩れ判定器。

ルールベース（デフォルト）と LSTM モデルの 2 モードを
`create_classifier()` ファクトリで切り替えられる。
判定ロジックの差し替えはこのファイルのみに閉じている。
"""

import collections

THRESHOLDS = {
    "knee_forward": 1.3,   # baseline の 1.3 倍以上前に出たとき
    "trunk_lean":   15.0,  # baseline から 15° 以上前傾したとき
    "back_round":   0.85,  # 肩腰距離が baseline の 85% 未満になったとき
}

LABEL_COLORS = {
    "knee_forward": (0, 60, 220),
    "trunk_lean":   (0, 140, 255),
    "back_round":   (180, 60, 200),
    "good_form":    (0, 200, 80),
}

_EMPTY_RESULT = {
    "knee_forward": False,
    "trunk_lean":   False,
    "back_round":   False,
    "any_error":    False,
}


# ---------------------------------------------------------------------------
# ルールベース単関数 API（後方互換・build_summary 等から直接呼ぶ用途も残す）
# ---------------------------------------------------------------------------

def classify_form(features: dict) -> dict:
    """1 フレームの特徴量からフォーム崩れをルールベースで判定する。

    Args:
        features: extract_form_features が返す特徴量辞書

    Returns:
        {"knee_forward": bool, "trunk_lean": bool, "back_round": bool, "any_error": bool}
    """
    knee_forward = features["knee_forward_ratio"] > THRESHOLDS["knee_forward"]
    trunk_lean   = features["trunk_lean_delta"]   > THRESHOLDS["trunk_lean"]
    back_round   = features["back_round_ratio"]   < THRESHOLDS["back_round"]
    return {
        "knee_forward": knee_forward,
        "trunk_lean":   trunk_lean,
        "back_round":   back_round,
        "any_error":    knee_forward or trunk_lean or back_round,
    }


# ---------------------------------------------------------------------------
# オブジェクト API — main.py は create_classifier() を通じてこちらを使う
# ---------------------------------------------------------------------------

class RuleBasedClassifier:
    """ルールベース判定器のオブジェクトラッパー。"""

    def classify(self, features: dict) -> dict:
        """単一フレームを判定する（リアルタイムループ用）。"""
        return classify_form(features)

    def classify_sequence(self, features_seq: list) -> dict:
        """レップ全体のシーケンスを判定する（後処理用）。

        いずれかのフレームで閾値超えがあれば True を返す。
        """
        if not features_seq:
            return dict(_EMPTY_RESULT)
        results = [classify_form(f) for f in features_seq]
        return {
            "knee_forward": any(r["knee_forward"] for r in results),
            "trunk_lean":   any(r["trunk_lean"]   for r in results),
            "back_round":   any(r["back_round"]   for r in results),
            "any_error":    any(r["any_error"]     for r in results),
        }

    def reset(self) -> None:
        pass


class LSTMClassifier:
    """学習済み LSTM モデルによる判定器。

    - classify(): スライディングウィンドウによるリアルタイム推論
    - classify_sequence(): レップ全体シーケンスでの後処理推論
    """

    INFER_THRESHOLD  = 0.5
    MIN_WINDOW_RATIO = 0.25  # ウィンドウが seq_len の 25% 未満なら判定保留

    def __init__(self, model_path: str) -> None:
        # torch は LSTM モード時のみ必要（ルールベース環境への影響を避けるため遅延 import）
        import torch
        import numpy as np
        from model import SquatLSTM, FEATURE_KEYS, INPUT_DIM

        self._torch        = torch
        self._np           = np
        self._FEATURE_KEYS = FEATURE_KEYS
        self._INPUT_DIM    = INPUT_DIM

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt   = torch.load(model_path, map_location=device, weights_only=True)

        self._model = SquatLSTM(
            hidden_size=ckpt.get("hidden_size", 64),
            num_layers=ckpt.get("num_layers",   2),
        ).to(device)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()

        self._device   = device
        self._seq_len  = int(ckpt.get("seq_len", 90))
        self._window: collections.deque = collections.deque(maxlen=self._seq_len)

        print(f"LSTMClassifier: {model_path} をロード (device={device}, seq_len={self._seq_len})")

    def _to_vec(self, features: dict) -> list:
        return [float(features.get(k, 0.0)) for k in self._FEATURE_KEYS]

    def _infer(self, seq: list) -> dict:
        """float ベクトルのリストを LSTM に通して判定結果を返す。"""
        arr = self._np.array(seq, dtype=self._np.float32)
        if len(arr) < self._seq_len:
            pad = self._np.zeros((self._seq_len - len(arr), self._INPUT_DIM), dtype=self._np.float32)
            arr = self._np.concatenate([pad, arr], axis=0)
        else:
            arr = arr[-self._seq_len:]
        x = self._torch.from_numpy(arr).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            probs = self._torch.sigmoid(self._model(x)).squeeze().cpu().tolist()
        if isinstance(probs, float):
            probs = [probs]
        kf = bool(probs[0] > self.INFER_THRESHOLD)
        tl = bool(probs[1] > self.INFER_THRESHOLD)
        br = bool(probs[2] > self.INFER_THRESHOLD)
        return {"knee_forward": kf, "trunk_lean": tl, "back_round": br, "any_error": kf or tl or br}

    def classify(self, features: dict) -> dict:
        """スライディングウィンドウによるリアルタイム推論。

        ウィンドウが MIN_WINDOW_RATIO 未満の場合は判定保留（全 False）を返す。
        """
        self._window.append(self._to_vec(features))
        if len(self._window) < int(self._seq_len * self.MIN_WINDOW_RATIO):
            return dict(_EMPTY_RESULT)
        return self._infer(list(self._window))

    def classify_sequence(self, features_seq: list) -> dict:
        """レップ全体のシーケンスを使った後処理推論（精度優先）。"""
        if not features_seq:
            return dict(_EMPTY_RESULT)
        seq = [self._to_vec(f) for f in features_seq]
        return self._infer(seq)

    def reset(self) -> None:
        """スライディングウィンドウをリセットする（レップ間で呼ぶ）。"""
        self._window.clear()


# ---------------------------------------------------------------------------
# ファクトリ
# ---------------------------------------------------------------------------

def create_classifier(mode: str = "rule", model_path: str | None = None):
    """判定器を生成するファクトリ関数。

    Args:
        mode:       "rule" または "lstm"
        model_path: LSTM モデルのパス（mode="lstm" のとき必須）

    Returns:
        RuleBasedClassifier または LSTMClassifier
    """
    if mode == "lstm":
        if not model_path:
            raise ValueError(
                "--classifier lstm を指定する場合は --model でモデルパスを指定してください"
            )
        return LSTMClassifier(model_path)
    return RuleBasedClassifier()
