# Squat Evaluation — スクワット有効試技判定 LSTM

スクワットの動作シーケンスから**有効試技 / 無効試技を判定する LSTM 二値分類モデル**です。

---

## 性能

| 手法 | AUC |
|---|---|
| ベースライン（全て有効と予測） | 0.500 |
| ルールベース分類器 | 0.562 |
| **本モデル（LSTM + データ拡張）** | **0.732 ± 0.016**（5-fold CV） |

詳細な実験記録は [experiment_report.md](experiment_report.md) を参照してください。

---

## モデル概要

- **アーキテクチャ**: LSTM（2層、隠れ次元 64、Dropout 0.3）
- **入力**: 11次元の時系列特徴量（膝角度・股関節角度・肩変位・足首変位など）
- **出力**: 有効 / 無効の二値ロジット（sigmoid > 0.5 で有効判定）
- **可変長対応**: `pack_padded_sequence` によりレップごとに異なる長さに対応
- **学習済みモデル**: `model_augmented_v3.pt`

### 入力特徴量（11次元）

| 特徴量 | 説明 |
|---|---|
| left / right\_knee\_angle | 膝関節角度（度） |
| left / right\_hip\_angle | 股関節角度（度） |
| left / right\_shoulder\_y\_delta | 肩の垂直変位（正規化座標） |
| left / right\_ankle\_x\_delta | 足首の水平変位（正規化座標） |
| lr\_knee\_diff | 左右膝角度の差 |
| left / right\_visibility | 関節の検出信頼度 |

---

## データセット

```
training_data/
  seq_{ID}.json      # レップごとの特徴量シーケンス（元データ: 346件）
  anno_{ID}.csv      # レップごとのアノテーション（有効/無効ラベル）
```

| 項目 | 件数 |
|---|---|
| 総レップ数（元データ） | 346件 |
| 有効試技 | 247件 |
| 無効試技 | 99件 |
| 拡張後の無効試技 | 792件（元 99 × 7手法） |

拡張データ（`seq_aug*.json`, `anno_aug*.csv`）は `.gitignore` で除外しています。
`augment_data.py` を実行することで再生成できます。

---

## セットアップ

Python 3.9 以上が必要です。

```bash
pip install -r requirements.txt
```

GPU（CUDA）が使える環境では自動選択し、ない場合は CPU にフォールバックします。

---

## 使い方

### 1. データ拡張（初回のみ）

```bash
python augment_data.py --data-dir training_data
```

無効試技 99件を7手法で拡張し `training_data/` に保存します。

| 手法 | 内容 | 件数 |
|---|---|---|
| time\_warp | 時系列を 80%/90%/110%/120% にリサンプル | ×4 |
| noise | ガウスノイズを加算（角度±1.5°、変位±0.002） | ×2 |
| lr\_flip | 左右対称に反転 | ×1 |

### 2. 学習

```bash
python train_binary.py --data-dir training_data --epochs 150 --output model_augmented_v3.pt
```

主なオプション:

| オプション | デフォルト | 説明 |
|---|---|---|
| `--data-dir` | `training_data` | データディレクトリ |
| `--epochs` | 100 | エポック数 |
| `--output` | `model_binary.pt` | モデル保存先 |
| `--hidden-size` | 64 | LSTM 隠れ次元 |
| `--num-layers` | 2 | LSTM 層数 |
| `--dropout` | 0.3 | ドロップアウト率 |
| `--val-ratio` | 0.2 | 検証データ割合 |

学習中は `val AUC` が最高のエポックのモデルを自動保存します。

### 3. 推論

```python
from classifier import create_classifier

clf = create_classifier(mode="lstm", model_path="model_augmented_v3.pt")
result = clf.classify_sequence(features_seq)
# result["valid"] -> True / False
```

`features_seq` は各フレームの特徴量 dict のリストです（`FEATURE_KEYS` の11次元）。

---

## ファイル構成

```
squat_demo_v2/
├── model.py              # SquatLSTM モデル定義
├── train_binary.py       # 二値分類学習スクリプト
├── augment_data.py       # 時系列データ拡張
├── classifier.py         # ルールベース / LSTM 推論インターフェース
├── requirements.txt      # 依存ライブラリ
├── model_augmented_v3.pt # 学習済みモデル（最良チェックポイント）
├── experiment_report.md  # 実験記録
└── training_data/        # 元データ（seq_*.json + anno_*.csv）
```

---

## ルールベース分類器について

`classifier.py` にはルールベース判定器（`RuleBasedClassifier`）も含まれています。
LSTM モデルと同じ `create_classifier()` インターフェースで切り替えられます。

```python
# ルールベース
clf = create_classifier(mode="rule")

# LSTM
clf = create_classifier(mode="lstm", model_path="model_augmented_v3.pt")
```

### ルールベースの判定閾値

| 失敗要素 | 判定条件 |
|---|---|
| 深さ不足 | シーケンス中に代表膝角度 ≤ 90° を達成しなければ無効 |
| ロックアウト不足 | 開始・終了付近3フレームで代表膝角度 ≥ 165° でなければ無効 |
| バー下降 | 上昇フェーズで肩 Y delta が前フレームより 0.03 以上増加すれば無効 |
| 反復動作 | 下降フェーズで股関節角度の方向逆転（≥10°）が2回以上で無効 |
| 足ずれ | 足首 X delta の絶対値 ≥ 0.05 のフレームがあれば無効 |

閾値は `classifier.py` の `THRESHOLDS` で変更できます。
