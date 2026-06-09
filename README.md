# スクワットフォーム崩れ検出デモ

MediaPipe Pose を使って左右両足の骨格を検出し、各レップの `start_frame` を baseline として以下の3種類のフォーム崩れを判定します。

- `knee_forward` : 膝の過度な前出し
- `trunk_lean`   : 過度な体幹前傾
- `back_round`   : 腰の丸まり

判定モードは **ルールベース**（デフォルト）と **LSTM モデル** の2種類をコマンドで切り替えられます。  
本デモは GW 本開発（LSTM マルチラベル分類）の前段として、パイプライン動作確認とアノテーション候補データ収集を目的としています。

---

## セットアップ

Python 3.9 以上が必要です。

```bash
# 仮想環境を作成（初回のみ）
python -m venv venv

# Windows PowerShell で仮想環境を有効化
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
Unblock-File .\venv\Scripts\Activate.ps1
.\venv\Scripts\Activate.ps1

# ルールベースデモに必要なライブラリ
pip install -r requirements.txt

# LSTM 学習・推論を使う場合は追加で torch をインストール
# GPU あり（CUDA 12.1）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU のみ
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 実行コマンド例

### ルールベースデモ（デフォルト）

```bash
# Webカメラで起動
python main.py

# 動画ファイルを入力
python main.py --input video.mp4

# 処理済み動画を output.mp4 に保存
python main.py --input video.mp4 --save

# 最深点フレームを PNG で保存
python main.py --input video.mp4 --save-frames

# スクワットスタイルをサマリに記録
python main.py --input video.mp4 --style HB_NS

# アノテーション候補 CSV と特徴量シーケンス JSON を出力（学習データ収集）
python main.py --input video.mp4 --export-annotation --style HB_NS
```

### ルーティーン除外・レップ境界レビュー

ラックアップ前の準備動作やラックダウン後の動作が誤ってレップとして検出される場合に使います。

```bash
# 最初の5秒と60秒以降をレップ検出対象外にする
python main.py --input video.mp4 --ignore-before-sec 5 --ignore-after-sec 60

# 検出後にレップ境界を対話的に確認・修正する
python main.py --input video.mp4 --review-annotation

# 組み合わせ例：除外 + レビュー + アノテーション出力
python main.py --input video.mp4 \
    --ignore-before-sec 5 --ignore-after-sec 60 \
    --review-annotation --export-annotation --style HB_NS
```

`--review-annotation` を指定すると、動画処理後に以下のようなプロンプトが表示されます。

```
======================================================
レップ境界レビュー
======================================================
修正する場合は「開始秒 終了秒」を入力（例: 1.5 3.2）
変更なしの場合は Enter を押してください

Rep 1:
  開始   : frame    30  (0:01.00)
  終了   : frame    95  (0:03.17)
  最深点 : frame    62  (0:02.07)  膝角度=82.3°
  修正 [開始秒 終了秒 / Enter でスキップ]: 0.8 3.5
  → 更新: frames 24–105  (0:00.80 – 0:03.50)
```

修正した境界は `summary.json` / `annotation_candidates.csv` / `feature_sequences.json` すべてに反映されます。

### LSTM モデルで推論

```bash
python main.py --input video.mp4 --classifier lstm --model model.pt
python main.py --input video.mp4 --classifier lstm --model model.pt --save
```

---

## LSTM モデルの学習

### 1. 学習データを収集する

複数の動画に対して `--export-annotation` を実行し、データを蓄積します。

```bash
python main.py --input video1.mp4 --export-annotation --style HB_NS
# → annotation_candidates.csv, feature_sequences.json が生成される

# ファイルを別名で保管して次の動画も処理
cp feature_sequences.json seq_video1.json
cp annotation_candidates.csv anno_video1.csv

python main.py --input video2.mp4 --export-annotation --style LB_WS
cp feature_sequences.json seq_video2.json
cp annotation_candidates.csv anno_video2.csv
```

生成された CSV を確認し、ルールベースの自動ラベルに誤りがあれば手動で修正してください。  
`start_mmss` / `end_mmss` 列を見ることで、動画の何秒付近のレップかを素早く確認できます。  
レップ境界自体を修正したい場合は `--review-annotation` を使って処理時に対話的に上書きするか、
再実行時に `--ignore-before-sec` / `--ignore-after-sec` で検出対象範囲を絞ってください。

### 2. 学習する

```bash
# 単一動画のデータで学習
python train.py \
    --sequences feature_sequences.json \
    --annotations annotation_candidates.csv

# 複数動画のデータを結合して学習
python train.py \
    --sequences seq_video1.json seq_video2.json \
    --annotations anno_video1.csv anno_video2.csv \
    --epochs 100 --output model.pt

# 主なオプション
#   --seq-len     シーケンス長（フレーム数、default: 90）
#   --epochs      エポック数（default: 50）
#   --batch-size  バッチサイズ（default: 32）
#   --hidden-size LSTM 隠れ層サイズ（default: 64）
#   --num-layers  LSTM 層数（default: 2）
#   --val-ratio   検証データ割合（default: 0.2）
```

GPU（CUDA）が使える環境では自動で GPU を選択し、ない場合は CPU にフォールバックします。

### 3. 学習済みモデルで推論する

```bash
python main.py --input video.mp4 --classifier lstm --model model.pt
```

---

## 判定の仕組み

### baseline の計算

- リアルタイム時は代表角度が `170°` 以上の最新フレームを暫定立位 baseline として使用
- 終了後は各レップの `start_frame` から正式 baseline を再計算し、レップ単位の特徴量と判定を確定

### 特徴量

| 特徴量 | 計算方法 |
|--------|----------|
| `knee_forward_ratio` | 現フレームの `knee_foot_diff` / baseline の `knee_foot_diff` |
| `trunk_lean_delta`   | 現フレームの `trunk_angle` − baseline の `trunk_angle` |
| `back_round_ratio`   | 現フレームの `shoulder_hip_dist` / baseline の `shoulder_hip_dist` |

### ルールベースの判定閾値

`classifier.py` の `THRESHOLDS` を変更することで感度を調整できます。

| 指標 | 判定条件 |
|------|----------|
| knee_forward | `knee_forward_ratio > 1.3` |
| trunk_lean   | `trunk_lean_delta > 15.0°` |
| back_round   | `back_round_ratio < 0.85` |

---

## 出力ファイル

| ファイル | 生成条件 | 内容 |
|----------|----------|------|
| `summary.json` | 常時 | レップごとのフレーム範囲・時間情報・baseline・特徴量・form_labels |
| `output.mp4` | `--save` | 描画付き動画 |
| `repXX_frameYYYY_labels.png` | `--save-frames` | 最深点フレーム画像 |
| `annotation_candidates.csv` | `--export-annotation` | アノテーション候補（フレーム・時間情報・ラベル付き） |
| `feature_sequences.json` | `--export-annotation` | フレーム単位特徴量シーケンス（時間情報付き、LSTM 学習用） |
| `model.pt` | `train.py` 実行後 | 学習済み LSTM モデル |

### summary.json のスキーマ

各レップエントリに以下のフィールドが含まれます。

```json
{
  "total_reps": 3,
  "squat_style": "HB_NS",
  "classifier": "rule",
  "reps": [
    {
      "rep_id": 1,
      "start_frame": 30,
      "end_frame": 95,
      "deepest_frame": 62,
      "start_time_sec": 1.0,
      "end_time_sec": 3.17,
      "deepest_time_sec": 2.07,
      "start_mmss": "0:01.00",
      "end_mmss": "0:03.17",
      "deepest_mmss": "0:02.07",
      "min_angle_repr": 82.3,
      "max_lr_diff": 4.1,
      "baseline": { "knee_foot_diff": 0.12, "trunk_angle": 5.3, "shoulder_hip_dist": 0.45 },
      "features_at_deepest": { "knee_forward_ratio": 1.1, "trunk_lean_delta": 18.2, "back_round_ratio": 0.97 },
      "form_labels": { "knee_forward": 0, "trunk_lean": 1, "back_round": 0 }
    }
  ]
}
```

### annotation_candidates.csv の列定義

| 列名 | 説明 |
|------|------|
| video_id | 入力動画ファイル名（拡張子除く）または `webcam` |
| rep_id | レップ番号 |
| squat_style | `--style` で指定したスタイル名 |
| start_frame | レップ開始フレーム番号 |
| end_frame | レップ終了フレーム番号 |
| deepest_frame | 最深点フレーム番号 |
| start_time_sec | レップ開始時刻（秒） |
| end_time_sec | レップ終了時刻（秒） |
| deepest_time_sec | 最深点時刻（秒） |
| start_mmss | レップ開始タイムスタンプ（`m:ss.ff` 形式） |
| end_mmss | レップ終了タイムスタンプ（`m:ss.ff` 形式） |
| deepest_mmss | 最深点タイムスタンプ（`m:ss.ff` 形式） |
| knee_forward_flag | 1 = そのレップで膝前出し検出あり |
| trunk_lean_flag | 1 = そのレップで体幹前傾検出あり |
| back_round_flag | 1 = そのレップで腰丸まり検出あり |
| needs_review | 1 = いずれかの崩れが検出されたレップ |
| knee_forward_ratio_max | レップ内の最大 `knee_forward_ratio` |
| trunk_lean_delta_max | レップ内の最大 `trunk_lean_delta` |
| back_round_ratio_min | レップ内の最小 `back_round_ratio` |

---

## スクワットスタイル一覧

| スタイル名 | バーポジション | スタンス幅 |
|-----------|-------------|---------|
| HB_NS     | ハイバー      | ナロー   |
| HB_WS     | ハイバー      | ワイド   |
| LB_NS     | ローバー      | ナロー   |
| LB_WS     | ローバー      | ワイド   |

---

## 撮影推奨条件

- 全身が映ること
- 真横もしくは斜め45°の角度で撮影すること
- 左右両足が視界に入ること
- できるだけ背景がシンプルで動作が見やすいこと

---

## LSTM モデルへの移行手順

`classifier.py` の `create_classifier()` が判定ロジックの入り口です。  
`--classifier lstm --model model.pt` を指定するだけで、他のモジュールへの変更なしに LSTM 推論へ切り替わります。

独自モデルに差し替える場合は `LSTMClassifier` を継承し、`classify()` と `classify_sequence()` を実装してください。

```python
# classifier.py の LSTMClassifier を独自モデルに差し替えるイメージ
class MyModelClassifier:
    def classify(self, features: dict) -> dict:
        # リアルタイム推論
        ...

    def classify_sequence(self, features_seq: list) -> dict:
        # レップ全体推論
        ...

    def reset(self) -> None:
        ...
```

---

## ディレクトリ構成

```
squat_demo_v2/
├── pose/
│   ├── __init__.py
│   ├── estimator.py       # MediaPipe ラッパー・両足角度計算
│   └── rep_detector.py    # レップ回数・最深点フレーム検出
├── features/
│   ├── __init__.py
│   └── extractor.py       # フォーム崩れ用特徴量・baseline 計算
├── model.py               # SquatLSTM モデル定義
├── dataset.py             # LSTM 学習用データセット
├── train.py               # LSTM 学習スクリプト
├── classifier.py          # ルールベース / LSTM 判定器（切り替え可能）
├── visualizer.py          # OpenCV オーバーレイ描画
├── main.py                # エントリポイント
├── requirements.txt       # ルールベースデモの依存ライブラリ
└── README.md
```
