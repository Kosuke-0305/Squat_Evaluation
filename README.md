# スクワット有効試技判定システム

MediaPipe Pose で骨格を検出し、パワーリフティング競技における
スクワット試技の有効/無効をリアルタイムで判定するシステムです。

判定対象の失敗要素（各要素を審判3人の多数決で判定）:
- **深さ不足**         — 大腿上面が膝の上端より高い位置で折り返し
- **ロックアウト不足** — 試技開始・完了時に膝が完全に伸展しない
- **バー下降**         — 挙上中に肩（バー）の高さが一時的に低下
- **反復動作**         — 下降から上昇への切り返し時の反動利用
- **足の横ずれ**       — 試技中に足首の位置が横方向にずれる

各失敗要素は審判3人のうち何人が抵触と判断したかを 0〜3 で記録します。
2人以上（過半数）が抵触と判断した場合に無効試技と判定します。

判定モードは **ルールベース**（デフォルト）と **LSTM モデル** の2種類をコマンドで切り替えられます。

> **注意**: 入力画角は **正面** を想定しています。  
> 足の前後ずれ・補助員との重なりに関しては検出精度が落ちる場合があります。

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

### YouTubeクリップ生成機能

新たに `download_clips.py` を追加し、CSVの各行に記録されたクリップ区間をYouTube動画から抽出できます。

- 単一動画を対象に複数クリップを一括抽出
- CSVの時間欄は `start_time_sec`/`end_time_sec` または `start_sec`/`end_sec` に対応
- `0:32:55` のような `HH:MM:SS` / `MM:SS` 形式も受け取る
- ファイル名には `video_id`、`rep_id`、`squat_style`、判定・各失敗要素を含む

CSV例:

```csv
video_id,rep_id,squat_style,start_sec,end_sec,valid,depth_score,lockout_score,bar_descent,bounce_score,foot_shift
1,1,LB_NS,0:32:55,0:33:08,3,0,0,0,0,0
1,2,HB_NS,0:33:45,0:34:00,3,0,0,0,0,0
1,3,LB_NS,0:34:37,0:34:53,0,0,0,0,3,3,0
```

使い方:

```bash
# 実際にダウンロード
python download_clips.py Test_clip.csv "https://www.youtube.com/watch?v=diTikfGNnAI"

# 出力先ディレクトリを指定
python download_clips.py Test_clip.csv "https://www.youtube.com/watch?v=diTikfGNnAI" --output clips

# まずはドライランで確認
python download_clips.py Test_clip.csv "https://www.youtube.com/watch?v=diTikfGNnAI" --dry-run
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
#   --max-seq-len  シーケンス最大長フレーム数（超えたら末尾を使用、default: 制限なし）
#   --epochs       エポック数（default: 50）
#   --batch-size   バッチサイズ（default: 32）
#   --hidden-size  LSTM 隠れ層サイズ（default: 64）
#   --num-layers   LSTM 層数（default: 2）
#   --val-ratio    検証データ割合（default: 0.2）
```

GPU（CUDA）が使える環境では自動で GPU を選択し、ない場合は CPU にフォールバックします。

### 3. 学習済みモデルで推論する

```bash
python main.py --input video.mp4 --classifier lstm --model model.pt
```

---

## アノテーション手順

1. `--export-annotation` で CSV と feature_sequences.json を出力する
2. 各レップの動画タイムスタンプ（`start_mmss`〜`end_mmss` 列）を確認する
3. YouTube 大会動画の審判判定（ホワイト/レッドランプ）を参照し、  
   `valid` 列に `1`（有効）または `0`（無効）を記入する
4. 無効試技（`valid=0`）について、どの失敗要素が原因かを  
   `depth_score`〜`foot_shift_score` 列に 0〜3 で記入する  
   （3人全員が問題と判断した要素は `3`、2人なら `2`、1人なら `1`、0人なら `0`）  
   ※ 初期値はルールベース結果（0 または 3）なので、明確な試技はそのまま使用可能
5. スコアがすべて `0` か `3`（意見が割れていない）なら `unanimous=1` を記入する  
   スコアに `1` や `2` を含む場合は `unanimous=0` を記入する
6. 完成した CSV を `train.py` の `--annotations` に渡す

---

## 判定の仕組み

### LSTM モデルの設計

- **可変長対応**: 各レップのシーケンスを実際のフレーム数のまま保持し、`pack_padded_sequence` でバッチ化する。固定長 90 フレームへの切り詰めを行わない。
- **多出力**: `valid`（有効/無効）に加えて 5 つの失敗スコアも出力する（合計 6 出力）。
- **補助学習**: 失敗スコアは未記入でも学習可能（損失マスクで除外）。記入済みの試技ほど失敗要素の判定精度が向上する。
- **ルールとの組み合わせ**: ルールベースが「確実に抵触（3）」と判定した要素は LSTM の予測に関わらず 3 を維持。ルールが 0 でも LSTM が抵触と判定した場合は 2（際どい失敗）に設定する。
- **後方互換**: 旧形式（1 出力）の checkpoint も読み込み可能。`valid` のみ LSTM を使用し、失敗スコアはルールベースのまま。

### baseline の計算

- リアルタイム時は代表角度が `170°` 以上の最新フレームを暫定立位 baseline として使用
- 終了後は各レップの `start_frame` から正式 baseline を再計算し、レップ単位の特徴量と判定を確定

### 特徴量（D=11）

| 特徴量 | 計算方法 | 対応する失敗要素 |
|--------|----------|----------------|
| `left_knee_angle` | 左膝の屈曲角度（度） | 深さ・ロックアウト |
| `right_knee_angle` | 右膝の屈曲角度（度） | 深さ・ロックアウト |
| `left_hip_angle` | 左股関節の屈曲角度（度） | 深さ・反復動作 |
| `right_hip_angle` | 右股関節の屈曲角度（度） | 深さ・反復動作 |
| `left_shoulder_y_delta` | 左肩 Y − baseline 左肩 Y（正規化） | バー下降 |
| `right_shoulder_y_delta` | 右肩 Y − baseline 右肩 Y（正規化） | バー下降 |
| `left_ankle_x_delta` | 左足首 X − baseline 左足首 X（正規化） | 足ずれ |
| `right_ankle_x_delta` | 右足首 X − baseline 右足首 X（正規化） | 足ずれ |
| `lr_knee_diff` | abs(左膝角度 − 右膝角度) | 左右非対称 |
| `left_visibility` | LEFT_KNEE の visibility | 欠損フラグ |
| `right_visibility` | RIGHT_KNEE の visibility | 欠損フラグ |

### ルールベースの判定閾値

`classifier.py` の `THRESHOLDS` を変更することで感度を調整できます。

| 失敗要素 | 判定条件 | 自動出力値 |
|---------|---------|----------|
| 深さ不足 (`depth_score`) | シーケンス中に代表膝角度 ≤ 90° を達成しなければ抵触 | 3（確実）/ 0（なし） |
| ロックアウト不足 (`lockout_score`) | 開始・終了付近3フレームで代表膝角度 ≥ 165° でなければ抵触 | 3（確実）/ 0（なし） |
| バー下降 (`bar_descent_score`) | 上昇フェーズで肩 Y delta が前フレームより 0.03 以上増加すれば抵触 | 3（確実）/ 0（なし） |
| 反復動作 (`bounce_score`) | 下降フェーズで股関節角度の方向逆転（≥10°）が2回以上で抵触 | 3（確実）/ 0（なし） |
| 足ずれ (`foot_shift_score`) | 足首 X delta の絶対値 ≥ 0.05 のフレームがあれば抵触 | 3（確実）/ 0（なし） |

**`valid` 列（人手アノテーション用）**

| 値 | 意味 |
|----|------|
| 1  | 有効試技 |
| 0  | 無効試技 |

`valid` は審判多数決（過半数のスコアが `INVALID_MAJORITY` 未満）から自動導出することもできます。  
LSTM モデル学習時は `valid=1` を正例（有効）として使用します。

**`unanimous` 列（人手アノテーション用）**

| 値 | 意味 | 学習時の sample_weight |
|----|------|----------------------|
| 1  | 全員一致（スコアがすべて 0 か 3） | 1.0（通常） |
| 0  | 意見が割れた（スコアに 1 か 2 を含む） | 0.5（際どい試技） |

ルールベース自動判定では `unanimous` は常に 1 になります（0 か 3 しか出力しないため）。
際どい試技は人間がスコアを 1 や 2 に修正することで表現し、`unanimous=0` を記入します。

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
      "start_frame": 30,   "end_frame": 95,   "deepest_frame": 62,
      "start_time_sec": 1.0, "end_time_sec": 3.17, "deepest_time_sec": 2.07,
      "start_mmss": "0:01.00", "end_mmss": "0:03.17", "deepest_mmss": "0:02.07",
      "min_angle_repr": 82.3,
      "max_lr_diff": 4.1,
      "baseline": {
        "left_knee_angle": 171.2, "right_knee_angle": 170.8,
        "left_ankle_x": 0.42, "right_ankle_x": 0.58,
        "left_shoulder_y": 0.21, "right_shoulder_y": 0.22
      },
      "features_at_deepest": {
        "left_knee_angle": 82.1, "right_knee_angle": 84.5,
        "left_hip_angle": 55.3, "right_hip_angle": 56.1,
        "left_shoulder_y_delta": 0.08, "right_shoulder_y_delta": 0.07,
        "left_ankle_x_delta": 0.01, "right_ankle_x_delta": -0.01,
        "lr_knee_diff": 2.4, "left_visibility": 0.97, "right_visibility": 0.96
      },
      "form_labels": {
        "valid":             null,
        "unanimous":         null,
        "depth_score":       0,
        "lockout_score":     0,
        "bar_descent_score": 0,
        "bounce_score":      0,
        "foot_shift_score":  0
      }
    }
  ]
}
```

- `valid` と `unanimous` は人間がアノテーション CSV で記入するため `null`（`--classifier lstm` 時は `valid` のみモデルの二値出力 `true/false` が入る）
- 失敗スコアはルールベースで自動記入される（条件抵触 → 3、非抵触 → 0）
- 人間がレビュー時にスコアを 1 や 2 に修正し、`unanimous=0` を記入することで際どい試技を表現できる（スコア ≥ 2 を抵触ありとして失敗 recall を計算）

### annotation_candidates.csv の列定義

| 列名 | 種別 | 説明 |
|------|------|------|
| video_id | 自動 | 入力動画ファイル名（拡張子除く）または `webcam` |
| rep_id | 自動 | レップ番号 |
| squat_style | 自動 | `--style` で指定したスタイル名 |
| **valid** | **人間** | 1=有効試技 / 0=無効試技（初期値：空欄） |
| **unanimous** | **人間** | 1=全員一致 / 0=意見が割れた（初期値：空欄） |
| **depth_score** | **人間が上書き** | 審判3人のうち何人が抵触と判断したか（0〜3）。初期値はルールベース結果（0 または 3） |
| **lockout_score** | **人間が上書き** | 同上 |
| **bar_descent_score** | **人間が上書き** | 同上 |
| **bounce_score** | **人間が上書き** | 同上 |
| **foot_shift_score** | **人間が上書き** | 同上 |
| start_frame | 自動 | レップ開始フレーム番号 |
| end_frame | 自動 | レップ終了フレーム番号 |
| deepest_frame | 自動 | 最深点フレーム番号 |
| start_time_sec | 自動 | レップ開始時刻（秒） |
| end_time_sec | 自動 | レップ終了時刻（秒） |
| deepest_time_sec | 自動 | 最深点時刻（秒） |
| start_mmss | 自動 | レップ開始タイムスタンプ（`m:ss.ff` 形式） |
| end_mmss | 自動 | レップ終了タイムスタンプ（`m:ss.ff` 形式） |
| deepest_mmss | 自動 | 最深点タイムスタンプ（`m:ss.ff` 形式） |
| min_angle_repr | 自動 | 最深点での代表膝角度（度） |
| max_lr_diff | 自動 | レップ内の左右膝角度差の最大値（度） |
| auto_depth_fail | 自動 | 角度閾値による深さ自動判定（参考用） |
| auto_lockout_fail | 自動 | 角度閾値によるロックアウト自動判定（参考用） |

**人間** 列と **人間が上書き** 列はアノテーション作業で記入します。YouTube 大会動画の審判ランプを参照して判断してください。  
スコア列の初期値はルールベース結果（0 または 3）ですが、際どい試技は 1 や 2 に修正することで意見の割れを表現できます。  
`auto_*` 列はモデルの正解ラベルには使用せず、ラベリング作業の参考のみに使用してください。

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

- **正面から撮影すること**（横・斜め45°は非推奨）
- 全身（頭頂から足首）がフレームに収まること
- 補助員がなるべく画角外に位置すること
- ズームなしの固定カメラが望ましい
- 背景がシンプルで動作が見やすいこと

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

## GitHub 共同編集の推奨設定

このリポジトリは GitHub 上で複数人が編集しやすいように、次の運用を推奨します。

### 1. main ブランチを保護する

GitHub のリポジトリ設定で次を有効にしてください。

- Settings → Branches → Add branch protection rule
- Branch name pattern: `main`
- 以下をチェック
  - Require a pull request before merging
  - Require approvals: 1 以上
  - Require status checks to pass before merging
  - Require branches to be up to date before merging
  - Do not allow bypassing the above settings

これにより、直接の手打ち push を防ぎ、レビューを経てから統合できます。

### 2. PR テンプレートを用意する

GitHub の `.github/pull_request_template.md` を追加すると、レビューしやすくなります。

例:

```md
## 変更内容
- 何を直したか

## 確認内容
- 動作確認済みか
- 影響範囲はどこか

## レビュー依頼
- 特に確認してほしい点
```

### 3. Issue テンプレートを用意する

`.github/ISSUE_TEMPLATE/` に次のようなテンプレートを置くと、バグ報告や要望の整理がしやすくなります。

- bug_report.md
- feature_request.md

### 4. 変更は小さな単位で commit する

1 PR につき 1 目的にします。

例:

- `fix: レップ境界の自動検出を調整`
- `docs: GitHub 運用手順を追記`
- `feat: annotation review モードを追加`

### 5. 大きなデータや生成物は Git に含めない

このリポジトリでは `.gitignore` により、以下を除外しています。

- `venv/`
- `__pycache__/`
- `output.mp4`
- `summary.json`
- `annotation_candidates.csv`
- `feature_sequences.json`
- `model.pt`
- `*.mp4`

---

## グループメンバー向けの利用手順

### 1. リポジトリを取得する

```bash
git clone https://github.com/<your-user>/<repo-name>.git
cd <repo-name>
```

### 2. Python 環境を作る

```bash
python -m venv venv
```

Windows PowerShell では次も実行します。

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
Unblock-File .\venv\Scripts\Activate.ps1
.\venv\Scripts\Activate.ps1
```

### 3. 依存ライブラリを入れる

```bash
pip install -r requirements.txt
```

LSTM を使う場合は追加で torch を入れます。

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. まずはサンプル実行を確認する

```bash
python main.py --input video.mp4
```

### 5. 変更は必ずブランチで行う

```bash
git checkout -b feature/your-change
```

### 6. 変更を commit して push する

```bash
git add .
git commit -m "変更内容の要約"
git push -u origin feature/your-change
```

### 7. GitHub で Pull Request を作る

GitHub 上で以下を確認します。

- 変更内容が分かるタイトル
- 何を確認してほしいかを記載
- 必要なら画像や実行結果のスクリーンショットを添付

### 8. レビュー後に main に統合する

レビューが通ったら GitHub 上で Merge pull request を押します。

---

## 共同編集でのおすすめフロー

1. まず Issue を立てる
2. 自分用ブランチを作る
3. 小さな変更を commit する
4. PR を作る
5. レビューを受ける
6. main にマージする

この流れを守ると、誰が何を変えたか追跡しやすくなり、競合も減ります。

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
