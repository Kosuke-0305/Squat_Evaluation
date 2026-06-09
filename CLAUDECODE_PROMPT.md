# スクワット フォーム崩れ検出デモ — ClaudeCode 実装依頼

## 前提

現在の squat_demo_v2 は、MediaPipe Pose で角度を計算し、ルールベースでフォーム崩れを判定するデモです。  
このリポジトリには、LSTM 学習・推定用の学習コード、モデル定義、推論コードはまだ含まれていません。

そのため、以下の仕様書に沿って、まずは「ルールベース判定デモ」を実装し、後で LSTM マルチラベル分類へ差し替えられるようにしてください。

---

## 実装目標

MediaPipe Pose で左右両足の骨格を検出し、各レップの start_frame を baseline として
3種のフォーム崩れ（膝の過度な前出し・過度な体幹前傾・腰の丸まり）を
ルールベースでリアルタイム判定するデモを実装してください。

本デモは GW 本開発（LSTM マルチラベル分類）の前段として、
パイプライン動作確認とアノテーション候補データの収集を目的とします。
判定ロジックは後でモデル推論に差し替えられるよう、独立した関数として実装してください。

---

## 前提条件

- Python 3.9 以上
- 必要ライブラリ: mediapipe, opencv-python, numpy, scipy
- ルールベースデモ本体は CPU でも動作可能
- ただし、GW 本開発の LSTM 学習・推定では CUDA 対応 GPU を優先して利用し、GPU が無い環境では CPU フォールバックで動作可能にすること
- 入力: Webカメラ または MP4 動画ファイル（コマンドライン引数で切り替え）

---

## GPU 学習要件（LSTM 本開発向け）

- LSTM 学習時は可能であれば CUDA 対応 GPU を使用すること
- 学習スクリプトは `cuda` が利用可能なら自動で GPU を選択し、利用不可時は CPU にフォールバックすること
- 推論時も同様に、GPU が使える場合は GPU を優先し、CPU 環境でも動作すること
- GPU 環境ではメモリ使用量とバッチサイズの設定を考慮し、必要に応じて軽量化オプションを用意すること

---

## ディレクトリ構成

```
squat_form_demo/
├── pose/
│   ├── __init__.py
│   ├── estimator.py       # MediaPipe ラッパー・両足角度計算
│   └── rep_detector.py    # レップ回数・最深点フレーム検出
├── features/
│   ├── __init__.py
│   └── extractor.py       # フォーム崩れ用特徴量・baseline 計算
├── classifier.py          # ルールベース判定器（後でモデルに差し替え予定）
├── visualizer.py          # OpenCV オーバーレイ描画
├── main.py                # エントリポイント
├── requirements.txt
└── README.md
```

---

## 各モジュールの仕様

### pose/estimator.py

前デモ（squat_demo_v2）から変更なし。以下の仕様を引き継ぐ。

#### calc_angle(a, b, c) -> float
- 引数: (x, y) タプル × 3点。b を頂点とする角度を度数法で返す
- numpy の arccos を使用し、np.clip で 0〜180° に保護する

#### PoseEstimator クラス

`process(frame) -> dict | None`
- 左右それぞれ LEFT_HIP/LEFT_KNEE/LEFT_ANKLE/LEFT_SHOULDER、RIGHT_HIP/RIGHT_KNEE/RIGHT_ANKLE/RIGHT_SHOULDER を処理
- 各足の visibility がすべて 0.5 以上のときのみ角度を計算
- 片足のみ検出できた場合も dict を返す（未検出側は None で埋める）
- 両足とも visibility 不足のとき None を返す
- 戻り値の dict 構造:

```python
{
  "left": {
    "knee_angle": float,
    "hip_angle": float,
    "ankle_angle": float,
    "knee_x": float,
    "ankle_x": float,
    "knee_y": float,
    "hip_y": float,
    "shoulder_x": float,
    "shoulder_y": float,
    "hip_x": float,
    "visibility": float,
  } | None,
  "right": { ... 同構造 ... } | None,
  "landmarks": results.pose_landmarks,
}
```

`process_video(path: str) -> list[dict | None]`
- 動画ファイルを受け取り全フレームを処理した結果リストを返す

---

### pose/rep_detector.py

前デモから変更なし。

#### get_representative_angle(frame_result) -> float | None
- 左右両足有効 → 平均 / 片足のみ → その値 / 両足 None → None

#### detect_reps(frame_results) -> list[dict]
- scipy.signal.find_peaks(prominence=20, distance=30) で谷を検出
- 戻り値:

```python
{
  "valley_frame": int,
  "start_frame": int,
  "end_frame": int,
  "min_angle_left": float | None,
  "min_angle_right": float | None,
  "min_angle_repr": float,
}
```

---

### features/extractor.py

#### compute_baseline(frame_result: dict) -> dict | None

1フレームの process() 戻り値から baseline 値を計算する。
frame_result が None のとき None を返す。

baseline dict の構造:

```python
{
  "knee_foot_diff": float,
  "trunk_angle": float,
  "shoulder_hip_dist": float,
}
```

各値の計算方法:

- 左右肩中点 = ((L_SHOULDER.x + R_SHOULDER.x)/2, (L_SHOULDER.y + R_SHOULDER.y)/2)
- 左右腰中点 = ((L_HIP.x + R_HIP.x)/2, (L_HIP.y + R_HIP.y)/2)
- knee_foot_diff = 有効な足の (knee_x - ankle_x) の平均
- trunk_angle = atan2(腰中点.x - 肩中点.x, 肩中点.y - 腰中点.y) を度数法に変換
  （鉛直方向 = 0°、前傾するほど大きくなる）
- shoulder_hip_dist = sqrt((肩中点.x - 腰中点.x)^2 + (肩中点.y - 腰中点.y)^2)

片足が None の場合は有効な側のみで計算する。両足 None なら None を返す。

#### extract_form_features(frame_result: dict | None, baseline: dict) -> dict

1フレームの特徴量を計算して返す。
frame_result が None のとき全値を 0.0 にした dict を返す。

```python
{
  "knee_forward_ratio": float,
  "trunk_lean_delta": float,
  "back_round_ratio": float,
  "left_knee_angle": float,
  "right_knee_angle": float,
  "left_hip_angle": float,
  "right_hip_angle": float,
  "left_ankle_angle": float,
  "right_ankle_angle": float,
  "lr_knee_diff": float,
  "left_visibility": float,
  "right_visibility": float,
}
```

片足が None の場合は該当側の角度を 0.0、visibility を 0.0 にする。

---

### classifier.py

#### 定数（ファイル先頭に定義）

```python
THRESHOLDS = {
    "knee_forward": 1.3,
    "trunk_lean":   15.0,
    "back_round":   0.85,
}
```

#### classify_form(features: dict) -> dict

特徴量 dict を受け取り、3種の崩れフラグと表示色を返す。

```python
{
  "knee_forward": bool,
  "trunk_lean": bool,
  "back_round": bool,
  "any_error": bool,
}
```

判定ロジック:

```python
knee_forward = features["knee_forward_ratio"] > THRESHOLDS["knee_forward"]
trunk_lean = features["trunk_lean_delta"] > THRESHOLDS["trunk_lean"]
back_round = features["back_round_ratio"] < THRESHOLDS["back_round"]
```

---

### visualizer.py

#### draw_overlay(frame, result, form_result, features, rep_count, rep_total, current_frame, rep_start, rep_end) -> frame

以下を OpenCV で描画してください。

1. 左右骨格線: HIP→KNEE→ANKLE を白線（太さ 3px）
2. 関節点: 6点に白い円（半径 6px）
3. 角度テキスト: 左膝の右上に L:87.3° R:88.1° 形式
4. 崩れラベル: フレーム左上に崩れを列挙。崩れなしなら GOOD FORM を緑表示
5. 特徴量スコア表示: kf / tl / br を小さく表示
6. 左右差アラート: lr_knee_diff >= 15° のとき黄色で警告
7. レップカウンタ: 右上に Rep: X / Y
8. 崩れ状態バー: フレーム最下部に3色バーを常時表示
9. Pose not detected: result が None のとき中央に赤で表示

MediaPipe ランドマークは正規化座標なので、ピクセル変換を忘れずに実装してください。

---

### main.py

#### コマンドライン引数

```bash
python main.py
python main.py --input video.mp4
python main.py --save
python main.py --save-frames
python main.py --style HB_NS
python main.py --export-annotation
```

#### 処理フロー

1. cv2.VideoCapture で入力ソースを開く
2. フレームごとに PoseEstimator.process を呼ぶ
3. get_representative_angle で代表角度を計算
4. 暫定 baseline を管理
5. extract_form_features で特徴量を計算
6. classify_form で崩れフラグを取得
7. draw_overlay で描画
8. 暫定レップカウント
9. 終了時に detect_reps でレップ情報を確定
10. 各レップの start_frame から正式 baseline を再計算
11. --save なら output.mp4 を出力
12. --save-frames なら valley_frame を PNG 保存
13. --export-annotation なら annotation_candidates.csv を出力
14. summary.json を保存

---

## 実装上の注意

- calc_angle は np.clip で 0〜180° に保護すること
- MediaPipe ランドマークは正規化座標のためピクセル変換を忘れないこと
- cv2.VideoWriter の fourcc は mp4v を使うこと
- baseline が取れていないフレームでは features を 0.0 とし、崩れなし判定とすること
- リアルタイム時の暫定 baseline と終了後の正式 baseline は別管理にすること
- THRESHOLDS と classify_form は後でモデル推論へ置き換えられるよう、疎結合に実装すること
- 各ファイルに docstring を書くこと

---

## 追加要望

- 既存のコードを壊さずに最小変更で整備すること
- ルールベース判定部分と可視化部分を分離して実装すること
- 今後の LSTM マルチラベル分類へ差し替えやすい形にしておくこと
- LSTM 学習は GPU 優先・CPU フォールバックに対応させること

---

## 最終確認

実装が完了したら、以下を確認してください。

1. Webカメラ / 動画入力で起動できる
2. 3種類の崩れラベルが表示される
3. summary.json と annotation_candidates.csv が生成される
4. ルールベース判定が独立した関数として実装されている
5. 将来の LSTM モデル差し替えに備えた構造になっている
