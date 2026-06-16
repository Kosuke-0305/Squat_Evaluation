"""CSVから複数クリップをダウンロード（単一動画対象）"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def parse_time_str(value: str) -> float:
    """HH:MM:SS、MM:SS、または秒数をfloatで返す。"""
    value = value.strip()
    if not value:
        return 0.0

    if ':' not in value:
        return float(value)

    parts = value.split(':')
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = 0.0
        minutes, seconds = parts
    else:
        raise ValueError(f"無効な時間形式: {value}")
    return hours * 3600.0 + minutes * 60.0 + seconds


def load_clip_specs(csv_path: str) -> list:
    """CSVからクリップ仕様を読み込む。"""
    specs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 必須フィールド
            video_id = row.get('video_id', '').strip()
            rep_id = row.get('rep_id', '').strip()
            squat_style = row.get('squat_style', '').strip()
            start_str = row.get('start_time_sec', row.get('start_sec', '')).strip()
            end_str = row.get('end_time_sec', row.get('end_sec', '')).strip()

            try:
                start_sec = parse_time_str(start_str)
                end_sec = parse_time_str(end_str)
            except ValueError as exc:
                print(f"警告: 時間の解析に失敗しました: {exc}", file=sys.stderr)
                continue

            # スコア
            valid = row.get('valid', '').strip()
            depth_score = row.get('depth_score', '').strip()
            lockout_score = row.get('lockout_score', '').strip()
            bar_descent_score = row.get('bar_descent', row.get('bar_descent_score', '')).strip()
            bounce_score = row.get('bounce_score', '').strip()
            foot_shift_score = row.get('foot_shift_score', '').strip()

            if start_sec >= end_sec:
                continue

            specs.append({
                'video_id': video_id,
                'rep_id': rep_id,
                'squat_style': squat_style,
                'start_sec': start_sec,
                'end_sec': end_sec,
                'valid': valid,
                'depth_score': depth_score,
                'lockout_score': lockout_score,
                'bar_descent_score': bar_descent_score,
                'bounce_score': bounce_score,
                'foot_shift_score': foot_shift_score,
            })
    return specs


def build_filename(spec: dict) -> str:
    """スペックからファイル名を生成。"""
    parts = [
        spec['video_id'],
        f"rep{spec['rep_id']}",
        spec['squat_style'],
        f"v{spec['valid']}",           # valid
        f"d{spec['depth_score']}",     # depth
        f"l{spec['lockout_score']}",   # lockout
        f"b{spec['bar_descent_score']}", # bar descent
        f"bo{spec['bounce_score']}",   # bounce
        f"f{spec['foot_shift_score']}", # foot shift
    ]
    return "_".join(parts) + ".mp4"


def download_clip(youtube_url: str, start_sec: float, end_sec: float, 
                  output_filename: str, output_dir: str = "clips") -> bool:
    """yt-dlpでクリップをダウンロード。"""
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = Path(output_dir) / output_filename
    
    # yt-dlpコマンド
    cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4]",
        "--download-sections", f"*{start_sec:.2f}-{end_sec:.2f}",
        youtube_url,
        "-o", str(output_path)
    ]
    
    print(f"ダウンロード中: {output_filename}")
    print(f"  時間範囲: {start_sec:.2f}s - {end_sec:.2f}s ({end_sec - start_sec:.2f}s)")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✓ 成功: {output_path}\n")
            return True
        else:
            print(f"✗ 失敗: {result.stderr}", file=sys.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ タイムアウト\n", file=sys.stderr)
        return False
    except Exception as e:
        print(f"✗ エラー: {e}\n", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="CSVから複数のクリップをダウンロード（単一動画対象）"
    )
    parser.add_argument("csv", help="入力CSVファイルパス")
    parser.add_argument("url", help="YouTube URL または video_id")
    parser.add_argument("--output", "-o", default="clips",
                        help="出力ディレクトリ（デフォルト: clips）")
    parser.add_argument("--dry-run", action="store_true",
                        help="実際にはダウンロードせず、対象を表示のみ")
    args = parser.parse_args()
    
    # URLの正規化
    youtube_url = args.url
    if not youtube_url.startswith("http"):
        youtube_url = f"https://www.youtube.com/watch?v={youtube_url}"
    
    # CSVから仕様を読み込む
    specs = load_clip_specs(args.csv)
    if not specs:
        print("エラー: CSVにクリップ情報が見つかりません", file=sys.stderr)
        sys.exit(1)
    
    print(f"対象クリップ数: {len(specs)}\n")
    
    if args.dry_run:
        print("【ドライラン】以下のクリップをダウンロード予定:")
        for spec in specs:
            filename = build_filename(spec)
            print(f"  {filename}")
            print(f"    時間: {spec['start_sec']:.2f}s - {spec['end_sec']:.2f}s")
        return
    
    # ダウンロード実行
    success_count = 0
    for spec in specs:
        filename = build_filename(spec)
        if download_clip(youtube_url, spec['start_sec'], 
                        spec['end_sec'], filename, args.output):
            success_count += 1
    
    print(f"完了: {success_count}/{len(specs)} 成功")


if __name__ == "__main__":
    main()
