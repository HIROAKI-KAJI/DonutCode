import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 設定
# ==========================================
#INPUT_FILE = "./test_result/test_results_QRCode_v2_L.csv"  # CSVファイル
INPUT_FILE = "./test_result/test_results_QRCode_v2_Q.csv"  # CSVファイル
OUTPUT_BASE_DIR = "test_result"  # グラフなどを出力するベースフォルダ

def parse_filename(filename):
    """画像名からテスト条件(パラメータ)を抽出する"""
    # サイズ (例: size100 -> 100)
    size = re.search(r'size(\d+)', filename)
    size = int(size.group(1)) if size else 0
    
    # Z軸回転 (例: z30 -> 30)
    rot_z = re.search(r'z(\d+)', filename)
    rot_z = int(rot_z.group(1)) if rot_z else 0
    
    # X軸ホモグラフィ(横の傾き) (例: x30 -> 30)
    homo_x = re.search(r'x(\d+)', filename)
    homo_x = int(homo_x.group(1)) if homo_x else 0
    
    # Y軸ホモグラフィ(縦の傾き) (例: y50 -> 50)
    homo_y = re.search(r'y(\d+)', filename)
    homo_y = int(homo_y.group(1)) if homo_y else 0
    
    return pd.Series([size, rot_z, homo_x, homo_y])

def create_visualizations(df, config_name, output_dir):
    print(f"[{config_name}] のデータを解析中...")
    
    # 1. データの前処理
    # 画像名からパラメータを抽出して新しい列にする
    df[['size', 'rot_z', 'homo_x', 'homo_y']] = df['画像名'].apply(parse_filename)
    
    # '成功' なら 1, '失敗' なら 0 として数値化 (成功率の計算用)
    df['is_success'] = df['読み取れたかどうか'].apply(lambda x: 1 if x == '成功' else 0)

    # ---------------------------------------------------------
    # 2. サイズ(解像度)ごとの成功率グラフ
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))
    size_success = df.groupby('size')['is_success'].mean() * 100
    sns.barplot(x=size_success.index, y=size_success.values, palette="Blues_d")
    plt.title(f"Success Rate by Image Size - {config_name}")
    plt.xlabel("Image Size (px)")
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 100)
    for i, v in enumerate(size_success.values):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
    plt.savefig(os.path.join(output_dir, f"{config_name}_size_success.png"), bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # 3. Z軸回転角度ごとの成功率グラフ
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))
    rot_success = df.groupby('rot_z')['is_success'].mean() * 100
    sns.barplot(x=rot_success.index, y=rot_success.values, palette="Blues_d") # こちらも統一感のためにBlues_dに変更しています
    plt.title(f"Success Rate by Z-Rotation - {config_name}")
    plt.xlabel("Z-Rotation Angle (deg)")
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 100)
    for i, v in enumerate(rot_success.values):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
    plt.savefig(os.path.join(output_dir, f"{config_name}_rotation_success.png"), bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # 4. ホモグラフィ(X, Y傾き)のヒートマップ
    # ---------------------------------------------------------
    plt.figure(figsize=(7, 6))
    # X傾きとY傾きの組み合わせによる平均成功率をピボットテーブル化
    homo_pivot = df.pivot_table(values='is_success', index='homo_y', columns='homo_x', aggfunc='mean') * 100
    
    # 🌟 ここを変更: cmap='RdYlGn' を cmap='Blues' に変更
    sns.heatmap(homo_pivot, annot=True, cmap='Blues', fmt=".1f", vmin=0, vmax=100)
    
    plt.title(f"Success Rate Heatmap by Homography (X, Y) - {config_name}")
    plt.xlabel("Homo X (Horizontal Tilt)")
    plt.ylabel("Homo Y (Vertical Tilt)")
    # Y軸を反転させて、上が0になるようにする
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(output_dir, f"{config_name}_homography_heatmap.png"), bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # 5. 詳細な統計サマリーの出力 (CSV)
    # ---------------------------------------------------------
    summary_path = os.path.join(output_dir, f"{config_name}_summary.csv")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"=== Config: {config_name} ===\n")
        f.write(f"Total Trials: {len(df)}\n")
        f.write(f"Overall Success Rate: {df['is_success'].mean() * 100:.1f}%\n\n")
        f.write("--- Success Rate by Size ---\n")
        f.write(size_success.to_csv(header=["Success Rate (%)"]))
        f.write("\n--- Success Rate by Rotation ---\n")
        f.write(rot_success.to_csv(header=["Success Rate (%)"]))
        f.write("\n--- Success Rate by Homography X & Y ---\n")
        f.write(homo_pivot.to_csv())
    
    print(f"  -> 可視化データの保存が完了しました: {output_dir}")

def main():

    filename = os.path.basename(INPUT_FILE)
    
    # ファイル名からコンフィグ名を抽出 (マッチしない場合は "General" とする)
    match = re.search(r'test_results_(.+)\.csv', filename)
    if match:
        config_name = match.group(1)
    else:
        # test_results.csv などの場合、中身のカラムから取得を試みる
        try:
            temp_df = pd.read_csv(INPUT_FILE)
            config_name = temp_df['コード形式（コンフィグ）'].iloc[0]
        except Exception:
            config_name = "General"

    # 出力先ディレクトリの作成 (例: test_results/D-25-11-Compact/)
    output_dir = os.path.join(OUTPUT_BASE_DIR, config_name)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(INPUT_FILE)
        create_visualizations(df, config_name, output_dir)
    except Exception as e:
        print(f" {filename} の処理中にエラーが発生しました: {e}")

    print("\n===== すべての可視化処理が完了しました =====")

if __name__ == "__main__":
    main()