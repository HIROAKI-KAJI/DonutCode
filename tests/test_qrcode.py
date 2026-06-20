import os
import glob
import cv2
import pandas as pd
from pyzbar.pyzbar import decode

# ==========================================
# 設定
# ==========================================
# QRコードのテスト画像が入っているディレクトリを指定してください
INPUT_DIR = "detect_test_samples"
OUTPUT_CSV = "test_results_QRCode_v2_Q.csv"

OUTPUT_FOLDER = "test_result"
# 可視化プログラムで表示される名前
CONFIG_NAME = "QRCode_Standard" 

def main():
    print("===== QR Code Benchmark Test =====")
    
    # 対象の画像ファイルを取得 (png, jpgなど)
    search_pattern = os.path.join(INPUT_DIR, "*.*")
    image_files = [f for f in glob.glob(search_pattern) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"エラー: '{INPUT_DIR}' に画像ファイルが見つかりません。")
        return

    print(f"合計 {len(image_files)} 件のテスト画像を処理します...\n")

    results = []

    for i, img_path in enumerate(image_files, 1):
        filename = os.path.basename(img_path)
        
        # 画像の読み込み
        img = cv2.imread(img_path)
        if img is None:
            print(f"[{i}/{len(image_files)}] ⚠️ 読み込み失敗: {filename}")
            continue

        try:
            # pyzbarによるデコード処理
            decoded_objects = decode(img)
            
            if decoded_objects:
                # 読み取れた場合 (複数ある場合は最初の1つを採用)
                raw_data = decoded_objects[0].data
                try:
                    text_data = raw_data.decode('utf-8')
                except UnicodeDecodeError:
                    # UTF-8でデコードできないバイナリの場合はHex表現にする
                    text_data = raw_data.hex()
                    
                print(f"[{i}/{len(image_files)}] ✅ 成功: {filename} -> {text_data}")
                results.append({
                    "画像名": filename,
                    "コード形式（コンフィグ）": CONFIG_NAME,
                    "読み取れたかどうか": "成功",
                    "読み取れたデータ": text_data
                })
            else:
                # 読み取れなかった場合
                print(f"[{i}/{len(image_files)}] ❌ 失敗: {filename}")
                results.append({
                    "画像名": filename,
                    "コード形式（コンフィグ）": CONFIG_NAME,
                    "読み取れたかどうか": "失敗",
                    "読み取れたデータ": None
                })
                
        except Exception as e:
            print(f"[{i}/{len(image_files)}] ⚠️ エラー: {filename} ({e})")
            results.append({
                "画像名": filename,
                "コード形式（コンフィグ）": CONFIG_NAME,
                "読み取れたかどうか": "失敗",
                "読み取れたデータ": None
            })

    # ==========================================
    # CSVへの書き出し (DonutCodeと互換性のあるフォーマット)
    # ==========================================
    df = pd.DataFrame(results, columns=["画像名", "コード形式（コンフィグ）", "読み取れたかどうか", "読み取れたデータ"])

    output_file = OUTPUT_FOLDER + "/" + OUTPUT_CSV
    df.to_csv(output_file, index=False, encoding='utf-8')

    print("\n===== ベンチマーク完了 =====")
    success_count = len(df[df['読み取れたかどうか'] == '成功'])
    print(f"全体成功率: {success_count} / {len(image_files)} ({success_count/len(image_files)*100:.1f}%)")
    print(f"結果を保存しました: {output_file}")
    print("-> 先ほどの 'analyze_results.py' を実行すると、QRCodeのグラフが生成されます！")

if __name__ == "__main__":
    main()