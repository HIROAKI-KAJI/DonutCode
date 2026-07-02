"""
DonutCode デコーダ単体テストスクリプト（ハードコード版）

【使い方】
python examples/decoder_test.py
"""

import os
import csv
import glob
from donutcode import Decoder

# ==========================================
# テストケースの定義 
# ==========================================
# テストしたい画像パスと、その画像のコンフィグ名の組み合わせを記述するようにしました。
"""
TEST_CASES = [
    {
        "image_path": "sample-result/test_fresh_donut.png",
        "config": "D-27-13"
    },
    {
        "image_path": "sample-result/test_D-27-13-Robust_donut.png",
        "config": "D-27-13-Robust"
    },
    {
        "image_path": "sample-result/test_D-25-11-Compact_donut.png",
        "config": "D-25-11-Compact"
    },
    # 必要に応じてテストケースを追加・コメントアウトしてください
]
"""

# 同じコーデックで一気にテスト用(analyze_results.pyで可視化するためのCSV出力用)
TEST_IMG_DIR = "detect_test_samples"
TEST_CONFIG = "D-25-11-Compact"

# ==========================================
# 辞書（テストケース一覧）の自動生成処理
# ==========================================
def generate_test_cases(img_dir, config):
    test_cases = []
    
    # 対象とする拡張子（大文字小文字を区別しないように対応）
    extensions = ['*.png', '*.jpg', '*.PNG', '*.JPG', '*.jpeg', '*.JPEG']
    
    # 指定フォルダ内の画像パスを格納するリスト
    image_paths = []
    for ext in extensions:
        # os.path.joinを使って環境に依存しないパスを生成
        search_path = os.path.join(img_dir, ext)
        # globでファイル名を取得し、リストに追加
        image_paths.extend(glob.glob(search_path))
        
    # 順不同にならないよう、ファイル名順でソート（任意）
    image_paths.sort()
    
    # 取得した画像パスから辞書を構成
    for path in image_paths:
        # Windows環境などでバックスラッシュ '\\' になるのを '/' に統一（必要に応じて）
        normalized_path = path.replace(os.sep, '/')
        
        case = {
            "image_path": normalized_path,
            "config": config
        }
        test_cases.append(case)
        
    return test_cases

# 実行
TEST_CASES = generate_test_cases(TEST_IMG_DIR, TEST_CONFIG)


def main():
    print("===== DonutCode Decoder Batch Test =====")
    print(f"登録されたテストケース数: {len(TEST_CASES)}件\n")

    success_count = 0
    
    # CSVに書き込む行データを保持するリスト
    csv_rows = []

    for i, test in enumerate(TEST_CASES, 1):
        image_path = test["image_path"]
        config_name = test["config"]

        # ファイル名のみを取り出す場合は os.path.basename(image_path) を使いますが、
        # ここでは元の画像パス（あるいは画像名）をそのまま記録します。
        image_name = os.path.basename(image_path)

        print(f"[{i}/{len(TEST_CASES)}] 対象画像: {image_path} (コンフィグ: {config_name})")

        # 1. ファイルの存在確認
        if not os.path.exists(image_path):
            print(f"  ->  エラー: 画像ファイルが見つかりません。スキップします。\n")
            # 読み取れたかどうか: エラー、読み取れたデータ: 空文字
            csv_rows.append([image_name, config_name, "エラー（ファイル未検出）", ""])
            continue

        # 2. デコード実行
        try:
            decoder = Decoder(config_type=config_name)
            result = decoder.decode_image(image_path)
            
            if result:
                print(f"  ->  デコード成功: 【 {result} 】\n")
                success_count += 1
                # 読み取れたかどうか: 成功、読み取れたデータ: デコード結果
                csv_rows.append([image_name, config_name, "成功", result])
            else:
                print("  ->  デコード失敗: コードが検出できないか、データが破損しています。\n")
                # 読み取れたかどうか: 失敗、読み取れたデータ: 空文字
                csv_rows.append([image_name, config_name, "失敗", ""])
                
        except Exception as e:
            print(f"  ->  予期せぬエラーが発生しました: {e}\n")
            # 例外が発生した場合も記録
            csv_rows.append([image_name, config_name, f"エラー（{type(e).__name__}）", str(e)])

    # 結果のサマリー表示
    print("===== テスト完了 =====")
    print(f"成功: {success_count} / 全体: {len(TEST_CASES)}")

    # ==========================================
    # CSVファイルの出力処理
    # ==========================================
    output_dir = "test_result"
    os.makedirs(output_dir, exist_ok=True) # フォルダがなければ自動作成
    csv_path = os.path.join(output_dir, "test_results.csv")
    
    # ヘッダーの定義
    headers = ["画像名", "コード形式（コンフィグ）", "読み取れたかどうか", "読み取れたデータ"]
    
    try:
        # Excelでの文字化けを防ぐため utf-8-sig を指定
        with open(csv_path, mode="w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)      # ヘッダーを書き込み
            writer.writerows(csv_rows)    # テスト結果の全行を書き込み
            
        print(f"\n[INFO] CSV結果を保存しました: {csv_path}")
    except Exception as e:
        print(f"\n[ERROR] CSVの保存中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()