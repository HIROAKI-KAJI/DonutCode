import os
import cv2
import numpy as np

# ==========================================
# donutcodeのインポート
from donutcode import Encoder, Decoder

# ==========================================
# テスト用設定 
# ==========================================
CONFIG_TYPE ="D-25-11-Compact" #"D-27-13-Compact" #"D-27-13-Robust" #"D-27-13" "D-21-7-Compact" #"
TEST_MESSAGE = "134.2335,133.6387"
OUTPUT_DIR = "sample-result"
OUTPUT_IMAGE = os.path.join(OUTPUT_DIR, f"test_{CONFIG_TYPE}_donut.png")
DEBUG_MAPPING_IMAGE = os.path.join(OUTPUT_DIR, f"01_{CONFIG_TYPE}_mapping_debug.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# スペック情報表示関数
# ==========================================
def print_capacity_info(encoder):
    """エンコーダのコンフィグから容量情報を計算して表示する"""
    config = encoder.config
    
    # CELL_REPEAT を取得 (定義がない場合は1)
    cell_repeat = getattr(config, 'CELL_REPEAT', 1)
    
    # 物理的な空きマスの数を取得
    available_physical_bits = len(config.get_mapping())
    
    # 論理ビット数・バイト数に換算
    available_logical_bits = available_physical_bits // cell_repeat
    total_logical_bytes = available_logical_bits // 8
    
    ecc_bytes = config.ECC_BYTES
    data_area_bytes = total_logical_bytes - ecc_bytes

    # PayloadBuilder に正確な最大文字数を計算させる
    if hasattr(config, 'PAYLOAD_BUILDER'):
        exact_max_chars = config.PAYLOAD_BUILDER.get_max_chars(data_area_bytes)
        codec_name = config.PAYLOAD_BUILDER.codec.__class__.__name__
        header_name = config.PAYLOAD_BUILDER.header_strategy.__class__.__name__
    else:
        exact_max_chars = 0
        codec_name = "不明"
        header_name = "不明"

    print("\n" + "="*40)
    print("DonutCode スペック情報")
    print("="*40)
    print(f" モード(Config)   : {config.__class__.__name__}")
    print(f" グリッドサイズ   : {config.GRID_SIZE} x {config.GRID_SIZE}")
    print(f" セル繰り返し     : {cell_repeat} 回 (空間ダイバーシティ)")
    print("-" * 40)
    print(f" 物理マッピング数 : {available_physical_bits} cells")
    print(f" 論理マッピング数 : {available_logical_bits} bits")
    print(f" 論理総容量       : {total_logical_bytes} bytes")
    print(f" 誤り訂正(ECC)    : {ecc_bytes} bytes")
    print("-" * 40)
    print(f" データ領域       : {data_area_bytes} bytes")
    print(f" 構成コーデック   : {codec_name}")
    print(f" 構成ヘッダ戦略   : {header_name}")
    print(f" 最大埋め込み文字 : 【 {exact_max_chars} 文字 】")
    print("="*40 + "\n")

# ==========================================
# メイン処理
# ==========================================
def main():
    print("===== DonutCode 統合テスト開始 =====")
    
    # ---------------------------------------------------------
    # 1. 画像の生成 (エンコード)
    # ---------------------------------------------------------
    print(f"\n[1] エンコードを実行します (メッセージ: '{TEST_MESSAGE}')")
    try:
        # コンフィグを指定してエンコーダを初期化
        encoder = Encoder(config_type=CONFIG_TYPE)
        
        # ここでスペック情報を表示
        print_capacity_info(encoder)
        
        # デバッグ用マッピング画像の生成
        encoder.save_mapping_debug_image(DEBUG_MAPPING_IMAGE, scale=20, padding=20)
        print(f" -> [デバッグ] マッピング確認画像を生成しました: {DEBUG_MAPPING_IMAGE}")

        # 本番のエンコード画像生成
        matrix = encoder.encode(TEST_MESSAGE)
        encoder.save_image(matrix, OUTPUT_IMAGE, scale=15, hole_color="#ffebee")
        print(f" -> 新しいコード画像の生成に成功しました: {OUTPUT_IMAGE}")
        
    except Exception as e:
        print(f" エンコード中にエラーが発生しました: {e}")
        return

    # ---------------------------------------------------------
    # 2. 画像の解析・デコード
    # ---------------------------------------------------------
    print(f"\n[2] デコードを実行します ('{OUTPUT_IMAGE}' を読み込み)")
    try:
        # コンフィグを指定してデコーダを初期化
        decoder = Decoder(config_type=CONFIG_TYPE)
        
        # 内部で VisionProcessor による補正からデータ復元まで一気通貫で行われます
        result = decoder.decode_image(OUTPUT_IMAGE)
        
        if result:
            print(f"\n 最終デコード成功！ 復元されたデータ: 【 {result} 】")
        else:
            print("\n デコード失敗 (データが見つからないか破損しています)")
            
    except Exception as e:
        print(f"\n デコード処理中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()