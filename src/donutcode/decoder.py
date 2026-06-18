# decoder.py
import cv2
import numpy as np
from .reedsolomon import _ReedSolomon
from .config import get_config
from .vision import VisionProcessor

class Decoder:
    def __init__(self, config_type="D-27-13"):
        self.config = get_config(config_type)
        self.grid_size = self.config.GRID_SIZE
        self.rs = _ReedSolomon()
        self.vision = VisionProcessor(config=self.config)

    def _fix_orientation(self, bit_map):
        """ファインダパタンの位置から画像の正しい向きを判定して回転を補正する"""
        gs = self.grid_size
        fp_template = np.array([
            [1,1,1,1,1,1,1],
            [1,0,0,0,0,0,1],
            [1,0,1,1,1,0,1],
            [1,0,1,1,1,0,1],
            [1,0,1,1,1,0,1],
            [1,0,0,0,0,0,1],
            [1,1,1,1,1,1,1]
        ])

        corners = {
            'TL': bit_map[0:7, 0:7],
            'TR': bit_map[0:7, gs-7:gs],
            'BL': bit_map[gs-7:gs, 0:7],
            'BR': bit_map[gs-7:gs, gs-7:gs]
        }

        # 各角がどれくらいファインダパタンに似ているかスコア化
        match_scores = {name: np.sum(img == fp_template) for name, img in corners.items()}
        print(f"[Debug] 回転判定スコア (TL, TR, BL, BR): {match_scores}")
        
        # 一番ファインダに似ていない角が「右下(BR)」になるように回転
        odd_corner = min(match_scores, key=match_scores.get)
        print(f"[Debug] ファインダのない角: {odd_corner}")

        if odd_corner == 'TR':
            return np.rot90(bit_map, -1)
        elif odd_corner == 'TL':
            return np.rot90(bit_map, 2)
        elif odd_corner == 'BL':
            return np.rot90(bit_map, 1)

        return bit_map

    def image_to_bitmap(self, square_img):
        """サンプリングしてビットマップを生成"""
        gray_crop = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
        _, thresh_crop = cv2.threshold(gray_crop, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        gs = self.grid_size
        cell_size = square_img.shape[0] / gs
        bit_map = np.zeros((gs, gs), dtype=int)

        for row in range(gs):
            for col in range(gs):
                center_x = int((col + 0.5) * cell_size)
                center_y = int((row + 0.5) * cell_size)
                
                # 黒(<128)なら1、白なら0
                if thresh_crop[center_y, center_x] < 128:  
                    bit_map[row, col] = 1
                else:  
                    bit_map[row, col] = 0

        # 回転補正してから返す
        fixed_bit_map = self._fix_orientation(bit_map)

        # デバッグ用のテキスト出力
        with open("sample-result/bit_map_debug.txt", "w") as f:
            f.write("   " + "".join([f"{i:2d}" for i in range(gs)]) + "\n")
            f.write("   " + "-" * (gs * 2) + "\n")
            for y in range(gs):
                # Numpy配列なので必ず [y, x]
                row_str = "".join([f"{fixed_bit_map[y, x]} " for x in range(gs)])
                f.write(f"{y:2d}| {row_str}\n")
        print(" -> [Debug] ビットマップを sample-result/bit_map_debug.txt に保存しました")

        return fixed_bit_map
    
    def _attempt_decode(self, bit_map):
        """与えられたビットマップからバイト列を抽出し、デコードを試みる内部関数"""
        # CELL_REPEAT を取得
        cell_repeat = getattr(self.config, 'CELL_REPEAT', 1)

        # ジグザグマッピングに従って抽出 (Numpyは [y, x] でアクセス)
        raw_bit_stream = [bit_map[y, x] for x, y in self.config.get_mapping()]

        # 多数決による論理ビットの復元
        logical_bit_stream = []
        for i in range(0, len(raw_bit_stream) - (len(raw_bit_stream) % cell_repeat), cell_repeat):
            chunk = raw_bit_stream[i : i + cell_repeat]
            # チャンク内の1の合計が半分以上なら1、それ以外なら0 (多数決)
            vote = 1 if sum(chunk) >= (cell_repeat / 2.0) else 0
            logical_bit_stream.append(vote)

        # バイト列の復元 (MSB First)
        byte_list = bytearray()
        for i in range(0, len(logical_bit_stream) - (len(logical_bit_stream) % 8), 8):
            byte_val = 0
            for j in range(8):
                byte_val = (byte_val << 1) | logical_bit_stream[i + j]
            byte_list.append(byte_val)

        # リードソロモンによる誤り訂正
        decoded_bytes = self.rs.decode(byte_list, self.config.ECC_BYTES)
        msg_bytes = bytes(decoded_bytes)

        # PayloadBuilderによるデータの解釈と文字列化
        return self.config.PAYLOAD_BUILDER.decode(msg_bytes)

    def _decode_from_bit_map(self, bit_map):
        """ビットマップからデコードを試みる関数"""
        # 斜めに撮ったときなど鏡像反転する場合がある（右上と左下が逆になる）ので両方試す。
        try:
            return self._attempt_decode(bit_map)
        except Exception as e:
            print(f"[Debug] 通常方向での読み取り失敗 ({e})。鏡像反転を試みます...")

        try:
            transposed_map = bit_map.T 
            result = self._attempt_decode(transposed_map)
            print(f"[Debug] 反転後読み取りに成功しました！")
            return result
        except Exception as e:
            print(f"[error] 鏡像反転でのデコード失敗: {e}")
            return None
    
    def decode_image(self, img_path: str|np.ndarray):
        try:
            square_img = self.vision.process(img_path, debug_mode=True)
            if square_img is None:
                return None
            bit_map = self.image_to_bitmap(square_img)
            return self._decode_from_bit_map(bit_map)
        except Exception as e:
            print(f"[error] 画像デコード失敗: {e}")
            return None