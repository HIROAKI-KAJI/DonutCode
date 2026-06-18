import colorsys
from PIL import Image, ImageDraw
from .reedsolomon import _ReedSolomon
from .config import get_config

class Encoder:
    def __init__(self, config_type="D-27-13"):
        self.config = get_config(config_type)
        self.grid_size = self.config.GRID_SIZE
        self.rs = _ReedSolomon()

    def _draw_finder(self, matrix, ox, oy):
        """
        全体で9x9の領域を処理します。
        一番外側の1マスをセパレータ（白）、内側7x7をファインダパタン（黒枠・白枠・黒中心）とします。
        配列の範囲外（はみ出した部分）は無視されるため、3つの角すべてに同じ関数を適用できます。
        """ 
        for dy in range(9):
            for dx in range(9):
                x = ox + dx
                y = oy + dy
                
                # Pythonは -1 を指定すると末尾を書き換えてしまうため、必ず範囲チェックを行う
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    # 1層目(最外周): セパレータ (白:0)
                    if dx == 0 or dx == 8 or dy == 0 or dy == 8:
                        matrix[y][x] = 0
                    # 2層目: ファインダパタン外枠 (黒:1)
                    elif dx == 1 or dx == 7 or dy == 1 or dy == 7:
                        matrix[y][x] = 1
                    # 3層目: ファインダパタン内枠 (白:0)
                    elif dx == 2 or dx == 6 or dy == 2 or dy == 6:
                        matrix[y][x] = 0
                    # 4層目(中心の3x3): (黒:1)
                    else:
                        matrix[y][x] = 1

    def _draw_fixed_patterns(self, matrix):
        """Configの判定メソッドと連携して固定パターンを描画"""
        
        # 1. ファインダパタンの描画 (基準点をはみ出させて9x9を共通描画)
        self._draw_finder(matrix, -1, -1)                              # 左上
        self._draw_finder(matrix, self.grid_size - 8, -1)              # 右上
        self._draw_finder(matrix, -1, self.grid_size - 8)              # 左下

        #　関数が定義されていなければfalseを返す関数にラップ
        is_finder = getattr(self.config, 'is_finder', lambda x, y: False)
        is_alignment = getattr(self.config, 'is_alignment', lambda x, y: False)
        is_timing = getattr(self.config, 'is_timing', lambda x, y: False)

        # 2. その他のパターンの描画
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if is_finder(x, y):
                    continue
                
                elif is_alignment(x, y):
                    # アライメント位置の定義がない場合も考慮して getattr
                    ax, ay = getattr(self.config, 'ALIGNMENT_POS', (0, 0))
                    dx, dy = x - ax, y - ay
                    if dx == 0 or dx == 4 or dy == 0 or dy == 4: matrix[y][x] = 1
                    elif dx == 2 and dy == 2: matrix[y][x] = 1
                    else: matrix[y][x] = 0

                elif is_timing(x, y):
                    matrix[y][x] = 1 if (x if y == 7 else y) % 2 == 0 else 0

    def encode(self, data_str):
        matrix = [[0] * self.grid_size for _ in range(self.grid_size)]
        
        # 固定パターンの描画
        self._draw_fixed_patterns(matrix)

        available_cells = self.config.get_mapping()
        
        # CELL_REPEAT を取得 (定義がないコンフィグの場合は 1 となる)
        cell_repeat = getattr(self.config, 'CELL_REPEAT', 1)

        # 物理マスを cell_repeat で割った数が、論理的な最大バイト数になる
        max_bytes = (len(available_cells) // cell_repeat) // 8
        data_bytes_len = max_bytes - self.config.ECC_BYTES
        
        if data_bytes_len <= 0:
            raise ValueError("データ領域が小さすぎます。")

        # メッセージのエンコード (文字数ヘッダはCodec内で付与されている前提です。)
        # 情報ビットとデータコード語の処理は codecにすべて任せています。
        msg_bytes = self.config.PAYLOAD_BUILDER.encode(data_str)    

        if len(msg_bytes) > data_bytes_len:
            raise ValueError("データが長すぎます。")

        # QR風パディング 231(0xEC) と17(0x11) で埋める。
        padding = bytes([0xEC if i % 2 == 0 else 0x11 for i in range(data_bytes_len - len(msg_bytes))])
        full_msg_bytes = msg_bytes + padding

        # RSエンコード
        encoded_bytes = self.rs.encode(full_msg_bytes, self.config.ECC_BYTES)

        # 論理ビット化 (MSB First)
        logical_bit_stream = [(byte >> i) & 1 for byte in encoded_bytes for i in range(7, -1, -1)]
        
        # 物理セルへの割り当て (ビットのコピー増殖)
        physical_bit_stream = []
        for bit in logical_bit_stream:
            physical_bit_stream.extend([bit] * cell_repeat)
            
        # 余りマスのゼロ埋め
        physical_bit_stream.extend([0] * (len(available_cells) - len(physical_bit_stream)))

        # Configのマッピング順序で配置
        for (x, y), bit in zip(available_cells, physical_bit_stream):
            matrix[y][x] = bit

        return matrix

    def save_mapping_debug_image(self, filename, scale=20, padding=20):
        img = Image.new("RGB", (self.grid_size * scale + 2 * padding, self.grid_size * scale + 2 * padding), "white")
        draw = ImageDraw.Draw(img)
        
        available_cells = self.config.get_mapping()
        
        #  デバッグ画像用にも CELL_REPEAT を取得
        cell_repeat = getattr(self.config, 'CELL_REPEAT', 1)
        
        # 論理的な総バイト数を計算
        total_bytes = (len(available_cells) // cell_repeat) // 8
        
        cell_to_byte_idx = {}
        for bit_idx, (x, y) in enumerate(available_cells):
            # 何番目の物理ビットかをcell_repeatで割り、さらに8で割って論理的なバイトインデックスを算出
            byte_idx = (bit_idx // cell_repeat) // 8
            cell_to_byte_idx[(x, y)] = byte_idx

        def get_gradient_color(idx, total):
            if total <= 0: return (128, 128, 128)
            hue = idx / total
            r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.9)
            return (int(r * 255), int(g * 255), int(b * 255))

        hx, hy, hw, hh = self.config.HOLE_RECT

        # 固定パタンの位置関数がないときはfalseを返す関数にラップ
        is_finder = getattr(self.config, 'is_finder', lambda x, y: False)
        is_alignment = getattr(self.config, 'is_alignment', lambda x, y: False)
        is_timing = getattr(self.config, 'is_timing', lambda x, y: False)

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                box = [x * scale + padding, y * scale + padding, (x + 1) * scale + padding, (y + 1) * scale + padding]
                
                if (x, y) in cell_to_byte_idx:
                    b_idx = cell_to_byte_idx[(x, y)]
                    if b_idx < total_bytes:
                        color = get_gradient_color(b_idx, total_bytes)
                    else:
                        color = (200, 200, 200)
                    draw.rectangle(box, fill=color, outline="white")
                
                elif is_finder(x, y) or is_alignment(x, y) or is_timing(x, y):
                    draw.rectangle(box, fill="black")
                
                elif hx <= x < hx + hw and hy <= y < hy + hh:
                    draw.rectangle(box, fill="#FFE4E1")
                
                else:
                    draw.rectangle(box, fill="#F0F0F0", outline="white")
                    
        img.save(filename)

    def save_image(self, matrix, filename, scale=20, hole_color="white", padding=20):
        size = self.grid_size
        img = Image.new("RGB", (size * scale + 2 * padding, size * scale + 2 * padding), "white")
        draw = ImageDraw.Draw(img)
        for y in range(size):
            for x in range(size):
                val = matrix[y][x]
                box = [x * scale + padding, y * scale + padding, (x + 1) * scale + padding, (y + 1) * scale + padding]
                if val == 1:
                    draw.rectangle(box, fill="black")
                elif val == 0:
                    draw.rectangle(box, fill="white")
                elif val == 2:
                    draw.rectangle(box, fill=hole_color) 
        img.save(filename)