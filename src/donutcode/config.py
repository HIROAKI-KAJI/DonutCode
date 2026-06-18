# config.py

from .msg_codec import AsciiCodec, Compact1Codec
from .payloadbuilder import PayloadBuilder
from .header_strategy import OneByteLengthHeader


class Config_D_27_13:
    GRID_SIZE = 27
    HOLE_RECT = (7, 7, 13, 13)
    ECC_BYTES = 24  # データエリアが小さいため、ECCは多めに取る
    PAYLOAD_BUILDER = PayloadBuilder(
        codec=AsciiCodec(),
        header_strategy=OneByteLengthHeader()
    ) # データのエンコーダとデコーダ（ascii)

    # アライメントパターンの左上座標
    ALIGNMENT_POS = (21, 21)

    @classmethod
    def is_finder(cls, x, y):
        gs = cls.GRID_SIZE
        if 0 <= x < 8 and 0 <= y < 8: return True
        if gs - 8 <= x < gs and 0 <= y < 8: return True
        if 0 <= x < 8 and gs - 8 <= y < gs: return True
        return False

    @classmethod
    def is_hole(cls, x, y):
        hx, hy, hw, hh = cls.HOLE_RECT
        return hx <= x < hx + hw and hy <= y < hy + hh

    @classmethod
    def is_alignment(cls, x, y):
        ax, ay = cls.ALIGNMENT_POS
        return ax <= x < ax + 5 and ay <= y < ay + 5
    


    @classmethod
    def get_mapping(cls):
        """右下から2列ずつ左へ進むジグザグスキャンの座標リストを取得"""
        available = []
        gs = cls.GRID_SIZE
        upward = True  
        
        for x_base in range(gs - 1, -1, -2):
            x_coords = [x_base, x_base - 1] if x_base > 0 else [x_base]
            y_range = range(gs - 1, -1, -1) if upward else range(gs)
            
            for y in y_range:
                for x in x_coords:
                    # 予約領域はスキップ
                    if cls.is_finder(x, y): continue
                    if cls.is_hole(x, y): continue
                    if cls.is_alignment(x, y): continue
                    
                    available.append((x, y))
            upward = not upward
            
        return available


class Config_D_27_13_Compact:
    """
    D-27-13の物理形状はそのままに、
    数字と一部記号に特化した高圧縮コーデック(Compact1Codec)を使用するモード
    """
    GRID_SIZE = 27

    HOLE_RECT = (7, 7, 13, 13)
    ECC_BYTES = 24  # データエリアが小さいため、ECCは多めに取る
    
    # AsciiCodec から Compact1Codec に差し替え
    PAYLOAD_BUILDER = PayloadBuilder(
        codec=Compact1Codec(),
        header_strategy=OneByteLengthHeader()
    )

    # アライメントパターンの左上座標
    ALIGNMENT_POS = (21, 21)

    @classmethod
    def is_finder(cls, x, y):
        gs = cls.GRID_SIZE
        if 0 <= x < 8 and 0 <= y < 8: return True
        if gs - 8 <= x < gs and 0 <= y < 8: return True
        if 0 <= x < 8 and gs - 8 <= y < gs: return True
        return False

    @classmethod
    def is_hole(cls, x, y):
        hx, hy, hw, hh = cls.HOLE_RECT
        return hx <= x < hx + hw and hy <= y < hy + hh

    @classmethod
    def is_alignment(cls, x, y):
        ax, ay = cls.ALIGNMENT_POS
        return ax <= x < ax + 5 and ay <= y < ay + 5
    
    @classmethod
    def get_mapping(cls):
        """右下から2列ずつ左へ進むジグザグスキャンの座標リストを取得"""
        available = []
        gs = cls.GRID_SIZE
        upward = True  
        
        for x_base in range(gs - 1, -1, -2):
            x_coords = [x_base, x_base - 1] if x_base > 0 else [x_base]
            y_range = range(gs - 1, -1, -1) if upward else range(gs)
            
            for y in y_range:
                for x in x_coords:
                    # 予約領域はスキップ
                    if cls.is_finder(x, y): continue
                    if cls.is_hole(x, y): continue
                    if cls.is_alignment(x, y): continue
                    
                    available.append((x, y))
            upward = not upward
            
        return available


class Config_D_25_11_Compact:
    """
    穴の大きさと、低解像度耐性を両立したベストバランスモデル。
    CELL_REPEAT(セル巨大化)は使わず、Vision層がファインダを検出しやすい均等なドット配置を維持。
    その代わり、余った容量をすべてリードソロモン(ECC)に注ぎ込み、
    画像がぼやけてデータがボロボロになっても数学的に復元できるようにしています。
    """
    GRID_SIZE = 25
    
    # 25x25のキャンバスに対して、11x11の十分な穴を中央に確保
    HOLE_RECT = (7, 7, 11, 11)
    
    # 物理総容量 39バイト のうち、なんと 24バイト(約61%) をエラー訂正に全振り！
    ECC_BYTES = 24  
    
    PAYLOAD_BUILDER = PayloadBuilder(
        codec=Compact1Codec(),
        header_strategy=OneByteLengthHeader()
    )

    # アライメントパターンの左上座標 (25x25に合わせて調整)
    ALIGNMENT_POS = (19, 19)

    @classmethod
    def is_finder(cls, x, y):
        gs = cls.GRID_SIZE
        if 0 <= x < 8 and 0 <= y < 8: return True
        if gs - 8 <= x < gs and 0 <= y < 8: return True
        if 0 <= x < 8 and gs - 8 <= y < gs: return True
        return False

    @classmethod
    def is_hole(cls, x, y):
        hx, hy, hw, hh = cls.HOLE_RECT
        return hx <= x < hx + hw and hy <= y < hy + hh

    @classmethod
    def is_alignment(cls, x, y):
        ax, ay = cls.ALIGNMENT_POS
        return ax <= x < ax + 5 and ay <= y < ay + 5

    @classmethod
    def get_mapping(cls):
        available = []
        gs = cls.GRID_SIZE
        upward = True  
        
        for x_base in range(gs - 1, -1, -2):
            x_coords = [x_base, x_base - 1] if x_base > 0 else [x_base]
            y_range = range(gs - 1, -1, -1) if upward else range(gs)
            
            for y in y_range:
                for x in x_coords:
                    if cls.is_finder(x, y): continue
                    if cls.is_hole(x, y): continue
                    if cls.is_alignment(x, y): continue
                    available.append((x, y))
            upward = not upward
            
        return available
   
class Config_D_21_7_Compact:
    """
    DonutCode 最小サイズのコンフィグ (21x21, 7x7穴)
    アライメントパターンありの小さいサイズ版。
    Compact1Codecとの組み合わせにより、極小サイズながら位置情報に十分な容量を確保。
    """
    GRID_SIZE = 21
    # 中心の穴
    HOLE_RECT = (6, 6, 9, 9) 
    
    ECC_BYTES = 9  

    PAYLOAD_BUILDER = PayloadBuilder(
        codec=Compact1Codec(),
        header_strategy=OneByteLengthHeader()
    )

    # アライメントパターンの左上座標
    ALIGNMENT_POS = (15, 15)

    @classmethod
    def is_finder(cls, x, y):
        # 8x8のファインダパタン (左上、右上、左下)
        gs = cls.GRID_SIZE
        if 0 <= x < 8 and 0 <= y < 8: return True
        if gs - 8 <= x < gs and 0 <= y < 8: return True
        if 0 <= x < 8 and gs - 8 <= y < gs: return True
        return False

    @classmethod
    def is_hole(cls, x, y):
        hx, hy, hw, hh = cls.HOLE_RECT
        return hx <= x < hx + hw and hy <= y < hy + hh

    @classmethod
    def is_timing(cls, x, y):
        return False

    @classmethod
    def is_alignment(cls, x, y):
        ax, ay = cls.ALIGNMENT_POS
        return ax <= x < ax + 5 and ay <= y < ay + 5

    @classmethod
    def get_mapping(cls):
        """右下から2列ずつ左へ進むジグザグスキャンの座標リストを取得"""
        available = []
        gs = cls.GRID_SIZE
        upward = True  
        
        for x_base in range(gs - 1, -1, -2):
            x_coords = [x_base, x_base - 1] if x_base > 0 else [x_base]
            y_range = range(gs - 1, -1, -1) if upward else range(gs)
            
            for y in y_range:
                for x in x_coords:
                    # 予約領域はスキップ
                    if cls.is_finder(x, y): continue
                    if cls.is_hole(x, y): continue
                    if cls.is_timing(x, y): continue
                    if cls.is_alignment(x, y): continue
                    
                    available.append((x, y))
            upward = not upward
            
        return available


class Config_D_27_13_Robust:
    """
    D-27-13の物理形状はそのままに、低解像度・ピンボケに強い大きい１セルの拡大は位置を採用したモード。
    CELL_REPEAT=2 により、1つの論理ビットを物理的な2セルにコピーして配置します。
    デコード時の「多数決」により、RS誤り訂正の前にカメラの読み取りミスを自己修復します。
    """
    GRID_SIZE = 27
    HOLE_RECT = (7, 7, 13, 13)
    
    # 堅牢化のための追加設定
    CELL_REPEAT = 3  
    ECC_BYTES = 4  # 多数決の時点で大半のエラーが消えるため、ECCは少なめでOK
    
    PAYLOAD_BUILDER = PayloadBuilder(
        codec=Compact1Codec(),
        header_strategy=OneByteLengthHeader()
    )

    # アライメントパターンの左上座標
    ALIGNMENT_POS = (21, 21)

    @classmethod
    def is_finder(cls, x, y):
        gs = cls.GRID_SIZE
        if 0 <= x < 8 and 0 <= y < 8: return True
        if gs - 8 <= x < gs and 0 <= y < 8: return True
        if 0 <= x < 8 and gs - 8 <= y < gs: return True
        return False

    @classmethod
    def is_hole(cls, x, y):
        hx, hy, hw, hh = cls.HOLE_RECT
        return hx <= x < hx + hw and hy <= y < hy + hh

    @classmethod
    def is_alignment(cls, x, y):
        ax, ay = cls.ALIGNMENT_POS
        return ax <= x < ax + 5 and ay <= y < ay + 5

    @classmethod
    def get_mapping(cls):
        """右下から2列ずつ左へ進むジグザグスキャンの座標リストを取得"""
        available = []
        gs = cls.GRID_SIZE
        upward = True  
        
        for x_base in range(gs - 1, -1, -2):
            x_coords = [x_base, x_base - 1] if x_base > 0 else [x_base]
            y_range = range(gs - 1, -1, -1) if upward else range(gs)
            
            for y in y_range:
                for x in x_coords:
                    # 予約領域はスキップ
                    if cls.is_finder(x, y): continue
                    if cls.is_hole(x, y): continue
                    if cls.is_alignment(x, y): continue
                    
                    available.append((x, y))
            upward = not upward
            
        return available




## これはひな型で残しています。 ========================
class Config_D_27_13_OLD:
    """
    タイミングパタンなどを残した、ひな型です。
    """
    GRID_SIZE = 27
    HOLE_RECT = (7, 7, 13, 13)
    ECC_BYTES = 24  # データエリアが小さいため、ECCは多めに取る
    PAYLOAD_BUILDER = PayloadBuilder(
        codec=AsciiCodec(),
        header_strategy=OneByteLengthHeader()
    ) # データのエンコーダとデコーダ（ascii)

    # アライメントパターンの左上座標
    ALIGNMENT_POS = (21, 21)

    @classmethod
    def is_finder(cls, x, y):
        gs = cls.GRID_SIZE
        if 0 <= x < 8 and 0 <= y < 8: return True
        if gs - 8 <= x < gs and 0 <= y < 8: return True
        if 0 <= x < 8 and gs - 8 <= y < gs: return True
        return False

    @classmethod
    def is_hole(cls, x, y):
        hx, hy, hw, hh = cls.HOLE_RECT
        return hx <= x < hx + hw and hy <= y < hy + hh

    @classmethod
    def is_alignment(cls, x, y):
        ax, ay = cls.ALIGNMENT_POS
        return ax <= x < ax + 5 and ay <= y < ay + 5
    
    @classmethod
    def is_timing(cls, x, y):
        if y == 7 and 8 <= x <= cls.GRID_SIZE - 9: return True
        if x == 7 and 8 <= y <= cls.GRID_SIZE - 9: return True
        return False


    @classmethod
    def get_mapping(cls):
        """右下から2列ずつ左へ進むジグザグスキャンの座標リストを取得"""
        available = []
        gs = cls.GRID_SIZE
        upward = True  
        
        for x_base in range(gs - 1, -1, -2):
            x_coords = [x_base, x_base - 1] if x_base > 0 else [x_base]
            y_range = range(gs - 1, -1, -1) if upward else range(gs)
            
            for y in y_range:
                for x in x_coords:
                    # 予約領域はスキップ
                    if cls.is_finder(x, y): continue
                    if cls.is_hole(x, y): continue
                    if cls.is_alignment(x, y): continue
                    if cls.is_timing(x, y): continue
                    
                    available.append((x, y))
            upward = not upward
            
        return available

CONFIG_REGISTRY = {
    "D-27-13": Config_D_27_13,
    "D-27-13-Compact": Config_D_27_13_Compact,
    "D-21-7-Compact": Config_D_21_7_Compact,
    "D-25-11-Compact": Config_D_25_11_Compact,
    "D-27-13-Robust": Config_D_27_13_Robust,
}

def get_config(type_name):
    if type_name not in CONFIG_REGISTRY:
        raise ValueError(f"未定義のコンフィグタイプです: {type_name}")
    return CONFIG_REGISTRY[type_name]