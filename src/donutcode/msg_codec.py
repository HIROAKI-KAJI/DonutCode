"""
ここには、情報をビット配列にエンコードするクラスをまとめています。
ここで定義したものをcongig.pyでENCODER = AsciiEncoder　のように設定することでそれぞれのエンコーダを呼び出します。

共通して　メッセージbyteと文字数を返すように設計します。
"""


class AsciiCodec:
    """純粋なASCIIエンコード/デコード"""
    @staticmethod
    def encode(data_str: str) -> bytes:
        return data_str.encode('ascii', errors='ignore')

    @staticmethod
    def decode(data_bytes: bytes) -> str:
        return data_bytes.decode('ascii', errors='ignore')
    
    @staticmethod
    def get_max_chars(max_payload_bytes: int) -> int:
        """与えられたバイト数で表現できる最大文字数を返す"""
        # ASCIIは1文字1バイトなのでそのまま
        return max_payload_bytes


""" 以下はサンプル 作成途中です。"""

class Compact1Codec:
    """
    数字10種と記号5種を4bitに圧縮し、1byteに2文字を入れる高効率コーデック。
    リードソロモンやゼロパディングとの干渉を防ぐため、0000(0)はデータとして使用せず、
    0001(1) 〜 1111(15) の15種類を文字に割り当てます。
    （0000は文字数が奇数の場合の終端パディングとして自動利用されます）
    """
    
    # 文字から4bit整数へのマッピング (1〜15)
    CHAR_TO_INT = {
        '0': 1,  '1': 2,  '2': 3,  '3': 4,  '4': 5, 
        '5': 6,  '6': 7,  '7': 8,  '8': 9,  '9': 10,
        '.': 11, ',': 12, '-': 13, '+': 14, ':': 15
    }
    
    # 4bit整数から文字への逆マッピング (デコード用)
    INT_TO_CHAR = {v: k for k, v in CHAR_TO_INT.items()}

    @staticmethod
    def encode(data_str: str) -> bytes:
        byte_list = bytearray()
        
        # 定義されていない文字はスキップ (AsciiCodecの errors='ignore' と同じ挙動)
        valid_chars = [c for c in data_str if c in Compact1Codec.CHAR_TO_INT]
        
        # 2文字ずつペアにして1バイトにパックする
        for i in range(0, len(valid_chars), 2):
            # 1文字目 (上位4ビット)
            high_nibble = Compact1Codec.CHAR_TO_INT[valid_chars[i]]
            
            # 2文字目 (下位4ビット) - 奇数文字で終わる場合は 0000 (0) で埋める
            if i + 1 < len(valid_chars):
                low_nibble = Compact1Codec.CHAR_TO_INT[valid_chars[i+1]]
            else:
                low_nibble = 0
            
            # 結合して1バイトにする (例: 0011 と 0100 -> 00110100)
            byte_val = (high_nibble << 4) | low_nibble
            byte_list.append(byte_val)
            
        return bytes(byte_list)

    @staticmethod
    def decode(data_bytes: bytes) -> str:
        chars = []
        for b in data_bytes:
            # 上位4ビットと下位4ビットに分解
            high_nibble = (b >> 4) & 0x0F
            low_nibble = b & 0x0F
            
            # 0000はパディング(空き)なので、0以外の時だけ文字に変換する
            if high_nibble != 0:
                chars.append(Compact1Codec.INT_TO_CHAR.get(high_nibble, ''))
            if low_nibble != 0:
                chars.append(Compact1Codec.INT_TO_CHAR.get(low_nibble, ''))
                
        return "".join(chars)

    @staticmethod
    def get_max_chars(max_payload_bytes: int) -> int:
        """1バイトに2文字入るため、最大文字数はバイト数の2倍"""
        return max_payload_bytes * 2

class FixedLengthNumericEncoder:
    """文字数なし・固定長・数字のみの高効率エンコーダの例"""
    @staticmethod
    def encode(data_str: str) -> bytes:
        # 例: 数字のみを対象とし、2桁の数字を1バイト(16進数)にパックする
        packed_bytes = b'' 
        char_count = 0
        
        return packed_bytes, char_count
