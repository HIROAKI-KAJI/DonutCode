"""
ここには、情報をビット配列にエンコードするクラスをまとめています。
ここで定義したものをcongig.pyでENCODER = AsciiEncoder　のように設定することでそれぞれのエンコーダを呼び出します。

共通して　メッセージbyteと文字数を返すように設計します。
"""


class AsciiCodec:
    """ASCII専用のエンコード・デコード処理"""
    
    @staticmethod
    def encode(data_str: str):
        msg_bytes = data_str.encode('ascii', errors='ignore')
        char_count = len(data_str)
        return msg_bytes, char_count

    @staticmethod
    def decode(msg_bytes: bytes, char_count: int = None) -> str:
        # 文字数(char_count)が指定されている場合は、その長さまでスライスするなどの処理を入れる
        # パディング(0xEC, 0x11など)はここの入力の前に切って置く
        if char_count is not None:
            msg_bytes = msg_bytes[:char_count]
            
        return msg_bytes.decode('ascii', errors='ignore')


""" 以下はサンプル 作成途中です。"""

class CustomCompactEncoder:
    """文字数ヘッダ付きや独自の圧縮を行うエンコーダの例"""
    @staticmethod
    def encode(data_str: str) -> bytes:
        # 例：数字だけなら2文字で1バイト(BCD表現など)にする、といった圧縮処理
        # ここでは「先頭1バイトに文字数を入れ、以降をASCIIにする」例
        length = len(data_str)
        payload = data_str.encode('ascii', errors='ignore')
        char_count = 0
        # 文字数(1バイト) + ペイロード
        return bytes([length]) + payload, char_count

class FixedLengthNumericEncoder:
    """文字数なし・固定長・数字のみの高効率エンコーダの例"""
    @staticmethod
    def encode(data_str: str) -> bytes:
        # 例: 数字のみを対象とし、2桁の数字を1バイト(16進数)にパックする
        packed_bytes = b'' 
        char_count = 0
        
        return packed_bytes, char_count
