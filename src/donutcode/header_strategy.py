"""
ペイロードのヘッダー情報を作る方法を定義しています。
"""

class OneByteLengthHeader:
    """先頭1バイトにデータ長(バイト数)を記録するヘッダ"""
    @staticmethod
    def add_header(data_bytes: bytes) -> bytes:
        length = len(data_bytes)
        if length > 255:
            raise ValueError("データ長が255バイトを超えています。")
        return bytes([length]) + data_bytes

    @staticmethod
    def extract_data(payload_bytes: bytes) -> bytes:
        if not payload_bytes:
            return b""
        length = payload_bytes[0]
        # 先頭1バイトを読み飛ばし、length分だけ切り出す（パディングを無視）
        return payload_bytes[1 : 1 + length]

class NoHeaderFixedLength:
    """ヘッダなし（固定長など、外部で長さを管理する前提）"""
    def __init__(self, fixed_length: int):
        self.fixed_length = fixed_length

    def add_header(self, data_bytes: bytes) -> bytes:
        # ヘッダは付けず、必要なら長さを切り詰めるかパディングする
        return data_bytes[:self.fixed_length]

    def extract_data(self, payload_bytes: bytes) -> bytes:
        # 指定された固定長だけを切り出す
        return payload_bytes[:self.fixed_length]