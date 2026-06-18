class PayloadBuilder:
    """CodecとHeaderStrategyを組み合わせてペイロードを管理する"""
    def __init__(self, codec, header_strategy):
        self.codec = codec
        self.header_strategy = header_strategy

    def encode(self, data_str: str) -> bytes:
        # 1. 文字列 -> バイト列
        raw_bytes = self.codec.encode(data_str)
        # 2. バイト列 -> ヘッダ付きバイト列
        return self.header_strategy.add_header(raw_bytes)

    def decode(self, payload_bytes: bytes) -> str:
        # 1. ヘッダ付きバイト列 -> データ本体のバイト列（パディング除去）
        raw_bytes = self.header_strategy.extract_data(payload_bytes)
        # 2. バイト列 -> 文字列
        return self.codec.decode(raw_bytes)