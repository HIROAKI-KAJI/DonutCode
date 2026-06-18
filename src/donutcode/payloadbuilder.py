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
    

    def get_max_chars(self, total_data_bytes: int) -> int:
        """
        物理層のデータ領域(total_data_bytes)に対して、
        現在の設定で埋め込み可能な最大文字数を算出する
        """
        if total_data_bytes <= 0:
            return 0
            
        # 1. ヘッダ戦略に問い合わせて、データ本体に使えるバイト数を割り出す
        payload_bytes = self.header_strategy.get_max_payload_bytes(total_data_bytes)
        
        # 2. そのバイト数で何文字いけるかをCodecに問い合わせる
        return self.codec.get_max_chars(payload_bytes)