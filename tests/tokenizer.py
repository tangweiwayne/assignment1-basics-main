import regex as re  # 使用 regex 而非内置 re，因为它支持 Unicode 类别（如 \p{L}）
from collections.abc import Iterable

class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None):

        self.vocab = vocab
        self.id2byte = vocab
        self.byte2id = {v:k for k ,v in vocab.items()}
        self.merges = {pair: i for i ,pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        if self.special_tokens:
            sorted_special = sorted(self.special_tokens,key=len,reverse=True)
            special_pat = "|".join(re.escape(t) for t in sorted_special)
            self.special_regex = re.compile(special_pat)
        else:
            self.special_regex = None
        self.gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|re|ve)|?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def encode(self, text: str) -> list[int]:
        pass

    def _encode_text_segment(self, text: str) -> list[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        pass






