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
        self.gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|re|ve)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        if not self.special_regex:
            return self._encode_text_segment(text)
        token =[]
        last_pos = 0

        for match in self.special_regex.finditer(text):
            pre_text = text[last_pos:match.start()]
            if pre_text:
                token.extend(self._encode_text_segment(pre_text))
            special_tok = match.group()
            token.append(self.byte2id[special_tok.encode('utf-8')])
            last_pos = match.end()

        remaining_text = text[last_pos:]
        if remaining_text:
            token.extend(self._encode_text_segment(remaining_text))
        return token

    def _encode_text_segment(self, text: str) -> list[int]:
        # 内部核心函数：对不含特殊 Token 的纯文本片段应用 BPE 合并逻辑。
        ids =[]
        pre_token = self.gpt2_pat.findall(text)
        # 例如："Hello world!" -> ["Hello", " world", "!"]

        for tokenstr in pre_token:
            byte_part = [bytes([b]) for b in tokenstr.encode('utf-8')]
            # 例如："Hello" -> [b'H', b'e', b'l', b'l', b'o']
            while len(byte_part) >1:
                i = 0
                min_index = float('inf')
                best_pair =None
                for i in range(len(byte_part)-1):
                    pair = (byte_part[i],byte_part[i+1])
                    if pair in self.merges:
                        if self.merges[pair] < min_index:
                            best_pair = pair
                            min_index = self.merges[pair]
                if min_index ==float('inf'):
                    break

                new_part = []
                j = 0
                while j < len(byte_part):
                    if j < len(byte_part)-1 and (byte_part[j],byte_part[j+1]) == best_pair:
                        new_part.append(byte_part[j]+byte_part[j+1])
                        j +=2
                    else:
                        new_part.append(byte_part[j])
                        j +=1
                byte_part = new_part

            ids.extend([self.byte2id[b] for b in byte_part])
        return ids

    def decode(self, ids: list[int]) -> str:
        byte_segments = [self.id2byte[number] for number in ids]
        full_bytes = b''.join(byte_segments)
        return (full_bytes.decode('utf-8',errors="replace"))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
            # 1. 遍历输入的可迭代对象（每次拿到一段文本，如一行）
        for text_chunk in iterable:
            # 2. 调用你现有的 encode 方法处理这一小段
            token_ids = self.encode(text_chunk)
            # 3. 使用 yield from 逐个产出 token，而不是一次性返回列表
            yield from token_ids
    # def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
    #     pass






