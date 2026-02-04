import os
import regex as re
from collections import defaultdict,Counter
import json

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    vocab = {i: bytes([i]) for i in range(256)}
    num_merges = vocab_size - 256 - len(special_tokens)
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    special_regex = "|".join([re.escape(t) for t in special_tokens])
    word_chunk = re.split(f"({special_regex})",text)
    train_segments = [chunk for chunk in word_chunk if chunk not in special_tokens]
    gpt2pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""") 
    
    # --- 1. 预处理阶段 & 2. 统计频率 ---
    token_count = Counter()
    for seg in train_segments:
        # seg: "construction and repair"
        raw_tokens = gpt2pat.findall(seg)    
        # raw_tokens: ['construction', ' and', ' repair'] (注意空格)
        
        for raw_token in raw_tokens:
            token = raw_token.encode("utf-8")
            token_count[token] += 1
    # token_count: {b' and': 100, b'the': 80} (统计单词总次数)

    # --- 3. 初始化拆分状态 ---
    token_list={}   
    # token_list: {b'Hi': [b'H', b'i']} (拆分为单字节列表)
    for token in token_count.keys():
        token_list[token] = [bytes([b]) for b in token]

    # --- 4. 统计初始 Pair 频率 ---
    pair_count = defaultdict(int)    # Pair 总频次
    pair_exist = defaultdict(set)   # Pair 倒排索引
    
    for token_name, token_b in token_list.items():
        # token_name: b'Hi', token_b: [b'H', b'i']
        for byte1, byte2 in zip(token_b[:-1], token_b[1:]):
            pair = (byte1, byte2)  
            pair_count[pair] += token_count[token_name]
            # pair_count: {(b'e', b'l'): 150} ("Hello"100次 + "help"50次)
            
            pair_exist[pair].add(token_name)
            # pair_exist: {(b'e', b'l'): [b'Hello', b'help']} (哪些词包含el)

    merge = []
    for j in range(num_merges):
        if not pair_count:
            break   
        max_pair = max(pair_count.items(),key=lambda x :(x[1],x[0]))[0]
        newtoken = max_pair[0]+max_pair[1]
        merge.append(max_pair)
        vocab[256+j] = newtoken
        
        for word_l in pair_exist[max_pair]:
            word = token_list[word_l]
            freq = token_count[word_l]
            i = 0
            while i < len(word):
                if i < len(word)-1 and (word[i],word[i+1]) == max_pair:
                    if i > 0:
                        prev_pair = (word[i-1],word[i])
                        pair_count[prev_pair] -= freq
                        if pair_count[prev_pair] ==0:
                            del pair_count[prev_pair] 
                    if i < len(word) -2:
                        next_pair = (word[i+1],word[i+2])
                        pair_count[next_pair] -= freq
                        if pair_count[next_pair] ==0:
                            del pair_count[next_pair]
                    word[i] = newtoken
                    del word[i+1]
                    if i > 0:
                        prev_pair = (word[i-1],word[i])
                        pair_count[prev_pair] += freq
                        pair_exist[prev_pair].add(word_l)
                    if i < len(word) -1:
                        next_pair = (word[i],word[i+1])
                        pair_count[next_pair] += freq
                        pair_exist[next_pair].add(word_l)
                else:
                    i +=1
            token_list[word_l] = word

        if max_pair in pair_count: 
            del pair_count[max_pair]
        if max_pair in pair_exist: 
            del pair_exist[max_pair]
    for s_tok in special_tokens:
        s_bytes = s_tok.encode("utf-8")
        vocab[len(vocab)] = s_bytes
    
    return(vocab,merge)

def bytes_to_unicode():
    """
    创建一个映射，将 0-255 字节映射为一组可见的 Unicode 字符。
    这是 GPT-2 源码中的标准做法。
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def save_tokenizer_files(vocab, merges, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 初始化映射表
    byte_encoder = bytes_to_unicode()

    # 词表保存
    # 使用 byte_encoder 将 bytes 转换为可见字符串
    json_vocab = {
        k: "".join(byte_encoder[b] for b in v) 
        for k, v in vocab.items()
    }
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(json_vocab, f, indent=4)
    
    # 合并规则保存
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            # 同样转换 p1 和 p2
            s1 = "".join(byte_encoder[b] for b in p1)
            s2 = "".join(byte_encoder[b] for b in p2)
            f.write(f"{s1} {s2}\n")

def main():
    input_path = "data/TinyStoriesV2-GPT4-train.txt" # 你的原始文本路径
    vocab_size = 10000 # 作业要求的词表大小
    # input_path = "data/owt_train.txt" 
    # input_path = "data/chinese.txt" 
    # vocab_size = 1000 # 作业要求的词表大小
    
    special_tokens = ["<|endoftext|>"]
    output_dir = "data/TinyStoriesV2-GPT4-train"

    print(f"开始训练 BPE 分词器 (目标词表大小: {vocab_size})...")
    print("这可能需要几分钟，具体取决于你的 CPU 速度和倒排索引的效率。")
    
    # 调用你之前写好的逻辑
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    
    # 保存结果
    save_tokenizer_files(vocab, merges, output_dir)

if __name__ == "__main__":
    main()