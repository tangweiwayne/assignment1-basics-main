import regex as re

def debug_bpe_logic():
    # 1. 模拟 init_vocab (展示前 5 个和后 5 个)
    init_vocab = {i: bytes([i]) for i in range(256)}
    print(f"init_vocab size: {len(init_vocab)}")
    print(f"init_vocab sample (0-4): {[init_vocab[i] for i in range(5)]}")
    print(f"init_vocab sample (65-70 'A'-'F'): {[init_vocab[i] for i in range(65, 71)]}")

    # 2. 定义正则 PAT (GPT-2 pattern)
    # 说明：
    # '(?:[sdmt]|ll|ve|re)  -> 匹配缩写，如 's, 'd, 'll, 've, 're
    # ?\p{L}+               -> 匹配单词（前面可选空格），\p{L} 是 Unicode 字母
    # ?\p{N}+               -> 匹配数字（前面可选空格）
    # ?[^\s\p{L}\p{N}]+     -> 匹配标点符号（非空格、非字母、非数字）
    # \s+(?!\S)             -> 匹配尾部空白
    # \s+                   -> 匹配其他空白
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # 3. 模拟输入文本
    sample_text = "Hello world! It's a GPT-2 style test. 12345  running."
    print(f"\nOriginal Text: '{sample_text}'")
    
    # 4. 执行分割
    pretokenized_text_list = []
    # 模拟逐行读取（这里只有一行）
    lines = [sample_text]
    
    for line in lines:
        it = re.finditer(PAT, line)
        for match in it:
            pretokenized_text_list.append(match.group())
            
    # 5. 展示结果
    print("\nSplit Results (Tokens):")
    for i, token in enumerate(pretokenized_text_list):
        print(f"[{i}]: '{token}'")

if __name__ == "__main__":
    debug_bpe_logic()
