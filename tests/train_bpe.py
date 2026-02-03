import os
import regex as re

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {i: bytes([i]) for i in range(256)}
    
    # 将特殊 token (如 <|endoftext|>) 加入词表，避免它们被拆分
    # 注意：特殊 token 的 ID 从 256 开始往后排
    for index, item in enumerate(special_tokens):
        vocab[256 + index] = item.encode('utf-8')
   
    num_merges = vocab_size - len(vocab)
    merge = {}
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    

    
    raw_segments = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            # finditer 返回所有非重叠匹配
            it = re.finditer(PAT, line)
            for match in it:
                raw_segments.append(match.group())
            #长这样['construction', ' and', ' repair', ' of', ' highways', ' and', ' ...']

    byte_segments = [segment.encode("utf-8") for segment in raw_segments]
        #长这样[b'construction', b' and', b' repair', b' of', b' highways', b' and', b' ...']
    segment_count = {}
    for seg in byte_segments:
        if seg in segment_count:
            segment_count[seg] += 1
        else:
            segment_count[seg] =1
        #segment_count   长这样{b'construction': 1,
                            #  b' and': 2,
                            #  b' repair': 1,
                            #  b' of': 1,
                            #  b' highways': 1,
                            #  b' ...': 1}

    splits={}   #初始化：给每一个 b" " 字节流 对应一个 int列表，一开始一byte对应一个int，在训练过程中，合并 相邻int，列表长度变小
                #相当于 原来多少个字母就多少个token，然后合并字母，这个字节流总token数变小
    for seg in segment_count.keys():
        splits[seg] = [b for b in seg]
        #splits  长这样{b'construction': [99, 111, 110, 115, 116, 114, 117, 99, 116, 105, 111, 110],
                    #  b' and': [32, 97, 110, 100],
                    #  b' repair': [32, 114, 101, 112, 97, 105, 114],
                    #  b' of': [32, 111, 102],
                    #  b' highways': [32, 104, 105, 103, 104, 119, 97, 121, 115],
                    #  b' ...': [32, 46, 46, 46]}
    result_merge = []

    for i in range(num_merges):
        pair_count = {}
        for bytes_name, int_list in splits.items():
            # 遍历当前片段中的所有相邻对
            # int_list 是当前片段的 ID 列表，如 [32, 97, 110, 100] -> (32, 97), (97, 110), ...
            for index1, index2 in zip(int_list[:-1], int_list[1:]):
                pair = (index1, index2)
                # 频率 = 该片段出现的次数
                if pair in pair_count:
                    pair_count[pair] += segment_count[bytes_name]
                else:
                    pair_count[pair] = segment_count[bytes_name]
            #pair_count  长这样{(99, 111): 1,
                            #  (111, 110): 2,
                            #  (110, 115): 1,
                            #  (115, 116): 1,
                            #  (116, 114): 1,
                            #  (114, 117): 1,
                            #  (117, 99): 1,}
        max_pair = max(pair_count.items(),key=lambda x :(x[1],x[0]))[0]
        merge[max_pair] = len(vocab) #merge 长这样{(111, 110): 256}
        result_merge.append(max_pair)
        vocab[len(vocab)] = vocab[max_pair[0]]+vocab[max_pair[1]]
        for bytes_name,int_list in splits.items():
            new_list=[]
            j = 0
            while j < len(int_list):
                if j < len(int_list)-1 and (int_list[j],int_list[j+1]) == max_pair:
                    new_list.append(merge[max_pair])
                    j +=2
                else:
                    new_list.append(int_list[j])
                    j +=1
            splits[bytes_name] = new_list
    
    result_merge = [(vocab[index1],vocab[index2]) for index1,index2 in result_merge]

    return(vocab,result_merge)
        
         